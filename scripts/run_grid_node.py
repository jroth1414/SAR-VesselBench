"""Parallel grid queue for a multi-GPU node (one cell per GPU).

Same cell list and skip-if-finished semantics as the dev-card queue
(scripts/run_grid_queue.py), but cells are dispatched to a pool of GPUs —
one training process per selected device via CUDA_VISIBLE_DEVICES. All cells
are independent and fill GPUs greedily, most expensive fractions first so
stragglers finish together.

The active V100 campaign uses ``--micro-batch 16`` (effective batch 16) under
the shared ``32-true`` recipe. Select only GPUs reserved for this job; there is
intentionally no implicit all-GPU default. GPU IDs are container-local (from
``nvidia-smi -L``), not physical lease IDs. Example:
    python scripts/run_grid_node.py --gpus 0 1 2 3 4 5 6 7 --micro-batch 16
Resumable: kill and relaunch freely.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from src.eval.result_contract import ResultContractError, load_completion_marker

SHORT = {
    "vit_random": "vitrand",
    "satdino_b": "satdino",
    "sarmae_b": "sarmae",
    "vit_imagenet": "vitin1k",
    "cnn_random": "cnnrand",
    "bigearthnet_s2": "beS2",
    "bigearthnet_s1": "beS1",
    "cnn_imagenet": "cnnin1k",
}
ALL_ARMS = [
    "vit_random", "satdino_b", "sarmae_b", "vit_imagenet",
    "cnn_random", "bigearthnet_s2", "bigearthnet_s1", "cnn_imagenet",
]
FRACS = [1.0, 0.5, 0.25, 0.1]  # remaining work: expensive first
PRIORITY_CELLS = [
    ("vit_imagenet", 0.1),
    ("cnn_imagenet", 0.1),
]
REPO = Path(__file__).resolve().parents[1]
DETECTOR_PATH = REPO / "configs" / "detector.yaml"
EXPECTED_DETECTOR_SHA256 = hashlib.sha256(DETECTOR_PATH.read_bytes()).hexdigest()
EXPECTED_PRECISION = yaml.safe_load(DETECTOR_PATH.read_text())["schedule"]["precision"]
CANDIDATE_FLOOR = yaml.safe_load(DETECTOR_PATH.read_text())["decode"]["candidate_floor"]
EXPECTED_GIT_SHA = subprocess.run(
    ["git", "-c", f"safe.directory={REPO}", "rev-parse", "HEAD"],
    cwd=REPO,
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()


def log(message: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def exp_id(init: str, frac: float, seed: int = 0) -> str:
    return f"{SHORT[init]}-f{int(round(frac * 100))}-s{seed}"


def cell_done(
    init: str,
    frac: float,
    runs_root: Path = Path("runs"),
) -> bool:
    """Validate a completed cell marker; malformed markers are a hard STOP."""

    final = runs_root / exp_id(init, frac) / "final_metrics.json"
    if not final.exists():
        return False
    try:
        load_completion_marker(
            final,
            candidate_floor=CANDIDATE_FLOOR,
            expected_recipe={
                "exp_id": exp_id(init, frac),
                "git_sha": EXPECTED_GIT_SHA,
                "detector_sha256": EXPECTED_DETECTOR_SHA256,
                "precision": EXPECTED_PRECISION,
                "micro_batch": 16,
                "gradient_accumulation": 1,
                "effective_batch": 16,
            },
        )
    except ResultContractError as exc:
        raise RuntimeError(
            f"invalid schema-2 completion marker contents or recipe: {final}"
        ) from exc
    return True


def matrix_cells() -> list[tuple[str, float]]:
    """Exact seed-0 core order, with the two replacement f10 cells first."""

    cells = list(PRIORITY_CELLS)
    cells.extend(
        (init, frac)
        for frac in FRACS
        for init in ALL_ARMS
        if (init, frac) not in PRIORITY_CELLS
    )
    return cells


def build_work(runs_root: Path = Path("runs")) -> list[dict]:
    return [
        {"init": init, "frac": frac}
        for init, frac in matrix_cells()
        if not cell_done(init, frac, runs_root)
    ]


def launch(
    item: dict,
    gpu: int,
    *,
    micro_batch: int,
    workers: int,
) -> subprocess.Popen:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    name = exp_id(item["init"], item["frac"])
    argv = [
        sys.executable,
        "-m",
        "src.train.finetune",
        "--init",
        item["init"],
        "--label_frac",
        str(item["frac"]),
        "--seed",
        "0",
        "--workers",
        str(workers),
        "--micro-batch",
        str(micro_batch),
    ]
    log(f"gpu{gpu}: launching {name}")
    log_dir = Path("runs/logs/node")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{name}.log").open("a") as out:
        return subprocess.Popen(
            argv,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
        )


def validate_hardware_args(gpus: list[int], micro_batch: int) -> None:
    if len(set(gpus)) != len(gpus):
        raise ValueError("--gpus contains duplicate container-local device IDs")
    if any(gpu < 0 for gpu in gpus):
        raise ValueError("--gpus must contain non-negative container-local IDs")
    if micro_batch != 16:
        raise ValueError("active V100 campaign requires --micro-batch 16")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    try:
        validate_hardware_args(args.gpus, args.micro_batch)
    except ValueError as exc:
        parser.error(str(exc))

    running: dict[int, tuple[subprocess.Popen, dict]] = {}
    failure_seen = False
    while True:
        for gpu in list(running):
            proc, item = running[gpu]
            if proc.poll() is None:
                continue
            name = exp_id(item["init"], item["frac"])
            valid_marker = False
            if proc.returncode == 0:
                try:
                    valid_marker = cell_done(item["init"], item["frac"])
                except RuntimeError as exc:
                    log(f"gpu{gpu}: {name}: {exc}")
            if proc.returncode != 0 or not valid_marker:
                failure_seen = True
                log(
                    f"gpu{gpu}: {name} FAILED (exit {proc.returncode}); "
                    "no new cells will launch"
                )
            else:
                log(f"gpu{gpu}: {name} complete")
            del running[gpu]

        if failure_seen:
            if not running:
                log("NODE QUEUE STOPPED AFTER FAILURE")
                return 1
            time.sleep(30)
            continue

        work = [
            item
            for item in build_work()
            if not any(r_item == item for _, r_item in running.values())
        ]
        free_gpus = [gpu for gpu in args.gpus if gpu not in running]
        for gpu in free_gpus:
            if not work:
                break
            item = work.pop(0)
            running[gpu] = (
                launch(
                    item,
                    gpu,
                    micro_batch=args.micro_batch,
                    workers=args.workers,
                ),
                item,
            )
        if not running and not work:
            log("NODE QUEUE DRAINED")
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
