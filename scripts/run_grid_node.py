"""Parallel grid queue for the 8x V100 node (one cell per GPU).

Same cell list and skip-if-finished semantics as the dev-card queue
(scripts/run_grid_queue.py), but cells are dispatched to a pool of GPUs —
one training process per V100 via CUDA_VISIBLE_DEVICES (the plan's "one
config per card per night", DEVPLAN §12). Dependencies are honored: the two
LS-SSDD pretrainings run before any vit_supervised / cnn_supervised cell;
everything else is independent and fills GPUs greedily, most expensive
fractions first so stragglers finish together.

V100 notes (Appendix C): plain recipe batch 16 (32 GB — no micro-batching),
fp16 + GradScaler, no bf16/FlashAttention. Run from the repo root:
    python scripts/run_grid_node.py --gpus 0 1 2 3 4 5 6 7
Resumable: kill and relaunch freely.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SHORT = {
    "vit_random": "vitrand",
    "satdino_b": "satdino",
    "sarmae_b": "sarmae",
    "vit_supervised": "vitsup",
    "cnn_random": "cnnrand",
    "bigearthnet_s2": "beS2",
    "bigearthnet_s1": "beS1",
    "cnn_supervised": "cnnsup",
}
DOWNLOADED_ARMS = [
    "vit_random", "satdino_b", "sarmae_b",
    "cnn_random", "bigearthnet_s2", "bigearthnet_s1",
]
SUPERVISED_ARMS = ["vit_supervised", "cnn_supervised"]
SUPERVISED_CKPT = {
    "vit_supervised": Path("runs/vitsup-lsssdd/checkpoints/best.ckpt"),
    "cnn_supervised": Path("runs/cnnsup-lsssdd/checkpoints/best.ckpt"),
}
FRACS = [1.0, 0.5, 0.25, 0.1]  # expensive first: stragglers finish together


def log(message: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def exp_id(init: str, frac: float, seed: int = 0) -> str:
    return f"{SHORT[init]}-f{int(round(frac * 100))}-s{seed}"


def cell_done(init: str, frac: float) -> bool:
    return (Path("runs") / exp_id(init, frac) / "final_metrics.json").exists()


def pretrain_done(backbone: str) -> bool:
    exp = "vitsup-lsssdd" if backbone == "vit" else "cnnsup-lsssdd"
    return (Path("runs") / exp / "final_metrics.json").exists()


def build_work() -> list[dict]:
    work: list[dict] = []
    for backbone in ("vit", "cnn"):
        if not pretrain_done(backbone):
            work.append({"kind": "pretrain", "backbone": backbone})
    for frac in FRACS:
        for init in DOWNLOADED_ARMS + SUPERVISED_ARMS:
            if not cell_done(init, frac):
                work.append({"kind": "cell", "init": init, "frac": frac})
    return work


def ready(item: dict) -> bool:
    if item["kind"] == "cell" and item["init"] in SUPERVISED_CKPT:
        return SUPERVISED_CKPT[item["init"]].exists()
    return True


def launch(item: dict, gpu: int) -> subprocess.Popen:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if item["kind"] == "pretrain":
        name = f"pretrain-{item['backbone']}"
        argv = [sys.executable, "-m", "src.train.pretrain_supervised", "--backbone", item["backbone"]]
    else:
        name = exp_id(item["init"], item["frac"])
        argv = [
            sys.executable, "-m", "src.train.finetune",
            "--init", item["init"], "--label_frac", str(item["frac"]),
            "--seed", "0", "--workers", "4",
        ]
        if item["init"] in SUPERVISED_CKPT:
            argv += ["--supervised-checkpoint", str(SUPERVISED_CKPT[item["init"]])]
    log(f"gpu{gpu}: launching {name}")
    log_dir = Path("runs/logs/node")
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / f"{name}.log", "a")
    return subprocess.Popen(argv, env=env, stdout=out, stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()

    running: dict[int, tuple[subprocess.Popen, dict]] = {}
    while True:
        # reap finished
        for gpu in list(running):
            proc, item = running[gpu]
            if proc.poll() is not None:
                name = item.get("init") or f"pretrain-{item['backbone']}"
                log(f"gpu{gpu}: {name} exited with {proc.returncode}")
                del running[gpu]

        work = [
            item for item in build_work()
            if ready(item)
            and not any(r_item == item for _, r_item in running.values())
        ]
        free_gpus = [g for g in args.gpus if g not in running]
        for gpu in free_gpus:
            if not work:
                break
            item = work.pop(0)
            running[gpu] = (launch(item, gpu), item)
        if not running and not work:
            log("NODE QUEUE DRAINED")
            return 0
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
