"""Sequential grid-cell queue for the active V100 recipe (Phase 4/5).

Runs recipe-conform grid cells (exact frozen detector.yaml: batch 16, early
stopping, plan-literal epochs) one at a time, cheapest fractions first so the
partial label-efficiency curves fill in early. Resumable: a cell whose
``final_metrics.json`` exists is skipped, so kill/relaunch is always safe.

Order: f10 x 8 -> f25 x 8 -> f50 x 8 -> f100 x 8. All eight arms are
ordinary random/downloaded initializations; no prerequisite training jobs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
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


def build_queue() -> list[dict]:
    return [
        {"init": arm, "frac": frac}
        for frac in (0.1, 0.25, 0.5, 1.0)
        for arm in ALL_ARMS
    ]


def cell_done(exp: str, runs_root: Path = Path("runs")) -> bool:
    """Validate a completed cell marker; malformed markers are a hard STOP."""

    final = runs_root / exp / "final_metrics.json"
    if not final.exists():
        return False
    try:
        load_completion_marker(
            final,
            candidate_floor=CANDIDATE_FLOOR,
            expected_recipe={
                "exp_id": exp,
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


def log(message: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def main() -> int:
    for item in build_queue():
        exp = f"{SHORT[item['init']]}-f{int(round(item['frac'] * 100))}-s0"
        final = Path("runs") / exp / "final_metrics.json"
        if cell_done(exp):
            log(f"{exp}: done — skip")
            continue
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
            "4",
            "--micro-batch",
            "16",
        ]
        log(f"{exp}: starting (init {item['init']}, frac {item['frac']})")
        code = subprocess.run(argv).returncode
        if code != 0:
            log(f"{exp}: FAILED with code {code} — queue stopped")
            return code
        if not cell_done(exp):
            log(f"{exp}: process exited 0 without a valid completion marker")
            return 1
        payload = json.loads(final.read_text())
        log(
            f"{exp}: done — best dev F1 {payload.get('best_dev_f1')}, "
            f"epochs {payload.get('epochs_run')}"
        )
    log("QUEUE DRAINED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
