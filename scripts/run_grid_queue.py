"""Sequential grid-cell queue for the dev card (Phase 4/5, option B).

Runs recipe-conform grid cells (exact frozen detector.yaml: batch 16, early
stopping, plan-literal epochs) one at a time, cheapest fractions first so the
partial label-efficiency curves fill in early. Resumable: a cell whose
``final_metrics.json`` exists is skipped, so kill/relaunch is always safe.

Order: f10 x 8 -> f25 x 8 -> f50 x 8 -> f100 x 8. All eight arms are
ordinary random/downloaded initializations; no prerequisite training jobs.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import sys
from pathlib import Path

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
        payload = json.loads(final.read_text())
        best_dev_f1 = float(payload["best_dev_f1"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid completion marker: {final}") from exc
    if payload.get("exp_id") != exp or not math.isfinite(best_dev_f1):
        raise RuntimeError(f"invalid completion marker contents: {final}")
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
        ]
        if item["init"].startswith(("cnn", "bigearthnet")):
            # ConvNeXt at the recipe batch overflows the 16 GB dev card
            # into shared memory (~20x slowdown); micro-batch 8 with
            # gradient accumulation keeps the effective batch at 16.
            argv += ["--micro-batch", "8"]
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
