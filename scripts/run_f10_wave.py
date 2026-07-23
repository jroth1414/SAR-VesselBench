"""One-shot gated runner: finish the two remaining f10 cells, then STOP.

Owner instruction (2026-07-23): another project's training currently holds
the dev GPU — hold off until it finishes, complete the f10 wave for arms 1-8
(only vitin1k-f10-s0 and cnnin1k-f10-s0 remain), then pause for owner review
before any f25/f50/f100 cell launches. The full resumable queue
(scripts/run_grid_queue.py) is NOT started by this script.

Gate: the other run's completion marker exists AND GPU memory stays below
the free threshold for three consecutive checks.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path

MARKER = Path(
    r"C:\Users\Admin\Documents\JHU-Computer-Vision"
    r"\module9\assignment5\vision_transformer\train_all_done.marker"
)
VRAM_FREE_MIB = 4096
GATE_CHECKS = 3
GATE_INTERVAL_S = 60

CELLS = [
    ("vit_imagenet", "vitin1k-f10-s0", []),
    # ConvNeXt on the 16 GB dev card: micro-batch 8 x accumulation 2 keeps
    # the recipe's effective batch 16 (see run_grid_queue.py).
    ("cnn_imagenet", "cnnin1k-f10-s0", ["--micro-batch", "8"]),
]


def log(message: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def cell_done(exp: str, runs_root: Path = Path("runs")) -> bool:
    """Same validation as run_grid_queue.py: malformed markers are a hard STOP."""

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


def wait_for_free_gpu() -> None:
    log(f"gate: waiting for {MARKER.name} + GPU below {VRAM_FREE_MIB} MiB")
    consecutive = 0
    while consecutive < GATE_CHECKS:
        if MARKER.exists() and gpu_used_mib() < VRAM_FREE_MIB:
            consecutive += 1
        else:
            consecutive = 0
        time.sleep(GATE_INTERVAL_S)
    log("gate: GPU free and other training complete")


def main() -> int:
    wait_for_free_gpu()
    for init, exp, extra in CELLS:
        if cell_done(exp):
            log(f"{exp}: done — skip")
            continue
        argv = [
            sys.executable, "-m", "src.train.finetune",
            "--init", init, "--label_frac", "0.1", "--seed", "0",
            "--workers", "4", *extra,
        ]
        log(f"{exp}: starting (init {init})")
        code = subprocess.run(argv).returncode
        if code != 0:
            log(f"{exp}: FAILED with code {code} — stopped")
            return code
        if not cell_done(exp):
            log(f"{exp}: process exited 0 without a valid completion marker")
            return 1
        payload = json.loads((Path("runs") / exp / "final_metrics.json").read_text())
        log(
            f"{exp}: done — best dev F1 {payload.get('best_dev_f1')}, "
            f"epochs {payload.get('epochs_run')}"
        )
    log("F10 WAVE COMPLETE (arms 1-8) — PAUSED for owner review; "
        "no f25/f50/f100 cell was launched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
