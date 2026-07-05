"""P3.6 early-signal checks: each downloaded backbone vs its track's floor.

Runs the four downloaded arms (satdino_b, sarmae_b, bigearthnet_s1,
bigearthnet_s2) plus both floors (vit_random, cnn_random) through the shared
fine-tune loop at 100% labels under IDENTICAL reduced settings (dev-card
budget: batch 8, capped samples/epoch, few epochs, full 8-scene dev eval at
the end). The check: every downloaded arm trains stably and beats its
track's floor at equal epochs — a catastrophic channel-format/stem mismatch
shows up here before the grid burns node-nights (DEVPLAN P3.6).

The frozen detector.yaml stays authoritative for the real grid; these
overrides exist only so the six short runs share one reduced budget.
Resumable: an arm whose final_metrics.json exists is skipped.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ARMS = [
    "vit_random",
    "satdino_b",
    "sarmae_b",
    "cnn_random",
    "bigearthnet_s2",
    "bigearthnet_s1",
]
SUFFIX = "p36"
SETTINGS = [
    "--label_frac", "1.0",
    "--epochs", "5",
    "--batch-size", "8",
    "--samples-per-epoch", "24000",
    "--dev-every", "5",
    "--n-dev-scenes", "8",
    "--exp-suffix", SUFFIX,
    "--workers", "4",
]
SHORT = {
    "vit_random": "vitrand", "satdino_b": "satdino", "sarmae_b": "sarmae",
    "cnn_random": "cnnrand", "bigearthnet_s2": "beS2", "bigearthnet_s1": "beS1",
}


def log(message: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def main() -> int:
    results: dict[str, dict] = {}
    for arm in ARMS:
        run_dir = Path("runs") / f"{SHORT[arm]}-f100-s0-{SUFFIX}"
        final = run_dir / "final_metrics.json"
        if final.exists():
            log(f"{arm}: final_metrics.json exists — skip")
            results[arm] = json.loads(final.read_text())
            continue
        log(f"{arm}: starting")
        proc = subprocess.run(
            [sys.executable, "-m", "src.train.finetune", "--init", arm, *SETTINGS],
        )
        if proc.returncode != 0:
            log(f"{arm}: FAILED with code {proc.returncode} — continuing with the rest")
            continue
        results[arm] = json.loads(final.read_text())
        log(f"{arm}: done — dev {results[arm]['last_dev']}")

    summary_path = Path("runs") / "p36_summary.json"
    summary_path.write_text(json.dumps(results, indent=1), newline="\n")

    log("=== P3.6 summary (dev F1 @ equal reduced budget) ===")
    for track, floor, downloaded in (
        ("ViT", "vit_random", ("satdino_b", "sarmae_b")),
        ("CNN", "cnn_random", ("bigearthnet_s2", "bigearthnet_s1")),
    ):
        floor_f1 = results.get(floor, {}).get("last_dev", {}).get("f1")
        log(f"{track} floor {floor}: dev F1 {floor_f1}")
        for arm in downloaded:
            arm_f1 = results.get(arm, {}).get("last_dev", {}).get("f1")
            verdict = "?" if None in (arm_f1, floor_f1) else ("BEATS floor" if arm_f1 > floor_f1 else "does NOT beat floor")
            log(f"  {arm}: dev F1 {arm_f1} — {verdict}")
    log(f"summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
