"""Sequential grid-cell queue for the dev card (Phase 4/5, option B).

Runs recipe-conform grid cells (exact frozen detector.yaml: batch 16, early
stopping, plan-literal epochs) one at a time, cheapest fractions first so the
partial label-efficiency curves fill in early. Resumable: a cell whose
``final_metrics.json`` exists is skipped, so kill/relaunch is always safe.
The special ``yolo-train`` item runs the R2 reference training between the
f10 and f25 waves.

Order: f10 x 6 arms -> yolo-train -> f25 x 6 -> f50 x 6 -> f100 x remaining.
(Arms 4/8 join in Phase 5 after the LS-SSDD pretrainings.)
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

SHORT = {
    "vit_random": "vitrand",
    "satdino_b": "satdino",
    "sarmae_b": "sarmae",
    "cnn_random": "cnnrand",
    "bigearthnet_s2": "beS2",
    "bigearthnet_s1": "beS1",
}
ARMS = ["vit_random", "satdino_b", "sarmae_b", "cnn_random", "bigearthnet_s2", "bigearthnet_s1"]


def build_queue() -> list[dict]:
    queue: list[dict] = []
    for frac in (0.1,):
        for arm in ARMS:
            queue.append({"kind": "cell", "init": arm, "frac": frac})
    queue.append({"kind": "yolo-train"})
    for frac in (0.25, 0.5, 1.0):
        for arm in ARMS:
            queue.append({"kind": "cell", "init": arm, "frac": frac})
    return queue


def log(message: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def main() -> int:
    for item in build_queue():
        if item["kind"] == "cell":
            exp = f"{SHORT[item['init']]}-f{int(round(item['frac'] * 100))}-s0"
            final = Path("runs") / exp / "final_metrics.json"
            if final.exists():
                log(f"{exp}: done — skip")
                continue
            log(f"{exp}: starting (init {item['init']}, frac {item['frac']})")
            code = subprocess.run(
                [
                    sys.executable, "-m", "src.train.finetune",
                    "--init", item["init"],
                    "--label_frac", str(item["frac"]),
                    "--seed", "0",
                    "--workers", "4",
                ]
            ).returncode
            if code != 0:
                log(f"{exp}: FAILED with code {code} — continuing with next cell")
                continue
            payload = json.loads(final.read_text())
            log(f"{exp}: done — best dev F1 {payload.get('best_dev_f1')}, epochs {payload.get('epochs_run')}")
        else:
            marker = Path("runs/yolo26-f100/weights/best.pt")
            if marker.exists():
                log("yolo-train: best.pt exists — skip")
                continue
            log("yolo-train: starting (R2 reference)")
            code = subprocess.run(
                [sys.executable, "-m", "src.references.yolo26_ref", "train"]
            ).returncode
            log(f"yolo-train: finished with code {code}")
    log("QUEUE DRAINED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
