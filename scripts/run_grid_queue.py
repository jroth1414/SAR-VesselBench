"""Sequential grid-cell queue for the dev card (Phase 4/5, option B).

Runs recipe-conform grid cells (exact frozen detector.yaml: batch 16, early
stopping, plan-literal epochs) one at a time, cheapest fractions first so the
partial label-efficiency curves fill in early. Resumable: a cell whose
``final_metrics.json`` exists is skipped, so kill/relaunch is always safe.

Order: f10 x 6 -> yolo-train -> pretrain-vit -> pretrain-cnn (P5.1)
-> vitsup/cnnsup f10 catch-up -> f25 x 8 -> f50 x 8 -> f100 x 8.
Arms 4/8 cells are skipped with a warning if their pretraining checkpoint
is missing (e.g. its run failed) rather than crashing the queue.
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


def build_queue() -> list[dict]:
    queue: list[dict] = []
    for arm in DOWNLOADED_ARMS:
        queue.append({"kind": "cell", "init": arm, "frac": 0.1})
    queue.append({"kind": "yolo-train"})
    queue.append({"kind": "pretrain", "backbone": "vit"})
    queue.append({"kind": "pretrain", "backbone": "cnn"})
    for arm in SUPERVISED_ARMS:
        queue.append({"kind": "cell", "init": arm, "frac": 0.1})
    for frac in (0.25, 0.5, 1.0):
        for arm in DOWNLOADED_ARMS + SUPERVISED_ARMS:
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
            argv = [
                sys.executable, "-m", "src.train.finetune",
                "--init", item["init"],
                "--label_frac", str(item["frac"]),
                "--seed", "0",
                "--workers", "4",
            ]
            if item["init"].startswith(("cnn", "bigearthnet")):
                # ConvNeXt at the recipe batch overflows the 16 GB dev card
                # into shared memory (~20x slowdown); micro-batch 8 with
                # gradient accumulation keeps the effective batch at 16.
                argv += ["--micro-batch", "8"]
            if item["init"] in SUPERVISED_CKPT:
                checkpoint = SUPERVISED_CKPT[item["init"]]
                if not checkpoint.exists():
                    log(f"{exp}: SKIPPED — pretraining checkpoint missing ({checkpoint})")
                    continue
                argv += ["--supervised-checkpoint", str(checkpoint)]
            log(f"{exp}: starting (init {item['init']}, frac {item['frac']})")
            code = subprocess.run(argv).returncode
            if code != 0:
                log(f"{exp}: FAILED with code {code} — continuing with next cell")
                continue
            payload = json.loads(final.read_text())
            log(f"{exp}: done — best dev F1 {payload.get('best_dev_f1')}, epochs {payload.get('epochs_run')}")
        elif item["kind"] == "pretrain":
            exp = "vitsup-lsssdd" if item["backbone"] == "vit" else "cnnsup-lsssdd"
            if (Path("runs") / exp / "final_metrics.json").exists():
                log(f"{exp}: done — skip")
                continue
            log(f"{exp}: starting (P5.1 LS-SSDD supervised pretraining)")
            argv = [
                sys.executable, "-m", "src.train.pretrain_supervised",
                "--backbone", item["backbone"],
            ]
            if item["backbone"] == "cnn":
                argv += ["--micro-batch", "8"]  # 16 GB dev card; node uses plain recipe
            code = subprocess.run(argv).returncode
            log(f"{exp}: finished with code {code}")
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
