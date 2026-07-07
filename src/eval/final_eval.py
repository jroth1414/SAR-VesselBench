"""FINAL EVAL — the once-only verified-scene evaluation (DEVPLAN P6.1).

The ~50 human-verified validation scenes (``eval_final`` in the frozen
splits.json) are touched EXACTLY ONCE, by this script, after the grid is
complete. Tripwires (ground rule 4):

- refuses to run without ``--i-am-sure``;
- refuses to run if the lockfile ``runs/final_eval.lock`` already exists —
  the eval has then already happened and MUST NOT be repeated (re-running
  after seeing numbers is tuning on the test of tests);
- writes the timestamped lockfile BEFORE scoring begins, so even a crashed
  attempt counts as the one touch (surface to a human, do not delete the
  lockfile yourself);
- entry preconditions are checked hard: grid complete (P5 acceptance),
  dev-selected thresholds present for every evaluated cell.

Scores the best config per study arm at the 10%, 25% and 100% cells on the
eval_final scenes with each cell's FROZEN dev threshold, through the SACRED
scorer, including the dark-vessel-recall and near-shore slices (dark GT
exists only in these scenes). Output: ``runs/summary/final_verified.csv``.
Nothing is tuned after this.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Sequence

LOCKFILE = Path("runs/final_eval.lock")
EVAL_FRACS = (0.1, 0.25, 1.0)


def main(argv: Sequence[str] | None = None) -> int:
    import pandas as pd
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--i-am-sure",
        action="store_true",
        help="required: this eval runs exactly once, ever",
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--arms-config", default="configs/arms.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if not args.i_am_sure:
        raise SystemExit(
            "REFUSING: the verified-scene eval runs exactly once. Re-invoke "
            "with --i-am-sure only when the grid is complete and reviewed "
            "(make final-eval CONFIRM=1)."
        )
    if LOCKFILE.exists():
        raise SystemExit(
            f"REFUSING: {LOCKFILE} exists — the once-only eval has already "
            "been performed. If you believe this is an error, STOP and "
            "consult a human; do not delete the lockfile."
        )

    data_cfg = yaml.safe_load(Path(args.data_config).read_text())
    det_cfg = yaml.safe_load(Path(args.detector_config).read_text())
    arms_cfg = yaml.safe_load(Path(args.arms_config).read_text())

    # Entry preconditions (P6): every evaluated cell finished with a frozen
    # dev threshold. Missing cells are a hard refusal, not a warning.
    cells = []
    missing = []
    for init_name, meta in arms_cfg["arms"].items():
        for frac in EVAL_FRACS:
            exp = f"{meta['short']}-f{int(round(frac * 100))}-s0"
            final = Path("runs") / exp / "final_metrics.json"
            if not final.exists():
                missing.append(exp)
                continue
            payload = json.loads(final.read_text())
            threshold = (payload.get("last_dev") or {}).get("threshold")
            checkpoint = Path("runs") / exp / "checkpoints" / "best.ckpt"
            if threshold is None or not checkpoint.exists():
                missing.append(exp)
                continue
            cells.append((exp, init_name, frac, threshold, checkpoint))
    if missing:
        raise SystemExit(
            f"REFUSING: {len(missing)} grid cells incomplete ({missing[:6]}…) "
            "— the final eval runs only after the grid is done (P6 preconditions)."
        )

    splits = json.loads(Path(data_cfg["paths"]["splits"]).read_text())["splits"]
    eval_scenes = splits["eval_final"]
    raw_root = Path(data_cfg["paths"]["raw_xview3"]) / "GRD"
    absent = [s for s in eval_scenes if not (raw_root / s / "VH_dB.tif").exists()]
    if absent:
        raise SystemExit(
            f"REFUSING: {len(absent)} eval_final scenes not extracted "
            f"(first: {absent[:3]}) — extract the validation archives first."
        )

    # --- point of no return: write the lockfile BEFORE scoring ---
    LOCKFILE.parent.mkdir(parents=True, exist_ok=True)
    LOCKFILE.write_text(
        json.dumps(
            {
                "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "cells": [c[0] for c in cells],
                "n_scenes": len(eval_scenes),
            },
            indent=1,
        ),
        newline="\n",
    )
    print(f"lockfile written: {LOCKFILE} — this is the one touch (ground rule 4)")

    from src.eval.infer_scene import ground_truth_from_labels, infer_scene
    from src.eval.scorer import score_dataset
    from src.eval.threshold import apply_threshold
    from src.train.lit_modules import HeatmapLitModule

    labels = pd.read_csv(
        Path(data_cfg["paths"]["raw_xview3"]) / "labels" / "validation.csv"
    )
    stats = json.loads(Path(data_cfg["paths"]["stats"]).read_text())

    gt_by_scene = {
        scene_id: ground_truth_from_labels(
            labels[labels["scene_id"] == scene_id].to_dict(orient="records")
        )
        for scene_id in eval_scenes
    }

    rows = []
    for exp, init_name, frac, threshold, checkpoint in cells:
        print(f"[final eval] {exp}")
        module = HeatmapLitModule.load_from_checkpoint(
            str(checkpoint), map_location=args.device
        ).eval()
        pred_by_scene = {
            scene_id: infer_scene(
                module,
                raw_root / scene_id,
                stats=stats,
                tau=det_cfg["decode"]["candidate_floor"],
                d_nms_m=det_cfg["decode"]["d_nms_m"],
                tile_px=det_cfg["eval"]["tile_px"],
                tile_stride_px=det_cfg["eval"]["tile_stride_px"],
                batch_size=det_cfg["eval"]["infer_batch"],
                device=args.device,
            )
            for scene_id in eval_scenes
        }
        result = score_dataset(gt_by_scene, apply_threshold(pred_by_scene, threshold))
        dark = result.slices["dark"]
        near = result.slices["near_shore"]
        rows.append(
            {
                "exp_id": exp,
                "init": init_name,
                "label_frac": frac,
                "threshold": threshold,
                "f1": result.aggregate.f1,
                "precision": result.aggregate.precision,
                "recall": result.aggregate.recall,
                "dark_recall": dark.recall,
                "dark_support": dark.tp + dark.fn,
                "near_shore_f1": near.f1,
            }
        )
        del module

    out = Path("runs/summary/final_verified.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"FINAL verified-scene results -> {out}. Nothing is tuned after this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
