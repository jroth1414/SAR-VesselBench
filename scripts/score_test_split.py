"""P5.4 test-split scoring for completed grid cells.

For every run named in the Section-12 manifest whose ``final_metrics.json``
exists but lacks ``test_f1``: load its best checkpoint, decode the 16 frozen
test scenes at the candidate floor, apply the run's DEV-SELECTED threshold
verbatim (frozen operating point — no re-selection on test, P2.2b), score
through the SACRED scorer, and write the aggregate + dark-recall +
near-shore-F1 slices back into ``final_metrics.json``. ``curves.py collect``
then picks up ``test_f1`` automatically.

Resumable and cheap to re-run: cells already carrying ``test_f1`` are
skipped. Runs on whatever GPU is free — safe alongside training (uses
~2 GB VRAM in short bursts) but nicer to run between queue waves.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Sequence

from scripts.h100.contracts import atomic_write_json


def log(message: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def score_run(
    run_dir: Path,
    *,
    data_cfg: dict,
    det_cfg: dict,
    detector_sha256: str,
    device: str,
) -> dict | None:
    import pandas as pd
    import torch

    from src.eval.infer_scene import ground_truth_from_labels, infer_scene
    from src.eval.scorer import score_dataset
    from src.eval.threshold import apply_threshold
    from src.train.lit_modules import HeatmapLitModule

    final_path = run_dir / "final_metrics.json"
    payload = json.loads(final_path.read_text())
    expected_precision = det_cfg["schedule"]["precision"]
    if (
        payload.get("precision") != expected_precision
        or payload.get("detector_sha256") != detector_sha256
    ):
        raise RuntimeError(
            f"{run_dir.name}: completion marker precision mismatch "
            f"({payload.get('precision')} != {expected_precision}; "
            f"detector hash {payload.get('detector_sha256')} != "
            f"{detector_sha256})"
        )
    if payload.get("test_f1") is not None:
        return None
    dev = payload.get("last_dev") or {}
    threshold = dev.get("threshold")
    if threshold is None:
        log(f"{run_dir.name}: no dev threshold recorded — skipping")
        return None
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    if not checkpoint.exists():
        log(f"{run_dir.name}: no best.ckpt — skipping")
        return None

    splits = json.loads(Path(data_cfg["paths"]["splits"]).read_text())["splits"]
    stats = json.loads(Path(data_cfg["paths"]["stats"]).read_text())
    labels = pd.read_csv(
        Path(data_cfg["paths"]["raw_xview3"]) / "labels" / "train.csv"
    )

    module = HeatmapLitModule.load_from_checkpoint(str(checkpoint), map_location=device)
    module.eval()

    gt_by_scene = {}
    pred_by_scene = {}
    for scene_id in sorted(splits["test"]):
        rows = labels[labels["scene_id"] == scene_id].to_dict(orient="records")
        gt_by_scene[scene_id] = ground_truth_from_labels(rows)
        pred_by_scene[scene_id] = infer_scene(
            module,
            Path(data_cfg["paths"]["raw_xview3"]) / "GRD" / scene_id,
            stats=stats,
            tau=det_cfg["decode"]["candidate_floor"],
            d_nms_m=det_cfg["decode"]["d_nms_m"],
            tile_px=det_cfg["eval"]["tile_px"],
            tile_stride_px=det_cfg["eval"]["tile_stride_px"],
            batch_size=det_cfg["eval"]["infer_batch"],
            device=device,
            precision=det_cfg["schedule"]["precision"],
        )

    result = score_dataset(gt_by_scene, apply_threshold(pred_by_scene, threshold))
    dark = result.slices["dark"]
    near = result.slices["near_shore"]
    payload.update(
        {
            "test_f1": result.aggregate.f1,
            "test_precision": result.aggregate.precision,
            "test_recall": result.aggregate.recall,
            "test_tp": result.aggregate.tp,
            "test_fp": result.aggregate.fp,
            "test_fn": result.aggregate.fn,
            "test_threshold_applied": threshold,
            "test_inference_precision": expected_precision,
            "test_dark_recall": dark.recall,
            "test_dark_support": dark.tp + dark.fn,
            "test_near_shore_f1": near.f1,
            "test_scored_at": dt.datetime.now().isoformat(timespec="seconds"),
        }
    )
    atomic_write_json(final_path, payload)
    del module
    torch.cuda.empty_cache()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--arms-config", default="configs/arms.yaml")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*", help="restrict to these exp ids")
    args = parser.parse_args(argv)

    data_cfg = yaml.safe_load(Path(args.data_config).read_text())
    detector_path = Path(args.detector_config)
    det_cfg = yaml.safe_load(detector_path.read_text())
    detector_sha256 = hashlib.sha256(detector_path.read_bytes()).hexdigest()
    arms_cfg = yaml.safe_load(Path(args.arms_config).read_text())

    exp_ids = []
    for meta in arms_cfg["arms"].values():
        for frac in arms_cfg["label_fracs"]:
            for seed in (
                list(arms_cfg["seeds"]["core"])
                + (list(arms_cfg["seeds"]["reruns"]) if frac in set(arms_cfg["seeds"]["rerun_fracs"]) else [])
            ):
                exp_ids.append(f"{meta['short']}-f{int(round(frac * 100))}-s{seed}")
    if args.only:
        exp_ids = [e for e in exp_ids if e in set(args.only)]

    scored = 0
    for exp_id in exp_ids:
        run_dir = Path(args.runs_root) / exp_id
        if not (run_dir / "final_metrics.json").exists():
            continue
        log(f"{exp_id}: scoring test split (16 scenes, frozen dev threshold)")
        payload = score_run(
            run_dir,
            data_cfg=data_cfg,
            det_cfg=det_cfg,
            detector_sha256=detector_sha256,
            device=args.device,
        )
        if payload:
            scored += 1
            log(
                f"{exp_id}: test F1 {payload['test_f1']:.4f} "
                f"(dark recall {payload['test_dark_recall']:.4f} on "
                f"{payload['test_dark_support']} dark GT; "
                f"near-shore F1 {payload['test_near_shore_f1']:.4f})"
            )
    log(f"scored {scored} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
