"""Score TEST only after freezing the exact 32-cell training cohort.

Training completion markers are immutable. Each held-out result is written
once to ``runs/<exp_id>/test_metrics.json`` and binds the cohort, completion
marker, best checkpoint, best-dev threshold, source, and precision.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.h100.contracts import load_cells
from src.eval.heldout_contract import (
    COHORT_FILENAME,
    TEST_RESULT_FILENAME,
    HeldoutContractError,
    build_test_result,
    cohort_record,
    validate_test_result,
    validate_training_cohort_cell,
    write_test_result,
)


def log(message: str) -> None:
    print(f"[{dt.datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HeldoutContractError("could not resolve the active source SHA") from exc


def _metric_payload(metric: object) -> dict[str, int | float]:
    return {
        "f1": float(getattr(metric, "f1")),
        "precision": float(getattr(metric, "precision")),
        "recall": float(getattr(metric, "recall")),
        "tp": int(getattr(metric, "tp")),
        "fp": int(getattr(metric, "fp")),
        "fn": int(getattr(metric, "fn")),
        "ignored_predictions": int(getattr(metric, "ignored_predictions")),
    }


def _per_scene_payload(result: object) -> dict[str, object]:
    return {
        scene_id: {
            "aggregate": _metric_payload(scene.aggregate),
            "slices": {
                name: _metric_payload(metric)
                for name, metric in sorted(scene.slices.items())
            },
        }
        for scene_id, scene in sorted(result.scene_results.items())
    }


def score_run(
    run_dir: Path,
    *,
    cohort: Mapping[str, object],
    cohort_sha256: str,
    data_cfg: Mapping[str, object],
    det_cfg: Mapping[str, object],
    labels: object,
    test_scene_ids: Sequence[str],
    device: str,
) -> dict | None:
    """Score one cohort-bound cell without mutating its training marker."""

    import torch

    from src.eval.ground_truth import ground_truth_from_labels
    from src.eval.infer_scene import infer_scene
    from src.eval.scorer import score_dataset
    from src.eval.threshold import apply_threshold
    from src.train.lit_modules import HeatmapLitModule

    exp_id = run_dir.name
    record = cohort_record(cohort, exp_id)
    test_path = run_dir / TEST_RESULT_FILENAME
    if test_path.exists() or test_path.is_symlink():
        validate_test_result(
            path=test_path,
            exp_id=exp_id,
            cohort=cohort,
            cohort_sha256=cohort_sha256,
            test_scene_ids=test_scene_ids,
        )
        return None

    best_dev = record["best_dev"]
    checkpoint_binding = record["best_checkpoint"]
    threshold = float(best_dev["threshold"])
    runs_root = run_dir.parent
    checkpoint = runs_root / str(checkpoint_binding["relative_path"])
    paths = data_cfg["paths"]
    raw_root = Path(str(paths["raw_xview3"]))
    stats = json.loads(Path(str(paths["stats"])).read_text())

    module = HeatmapLitModule.load_from_checkpoint(str(checkpoint), map_location=device)
    module.eval()
    try:
        gt_by_scene = {}
        pred_by_scene = {}
        for scene_id in test_scene_ids:
            rows = labels[labels["scene_id"] == scene_id].to_dict(orient="records")
            gt_by_scene[scene_id] = ground_truth_from_labels(rows)
            pred_by_scene[scene_id] = infer_scene(
                module,
                raw_root / "GRD" / scene_id,
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
        aggregate = _metric_payload(result.aggregate)
        dark = result.slices["dark"]
        near = result.slices["near_shore"]
        metrics = {
            **aggregate,
            "dark_recall": float(dark.recall),
            "dark_support": int(dark.tp + dark.fn),
            "near_shore_f1": float(near.f1),
            "near_shore_support": int(near.tp + near.fn),
        }
        payload = build_test_result(
            exp_id=exp_id,
            cohort_sha256=cohort_sha256,
            cohort_cell=record,
            inference_precision=str(det_cfg["schedule"]["precision"]),
            metrics=metrics,
            per_scene=_per_scene_payload(result),
            test_scene_ids=test_scene_ids,
        )
        write_test_result(test_path, payload)
        validate_test_result(
            path=test_path,
            exp_id=exp_id,
            cohort=cohort,
            cohort_sha256=cohort_sha256,
            test_scene_ids=test_scene_ids,
        )
        return payload
    finally:
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    import pandas as pd
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--cohort-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", nargs="*", help="restrict to these exp ids")
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    runs_root = args.runs_root.resolve()
    data_path = repo / args.data_config
    detector_path = repo / args.detector_config
    data_cfg = yaml.safe_load(data_path.read_text())
    det_cfg = yaml.safe_load(detector_path.read_text())
    detector_sha256 = hashlib.sha256(detector_path.read_bytes()).hexdigest()
    cells = load_cells(repo)
    cohort_path = (
        args.cohort.resolve()
        if args.cohort is not None
        else runs_root / ".h100" / COHORT_FILENAME
    )
    git_sha = _git_sha(repo)
    requested = set(args.only or [cell.exp_id for cell in cells])
    unknown = requested - {cell.exp_id for cell in cells}
    if unknown:
        raise HeldoutContractError(f"unknown core run IDs: {sorted(unknown)}")
    if not requested:
        raise HeldoutContractError("at least one cohort cell must be requested")

    # The controller has already validated and frozen all 32 bytes. Recheck the
    # immutable cohort identity plus each requested marker/checkpoint before any
    # split, label, stats, or TEST raster is opened. This stays O(requested).
    cohort: dict[str, object] | None = None
    cohort_sha256: str | None = None
    for cell in cells:
        if cell.exp_id not in requested:
            continue
        validated, validated_sha256, _record, _training = (
            validate_training_cohort_cell(
                path=cohort_path,
                expected_sha256=args.cohort_sha256,
                cells=cells,
                runs_root=runs_root,
                git_sha=git_sha,
                detector_sha256=detector_sha256,
                candidate_floor=float(det_cfg["decode"]["candidate_floor"]),
                exp_id=cell.exp_id,
            )
        )
        if cohort is None:
            cohort, cohort_sha256 = validated, validated_sha256
        elif cohort != validated or cohort_sha256 != validated_sha256:
            raise HeldoutContractError("training cohort changed during validation")
    assert cohort is not None and cohort_sha256 is not None
    splits_path = repo / str(data_cfg["paths"]["splits"])
    splits = json.loads(splits_path.read_text())["splits"]
    test_scene_ids = tuple(sorted(map(str, splits["test"])))
    if len(test_scene_ids) != 16:
        raise HeldoutContractError("frozen test partition must contain exactly 16 scenes")
    labels_path = repo / str(data_cfg["paths"]["raw_xview3"]) / "labels" / "train.csv"
    from src.eval.ground_truth_audit import audit_ground_truth_dataset

    audit_ground_truth_dataset(
        train_csv=labels_path,
        splits_json=splits_path,
    )
    labels = pd.read_csv(labels_path)
    labels["scene_id"] = labels["scene_id"].astype(str)
    absolute_data_cfg = {
        **data_cfg,
        "paths": {
            **data_cfg["paths"],
            "raw_xview3": str(repo / str(data_cfg["paths"]["raw_xview3"])),
            "stats": str(repo / str(data_cfg["paths"]["stats"])),
        },
    }

    scored = 0
    for cell in cells:
        if cell.exp_id not in requested:
            continue
        log(f"{cell.exp_id}: scoring frozen TEST with cohort-bound best-dev threshold")
        payload = score_run(
            runs_root / cell.exp_id,
            cohort=cohort,
            cohort_sha256=cohort_sha256,
            data_cfg=absolute_data_cfg,
            det_cfg=det_cfg,
            labels=labels,
            test_scene_ids=test_scene_ids,
            device=args.device,
        )
        if payload is not None:
            scored += 1
            metrics = payload["metrics"]
            log(
                f"{cell.exp_id}: test F1 {metrics['f1']:.4f}; "
                f"near-shore F1 {metrics['near_shore_f1']:.4f} "
                f"on {metrics['near_shore_support']} GT"
            )
    log(f"scored {scored} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
