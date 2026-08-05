"""Once-only verified-scene evaluation after the complete TEST cohort gate.

The final labels are opened only after all 32 training bindings and all 32
immutable TEST results validate.  The once-only lock is then created
exclusively before ``validation.csv`` is read.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Sequence

from scripts.h100.contracts import load_cells
from src.analysis.curves import (
    GRID_COLUMNS,
    GRID_COUNT_COLUMNS,
    MONOTONICITY_TOLERANCE,
    training_fraction_counts,
)
from src.eval.heldout_contract import (
    COHORT_FILENAME,
    TEST_RESULT_FILENAME,
    HeldoutContractError,
    cohort_record,
    validate_complete_test_cohort,
    validate_training_cohort,
)
from src.eval.result_contract import sha256_file

LOCKFILE = Path("runs/final_eval.lock")
EVAL_FRACS = (0.1, 0.25, 1.0)


def _repo_path(repo: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo / path


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


def _write_once_lock(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise HeldoutContractError(
            f"{path} exists; the once-only final evaluation was already attempted"
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=1, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # The lock intentionally remains after any post-create failure: opening
        # verified-final resources is a once-only event requiring human review.
        raise


def _finite_grid_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HeldoutContractError(f"grid.csv {field} must be finite") from exc
    if not math.isfinite(result):
        raise HeldoutContractError(f"grid.csv {field} must be finite")
    return result


def validate_grid_gate(
    path: Path,
    *,
    cells: Sequence[object],
    test_results: dict[str, dict],
    arms: dict[str, dict],
    fraction_counts: dict[float, dict[str, int]],
    git_sha: str,
    detector_sha256: str,
    precision: str,
) -> str:
    """Validate the complete TEST grid and its predeclared STOP diagnostic."""

    if path.is_symlink() or not path.is_file():
        raise HeldoutContractError(
            "verified final evaluation requires a regular non-symlink grid.csv"
        )
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != GRID_COLUMNS:
                raise HeldoutContractError(
                    "grid.csv columns do not match the exact reportable schema"
                )
            rows = list(reader)
    except OSError as exc:
        raise HeldoutContractError("could not read grid.csv") from exc

    expected = {str(cell.exp_id): cell for cell in cells}
    if len(rows) != 32 or len(expected) != 32:
        raise HeldoutContractError("grid.csv must contain the exact 32-cell cohort")
    observed_ids = [row["exp_id"] for row in rows]
    if len(set(observed_ids)) != 32 or set(observed_ids) != set(expected):
        raise HeldoutContractError("grid.csv IDs do not match the exact 32-cell cohort")

    points: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        exp_id = row["exp_id"]
        cell = expected[exp_id]
        try:
            arm = arms[str(cell.init)]
            expected_role = str(arm["role"])
            test_f1 = _finite_grid_float(row["test_f1"], field=f"{exp_id}.test_f1")
            fraction = _finite_grid_float(
                row["label_frac"], field=f"{exp_id}.label_frac"
            )
            seed = int(row["seed"])
            epochs_run = int(row["epochs_run"])
            counts = {name: int(row[name]) for name in GRID_COUNT_COLUMNS}
            dev_f1 = _finite_grid_float(row["dev_f1"], field=f"{exp_id}.dev_f1")
            dev_threshold = _finite_grid_float(
                row["dev_threshold"], field=f"{exp_id}.dev_threshold"
            )
            bound_test_f1 = _finite_grid_float(
                test_results[exp_id]["metrics"]["f1"],
                field=f"{exp_id}.bound_test_f1",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HeldoutContractError(
                f"grid.csv row for {exp_id} is malformed or unbound"
            ) from exc
        if (
            row["init"] != str(cell.init)
            or row["track"] != str(cell.track)
            or row["role"] != expected_role
            or not math.isclose(fraction, float(cell.fraction), abs_tol=0.0)
            or seed != int(cell.seed)
            or row["precision"] != precision
            or row["detector_sha256"] != detector_sha256
            or row["git_sha"] != git_sha
            or not 0.0 <= dev_f1 <= 1.0
            or not 0.0 <= dev_threshold <= 1.0
            or not 0.0 <= test_f1 <= 1.0
            or not math.isclose(test_f1, bound_test_f1, rel_tol=1e-12, abs_tol=1e-12)
            or epochs_run <= 0
            or counts != fraction_counts.get(float(cell.fraction))
            or counts["train_scene_count"] <= 0
            or counts["train_vessel_count"] <= 0
            or counts["train_dark_vessel_count"] < 0
            or counts["train_near_shore_vessel_count"] < 0
            or counts["train_dark_vessel_count"] > counts["train_vessel_count"]
            or counts["train_near_shore_vessel_count"]
            > counts["train_vessel_count"]
            or row["monotonicity_ok"].strip().casefold() != "true"
        ):
            raise HeldoutContractError(
                f"grid.csv row for {exp_id} disagrees with the frozen cohort"
            )
        points.setdefault(str(cell.init), []).append((fraction, test_f1))

    if set(points) != {str(cell.init) for cell in cells}:
        raise HeldoutContractError("grid.csv arm set is incomplete")
    for init_name, arm_points in points.items():
        ordered = sorted(arm_points)
        if [point[0] for point in ordered] != [0.1, 0.25, 0.5, 1.0]:
            raise HeldoutContractError(
                f"grid.csv fractions are incomplete for {init_name}"
            )
        if any(
            current < previous - MONOTONICITY_TOLERANCE
            for (_, previous), (_, current) in zip(ordered, ordered[1:])
        ):
            raise HeldoutContractError(
                f"grid.csv monotonicity STOP condition is false for {init_name}"
            )
    return sha256_file(path)


def main(argv: Sequence[str] | None = None) -> int:
    import pandas as pd
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-am-sure", action="store_true")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    if not args.i_am_sure:
        raise SystemExit(
            "REFUSING: verified final evaluation is once-only; pass --i-am-sure "
            "only after the complete TEST cohort has been reviewed"
        )
    repo = args.repo.resolve()
    runs_root = args.runs_root.resolve()
    lockfile = runs_root / "final_eval.lock"
    if lockfile.exists() or lockfile.is_symlink():
        raise SystemExit(
            f"REFUSING: {lockfile} exists; STOP and consult a human, never delete it"
        )

    data_path = _repo_path(repo, args.data_config)
    detector_path = _repo_path(repo, args.detector_config)
    data_cfg = yaml.safe_load(data_path.read_text())
    det_cfg = yaml.safe_load(detector_path.read_text())
    detector_sha256 = hashlib.sha256(detector_path.read_bytes()).hexdigest()
    git_sha = _git_sha(repo)
    cells = load_cells(repo)
    cohort_path = runs_root / ".h100" / COHORT_FILENAME

    # First barrier: validate every training marker/checkpoint byte without
    # touching TEST or verified-final data.
    cohort, cohort_sha256 = validate_training_cohort(
        path=cohort_path,
        cells=cells,
        runs_root=runs_root,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        candidate_floor=float(det_cfg["decode"]["candidate_floor"]),
    )

    # Reading the frozen split manifest is safe. Every TEST result must then
    # validate before eval-final raster checks, lock creation, or validation.csv.
    splits_path = _repo_path(repo, data_cfg["paths"]["splits"])
    splits = json.loads(splits_path.read_text())["splits"]
    test_scene_ids = tuple(sorted(map(str, splits["test"])))
    test_results = validate_complete_test_cohort(
        cells=cells,
        runs_root=runs_root,
        cohort=cohort,
        cohort_sha256=cohort_sha256,
        test_scene_ids=test_scene_ids,
    )

    # Third barrier: the complete reportable TEST grid and the predeclared
    # 0.02 monotonicity diagnostic must be green before any verified-final
    # raster, label, or once-only lock is touched.
    arms = yaml.safe_load((repo / "configs/arms.yaml").read_text())["arms"]
    fraction_counts = training_fraction_counts(
        repo=repo,
        fractions=tuple(sorted({float(cell.fraction) for cell in cells})),
    )
    grid_path = runs_root / "summary" / "grid.csv"
    grid_sha256 = validate_grid_gate(
        grid_path,
        cells=cells,
        test_results=test_results,
        arms=arms,
        fraction_counts=fraction_counts,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        precision=str(det_cfg["schedule"]["precision"]),
    )

    selected = [cell for cell in cells if float(cell.fraction) in EVAL_FRACS]
    if len(selected) != 24 or len({cell.exp_id for cell in selected}) != 24:
        raise HeldoutContractError("verified final evaluation requires exactly 24 cells")
    eval_scenes = tuple(sorted(map(str, splits["eval_final"])))
    if len(eval_scenes) != 50 or len(set(eval_scenes)) != 50:
        raise HeldoutContractError("verified final split must contain exactly 50 scenes")
    raw_root = _repo_path(repo, data_cfg["paths"]["raw_xview3"]) / "GRD"
    absent = [
        scene_id
        for scene_id in eval_scenes
        if not (raw_root / scene_id / "VH_dB.tif").is_file()
        or not (raw_root / scene_id / "VV_dB.tif").is_file()
    ]
    if absent:
        raise HeldoutContractError(
            f"{len(absent)} eval_final scenes are not extracted (first: {absent[:3]})"
        )

    test_bindings = [
        {
            "exp_id": cell.exp_id,
            "relative_path": f"{cell.exp_id}/{TEST_RESULT_FILENAME}",
            "sha256": sha256_file(runs_root / cell.exp_id / TEST_RESULT_FILENAME),
        }
        for cell in cells
    ]
    _write_once_lock(
        lockfile,
        {
            "lock_schema": 2,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "policy": "all-32-test-complete-before-once-only-final-access",
            "git_sha": git_sha,
            "detector_sha256": detector_sha256,
            "cohort_sha256": cohort_sha256,
            "grid": {
                "relative_path": "summary/grid.csv",
                "sha256": grid_sha256,
            },
            "test_results": test_bindings,
            "selected_cells": [cell.exp_id for cell in selected],
            "eval_scene_count": len(eval_scenes),
            "inference_precision": det_cfg["schedule"]["precision"],
        },
    )
    print(f"lockfile written: {lockfile}; verified-final access is now consumed")

    from src.eval.ground_truth import ground_truth_from_labels
    from src.eval.infer_scene import infer_scene
    from src.eval.scorer import score_dataset
    from src.eval.threshold import apply_threshold
    from src.train.lit_modules import HeatmapLitModule

    labels = pd.read_csv(
        _repo_path(repo, data_cfg["paths"]["raw_xview3"])
        / "labels"
        / "validation.csv"
    )
    labels["scene_id"] = labels["scene_id"].astype(str)
    stats = json.loads(_repo_path(repo, data_cfg["paths"]["stats"]).read_text())
    gt_by_scene = {
        scene_id: ground_truth_from_labels(
            labels[labels["scene_id"] == scene_id].to_dict(orient="records")
        )
        for scene_id in eval_scenes
    }

    rows = []
    for cell in selected:
        exp_id = cell.exp_id
        record = cohort_record(cohort, exp_id)
        threshold = float(record["best_dev"]["threshold"])
        checkpoint_binding = record["best_checkpoint"]
        checkpoint = runs_root / str(checkpoint_binding["relative_path"])
        print(f"[final eval] {exp_id}")
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
                precision=det_cfg["schedule"]["precision"],
            )
            for scene_id in eval_scenes
        }
        result = score_dataset(gt_by_scene, apply_threshold(pred_by_scene, threshold))
        dark = result.slices["dark"]
        near = result.slices["near_shore"]
        rows.append(
            {
                "exp_id": exp_id,
                "init": cell.init,
                "label_frac": cell.fraction,
                "threshold": threshold,
                "threshold_source": "best-dev-checkpoint-bound",
                "dev_epoch": record["best_dev"]["epoch"],
                "checkpoint_relative_path": checkpoint_binding["relative_path"],
                "checkpoint_sha256": checkpoint_binding["sha256"],
                "checkpoint_epoch": checkpoint_binding["epoch"],
                "cohort_sha256": cohort_sha256,
                "test_result_sha256": sha256_file(
                    runs_root / exp_id / TEST_RESULT_FILENAME
                ),
                "git_sha": git_sha,
                "detector_sha256": detector_sha256,
                "inference_precision": det_cfg["schedule"]["precision"],
                "f1": result.aggregate.f1,
                "precision": result.aggregate.precision,
                "recall": result.aggregate.recall,
                "tp": result.aggregate.tp,
                "fp": result.aggregate.fp,
                "fn": result.aggregate.fn,
                "ignored_predictions": result.aggregate.ignored_predictions,
                "dark_recall": dark.recall,
                "dark_support": dark.tp + dark.fn,
                "near_shore_f1": near.f1,
                "near_shore_support": near.tp + near.fn,
            }
        )
        del module

    out = runs_root / "summary" / "final_verified.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"FINAL verified-scene results -> {out}. Nothing is tuned after this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
