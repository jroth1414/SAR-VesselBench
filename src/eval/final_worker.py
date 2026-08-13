"""One isolated GPU worker for the already-consumed verified-final run.

The controller supplies normalized ground truth through stdin. This module has
no validation-label argument and never opens ``validation.csv``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from scripts.h100.contracts import load_cells
from src.eval.final_eval import (
    FINAL_CELL_RESULT_FILENAME,
    FINAL_CONSUMPTION_FILENAME,
    _git_sha,
    _regular_json,
    score_final_cell,
)
from src.eval.final_authorization import AUTHORIZATION_FILENAME
from src.eval.ground_truth import GroundTruthPoint
from src.eval.heldout_contract import (
    COHORT_FILENAME,
    TEST_RESULT_FILENAME,
    HeldoutContractError,
    validate_test_result,
    validate_training_cohort_cell,
)
from src.eval.result_contract import sha256_file


def _canonical_directory(path: Path, description: str) -> Path:
    absolute = path.absolute()
    if (
        absolute.is_symlink()
        or not absolute.is_dir()
        or absolute.resolve() != absolute
    ):
        raise HeldoutContractError(
            f"{description} must be an existing canonical non-symlink directory"
        )
    return absolute


def _ground_truth_from_stdin(
    *, expected_sha256: str, expected_scene_ids: Sequence[str]
) -> dict[str, list[GroundTruthPoint]]:
    import sys

    raw = sys.stdin.buffer.read()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise HeldoutContractError("normalized final ground truth SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HeldoutContractError("normalized final ground truth is invalid JSON") from exc
    if not isinstance(payload, Mapping) or list(sorted(payload)) != list(
        expected_scene_ids
    ):
        raise HeldoutContractError(
            "normalized final ground truth differs from the exact 50 scenes"
        )
    expected_keys = {
        "x_m",
        "y_m",
        "confidence",
        "source",
        "distance_from_shore_km",
    }
    normalized: dict[str, list[GroundTruthPoint]] = {}
    for scene_id in expected_scene_ids:
        rows = payload.get(scene_id)
        if not isinstance(rows, list):
            raise HeldoutContractError(
                f"normalized final ground truth rows are invalid for {scene_id}"
            )
        points = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != expected_keys:
                raise HeldoutContractError(
                    f"normalized final ground truth point is invalid for {scene_id}"
                )
            points.append(
                GroundTruthPoint(
                    x_m=float(row["x_m"]),
                    y_m=float(row["y_m"]),
                    confidence=str(row["confidence"]),
                    source=(
                        str(row["source"])
                        if row["source"] is not None
                        else None
                    ),
                    distance_from_shore_km=(
                        float(row["distance_from_shore_km"])
                        if row["distance_from_shore_km"] is not None
                        else None
                    ),
                )
            )
        normalized[scene_id] = points
    return normalized


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--lock-sha256", required=True)
    parser.add_argument("--consumption-sha256", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    repo = _canonical_directory(args.repo, "repository")
    runs_root = _canonical_directory(args.runs_root, "runs root")
    output = runs_root / args.exp_id / FINAL_CELL_RESULT_FILENAME
    if output.exists() or output.is_symlink():
        raise HeldoutContractError(
            f"final result already exists; workers are never retried: {output}"
        )

    lockfile = runs_root / "final_eval.lock"
    consumed_path = runs_root / ".h100" / FINAL_CONSUMPTION_FILENAME
    if sha256_file(lockfile) != args.lock_sha256:
        raise HeldoutContractError("final-eval lock SHA-256 mismatch")
    if sha256_file(consumed_path) != args.consumption_sha256:
        raise HeldoutContractError("final ground-truth receipt SHA-256 mismatch")
    lock = _regular_json(lockfile, "final-eval lock")
    consumed = _regular_json(consumed_path, "final ground-truth receipt")
    if lockfile.stat().st_mode & 0o222 or consumed_path.stat().st_mode & 0o222:
        raise HeldoutContractError("final lock and consumption receipt must be immutable")
    owner_amendment = lock.get("owner_amendment")
    final_data_view = lock.get("final_data_view")
    selected_cells = lock.get("selected_cells")
    if (
        lock.get("lock_schema") != 3
        or not isinstance(selected_cells, list)
        or len(selected_cells) != 32
        or len(set(selected_cells)) != 32
        or args.exp_id not in selected_cells
        or not isinstance(owner_amendment, Mapping)
        or not isinstance(final_data_view, Mapping)
        or consumed.get("consumption_schema") != 1
        or consumed.get("status") != "verified-final-ground-truth-consumed"
        or consumed.get("campaign_git_sha") != lock.get("campaign_git_sha")
        or consumed.get("evaluator_git_sha") != lock.get("evaluator_git_sha")
        or consumed.get("owner_authorization_sha256")
        != owner_amendment.get("sha256")
        or consumed.get("final_data_view") != final_data_view
        or consumed.get("lock", {}).get("sha256") != args.lock_sha256
    ):
        raise HeldoutContractError("final worker lock/consumption identity is invalid")
    amendment_path = runs_root / ".h100" / AUTHORIZATION_FILENAME
    data_view_path = runs_root / ".h100" / "FINAL_DATA_VIEW.json"
    if (
        sha256_file(amendment_path) != owner_amendment.get("sha256")
        or sha256_file(data_view_path) != final_data_view.get("sha256")
    ):
        raise HeldoutContractError("final worker immutable authorization/view drifted")
    campaign_git_sha = str(lock["campaign_git_sha"])
    evaluator_git_sha = str(lock["evaluator_git_sha"])
    if _git_sha(repo) != evaluator_git_sha:
        raise HeldoutContractError("final worker evaluator Git SHA mismatch")

    cells = load_cells(repo)
    matches = [cell for cell in cells if cell.exp_id == args.exp_id]
    if len(matches) != 1:
        raise HeldoutContractError(f"unknown final cell: {args.exp_id}")
    cell = matches[0]
    detector_path = repo / "configs/detector.yaml"
    detector_sha256 = sha256_file(detector_path)
    if detector_sha256 != lock.get("detector_sha256"):
        raise HeldoutContractError("final worker detector SHA-256 mismatch")
    det_cfg = yaml.safe_load(detector_path.read_text(encoding="utf-8"))
    cohort, cohort_sha256, record, _training = validate_training_cohort_cell(
        path=runs_root / ".h100" / COHORT_FILENAME,
        expected_sha256=str(lock["cohort_sha256"]),
        cells=cells,
        runs_root=runs_root,
        git_sha=campaign_git_sha,
        detector_sha256=detector_sha256,
        candidate_floor=float(det_cfg["decode"]["candidate_floor"]),
        exp_id=args.exp_id,
    )
    data_cfg = yaml.safe_load(
        (repo / "configs/data.yaml").read_text(encoding="utf-8")
    )
    splits = json.loads(
        (repo / str(data_cfg["paths"]["splits"])).read_text(encoding="utf-8")
    )["splits"]
    test_scene_ids = tuple(sorted(map(str, splits["test"])))
    test_path = runs_root / args.exp_id / TEST_RESULT_FILENAME
    validate_test_result(
        path=test_path,
        exp_id=args.exp_id,
        cohort=cohort,
        cohort_sha256=cohort_sha256,
        test_scene_ids=test_scene_ids,
    )
    test_bindings = {
        str(item["exp_id"]): item
        for item in lock.get("test_results", [])
        if isinstance(item, Mapping) and "exp_id" in item
    }
    test_binding = test_bindings.get(args.exp_id)
    test_sha256 = sha256_file(test_path)
    if (
        not isinstance(test_binding, Mapping)
        or test_binding.get("sha256") != test_sha256
        or test_binding.get("relative_path")
        != f"{args.exp_id}/{TEST_RESULT_FILENAME}"
    ):
        raise HeldoutContractError("final worker TEST binding mismatch")

    eval_scenes = tuple(sorted(map(str, splits["eval_final"])))
    if len(eval_scenes) != 50:
        raise HeldoutContractError("final worker requires exactly 50 scenes")
    normalized_sha256 = str(consumed.get("normalized_ground_truth_sha256", ""))
    gt_by_scene = _ground_truth_from_stdin(
        expected_sha256=normalized_sha256,
        expected_scene_ids=eval_scenes,
    )
    paths = data_cfg["paths"]
    raw_root = repo / str(paths["raw_xview3"]) / "GRD"
    stats = json.loads((repo / str(paths["stats"])).read_text(encoding="utf-8"))
    payload = score_final_cell(
        cell=cell,
        record=record,
        runs_root=runs_root,
        raw_root=raw_root,
        stats=stats,
        det_cfg=det_cfg,
        gt_by_scene=gt_by_scene,
        eval_scenes=eval_scenes,
        device=args.device,
        campaign_git_sha=campaign_git_sha,
        evaluator_git_sha=evaluator_git_sha,
        detector_sha256=detector_sha256,
        cohort_sha256=cohort_sha256,
        test_result_sha256=test_sha256,
        grid_audit={
            "monotonicity_ok": lock["grid"]["monotonicity_ok"],
            "monotonicity_tolerance": lock["grid"]["monotonicity_tolerance"],
            "violations": lock["grid"]["violations"],
        },
        authorization_sha256=str(owner_amendment["sha256"]),
        final_access={
            "lock_sha256": args.lock_sha256,
            "consumption_sha256": args.consumption_sha256,
            "data_view_sha256": str(lock["final_data_view"]["sha256"]),
        },
    )
    print(
        json.dumps(
            {
                "status": "final-cell-complete",
                "exp_id": args.exp_id,
                "f1": payload["metrics"]["f1"],
                "sha256": sha256_file(output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
