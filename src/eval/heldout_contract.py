"""Immutable cohort and held-out result contracts for the core grid.

Test labels may not be opened until all 32 schema-2 training completion
markers have been validated and frozen into one content-addressed cohort.
Training markers remain immutable; held-out metrics live in separate files.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval.result_contract import (
    RESULT_SCHEMA,
    ResultContractError,
    atomic_write_json,
    load_completion_marker,
    sha256_file,
)

COHORT_SCHEMA = 1
TEST_RESULT_SCHEMA = 1
COHORT_FILENAME = "TRAINING_COHORT.json"
TEST_RESULT_FILENAME = "test_metrics.json"
EXPECTED_TEST_SCENE_COUNT = 16
EXPECTED_TEST_POSITIVE_SUPPORT = 1165
EXPECTED_TEST_DARK_SUPPORT = 0
EXPECTED_TEST_NEAR_SHORE_SUPPORT = 2
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class HeldoutContractError(RuntimeError):
    """A cohort or held-out result is unsafe or inconsistent."""


def _regular_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise HeldoutContractError(f"{description} must be a regular non-symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeldoutContractError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise HeldoutContractError(f"{description} root must be a JSON object")
    return payload


def _write_new_immutable(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise HeldoutContractError(f"immutable artifact already exists: {path}")
    try:
        atomic_write_json(path, payload)
    except ResultContractError as exc:
        raise HeldoutContractError(str(exc)) from exc
    path.chmod(0o444)


def _expected_recipe(cell: object, git_sha: str, detector_sha256: str) -> dict[str, object]:
    return {
        "exp_id": str(getattr(cell, "exp_id")),
        "git_sha": git_sha,
        "detector_sha256": detector_sha256,
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
    }


def create_training_cohort(
    *,
    cells: Sequence[object],
    runs_root: Path,
    output: Path,
    git_sha: str,
    detector_sha256: str,
    candidate_floor: float,
) -> dict[str, object]:
    """Validate all 32 cells and freeze their exact training artifacts once."""

    if len(cells) != 32 or len({str(getattr(cell, "exp_id")) for cell in cells}) != 32:
        raise HeldoutContractError("training cohort requires exactly 32 unique core cells")
    if not _HEX40.fullmatch(git_sha) or not _HEX64.fullmatch(detector_sha256):
        raise HeldoutContractError("cohort source bindings are malformed")
    output = output.absolute()
    expected_output = (runs_root.absolute() / ".h100" / COHORT_FILENAME)
    if output != expected_output:
        raise HeldoutContractError(f"cohort must use the canonical path: {expected_output}")

    records: list[dict[str, object]] = []
    for cell in cells:
        exp_id = str(getattr(cell, "exp_id"))
        marker = runs_root / exp_id / "final_metrics.json"
        try:
            payload, checkpoint = load_completion_marker(
                marker,
                candidate_floor=candidate_floor,
                expected_recipe=_expected_recipe(cell, git_sha, detector_sha256),
            )
        except ResultContractError as exc:
            raise HeldoutContractError(f"{exp_id}: invalid training marker: {exc}") from exc
        if payload.get("result_schema") != RESULT_SCHEMA:
            raise HeldoutContractError(f"{exp_id}: unsupported training result schema")
        if marker.stat().st_mode & 0o222:
            raise HeldoutContractError(
                f"{exp_id}: training marker must be immutable before cohort freeze"
            )
        test_path = runs_root / exp_id / TEST_RESULT_FILENAME
        if test_path.exists() or test_path.is_symlink():
            raise HeldoutContractError(
                f"{exp_id}: held-out result exists before cohort freeze"
            )
        binding = payload["best_checkpoint"]
        records.append(
            {
                "exp_id": exp_id,
                "completion_marker": {
                    "relative_path": marker.relative_to(runs_root).as_posix(),
                    "sha256": sha256_file(marker),
                },
                "best_checkpoint": {
                    "relative_path": checkpoint.relative_to(runs_root).as_posix(),
                    "sha256": binding["sha256"],
                    "epoch": binding["epoch"],
                },
                "best_dev": payload["best_dev"],
                "recipe": _expected_recipe(cell, git_sha, detector_sha256),
            }
        )
    payload = {
        "cohort_schema": COHORT_SCHEMA,
        "status": "training-cohort-frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": "all-32-training-complete-before-any-test-access",
        "git_sha": git_sha,
        "detector_sha256": detector_sha256,
        "candidate_floor": float(candidate_floor),
        "cell_count": 32,
        "cells": records,
    }
    _write_new_immutable(output, payload)
    return payload


def validate_training_cohort(
    *,
    path: Path,
    cells: Sequence[object],
    runs_root: Path,
    git_sha: str,
    detector_sha256: str,
    candidate_floor: float,
) -> tuple[dict[str, object], str]:
    """Validate the frozen cohort and every referenced byte on disk."""

    expected_path = runs_root.absolute() / ".h100" / COHORT_FILENAME
    if path.absolute() != expected_path:
        raise HeldoutContractError(f"cohort must use the canonical path: {expected_path}")
    payload = _regular_json(path, "training cohort")
    if path.stat().st_mode & 0o222:
        raise HeldoutContractError("training cohort is not immutable")
    expected_top = {
        "cohort_schema", "status", "created_utc", "policy", "git_sha",
        "detector_sha256", "candidate_floor", "cell_count", "cells",
    }
    if set(payload) != expected_top:
        raise HeldoutContractError("training cohort keys do not match the contract")
    expected_ids = [str(getattr(cell, "exp_id")) for cell in cells]
    if (
        len(cells) != 32
        or len(set(expected_ids)) != 32
        or payload.get("cohort_schema") != COHORT_SCHEMA
        or payload.get("status") != "training-cohort-frozen"
        or payload.get("policy") != "all-32-training-complete-before-any-test-access"
        or payload.get("git_sha") != git_sha
        or payload.get("detector_sha256") != detector_sha256
        or payload.get("candidate_floor") != float(candidate_floor)
        or payload.get("cell_count") != 32
    ):
        raise HeldoutContractError("training cohort identity is invalid")
    records = payload.get("cells")
    if not isinstance(records, list) or [r.get("exp_id") if isinstance(r, Mapping) else None for r in records] != expected_ids:
        raise HeldoutContractError("training cohort is not the exact ordered 32-cell matrix")

    for cell, record in zip(cells, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != {
            "exp_id", "completion_marker", "best_checkpoint", "best_dev", "recipe"
        }:
            raise HeldoutContractError("training cohort cell keys are invalid")
        exp_id = str(getattr(cell, "exp_id"))
        marker_binding = record.get("completion_marker")
        checkpoint_binding = record.get("best_checkpoint")
        if not isinstance(marker_binding, Mapping) or set(marker_binding) != {"relative_path", "sha256"}:
            raise HeldoutContractError(f"{exp_id}: cohort marker binding is invalid")
        marker = runs_root / str(marker_binding["relative_path"])
        expected_marker = runs_root / exp_id / "final_metrics.json"
        if (
            marker.absolute() != expected_marker.absolute()
            or marker_binding.get("sha256") != sha256_file(marker)
            or marker.stat().st_mode & 0o222
        ):
            raise HeldoutContractError(
                f"{exp_id}: completion marker drifted after cohort freeze"
            )
        try:
            training, checkpoint = load_completion_marker(
                marker,
                candidate_floor=candidate_floor,
                expected_recipe=_expected_recipe(cell, git_sha, detector_sha256),
            )
        except ResultContractError as exc:
            raise HeldoutContractError(f"{exp_id}: training marker no longer validates: {exc}") from exc
        expected_checkpoint = {
            "relative_path": checkpoint.relative_to(runs_root).as_posix(),
            "sha256": training["best_checkpoint"]["sha256"],
            "epoch": training["best_checkpoint"]["epoch"],
        }
        if checkpoint_binding != expected_checkpoint or record.get("best_dev") != training["best_dev"] or record.get("recipe") != _expected_recipe(cell, git_sha, detector_sha256):
            raise HeldoutContractError(f"{exp_id}: cohort checkpoint/dev/recipe binding mismatch")
    return payload, sha256_file(path)



def validate_training_cohort_cell(
    *,
    path: Path,
    expected_sha256: str,
    cells: Sequence[object],
    runs_root: Path,
    git_sha: str,
    detector_sha256: str,
    candidate_floor: float,
    exp_id: str,
) -> tuple[dict[str, object], str, dict[str, object], dict[str, object]]:
    """Validate the cohort envelope and one cell binding in constant grid work.

    The controller performs the full all-32 byte validation at phase boundaries.
    A scoring process uses this narrower check immediately before inference so it
    hashes only its own completion marker and selected checkpoint.
    """

    expected_path = runs_root.absolute() / ".h100" / COHORT_FILENAME
    if path.absolute() != expected_path:
        raise HeldoutContractError(f"cohort must use the canonical path: {expected_path}")
    if not isinstance(expected_sha256, str) or not _HEX64.fullmatch(expected_sha256):
        raise HeldoutContractError("expected cohort SHA-256 is malformed")
    payload = _regular_json(path, "training cohort")
    if path.stat().st_mode & 0o222:
        raise HeldoutContractError("training cohort is not immutable")
    try:
        actual_sha256 = sha256_file(path)
    except ResultContractError as exc:
        raise HeldoutContractError("could not hash the training cohort") from exc
    if actual_sha256 != expected_sha256:
        raise HeldoutContractError("training cohort SHA-256 differs from its binding")

    expected_top = {
        "cohort_schema", "status", "created_utc", "policy", "git_sha",
        "detector_sha256", "candidate_floor", "cell_count", "cells",
    }
    expected_ids = [str(getattr(cell, "exp_id")) for cell in cells]
    records = payload.get("cells")
    observed_ids = (
        [
            record.get("exp_id") if isinstance(record, Mapping) else None
            for record in records
        ]
        if isinstance(records, list)
        else []
    )
    if (
        set(payload) != expected_top
        or len(cells) != 32
        or len(set(expected_ids)) != 32
        or payload.get("cohort_schema") != COHORT_SCHEMA
        or payload.get("status") != "training-cohort-frozen"
        or payload.get("policy")
        != "all-32-training-complete-before-any-test-access"
        or payload.get("git_sha") != git_sha
        or payload.get("detector_sha256") != detector_sha256
        or payload.get("candidate_floor") != float(candidate_floor)
        or payload.get("cell_count") != 32
        or observed_ids != expected_ids
    ):
        raise HeldoutContractError(
            "training cohort is not the exact ordered 32-cell matrix"
        )
    if exp_id not in expected_ids:
        raise HeldoutContractError(f"unknown cohort cell: {exp_id}")
    index = expected_ids.index(exp_id)
    cell = cells[index]
    record = records[index]
    if not isinstance(record, Mapping) or set(record) != {
        "exp_id", "completion_marker", "best_checkpoint", "best_dev", "recipe"
    }:
        raise HeldoutContractError(f"{exp_id}: cohort cell keys are invalid")
    marker_binding = record.get("completion_marker")
    checkpoint_binding = record.get("best_checkpoint")
    if not isinstance(marker_binding, Mapping) or set(marker_binding) != {
        "relative_path", "sha256"
    }:
        raise HeldoutContractError(f"{exp_id}: cohort marker binding is invalid")
    marker_path = runs_root / str(marker_binding["relative_path"])
    expected_marker = runs_root / exp_id / "final_metrics.json"
    try:
        marker_sha256 = sha256_file(marker_path)
    except ResultContractError as exc:
        raise HeldoutContractError(
            f"{exp_id}: completion marker is absent or unsafe"
        ) from exc
    if (
        marker_path.absolute() != expected_marker.absolute()
        or marker_binding.get("sha256") != marker_sha256
        or marker_path.stat().st_mode & 0o222
    ):
        raise HeldoutContractError(
            f"{exp_id}: completion marker drifted after cohort freeze"
        )
    try:
        training, checkpoint = load_completion_marker(
            marker_path,
            candidate_floor=candidate_floor,
            expected_recipe=_expected_recipe(cell, git_sha, detector_sha256),
        )
    except ResultContractError as exc:
        raise HeldoutContractError(
            f"{exp_id}: training marker no longer validates: {exc}"
        ) from exc
    expected_checkpoint = {
        "relative_path": checkpoint.relative_to(runs_root).as_posix(),
        "sha256": training["best_checkpoint"]["sha256"],
        "epoch": training["best_checkpoint"]["epoch"],
    }
    if (
        checkpoint_binding != expected_checkpoint
        or record.get("best_dev") != training["best_dev"]
        or record.get("recipe") != _expected_recipe(
            cell, git_sha, detector_sha256
        )
    ):
        raise HeldoutContractError(
            f"{exp_id}: cohort checkpoint/dev/recipe binding mismatch"
        )
    return (
        payload,
        actual_sha256,
        dict(record),
        dict(training),
    )

def cohort_record(cohort: Mapping[str, object], exp_id: str) -> Mapping[str, object]:
    records = cohort.get("cells")
    if not isinstance(records, list):
        raise HeldoutContractError("cohort cells are absent")
    matches = [record for record in records if isinstance(record, Mapping) and record.get("exp_id") == exp_id]
    if len(matches) != 1:
        raise HeldoutContractError(f"cohort lacks exactly one record for {exp_id}")
    return matches[0]


_POINT_METRIC_FIELDS = frozenset(
    {"f1", "precision", "recall", "tp", "fp", "fn", "ignored_predictions"}
)


def _validate_point_metric(
    metric: object, *, description: str
) -> dict[str, int | float]:
    if not isinstance(metric, Mapping) or set(metric) != _POINT_METRIC_FIELDS:
        raise HeldoutContractError(f"{description} metric keys do not match the contract")
    counts: dict[str, int] = {}
    for key in ("tp", "fp", "fn", "ignored_predictions"):
        value = metric[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HeldoutContractError(
                f"{description}.{key} must be a non-negative integer"
            )
        counts[key] = value
    values: dict[str, float] = {}
    for key in ("f1", "precision", "recall"):
        value = metric[key]
        if isinstance(value, bool):
            raise HeldoutContractError(f"{description}.{key} is not numeric")
        try:
            values[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise HeldoutContractError(f"{description}.{key} is not numeric") from exc
        if not math.isfinite(values[key]) or not 0.0 <= values[key] <= 1.0:
            raise HeldoutContractError(f"{description}.{key} is outside [0,1]")
    expected_precision = (
        counts["tp"] / (counts["tp"] + counts["fp"])
        if counts["tp"] + counts["fp"]
        else 0.0
    )
    expected_recall = (
        counts["tp"] / (counts["tp"] + counts["fn"])
        if counts["tp"] + counts["fn"]
        else 0.0
    )
    expected_f1 = (
        2.0 * expected_precision * expected_recall
        / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    for key, expected in (
        ("precision", expected_precision),
        ("recall", expected_recall),
        ("f1", expected_f1),
    ):
        if not math.isclose(values[key], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise HeldoutContractError(
                f"{description}.{key} is inconsistent with TP/FP/FN"
            )
    return {**values, **counts}


def _validate_test_components(
    *,
    metrics: Mapping[str, object],
    per_scene: Mapping[str, object],
    test_scene_ids: Sequence[str],
) -> None:
    scene_ids = tuple(map(str, test_scene_ids))
    if (
        len(scene_ids) != EXPECTED_TEST_SCENE_COUNT
        or len(set(scene_ids)) != EXPECTED_TEST_SCENE_COUNT
        or tuple(sorted(scene_ids)) != scene_ids
    ):
        raise HeldoutContractError("test result requires the exact 16 sorted scene IDs")
    required_metrics = _POINT_METRIC_FIELDS | {
        "dark_recall", "dark_support", "near_shore_f1", "near_shore_support",
    }
    if set(metrics) != required_metrics:
        raise HeldoutContractError("test metric keys do not match the contract")
    aggregate = _validate_point_metric(
        {key: metrics[key] for key in _POINT_METRIC_FIELDS},
        description="test.aggregate",
    )
    for key in ("dark_support", "near_shore_support"):
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HeldoutContractError(f"test metric {key} must be a non-negative integer")
    for key in ("dark_recall", "near_shore_f1"):
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise HeldoutContractError(f"test metric {key} is not numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise HeldoutContractError(f"test metric {key} is outside [0,1]")
    if aggregate["tp"] + aggregate["fn"] != EXPECTED_TEST_POSITIVE_SUPPORT:
        raise HeldoutContractError("test positive support must equal 1165")
    if metrics["dark_support"] != EXPECTED_TEST_DARK_SUPPORT:
        raise HeldoutContractError("test dark support must equal 0")
    if metrics["near_shore_support"] != EXPECTED_TEST_NEAR_SHORE_SUPPORT:
        raise HeldoutContractError("test near-shore support must equal 2")
    if float(metrics["dark_recall"]) != 0.0:
        raise HeldoutContractError("zero-support test dark recall must equal 0")
    if set(per_scene) != set(scene_ids):
        raise HeldoutContractError("test per-scene keys differ from the frozen test split")

    summed = {
        scope: {key: 0 for key in ("tp", "fp", "fn", "ignored_predictions")}
        for scope in ("aggregate", "dark", "near_shore")
    }
    for scene_id in scene_ids:
        scene = per_scene[scene_id]
        if not isinstance(scene, Mapping) or set(scene) != {"aggregate", "slices"}:
            raise HeldoutContractError(f"{scene_id}: per-scene result keys are invalid")
        slices = scene["slices"]
        if not isinstance(slices, Mapping) or set(slices) != {"dark", "near_shore"}:
            raise HeldoutContractError(f"{scene_id}: slice result keys are invalid")
        for scope, raw in (
            ("aggregate", scene["aggregate"]),
            ("dark", slices["dark"]),
            ("near_shore", slices["near_shore"]),
        ):
            validated = _validate_point_metric(
                raw, description=f"{scene_id}.{scope}"
            )
            for key in summed[scope]:
                summed[scope][key] += int(validated[key])
    for key in summed["aggregate"]:
        if summed["aggregate"][key] != aggregate[key]:
            raise HeldoutContractError(f"test per-scene aggregate mismatch at {key}")
    dark_support = summed["dark"]["tp"] + summed["dark"]["fn"]
    near_support = summed["near_shore"]["tp"] + summed["near_shore"]["fn"]
    if dark_support != metrics["dark_support"] or near_support != metrics["near_shore_support"]:
        raise HeldoutContractError("test slice support differs from per-scene metrics")
    near = _validate_point_metric(
        {
            **summed["near_shore"],
            "precision": (
                summed["near_shore"]["tp"]
                / (summed["near_shore"]["tp"] + summed["near_shore"]["fp"])
                if summed["near_shore"]["tp"] + summed["near_shore"]["fp"]
                else 0.0
            ),
            "recall": (
                summed["near_shore"]["tp"]
                / near_support if near_support else 0.0
            ),
            "f1": float(metrics["near_shore_f1"]),
        },
        description="test.near_shore",
    )
    if not math.isclose(float(metrics["near_shore_f1"]), float(near["f1"])):
        raise HeldoutContractError("test near-shore F1 differs from per-scene metrics")


def build_test_result(
    *,
    exp_id: str,
    cohort_sha256: str,
    cohort_cell: Mapping[str, object],
    inference_precision: str,
    metrics: Mapping[str, object],
    per_scene: Mapping[str, object],
    test_scene_ids: Sequence[str],
) -> dict[str, object]:
    """Create one held-out result bound to the frozen training cohort."""

    if not _HEX64.fullmatch(cohort_sha256):
        raise HeldoutContractError("test result cohort SHA-256 is malformed")
    recipe = cohort_cell.get("recipe")
    marker = cohort_cell.get("completion_marker")
    checkpoint = cohort_cell.get("best_checkpoint")
    best_dev = cohort_cell.get("best_dev")
    if not all(isinstance(value, Mapping) for value in (recipe, marker, checkpoint, best_dev)):
        raise HeldoutContractError("cohort cell is incomplete")
    if recipe.get("exp_id") != exp_id or recipe.get("precision") != inference_precision:
        raise HeldoutContractError("test inference recipe differs from its cohort cell")
    _validate_test_components(
        metrics=metrics,
        per_scene=per_scene,
        test_scene_ids=test_scene_ids,
    )
    return {
        "test_result_schema": TEST_RESULT_SCHEMA,
        "status": "test-complete",
        "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exp_id": exp_id,
        "cohort_sha256": cohort_sha256,
        "completion_marker_sha256": marker["sha256"],
        "git_sha": recipe["git_sha"],
        "detector_sha256": recipe["detector_sha256"],
        "inference_precision": inference_precision,
        "threshold_source": {
            "kind": "best-dev-checkpoint-bound",
            "threshold": best_dev["threshold"],
            "dev_epoch": best_dev["epoch"],
            "checkpoint_relative_path": checkpoint["relative_path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_epoch": checkpoint["epoch"],
        },
        "metrics": dict(metrics),
        "per_scene": dict(per_scene),
    }


def write_test_result(path: Path, payload: Mapping[str, object]) -> None:
    _write_new_immutable(path, payload)


def validate_test_result(
    *,
    path: Path,
    exp_id: str,
    cohort: Mapping[str, object],
    cohort_sha256: str,
    test_scene_ids: Sequence[str],
) -> dict[str, object]:
    payload = _regular_json(path, "test result")
    if path.stat().st_mode & 0o222:
        raise HeldoutContractError(f"{exp_id}: test result is not immutable")
    expected_keys = {
        "test_result_schema", "status", "scored_utc", "exp_id", "cohort_sha256",
        "completion_marker_sha256", "git_sha", "detector_sha256",
        "inference_precision", "threshold_source", "metrics", "per_scene",
    }
    if set(payload) != expected_keys or payload.get("test_result_schema") != TEST_RESULT_SCHEMA or payload.get("status") != "test-complete" or payload.get("exp_id") != exp_id or payload.get("cohort_sha256") != cohort_sha256:
        raise HeldoutContractError(f"{exp_id}: test result identity is invalid")
    record = cohort_record(cohort, exp_id)
    rebuilt = build_test_result(
        exp_id=exp_id,
        cohort_sha256=cohort_sha256,
        cohort_cell=record,
        inference_precision=str(payload.get("inference_precision", "")),
        metrics=payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {},
        per_scene=payload.get("per_scene") if isinstance(payload.get("per_scene"), Mapping) else {},
        test_scene_ids=test_scene_ids,
    )
    for key in expected_keys - {"scored_utc"}:
        if payload.get(key) != rebuilt.get(key):
            raise HeldoutContractError(f"{exp_id}: test result binding mismatch at {key}")
    return payload


def validate_complete_test_cohort(
    *,
    cells: Sequence[object],
    runs_root: Path,
    cohort: Mapping[str, object],
    cohort_sha256: str,
    test_scene_ids: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Require one valid, cohort-bound TEST result for every core cell."""

    exp_ids = [str(getattr(cell, "exp_id")) for cell in cells]
    if len(exp_ids) != 32 or len(set(exp_ids)) != 32:
        raise HeldoutContractError("complete test cohort requires 32 unique core cells")
    records = cohort.get("cells")
    if (
        not isinstance(records, list)
        or [
            record.get("exp_id") if isinstance(record, Mapping) else None
            for record in records
        ]
        != exp_ids
    ):
        raise HeldoutContractError("test cohort differs from the frozen 32-cell matrix")
    scene_ids = tuple(map(str, test_scene_ids))
    if (
        len(scene_ids) != EXPECTED_TEST_SCENE_COUNT
        or len(set(scene_ids)) != EXPECTED_TEST_SCENE_COUNT
        or tuple(sorted(scene_ids)) != scene_ids
    ):
        raise HeldoutContractError("complete test cohort requires 16 sorted scene IDs")

    validated: dict[str, dict[str, object]] = {}
    for exp_id in exp_ids:
        path = runs_root / exp_id / TEST_RESULT_FILENAME
        try:
            validated[exp_id] = validate_test_result(
                path=path,
                exp_id=exp_id,
                cohort=cohort,
                cohort_sha256=cohort_sha256,
                test_scene_ids=scene_ids,
            )
        except HeldoutContractError as exc:
            raise HeldoutContractError(
                f"complete test cohort is invalid at {exp_id}: {exc}"
            ) from exc
    return validated
