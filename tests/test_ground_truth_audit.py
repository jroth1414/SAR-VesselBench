from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.eval.ground_truth_audit import audit_ground_truth_dataset


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, int]]]:
    splits = {
        "splits": {
            "train": ["train-scene"],
            "dev": [f"dev-{index:02d}" for index in reversed(range(9))],
            "test": ["test-01", "test-00"],
            "eval_final": ["final-scene"],
        }
    }
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits, indent=1), encoding="utf-8")

    labels_path = tmp_path / "train.csv"
    fieldnames = [
        "scene_id",
        "confidence",
        "is_vessel",
        "distance_from_shore_km",
    ]
    rows = [
        *[[f"dev-{index:02d}", "LOW", "", ""] for index in range(1, 8)],
        ["dev-00", "HIGH", "True", "1.0"],
        ["dev-00", "HIGH", "False", "4.0"],
        ["dev-00", "LOW", "", ""],
        ["dev-08", "MEDIUM", "true", "3.0"],
        ["dev-08", "MEDIUM", "false", "1.5"],
        ["dev-08", "LOW", "nonsense", "0.5"],
        ["test-00", "HIGH", "TRUE", "2.0"],
        ["test-00", "LOW", "", "3.0"],
        ["test-01", "MEDIUM", "False", "4.0"],
        ["test-01", "MEDIUM", "True", "4.0"],
        ["train-scene", "not-inspected", "not-inspected", "not-inspected"],
    ]
    with labels_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fieldnames)
        writer.writerows(rows)

    expected = {
        "dev8": {
            "scene_count": 8,
            "rows": 10,
            "positive": 1,
            "background": 1,
            "ignore": 8,
            "near_shore_rows": 1,
            "near_shore_positive": 1,
            "near_shore_background": 0,
            "near_shore_ignore": 0,
        },
        "dev23": {
            "scene_count": 9,
            "rows": 13,
            "positive": 2,
            "background": 2,
            "ignore": 9,
            "near_shore_rows": 3,
            "near_shore_positive": 1,
            "near_shore_background": 1,
            "near_shore_ignore": 1,
        },
        "test": {
            "scene_count": 2,
            "rows": 4,
            "positive": 2,
            "background": 1,
            "ignore": 1,
            "near_shore_rows": 1,
            "near_shore_positive": 1,
            "near_shore_background": 0,
            "near_shore_ignore": 0,
        },
    }
    return labels_path, splits_path, expected


def test_audit_receipt_binds_inputs_scene_lists_counts_and_near_shore(tmp_path):
    labels_path, splits_path, expected = _write_fixture(tmp_path)
    receipt = audit_ground_truth_dataset(
        train_csv=labels_path,
        splits_json=splits_path,
        expected_counts=expected,
    )

    assert receipt["verified"] is True
    assert receipt["inputs"]["train_csv"]["sha256"] == hashlib.sha256(
        labels_path.read_bytes()
    ).hexdigest()
    assert receipt["inputs"]["splits_json"]["sha256"] == hashlib.sha256(
        splits_path.read_bytes()
    ).hexdigest()

    dev8 = receipt["scopes"]["dev8"]
    assert dev8["scene_ids"] == [f"dev-{index:02d}" for index in range(8)]
    assert dev8["scene_ids_sha256"] == hashlib.sha256(
        (json.dumps(dev8["scene_ids"], separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    assert dev8["counts"] == {
        "rows": 10,
        "positive": 1,
        "background": 1,
        "ignore": 8,
        "near_shore": {
            "rows": 1,
            "positive": 1,
            "background": 0,
            "ignore": 0,
        },
    }
    assert receipt["scopes"]["test"]["counts"]["near_shore"]["positive"] == 1
    assert receipt["expected_counts"] == expected


def test_audit_fails_closed_on_any_expected_count_drift(tmp_path):
    labels_path, splits_path, expected = _write_fixture(tmp_path)
    expected["test"]["positive"] += 1
    with pytest.raises(ValueError, match="ground-truth count mismatch for test"):
        audit_ground_truth_dataset(
            train_csv=labels_path,
            splits_json=splits_path,
            expected_counts=expected,
        )


def test_audit_rejects_missing_scope_scenes(tmp_path):
    labels_path, splits_path, expected = _write_fixture(tmp_path)
    lines = labels_path.read_text().splitlines()
    labels_path.write_text("\n".join(line for line in lines if "test-01" not in line) + "\n")
    with pytest.raises(ValueError, match="scenes with no train CSV rows"):
        audit_ground_truth_dataset(
            train_csv=labels_path,
            splits_json=splits_path,
            expected_counts=expected,
        )
