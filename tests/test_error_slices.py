from __future__ import annotations

import json
from pathlib import Path

from src.analysis.error_slices import collect_slice_metrics, compute_role_deltas


def _write_final(root: Path, exp_id: str, payload: dict) -> None:
    run_dir = root / exp_id
    run_dir.mkdir(parents=True)
    (run_dir / "final_metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def _arms_config() -> dict:
    return {
        "arms": {
            "vit_random": {"short": "vitrand", "track": "vit", "role": "floor"},
            "satdino_b": {"short": "satdino", "track": "vit", "role": "optical"},
            "sarmae_b": {"short": "sarmae", "track": "vit", "role": "sar"},
            "cnn_random": {"short": "cnnrand", "track": "cnn", "role": "floor"},
        },
        "label_fracs": [0.1],
        "seeds": {"core": [0], "reruns": [], "rerun_fracs": []},
    }


def test_collect_slice_metrics_flags_threshold_transfer(tmp_path: Path):
    _write_final(
        tmp_path,
        "vitrand-f10-s0",
        {
            "best_dev_f1": 0.90,
            "last_dev": {"threshold": 0.7, "precision": 0.9, "recall": 0.9},
            "test_f1": 0.60,
            "test_precision": 0.75,
            "test_recall": 0.50,
            "test_threshold_applied": 0.7,
            "test_dark_recall": 0.25,
            "test_dark_support": 4,
            "test_near_shore_f1": 0.10,
            "test_oracle_f1": 0.0,
            "test_oracle_threshold": 0.0,
        },
    )

    table = collect_slice_metrics(_arms_config(), tmp_path)

    assert len(table) == 1
    row = table[0]
    assert row["exp_id"] == "vitrand-f10-s0"
    assert round(row["dev_to_test_f1_gap"], 2) == 0.30
    assert bool(row["threshold_transfer_flag"]) is True
    assert bool(row["dark_slice_available"]) is True
    assert row["test_oracle_f1"] == 0.0
    assert row["test_oracle_threshold"] == 0.0


def test_zero_support_dark_recall_is_missing(tmp_path: Path):
    _write_final(
        tmp_path,
        "vitrand-f10-s0",
        {
            "best_dev_f1": 0.70,
            "last_dev": {},
            "test_f1": 0.60,
            "test_dark_recall": 0.0,
            "test_dark_support": 0,
        },
    )

    row = collect_slice_metrics(_arms_config(), tmp_path)[0]

    assert row["test_dark_recall"] is None
    assert bool(row["dark_slice_available"]) is False


def test_compute_role_deltas_stays_within_track(tmp_path: Path):
    _write_final(
        tmp_path,
        "vitrand-f10-s0",
        {"best_dev_f1": 0.7, "last_dev": {}, "test_f1": 0.60, "test_near_shore_f1": 0.1},
    )
    _write_final(
        tmp_path,
        "satdino-f10-s0",
        {"best_dev_f1": 0.75, "last_dev": {}, "test_f1": 0.65, "test_near_shore_f1": 0.2},
    )
    _write_final(
        tmp_path,
        "sarmae-f10-s0",
        {"best_dev_f1": 0.8, "last_dev": {}, "test_f1": 0.72, "test_near_shore_f1": 0.3},
    )
    _write_final(
        tmp_path,
        "cnnrand-f10-s0",
        {"best_dev_f1": 0.5, "last_dev": {}, "test_f1": 0.40, "test_near_shore_f1": 0.0},
    )

    table = collect_slice_metrics(_arms_config(), tmp_path)
    deltas = compute_role_deltas(table)

    vit_sar = next(row for row in deltas if row["track"] == "vit" and row["role"] == "sar")
    contrast = next(row for row in deltas if row["role"] == "sar_minus_optical")

    assert round(vit_sar["test_f1_vs_floor"], 2) == 0.12
    assert round(contrast["test_f1"], 2) == 0.07
    assert "cnn" in {row["track"] for row in deltas}
