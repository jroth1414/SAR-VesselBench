"""Pure decision and safety tests for the BES2 diagnostic contract."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.analysis.bes2_contract import (
    MAX_APPROVED_GPU_HOURS,
    RANDOM_FROZEN_DEV_F1,
    REPLAY_TOLERANCE,
    S2_FROZEN_DEV_F1,
    BES2ContractError,
    assert_no_final_consumption,
    classify_cause,
    validate_forecast_hours,
    verify_training_path_identity,
)


def _reset_for(replay: float, recovery: float) -> float:
    return replay + recovery * (RANDOM_FROZEN_DEV_F1 - replay)


@pytest.mark.parametrize(
    "recovery,classification,eligibility",
    [
        (
            0.50,
            "stem-conversion-explains-most-deficit",
            "conditional-owner-amendment-required",
        ),
        (
            0.20,
            "first-convolution-reset-insufficient",
            "blocked",
        ),
        (0.35, "mixed-mechanism", "blocked"),
        (-0.10, "stem-conversion-not-performance-cause", "blocked"),
    ],
)
def test_recovery_classification_boundaries(
    recovery,
    classification,
    eligibility,
):
    replay = S2_FROZEN_DEV_F1
    decision = classify_cause(replay, _reset_for(replay, recovery))

    assert decision["status"] == "determinate"
    assert decision["classification"] == classification
    assert decision["replacement_eligibility"] == eligibility
    assert decision["recovery"] == pytest.approx(recovery)


def test_replay_outside_tolerance_is_indeterminate_before_other_comparisons():
    replay = S2_FROZEN_DEV_F1 + REPLAY_TOLERANCE + 1.0e-9
    decision = classify_cause(replay, replay)

    assert decision["status"] == "indeterminate"
    assert decision["classification"] == "replay-not-reproducible"
    assert decision["recovery"] is None


def test_replay_tolerance_boundary_is_inclusive():
    replay = S2_FROZEN_DEV_F1 + REPLAY_TOLERANCE
    decision = classify_cause(replay, replay)

    assert decision["status"] == "determinate"
    assert decision["classification"] == "first-convolution-reset-insufficient"
    assert decision["recovery"] == pytest.approx(0.0)


def test_forecast_enforces_owner_budget():
    assert validate_forecast_hours(MAX_APPROVED_GPU_HOURS) == 125.0
    with pytest.raises(BES2ContractError, match="exceeds"):
        validate_forecast_hours(math.nextafter(MAX_APPROVED_GPU_HOURS, math.inf))
    with pytest.raises(BES2ContractError, match="mandatory"):
        validate_forecast_hours(251.0)


def test_final_consumption_gate_distinguishes_unconsumed_authorization(tmp_path):
    meta = tmp_path / ".h100"
    meta.mkdir()
    authorization = meta / "FINAL_EVAL_OWNER_AMENDMENT.json"
    authorization.write_text("{}\n")

    evidence = assert_no_final_consumption(tmp_path)
    assert evidence["status"] == "unconsumed"
    assert evidence["owner_authorization"]["path"] == (
        ".h100/FINAL_EVAL_OWNER_AMENDMENT.json"
    )

    consumed = meta / "FINAL_GROUND_TRUTH_CONSUMED.json"
    consumed.write_text("{}\n")
    with pytest.raises(BES2ContractError, match="consumed"):
        assert_no_final_consumption(tmp_path)


def test_final_consumption_gate_rejects_per_cell_final_results(tmp_path):
    result = tmp_path / "beS2-f100-s0" / "final_verified_metrics.json"
    result.parent.mkdir()
    result.write_text("{}\n")
    with pytest.raises(BES2ContractError, match="consumed"):
        assert_no_final_consumption(tmp_path)


def test_dev_training_path_is_byte_identical_to_h100_campaign_revision():
    repo = Path(__file__).resolve().parents[1]
    identity = verify_training_path_identity(repo)

    assert identity["status"] == "byte-identical"
    assert identity["blob_count"] > 0
    assert len(identity["blob_manifest_sha256"]) == 64
