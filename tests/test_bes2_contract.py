"""Pure decision and safety tests for the BES2 diagnostic contract."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.analysis.bes2_contract import (
    DIAGNOSTIC_FRACTIONS,
    FROZEN_COMPARATOR_EXPERIMENTS,
    H100_CAMPAIGN_GIT_SHA,
    MAX_APPROVED_GPU_HOURS,
    REPLAY_TOLERANCE,
    BES2ContractError,
    aggregate_fraction_decisions,
    assert_no_final_consumption,
    classify_cause,
    diagnostic_run_id,
    fraction_tag,
    frozen_comparator_payload,
    stage_fractions,
    validate_forecast_hours,
    verify_training_path_identity,
)


def _comparator(fraction: float, s2: float, random: float) -> dict:
    return {
        "fraction": fraction,
        "bigearthnet_s2": {"best_dev_f1": s2},
        "cnn_random": {"best_dev_f1": random},
    }


def _decision(fraction: float, recovery: float) -> dict:
    s2 = 0.80 + fraction / 10.0
    random = s2 + 0.08
    reset = s2 + recovery * (random - s2)
    return classify_cause(
        s2,
        reset,
        fraction=fraction,
        comparator=_comparator(fraction, s2, random),
    )


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
    for fraction in DIAGNOSTIC_FRACTIONS:
        decision = _decision(fraction, recovery)
        assert decision["status"] == "determinate"
        assert decision["classification"] == classification
        assert decision["replacement_eligibility"] == eligibility
        assert decision["recovery"] == pytest.approx(recovery)


def test_replay_outside_tolerance_is_indeterminate_before_other_comparisons():
    s2 = 0.80
    replay = s2 + REPLAY_TOLERANCE + 1.0e-9
    decision = classify_cause(
        replay,
        replay,
        fraction=0.1,
        comparator=_comparator(0.1, s2, 0.90),
    )

    assert decision["status"] == "indeterminate"
    assert decision["classification"] == "replay-not-reproducible"
    assert decision["recovery"] is None


def test_replay_tolerance_boundary_is_inclusive():
    s2 = 0.80
    replay = s2 + REPLAY_TOLERANCE
    decision = classify_cause(
        replay,
        replay,
        fraction=0.5,
        comparator=_comparator(0.5, s2, 0.90),
    )

    assert decision["status"] == "determinate"
    assert decision["classification"] == "first-convolution-reset-insufficient"
    assert decision["recovery"] == pytest.approx(0.0)


def test_fraction_matrix_authorizes_f50_probe_and_f10_f50_full_only():
    assert tuple(fraction_tag(value) for value in DIAGNOSTIC_FRACTIONS) == (
        "f10",
        "f50",
    )
    assert stage_fractions("probe") == (0.5,)
    assert stage_fractions("full") == (0.1, 0.5)
    assert diagnostic_run_id("current_replay", "probe", 0.5) == (
        "bes2-current_replay-probe-f50-s0"
    )
    with pytest.raises(BES2ContractError, match="does not authorize"):
        diagnostic_run_id("current_replay", "probe", 0.1)
    with pytest.raises(BES2ContractError, match="not one of"):
        diagnostic_run_id("current_replay", "full", 1.0)


def test_aggregate_requires_concordant_fraction_classification():
    concordant = {
        "f10": _decision(0.1, 0.5),
        "f50": _decision(0.5, 0.6),
    }
    aggregate = aggregate_fraction_decisions(concordant)
    assert aggregate["concordant"] is True
    assert aggregate["replacement_eligibility"] == (
        "conditional-owner-amendment-required"
    )

    mixed = dict(concordant)
    mixed["f10"] = _decision(0.1, 0.1)
    aggregate = aggregate_fraction_decisions(mixed)
    assert aggregate["classification"] == "fraction-dependent-mixed-mechanism"
    assert aggregate["concordant"] is False
    assert aggregate["replacement_eligibility"] == "blocked"


def _canonical_h100_result(exp_id: str, counts: tuple[int, int, int]) -> dict:
    tp, fp, fn = counts
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall)
    best = {
        "epoch": 9,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": 0,
        "threshold": 0.5,
        "n_candidates": tp + fp,
    }
    strict = {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }
    return {
        "result_schema": 2,
        "exp_id": exp_id,
        "git_sha": H100_CAMPAIGN_GIT_SHA,
        "detector_sha256": "d" * 64,
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "best_dev_f1": f1,
        "best_dev": best,
        "h100_runtime_contract": {
            "schema": 1,
            "status": "verified",
            "pre_trainer": {
                "devices": 1,
                "precision": "32-true",
                "micro_batch": 16,
                "gradient_accumulation": 1,
                "effective_batch": 16,
                "strict_fp32": strict,
            },
            "resolved_trainer": {
                "num_devices": 1,
                "world_size": 1,
                "gradient_accumulation": 1,
            },
        },
    }


def test_frozen_comparator_snapshots_only_four_canonical_h100_dev_results(
    tmp_path,
):
    counts = {
        "f10": {"bigearthnet_s2": (2, 1, 1), "cnn_random": (4, 1, 1)},
        "f50": {"bigearthnet_s2": (4, 1, 1), "cnn_random": (8, 1, 1)},
    }
    paths = {}
    for tag, specification in FROZEN_COMPARATOR_EXPERIMENTS.items():
        for role in ("bigearthnet_s2", "cnn_random"):
            exp_id = specification[role]
            path = tmp_path / exp_id / "final_metrics.json"
            path.parent.mkdir()
            path.write_text(
                json.dumps(_canonical_h100_result(exp_id, counts[tag][role]))
                + "\n",
                encoding="utf-8",
            )
            path.chmod(0o444)
            paths[(tag, role)] = path

    snapshot = frozen_comparator_payload(tmp_path)

    assert set(snapshot["fractions"]) == {"f10", "f50"}
    assert snapshot["read_scope"]["final_metrics_json"] == sorted(
        f"{specification[role]}/final_metrics.json"
        for specification in FROZEN_COMPARATOR_EXPERIMENTS.values()
        for role in ("bigearthnet_s2", "cnn_random")
    )
    assert snapshot["read_scope"]["checkpoint_bytes"] is False
    assert snapshot["read_scope"]["training_cohort"] is False
    assert snapshot["read_scope"]["test_metrics"] is False
    assert snapshot["read_scope"]["verified_final"] is False

    paths[("f10", "bigearthnet_s2")].chmod(0o644)
    with pytest.raises(BES2ContractError, match="remains writable"):
        frozen_comparator_payload(tmp_path)


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
    result = tmp_path / "beS2-f50-s0" / "final_verified_metrics.json"
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
