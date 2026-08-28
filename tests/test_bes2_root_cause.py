"""Fixture tests for the BES2 audit/run/summarize orchestration contracts."""

from __future__ import annotations

import inspect
import json
from fractions import Fraction
from pathlib import Path

import pytest

from src.analysis.bes2_contract import (
    BES2ContractError,
    FROZEN_COMPARATOR_EXPERIMENTS,
    H100_CAMPAIGN_GIT_SHA,
    OPTIMIZATION_TRACE_STEPS,
    hash_binding,
    read_regular_json,
    write_new_immutable,
)
from src.analysis.bes2_root_cause import (
    COMPARATOR_FILENAME,
    READY_FILENAME,
    _audit,
    _diagnostic_recipe,
    _readiness_strict_fp32,
    _scope_identity,
    main,
    validate_readiness,
)
from src.models.bes2_diagnostic import SOURCE_CHECKPOINT_SHA256


STRICT_FP32 = {
    "cuda_matmul_fp32_precision": "ieee",
    "cudnn_conv_fp32_precision": "ieee",
    "cudnn_rnn_fp32_precision": "ieee",
}


def test_readiness_flattens_and_cross_checks_launch_backend_state():
    launch = {
        "strict_fp32": STRICT_FP32,
        "autocast": {"global": False, "cuda": False, "cpu": False},
        "process": {
            "WORLD_SIZE": "unset",
            "SLURM_NTASKS": "1",
            "effective_world_size": 1,
        },
    }
    observed = _readiness_strict_fp32(
        {"strict_fp32": launch},
        {"strict_fp32": STRICT_FP32},
    )

    assert observed == STRICT_FP32
    assert observed is not launch["strict_fp32"]


@pytest.mark.parametrize(
    "runtime,accepted",
    [
        ({"strict_fp32": STRICT_FP32}, STRICT_FP32),
        (
            {"strict_fp32": {"strict_fp32": STRICT_FP32}},
            {
                **STRICT_FP32,
                "cuda_matmul_fp32_precision": "tf32",
            },
        ),
    ],
)
def test_readiness_rejects_unstructured_or_mismatched_backend_state(
    runtime, accepted
):
    with pytest.raises(BES2ContractError, match="strict-FP32 states differ"):
        _readiness_strict_fp32(runtime, {"strict_fp32": accepted})


def test_audit_uses_original_h100_results_as_numerical_comparators_only():
    source = inspect.getsource(_audit)
    assert '"strict_fp32": readiness_strict_fp32' in source
    assert 'strict_runtime["strict_fp32"]' not in source
    assert "write_new_immutable(ready_path, ready)" in source
    assert "validated_ready = validate_readiness(" in source
    assert "print(json.dumps(validated_ready, indent=1))" in source
    assert "load_completion_marker" not in source
    assert "validate_training_cohort" not in source
    assert "original_checkpoint" not in source
    assert "frozen_comparator_payload(args.original_runs_root)" in source
    assert (
        "comparators[\"detector_sha256\"] != sha256_file(detector_path)"
        in source
    )


def _bound_file(root: Path, name: str) -> Path:
    path = root / name
    path.write_text("{}\n", encoding="utf-8")
    return path


def _best(f1: float) -> dict[str, object]:
    rational = Fraction(f1).limit_denominator(1000)
    tp = rational.numerator
    error = rational.denominator - rational.numerator
    value = tp / rational.denominator
    return {
        "epoch": 9,
        "f1": value,
        "precision": value,
        "recall": value,
        "tp": tp,
        "fp": error,
        "fn": error,
        "ignored_predictions": 0,
        "threshold": 0.5,
        "n_candidates": rational.denominator,
    }


def _comparator_receipt() -> dict[str, object]:
    fractions: dict[str, object] = {}
    values = {"f10": (0.70, 0.80), "f50": (0.80, 0.90)}
    for tag, specification in FROZEN_COMPARATOR_EXPERIMENTS.items():
        s2, random = values[tag]
        records = {}
        for role, f1 in (
            ("bigearthnet_s2", s2),
            ("cnn_random", random),
        ):
            exp_id = specification[role]
            records[role] = {
                "exp_id": exp_id,
                "result": {
                    "path": f"{exp_id}/final_metrics.json",
                    "sha256": (tag + role).encode().hex().ljust(64, "0")[:64],
                },
                "best_dev_f1": f1,
                "best_dev": _best(f1),
            }
        fractions[tag] = {
            "fraction": specification["fraction"],
            **records,
            "frozen_random_minus_s2_f1": random - s2,
        }
    return {
        "schema": 1,
        "status": "frozen",
        "purpose": "bes2-h100-best-dev-numerical-comparators",
        "created_utc": "2026-08-27T00:00:00+00:00",
        "campaign_git_sha": H100_CAMPAIGN_GIT_SHA,
        "detector_sha256": "d" * 64,
        "read_scope": {
            "final_metrics_json": sorted(
                f"{specification[role]}/final_metrics.json"
                for specification in FROZEN_COMPARATOR_EXPERIMENTS.values()
                for role in ("bigearthnet_s2", "cnn_random")
            ),
            "checkpoint_bytes": False,
            "training_cohort": False,
            "test_metrics": False,
            "verified_final": False,
        },
        "fractions": fractions,
    }


def _scope() -> dict[str, object]:
    return _scope_identity(
        {
            "splits_sha256": "4" * 64,
            "stats_sha256": "5" * 64,
            "train_scene_ids": [f"train-{index:03d}" for index in range(111)],
            "dev_scene_ids": [f"dev-{index:03d}" for index in range(8)],
            "label_scene_count": 119,
        }
    )


def test_readiness_rehashes_every_bound_control(tmp_path):
    root = tmp_path / "diagnostics"
    control = root / ".control"
    control.mkdir(parents=True)
    sample = _bound_file(control, "sample.json")
    audit = _bound_file(control, "audit.json")
    tests = _bound_file(control, "tests.json")
    h100_ready = _bound_file(control, "canonical-h100-ready.json")
    comparator = control / COMPARATOR_FILENAME
    comparator.write_text(
        json.dumps(_comparator_receipt()) + "\n",
        encoding="utf-8",
    )
    h100_sha256 = hash_binding(h100_ready)["sha256"]
    source = "a" * 40
    scope_identity = _scope()
    recipe = _diagnostic_recipe(scope_identity)
    assert recipe["train_scenes"]["f10"]["scene_count"] == 12
    assert recipe["train_scenes"]["f50"]["scene_count"] == 56
    assert set(recipe["train_scenes"]["f10"]["scene_ids"]) < set(
        recipe["train_scenes"]["f50"]["scene_ids"]
    )
    payload = {
        "diagnostic_ready_schema": 2,
        "status": "ready",
        "purpose": "bes2-root-cause-train-dev-only",
        "created_utc": "2026-08-27T00:00:00+00:00",
        "source": {
            "git_sha": source,
            "training_path_identity": {
                "status": "byte-identical",
                "h100_campaign_git_sha": H100_CAMPAIGN_GIT_SHA,
            },
        },
        "diagnostic_root": str(root),
        "h100_acceptance": {
            "sha256": h100_sha256,
            "strict_fp32": STRICT_FP32,
        },
        "strict_fp32": STRICT_FP32,
        "data_view": {
            "contract": "train111-fixed-dev8-no-test-v1",
            "receipt_sha256": "1" * 64,
            "staging_receipt_sha256": "2" * 64,
            "data_config_sha256": "3" * 64,
            "scope_identity": scope_identity,
        },
        "checkpoint": {
            "relative_path": "bigearthnet_s2/model.safetensors",
            "sha256": SOURCE_CHECKPOINT_SHA256,
        },
        "sample_manifest": hash_binding(sample),
        "audit": hash_binding(audit),
        "test_suite": hash_binding(tests),
        "frozen_comparators": hash_binding(comparator),
        "recipe": recipe,
        "forecast_gpu_hours": 125.0,
        "final_access_gate": {"status": "unconsumed"},
        "canonical_h100_ready_unchanged": {
            "path": str(h100_ready),
            "sha256": h100_sha256,
        },
    }
    ready = control / READY_FILENAME
    write_new_immutable(ready, payload)

    assert (
        validate_readiness(
            ready,
            expected_source_sha=source,
            diagnostic_root=root,
        )
        == payload
    )

    audit.chmod(0o644)
    audit.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(Exception, match="binding drifted"):
        validate_readiness(
            ready,
            expected_source_sha=source,
            diagnostic_root=root,
        )


def _optimization_trace() -> dict[str, object]:
    group = {
        "parameter_count": 1,
        "parameter_norm_before": 1.0,
        "gradient_norm": 0.5,
        "gradient_to_parameter_norm": 0.5,
        "lr_min": 0.001,
        "lr_max": 0.001,
        "lr_scale_min": 0.65,
        "lr_scale_max": 0.65,
        "update_norm": 0.001,
        "relative_update_norm": 0.001,
    }
    first = {
        "name": "backbone.model.stem.0.weight",
        "parameter_count": 1,
        "parameter_norm_before": 1.0,
        "gradient_norm": 0.5,
        "gradient_to_parameter_norm": 0.5,
        "lr": 0.001,
        "lr_scale": 0.65,
        "update_norm": 0.001,
        "relative_update_norm": 0.001,
    }
    groups = {
        name: dict(group)
        for name in (
            "stem",
            "stage_0",
            "stage_1",
            "stage_2",
            "stage_3",
            "detector_head",
        )
    }
    return {
        "schema": 1,
        "milestones": list(OPTIMIZATION_TRACE_STEPS),
        "records": [
            {
                "optimizer_step": step,
                "groups": groups,
                "first_convolution": first,
            }
            for step in OPTIMIZATION_TRACE_STEPS
        ],
    }


def _metrics(
    root: Path,
    ready: Path,
    variant: str,
    fraction: float,
    f1: float,
) -> Path:
    best = _best(f1)
    payload = {
        "diagnostic_metrics_schema": 2,
        "status": "complete",
        "purpose": "bes2-root-cause",
        "variant": variant,
        "stage": "full",
        "fraction": fraction,
        "created_utc": "2026-08-27T00:00:00+00:00",
        "readiness": {
            "path": str(ready),
            "sha256": hash_binding(ready)["sha256"],
        },
        "training": {
            "variant": variant,
            "stage": "full",
            "fraction": fraction,
            "training_result": {
                "best_dev_f1": f1,
                "best_dev": best,
            },
            "optimization_trace": _optimization_trace(),
        },
        "dev_evidence": {
            "selected_operating_point": best,
            "leave_one_dev_scene_out": {
                "scene": {
                    "threshold": 0.5,
                    "retained_seven": {"f1": f1},
                }
            },
        },
        "activation_statistics": {"post_stem": {}},
        "layer_drift": {"first_convolution": {}},
        "runtime_provenance": {},
        "gpu_hours": 1.0,
    }
    tag = f"f{int(round(fraction * 100))}"
    path = root / f"bes2-{variant}-full-{tag}-s0/diagnostic_metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_summarize_publishes_no_pooling_json_and_reproducible_markdown(
    tmp_path,
):
    root = tmp_path / "diagnostics"
    control = root / ".control"
    control.mkdir(parents=True)
    comparator = control / COMPARATOR_FILENAME
    comparator.write_text(
        json.dumps(_comparator_receipt()) + "\n",
        encoding="utf-8",
    )
    ready = control / READY_FILENAME
    ready.write_text(
        json.dumps({"frozen_comparators": hash_binding(comparator)}) + "\n",
        encoding="utf-8",
    )

    comparator_payload = _comparator_receipt()
    for tag, specification in FROZEN_COMPARATOR_EXPERIMENTS.items():
        fraction = float(specification["fraction"])
        s2 = comparator_payload["fractions"][tag]["bigearthnet_s2"][
            "best_dev_f1"
        ]
        random = comparator_payload["fractions"][tag]["cnn_random"][
            "best_dev_f1"
        ]
        _metrics(root, ready, "current_replay", fraction, s2)
        _metrics(
            root,
            ready,
            "first_conv_reset",
            fraction,
            s2 + 0.5 * (random - s2),
        )

    assert main(["summarize", "--diagnostic-root", str(root)]) == 0

    result = read_regular_json(
        control / "BES2_ROOT_CAUSE.json",
        "root cause",
    )
    report = (control / "BES2_ROOT_CAUSE.md").read_text(encoding="utf-8")
    assert result["decision"]["aggregate"]["classification"] == (
        "stem-conversion-explains-most-deficit"
    )
    assert result["decision"]["aggregate"]["concordant"] is True
    assert set(result["decision"]["per_fraction"]) == {"f10", "f50"}
    for decision in result["decision"]["per_fraction"].values():
        assert decision["recovery"] == pytest.approx(0.5)
    assert result["scope"]["analysis_unit"] == "per-fraction-no-pooling"
    assert "never pooled" in report
    assert "no significance" in report
    assert "BigEarthNet-S1 was not modified or tested" in report

    report_path = control / "BES2_ROOT_CAUSE.md"
    report_path.chmod(0o644)
    report_path.write_text(report + "tampered\n", encoding="utf-8")
    with pytest.raises(Exception, match="not reproducible"):
        main(["summarize", "--diagnostic-root", str(root)])
