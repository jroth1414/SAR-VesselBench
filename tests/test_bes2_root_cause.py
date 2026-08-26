"""Fixture tests for the BES2 audit/run/summarize orchestration contracts."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.analysis.bes2_contract import (
    S2_FROZEN_DEV_F1,
    hash_binding,
    read_regular_json,
    write_new_immutable,
)
from src.analysis.bes2_root_cause import (
    READY_FILENAME,
    _audit,
    _scope_identity,
    main,
    validate_readiness,
)
from src.models.bes2_diagnostic import SOURCE_CHECKPOINT_SHA256


def test_audit_uses_original_h100_results_as_numerical_comparators_only():
    source = inspect.getsource(_audit)
    assert "load_completion_marker" not in source
    assert "validate_training_cohort" not in source
    assert "original_checkpoint" not in source
    assert '"original_campaign_artifacts_read": False' in source


def _bound_file(root: Path, name: str) -> Path:
    path = root / name
    path.write_text("{}\n")
    return path


def test_readiness_rehashes_every_bound_control(tmp_path):
    root = tmp_path / "diagnostics"
    control = root / ".control"
    control.mkdir(parents=True)
    sample = _bound_file(control, "sample.json")
    audit = _bound_file(control, "audit.json")
    tests = _bound_file(control, "tests.json")
    h100_ready = _bound_file(control, "canonical-h100-ready.json")
    h100_sha256 = hash_binding(h100_ready)["sha256"]
    source = "a" * 40
    strict_fp32 = {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }
    payload = {
        "diagnostic_ready_schema": 1,
        "status": "ready",
        "purpose": "bes2-root-cause-train-dev-only",
        "created_utc": "2026-08-26T00:00:00+00:00",
        "source": {
            "git_sha": source,
            "training_path_identity": {
                "status": "byte-identical",
                "h100_campaign_git_sha": (
                    "1a82d508fbeb9fdf6868a9637611e9018952fb43"
                ),
            },
        },
        "diagnostic_root": str(root),
        "h100_acceptance": {
            "sha256": h100_sha256,
            "strict_fp32": strict_fp32,
        },
        "strict_fp32": strict_fp32,
        "data_view": {
            "contract": "train111-fixed-dev8-no-test-v1",
            "receipt_sha256": "1" * 64,
            "staging_receipt_sha256": "2" * 64,
            "data_config_sha256": "3" * 64,
            "scope_identity": _scope_identity(
                {
                    "splits_sha256": "4" * 64,
                    "stats_sha256": "5" * 64,
                    "train_scene_ids": [],
                    "dev_scene_ids": [],
                    "label_scene_count": 0,
                }
            ),
        },
        "checkpoint": {
            "relative_path": "bigearthnet_s2/model.safetensors",
            "sha256": SOURCE_CHECKPOINT_SHA256,
        },
        "sample_manifest": hash_binding(sample),
        "audit": hash_binding(audit),
        "test_suite": hash_binding(tests),
        "recipe": {
            "variants": ["current_replay", "first_conv_reset"],
            "stages": {"probe_epochs": 5, "full_max_epochs": 50},
            "schedule_horizon_epochs": 50,
            "fraction": 1.0,
            "seed": 0,
            "train_scene_count": 111,
            "dev_scene_count": 8,
            "precision": "32-true",
            "micro_batch": 16,
            "gradient_accumulation": 1,
            "effective_batch": 16,
            "layer_decay": 0.65,
            "devices_per_run": 1,
            "ddp": False,
        },
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
    audit.write_text('{"changed":true}\n')
    with pytest.raises(Exception, match="binding drifted"):
        validate_readiness(
            ready,
            expected_source_sha=source,
            diagnostic_root=root,
        )


def _best(f1: float):
    return {
        "epoch": 9,
        "f1": f1,
        "precision": f1,
        "recall": f1,
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "ignored_predictions": 0,
        "threshold": 0.5,
        "n_candidates": 1,
    }


def _metrics(root: Path, variant: str, f1: float):
    best = _best(f1)
    # summarize consumes bound evidence, while deep score-count consistency is
    # enforced earlier by validate_training_metrics on Judy.
    payload = {
        "diagnostic_metrics_schema": 1,
        "status": "complete",
        "purpose": "bes2-root-cause",
        "variant": variant,
        "stage": "full",
        "created_utc": "2026-08-26T00:00:00+00:00",
        "readiness": {"path": "ready", "sha256": "b" * 64},
        "training": {
            "training_result": {
                "best_dev_f1": f1,
                "best_dev": best,
            }
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
    path = root / f"bes2-{variant}-full-f100-s0" / "diagnostic_metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def test_summarize_publishes_json_and_reproducible_markdown(tmp_path):
    root = tmp_path / "diagnostics"
    replay = S2_FROZEN_DEV_F1
    random_f1 = 0.8919449902
    reset = replay + 0.5 * (random_f1 - replay)
    _metrics(root, "current_replay", replay)
    _metrics(root, "first_conv_reset", reset)

    assert main(["summarize", "--diagnostic-root", str(root)]) == 0

    result = read_regular_json(
        root / ".control/BES2_ROOT_CAUSE.json",
        "root cause",
    )
    report = (root / ".control/BES2_ROOT_CAUSE.md").read_text()
    assert (
        result["decision"]["classification"]
        == "stem-conversion-explains-most-deficit"
    )
    assert result["decision"]["recovery"] == pytest.approx(0.5)
    assert "single-seed TRAIN+fixed-DEV8" in report
    assert "no significance" in report
    assert "BigEarthNet-S1 was not modified or tested" in report

    report_path = root / ".control/BES2_ROOT_CAUSE.md"
    report_path.chmod(0o644)
    report_path.write_text(report + "tampered\n", encoding="utf-8")
    with pytest.raises(Exception, match="not reproducible"):
        main(["summarize", "--diagnostic-root", str(root)])
