from __future__ import annotations

import inspect
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.h100 import bes2_diagnostic
from src.analysis import bes2_root_cause
from src.analysis.bes2_contract import BES2ContractError, payload_sha256


def _worker_args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        repo=tmp_path / "repo",
        diagnostic_root=tmp_path / "diagnostic",
        original_runs_root=tmp_path / "original",
        expected_git_sha="1" * 40,
        stage="probe",
        fraction=0.5,
        data_view_root=tmp_path / "view",
        data_view_receipt=tmp_path / "view/BES2_DATA_VIEW_READY.json",
        data_config=tmp_path / "repo/configs/data.yaml",
        weights_root=tmp_path / "view/weights",
        workers=6,
    )


def test_diagnostic_view_receipt_is_path_independent() -> None:
    staging = {
        "schema": 1,
        "status": "ready",
        "phase": "train",
        "purpose": "campaign",
        "contract": "train111-fixed-dev8-no-test-v1",
        "chips": ["train-scene"],
        "rasters": ["dev-scene"],
        "weights": ["checkpoint"],
        "training_cohort": None,
        "selected_artifacts": [],
    }
    payload = bes2_diagnostic.diagnostic_view_payload(
        source_git_sha="1" * 40,
        h100_ready_sha256="2" * 64,
        data_config_sha256="3" * 64,
        staging_receipt=staging,
    )

    assert set(payload) == {
        "schema",
        "status",
        "purpose",
        "source_git_sha",
        "h100_ready_sha256",
        "data_config_sha256",
        "authorized_splits",
        "forbidden_splits",
        "staging_receipt_sha256",
        "staging_receipt",
    }
    assert payload["authorized_splits"] == ["train", "dev8"]
    assert payload["forbidden_splits"] == ["test", "eval_final"]
    assert payload["staging_receipt_sha256"] == payload_sha256(staging)
    assert "data_view_root" not in json.dumps(payload)


def test_complete_test_receipt_is_exact_and_positive(tmp_path: Path) -> None:
    path = tmp_path / "BES2_TEST_GATE.json"
    payload = {
        "schema": 1,
        "status": "passed",
        "purpose": "bes2-diagnostic-complete-tests",
        "source_git_sha": "1" * 40,
        "command": ["/sealed/bin/python", "-B", "-m", "pytest", "-q"],
        "exit_code": 0,
        "passed": 407,
        "duration_seconds": 42.5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert bes2_diagnostic._test_receipt_valid(path, "1" * 40) == payload
    payload["passed"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(BES2ContractError, match="invalid"):
        bes2_diagnostic._test_receipt_valid(path, "1" * 40)


def test_worker_requeue_request_is_atomic_job_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = tmp_path / "requests"
    requests.mkdir(mode=0o700)
    monkeypatch.setenv("H100_REQUEUE_REQUEST_DIR", str(requests))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("H100_DIAGNOSTIC_GPU_SLOT", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "540001")

    bes2_diagnostic._request_requeue()

    marker = requests / "gpu-1.request"
    assert marker.read_text(encoding="utf-8") == "540001\n"
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600


def test_worker_requeue_rejects_symlink_request_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("H100_REQUEUE_REQUEST_DIR", str(link))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("H100_DIAGNOSTIC_GPU_SLOT", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "540001")

    with pytest.raises(BES2ContractError, match="unsafe"):
        bes2_diagnostic._request_requeue()
    assert list(real.iterdir()) == []


def test_worker_command_rebinds_current_train_dev_view(tmp_path: Path) -> None:
    command = bes2_diagnostic._worker_command(
        _worker_args(tmp_path), "current_replay", 0.5
    )
    joined = " ".join(command)

    for option in (
        "--data-view-root",
        "--data-view-receipt",
        "--data-config",
        "--weights-root",
    ):
        assert option in command
    assert "current_replay" in command
    assert "test_metrics.json" not in joined
    assert "final_metrics.json" not in joined


def test_completion_rejects_stale_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bes2_diagnostic,
        "validate_diagnostic_metrics",
        lambda *_args, **_kwargs: {
            "readiness": {"sha256": "a" * 64},
            "gpu_hours": 1.0,
        },
    )
    with pytest.raises(BES2ContractError, match="current BES2 readiness"):
        bes2_diagnostic._validated_completion(
            tmp_path,
            "current_replay",
            "probe",
            0.5,
            readiness_sha256="b" * 64,
        )


def test_stage_receipt_recomputes_pair_gpu_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for variant in bes2_diagnostic.DIAGNOSTIC_VARIANTS:
        path = bes2_diagnostic._metrics_path(
            tmp_path, variant, "probe", 0.5
        )
        path.parent.mkdir(parents=True)
        path.write_text(f"{variant}\n", encoding="utf-8")
    hours = {"current_replay": 2.5, "first_conv_reset": 3.5}

    def validated(
        _root,
        variant,
        _stage,
        fraction,
        *,
        readiness_sha256=None,
    ):
        assert fraction == 0.5
        assert readiness_sha256 == "c" * 64
        return {"gpu_hours": hours[variant]}

    monkeypatch.setattr(
        bes2_diagnostic, "_validated_completion", validated
    )
    receipt = bes2_diagnostic._stage_receipt(
        tmp_path,
        "probe",
        "c" * 64,
    )
    assert receipt["gpu_hours"] == 6.0
    assert receipt["fractions"] == [0.5]
    assert receipt["run_count"] == 2
    assert set(receipt["runs"]["f50"]) == set(
        bes2_diagnostic.DIAGNOSTIC_VARIANTS
    )


def test_full_stage_requires_both_fraction_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        bes2_diagnostic,
        "_validated_completion",
        lambda *_args, **_kwargs: {"gpu_hours": 1.0},
    )
    readiness_sha256 = "c" * 64
    for variant in bes2_diagnostic.DIAGNOSTIC_VARIANTS:
        path = bes2_diagnostic._metrics_path(
            tmp_path, variant, "full", 0.1
        )
        path.parent.mkdir(parents=True)
        path.write_text("f10 complete\n", encoding="utf-8")

    assert not bes2_diagnostic._stage_is_complete(
        root=tmp_path,
        stage="full",
        readiness_sha256=readiness_sha256,
    )

    for variant in bes2_diagnostic.DIAGNOSTIC_VARIANTS:
        path = bes2_diagnostic._metrics_path(
            tmp_path, variant, "full", 0.5
        )
        path.parent.mkdir(parents=True)
        path.write_text("f50 complete\n", encoding="utf-8")

    assert bes2_diagnostic._stage_is_complete(
        root=tmp_path,
        stage="full",
        readiness_sha256=readiness_sha256,
    )


def test_controller_requires_two_h100s_and_host_owned_requeue() -> None:
    controller = inspect.getsource(bes2_diagnostic.run_controller)
    worker = inspect.getsource(bes2_diagnostic.run_worker)
    receipt = inspect.getsource(bes2_diagnostic._stage_receipt)

    assert "_require_h100_devices(2)" in controller
    assert "_require_h100_devices(1)" in worker
    assert controller.index("signal.signal(signal.SIGUSR1") < controller.index(
        "subprocess.Popen"
    )
    assert "enumerate(incomplete)" in controller
    assert "CUDA_VISIBLE_DEVICES" in controller
    assert "H100_DIAGNOSTIC_GPU_SLOT" in controller
    assert "promote_hpc_checkpoint" in controller
    assert "scontrol" not in controller
    assert "state_path" not in receipt


def test_diagnostic_controller_cannot_emit_core_or_heldout_markers() -> None:
    source = inspect.getsource(bes2_diagnostic)
    forbidden = (
        "TRAINING_COHORT.json",
        "test_metrics.json",
        "final_metrics.json",
        "FINAL_EVAL",
    )
    assert not any(name in source for name in forbidden)


def test_root_cause_run_parser_requires_reconstructed_view() -> None:
    parser = bes2_root_cause._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--repo",
                "/repo",
                "--diagnostic-root",
                "/diagnostic",
                "--original-runs-root",
                "/original",
                "--variant",
                "current_replay",
                "--stage",
                "probe",
            ]
        )

    parsed = parser.parse_args(
        [
            "run",
            "--repo",
            "/repo",
            "--diagnostic-root",
            "/diagnostic",
            "--original-runs-root",
            "/original",
            "--variant",
            "current_replay",
            "--stage",
            "probe",
            "--fraction",
            "0.5",
            "--data-view-root",
            "/view",
            "--data-view-receipt",
            "/view/BES2_DATA_VIEW_READY.json",
            "--data-config",
            "/repo/configs/data.yaml",
            "--weights-root",
            "/view/weights",
        ]
    )
    assert parsed.data_view_root == Path("/view")
    assert parsed.fraction == 0.5
