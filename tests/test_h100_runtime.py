"""CPU-only fixture/static guards for the strict-FP32 H100 Slurm lane."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.h100 import (
    acceptance,
    build_container,
    campaign,
    contracts,
    cutover,
    host_test_gate,
    operator_cutover,
    slurm_smoke,
    source_validation,
)
from scripts.h100.precision import (
    STRICT_SENTINEL,
    apply_strict_fp32,
    assert_sitecustomize_active,
)
from scripts.h100.reverse_results import command as reverse_command

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "slurm/h100/campaign.sbatch"
SMOKE_SBATCH = REPO / "slurm/h100/smoke.sbatch"
SUBMIT = REPO / "slurm/h100/submit.sh"
SITE_EXAMPLE = REPO / "slurm/h100/site.env.example"
SHIM = REPO / "slurm/h100/shims/scontrol"
DEFINITION = REPO / "containers/h100-strict-fp32.def"
V100_RECEIPT_SCHEMA = REPO / "slurm/h100/V100_CORE_ARCHIVED.schema.json"
V100_ARCHIVE_SCHEMA = REPO / "slurm/h100/V100_CORE_ARCHIVE_MANIFEST.schema.json"


def fake_torch():
    return SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(fp32_precision="tf32")),
            cudnn=SimpleNamespace(
                conv=SimpleNamespace(fp32_precision="tf32"),
                rnn=SimpleNamespace(fp32_precision="tf32"),
            ),
        )
    )


def h100_devices():
    return [
        {
            "name": "NVIDIA H100 80GB HBM3",
            "uuid": f"GPU-H100-{index}",
            "compute_capability": [9, 0],
            "total_memory_bytes": 85_000_000_000,
        }
        for index in range(8)
    ]


def reference_provenance(git_sha="v100-sha"):
    return {
        "campaign_id": "fresh34-v100-fp32-20260726",
        "git_sha": git_sha,
        "hardware": "Tesla V100-SXM2-32GB",
        "container_local_gpu": 1,
        "gpu_uuid": "GPU-v100",
        "started_utc": "2026-07-26T00:00:00Z",
        "finished_utc": "2026-07-26T01:00:00Z",
        "elapsed_hours": 1.0,
        "gpu_hours": 1.0,
        "reference_precision": "published",
        "environment_sha256": "a" * 64,
        "environment_lock_sha256": "b" * 64,
        "campaign_manifest_sha256": "c" * 64,
        "runtime_launcher_sha256": "d" * 64,
    }


def strict_backend():
    return {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }


def h100_hardware():
    return {
        "torch": "2.11.0+cu126",
        "cuda_build": "12.6",
        "driver_version": "590.1",
        "backend": strict_backend(),
        "devices": h100_devices(),
    }


def completion_payload(cell, *, scored=False):
    payload = {
        "exp_id": cell.exp_id,
        "git_sha": "git",
        "detector_sha256": "detector",
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "epochs_run": 50,
        "best_dev_f1": 0.5,
    }
    if scored:
        payload.update(
            {
                "test_inference_precision": "32-true",
                "test_f1": 0.4,
                "test_precision": 0.5,
                "test_recall": 0.3,
                "test_near_shore_f1": 0.2,
            }
        )
    return payload


def runtime_provenance(cell, *, finalized=False):
    accepted_class = campaign.hardware_class(h100_hardware())
    payload = {
        "campaign_id": "campaign",
        "exp_id": cell.exp_id,
        "git_sha": "git",
        "detector_sha256": "detector",
        "sif_sha256": "sif",
        "base_oci": "oci",
        "package": dict(smoke_bindings()["package"]),
        "acceptance_uuid": "acceptance",
        "source_validation_sha256": "source-receipt",
        "cutover_ready_sha256": "cutover-receipt",
        "v100_core_archived_sha256": "v100-receipt",
        "strict_fp32": strict_backend(),
        "accepted_hardware_class": accepted_class,
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "gpu_name": accepted_class["gpu_name"],
        "gpu_total_memory_bytes": accepted_class["total_memory_bytes"],
        "compute_capability": accepted_class["compute_capability"],
        "driver_version": accepted_class["driver_version"],
        "torch": accepted_class["torch"],
        "cuda_build": accepted_class["cuda_build"],
        "gpu_uuid": "GPU-H100-live",
        "attempts": [{"gpu_uuid": "GPU-H100-live"}],
    }
    if finalized:
        payload.update(
            {
                "attempts": [
                    {
                        "gpu_uuid": "GPU-H100-live",
                        "exit_code": 0,
                        "finished_utc": "2026-07-27T01:00:00+00:00",
                    }
                ],
                "accumulated_active_seconds": 3600.0,
                "elapsed_hours": 1.0,
                "completed_utc": "2026-07-27T01:00:00+00:00",
                "epochs_run": 50,
                "best_dev_f1": 0.5,
                "test_f1": 0.4,
            }
        )
    return payload


def existing_state_kwargs(runs):
    return {
        "runs_root": runs,
        "campaign_id": "campaign",
        "git_sha": "git",
        "detector_sha256": "detector",
        "sif_sha256": "sif",
        "base_oci": "oci",
        "package": dict(smoke_bindings()["package"]),
        "acceptance_uuid": "acceptance",
        "source_validation_sha256": "source-receipt",
        "cutover_ready_sha256": "cutover-receipt",
        "v100_core_archived_sha256": "v100-receipt",
        "strict_fp32": strict_backend(),
        "accepted_hardware_class": campaign.hardware_class(h100_hardware()),
    }


def smoke_bindings():
    return slurm_smoke.make_bindings(
        git_sha="1" * 40,
        detector_sha256="2" * 64,
        sif_sha256="3" * 64,
        container_build_sha256="4" * 64,
        package_manifest_sha256="5" * 64,
        package_ready_sha256="6" * 64,
        package_sha256sums_sha256="7" * 64,
        package_repo_bundle_sha256="8" * 64,
    )


def completed_smoke(tmp_path):
    bindings = smoke_bindings()
    runs = tmp_path / "runs"
    assert slurm_smoke.run_allocation(
        runs_root=runs,
        bindings=bindings,
        job_id="1234",
        restart_count=0,
    ) == slurm_smoke.HOST_REQUEUE_EXIT_CODE
    root = slurm_smoke.smoke_root(runs)
    slurm_smoke.authorize_host_requeue(
        root=root,
        bindings=bindings,
        job_id="1234",
        restart_count=0,
    )
    assert slurm_smoke.run_allocation(
        runs_root=runs,
        bindings=bindings,
        job_id="1234",
        restart_count=1,
    ) == 0
    ready = root / slurm_smoke.READY_NAME
    return ready, bindings, json.loads(ready.read_text())


def add_host_acceptance_receipts(root: Path, ready: dict) -> None:
    source_path = root / source_validation.SOURCE_RECEIPT_NAME
    source_receipt = source_validation.expected_receipt(
        git_sha=ready["source"]["git_sha"],
        frozen_sha256=ready["source"]["frozen_sha256"],
        package=ready["package"],
    )
    contracts.atomic_write_json(source_path, source_receipt)
    source_sha = contracts.sha256_file(source_path)

    host_log = root / "acceptance-logs/pytest-handoff-host.log"
    host_log.parent.mkdir(parents=True, exist_ok=True)
    host_log.write_text("host tests passed\n")
    host_receipt_path = root / "HOST_HANDOFF_TESTS.json"
    host_receipt = {
        "schema": 1,
        "status": "passed",
        "slice": "host-handoff",
        "command": host_test_gate.HOST_COMMAND,
        "source_validation_sha256": source_sha,
        "duration_seconds": 1.0,
        "log": {
            "path": str(host_log),
            "sha256": contracts.sha256_file(host_log),
        },
    }
    contracts.atomic_write_json(host_receipt_path, host_receipt)

    sif_log = root / "acceptance-logs/pytest-sif-remaining.log"
    sif_log.write_text("SIF tests passed\n")
    sif_command = [
        "-m",
        "pytest",
        "-q",
        *(f"--ignore={path}" for path in host_test_gate.HOST_TESTS),
    ]
    suite = {
        "schema": 1,
        "status": "passed",
        "source_validation_sha256": source_sha,
        "coverage": {
            "host": host_test_gate.HOST_TESTS,
            "sif": "all pytest collection except the two host-only files",
            "aggregate": "entire repository pytest suite",
        },
        "host_handoff": {
            "receipt_path": str(host_receipt_path),
            "receipt_sha256": contracts.sha256_file(host_receipt_path),
            "receipt": host_receipt,
        },
        "sif_remaining": {
            "command": sif_command,
            "duration_seconds": 2.0,
            "log": {
                "path": str(sif_log),
                "sha256": contracts.sha256_file(sif_log),
            },
        },
        "aggregate_duration_seconds": 3.0,
    }
    suite_path = root / "PYTEST_ACCEPTANCE.json"
    contracts.atomic_write_json(suite_path, suite)
    ready["gates"]["pytest_seconds"] = 3.0
    ready["source_validation"] = {
        "path": str(source_path),
        "sha256": source_sha,
        "receipt": source_receipt,
    }
    ready["test_suite"] = {
        "path": str(suite_path),
        "sha256": contracts.sha256_file(suite_path),
        "receipt": suite,
    }
    steps = {
        label: int(item["steps_per_epoch"])
        for label, item in acceptance.EXPECTED_FRACTION_WORKLOAD.items()
    }
    ready["projection"].update(
        {
            "fraction_workload": acceptance.EXPECTED_FRACTION_WORKLOAD,
            "steps_per_epoch": steps,
            "grid_steps_per_epoch": 8 * sum(steps.values()),
        }
    )


def operator_evidence_fixture(tmp_path: Path) -> dict:
    meta = tmp_path / ".h100"
    external = tmp_path / "external"
    meta.mkdir()
    external.mkdir()
    h100_git = "1" * 40
    v100_git = "9" * 40
    package = {
        "manifest_sha256": "2" * 64,
        "ready_sha256": "3" * 64,
        "sha256sums_sha256": "4" * 64,
        "repo_bundle_sha256": "5" * 64,
    }
    cutover_path = meta / "CUTOVER_READY.json"
    cutover_payload = {
        "schema": 1,
        "status": "cutover-ready",
        "created_utc": "2026-07-27T00:00:00+00:00",
        "acceptance": {
            "uuid": "acceptance-uuid",
            "source": {"git_sha": h100_git},
            "sif": {"sha256": "6" * 64},
            "package": package,
            "projection": {
                "conservative_h100_wall_hours": 2.0,
                "remaining_v100_wall_hours": 3.0,
            },
        },
        "cutover_forecast": {
            "conservative_h100_wall_hours": 2.0,
            "acceptance_remaining_v100_wall_hours": 3.0,
            "current_remaining_v100_wall_hours": 2.5,
        },
        "references": {
            name: {
                "metrics": {"exp_id": name},
                "metrics_sha256": "7" * 64,
                "provenance": {
                    "git_sha": v100_git,
                    "campaign_id": "v100-campaign",
                },
                "provenance_sha256": "8" * 64,
            }
            for name in ("r2", "r3")
        },
        "v100_action": "none; this guard never stops or signals V100 processes",
    }
    contracts.atomic_write_json(cutover_path, cutover_payload)
    canonical_manifest = meta / "V100_CORE_ARCHIVE_MANIFEST.json"
    source_manifest = external / "archive-manifest.json"
    manifest_payload = {
        "schema": 1,
        "status": "v100-core-diagnostics-archived",
        "scope": "v100-core-diagnostics",
        "diagnostic_status": "non-reportable-diagnostic",
        "git_sha": v100_git,
        "campaign_id": "v100-campaign",
        "stopped_utc": "2026-07-27T01:00:00+00:00",
        "archived_utc": "2026-07-27T01:10:00+00:00",
        "file_count": 2,
        "total_bytes": 20,
    }
    contracts.atomic_write_json(source_manifest, manifest_payload)
    manifest_sha = contracts.sha256_file(source_manifest)
    source_receipt = external / "V100_CORE_ARCHIVED.json"
    receipt_payload = {
        "schema": 1,
        "status": "v100-core-archived",
        "created_utc": "2026-07-27T01:15:00+00:00",
        "attestation": "external-human-operator",
        "cutover_ready_sha256": contracts.sha256_file(cutover_path),
        "h100": {
            "acceptance_uuid": "acceptance-uuid",
            "git_sha": h100_git,
            "sif_sha256": "6" * 64,
            "package": package,
        },
        "v100": {
            "git_sha": v100_git,
            "campaign_id": "v100-campaign",
            "stopped_utc": "2026-07-27T01:00:00+00:00",
            "stop_mode": "graceful",
            "running_core_processes": 0,
            "diagnostic_status": "non-reportable-diagnostic",
        },
        "archive": {
            "manifest_path": str(canonical_manifest),
            "manifest_sha256": manifest_sha,
        },
    }
    contracts.atomic_write_json(source_receipt, receipt_payload)
    return {
        "meta_root": meta,
        "cutover_ready": cutover_path,
        "cutover_ready_sha256": contracts.sha256_file(cutover_path),
        "receipt": source_receipt,
        "receipt_sha256": contracts.sha256_file(source_receipt),
        "archive_manifest": source_manifest,
        "archive_manifest_sha256": manifest_sha,
        "bound_archive_manifest": canonical_manifest,
        "expected_h100_git_sha": h100_git,
        "expected_sif_sha256": "6" * 64,
        "expected_package_hashes": package,
        "expected_reference_git_sha": v100_git,
        "expected_reference_campaign_id": "v100-campaign",
    }


def test_matrix_is_exactly_32_and_expensive_first():
    cells = contracts.load_cells(REPO)
    assert len(cells) == len({cell.exp_id for cell in cells}) == 32
    assert [cell.fraction for cell in cells] == (
        [1.0] * 8 + [0.5] * 8 + [0.25] * 8 + [0.1] * 8
    )
    assert [cell.track for cell in cells[:8]] == ["cnn"] * 4 + ["vit"] * 4


def test_failure_or_preemption_fail_stops_new_launches():
    cells = contracts.load_cells(REPO)
    common = {
        "cells": cells,
        "running_ids": set(),
        "complete_ids": set(),
        "free_gpus": list(range(8)),
    }
    assert campaign.next_launches(**common, failure_seen=True, preemption_seen=False) == []
    assert campaign.next_launches(**common, failure_seen=False, preemption_seen=True) == []
    launches = campaign.next_launches(
        **common, failure_seen=False, preemption_seen=False
    )
    assert [cell.exp_id for _gpu, cell in launches] == [
        cell.exp_id for cell in cells[:8]
    ]
    allowed = {cells[1].exp_id, cells[2].exp_id}
    resumed = campaign.fail_stop_launches(
        cells,
        allowed_ids=allowed,
        failed_ids={cells[0].exp_id},
        running_ids=set(),
        complete_ids={cells[1].exp_id},
        free_gpus=list(range(8)),
        preemption_seen=False,
    )
    assert [cell.exp_id for _gpu, cell in resumed] == [cells[2].exp_id]
    prior = {
        "fail_stop": {
            "engaged": True,
            "failed": [cells[0].exp_id],
            "allowed_to_finish": sorted(allowed),
        }
    }
    assert campaign.restore_fail_stop(prior, cells) == (
        True,
        {cells[0].exp_id},
        allowed,
    )
    with pytest.raises(RuntimeError, match="inconsistent"):
        campaign.restore_fail_stop(
            {
                "fail_stop": {
                    "engaged": True,
                    "failed": [cells[0].exp_id],
                    "allowed_to_finish": [cells[0].exp_id],
                }
            },
            cells,
        )


def test_final_grid_requires_exact_32_finite_test_rows_and_monotonicity(
    tmp_path, monkeypatch
):
    cells = contracts.load_cells(REPO)
    runs = tmp_path / "runs"
    rows = [
        {
            "exp_id": cell.exp_id,
            "test_f1": "0.5",
            "monotonicity_ok": "True",
            "git_sha": "git",
            "detector_sha256": "detector",
        }
        for cell in cells
    ]

    def fake_collect(_command, *, cwd, check):
        assert cwd == REPO
        assert check is True
        grid = runs / "summary/grid.csv"
        grid.parent.mkdir(parents=True, exist_ok=True)
        header = ",".join(rows[0])
        lines = [header] + [",".join(row.values()) for row in rows]
        grid.write_text("\n".join(lines) + "\n")
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(campaign.subprocess, "run", fake_collect)
    grid = campaign.collect_and_validate_grid(
        repo=REPO,
        runs_root=runs,
        cells=cells,
        git_sha="git",
        detector_sha256="detector",
    )
    assert grid == runs / "summary/grid.csv"
    rows[0]["monotonicity_ok"] = "False"
    with pytest.raises(RuntimeError, match="monotonicity STOP"):
        campaign.collect_and_validate_grid(
            repo=REPO,
            runs_root=runs,
            cells=cells,
            git_sha="git",
            detector_sha256="detector",
        )


def test_campaign_wires_train_then_test_wrapper_and_single_controller_lock():
    source = (REPO / "scripts/h100/campaign.py").read_text()
    wrapper = (REPO / "scripts/h100/cell.py").read_text()
    assert '"scripts.h100.cell"' in source
    assert '"scripts/score_test_split.py"' in wrapper
    assert '"--only"' in wrapper and '"--device",\n            "cuda"' in wrapper
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in source
    assert '"src.analysis.curves",\n            "collect"' in source


def test_strict_policy_uses_new_pytorch_backend_api(monkeypatch):
    torch = fake_torch()
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    state = apply_strict_fp32(torch)
    assert set(state.values()) == {"ieee"}
    assert os.environ[STRICT_SENTINEL] == "1"
    assert_sitecustomize_active(torch)
    source = (REPO / "scripts/h100/precision.py").read_text()
    assert ".allow_tf32 =" not in source
    assert "cudnn.conv.fp32_precision" in source
    assert "cudnn.rnn.fp32_precision" in source


def test_sitecustomize_fails_closed_without_override():
    env = dict(os.environ)
    env.pop("NVIDIA_TF32_OVERRIDE", None)
    env.pop(STRICT_SENTINEL, None)
    env["PYTHONPATH"] = f"{REPO / 'scripts/h100'}:{REPO}"
    completed = subprocess.run(
        [sys.executable, "-c", "print('must-not-run')"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 86
    assert "FATAL H100 strict-FP32 initialization failed" in completed.stderr
    assert "must-not-run" not in completed.stdout


def test_h100_inventory_requires_eight_cc90_unique_uuids():
    assert len(contracts.validate_gpu_inventory(h100_devices())) == 8
    with pytest.raises(RuntimeError, match="exactly 8"):
        contracts.validate_gpu_inventory(h100_devices()[:7])
    bad = h100_devices()
    bad[0]["compute_capability"] = [8, 0]
    with pytest.raises(RuntimeError, match="9.0"):
        contracts.validate_gpu_inventory(bad)
    bad = h100_devices()
    bad[0]["uuid"] = ""
    with pytest.raises(RuntimeError, match="UUID"):
        contracts.validate_gpu_inventory(bad)
    bad = h100_devices()
    bad[0]["total_memory_bytes"] -= 1
    with pytest.raises(RuntimeError, match="exact GPU name and memory"):
        contracts.validate_gpu_inventory(bad)


def test_completion_marker_is_recipe_and_hash_checked(tmp_path):
    cell = contracts.load_cells(REPO)[0]
    marker = tmp_path / "final_metrics.json"
    payload = {
        "exp_id": cell.exp_id,
        "git_sha": "git",
        "detector_sha256": "detector",
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "best_dev_f1": 0.5,
    }
    marker.write_text(json.dumps(payload))
    assert contracts.validate_completion_marker(
        marker, cell=cell, git_sha="git", detector_sha256="detector"
    ) == payload
    payload["precision"] = "16-mixed"
    marker.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="recipe-matched"):
        contracts.validate_completion_marker(
            marker, cell=cell, git_sha="git", detector_sha256="detector"
        )


def test_completed_cell_reuse_requires_scoring_checkpoints_and_full_provenance(tmp_path):
    cell = contracts.load_cells(REPO)[0]
    runs = tmp_path / "runs"
    run = runs / cell.exp_id
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "best.ckpt").write_bytes(b"best")
    (checkpoints / "last.ckpt").write_bytes(b"last")
    marker = {
        "exp_id": cell.exp_id,
        "git_sha": "git",
        "detector_sha256": "detector",
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "epochs_run": 50,
        "best_dev_f1": 0.5,
        "test_f1": 0.4,
        "test_precision": 0.5,
        "test_recall": 0.3,
        "test_near_shore_f1": 0.2,
        "test_inference_precision": "32-true",
    }
    (run / "final_metrics.json").write_text(json.dumps(marker))
    accepted_class = campaign.hardware_class(h100_hardware())
    package = dict(smoke_bindings()["package"])
    provenance = {
        "campaign_id": "campaign",
        "exp_id": cell.exp_id,
        "git_sha": "git",
        "detector_sha256": "detector",
        "sif_sha256": "sif",
        "base_oci": "oci",
        "package": package,
        "acceptance_uuid": "acceptance",
        "source_validation_sha256": "source-receipt",
        "cutover_ready_sha256": "cutover-receipt",
        "v100_core_archived_sha256": "v100-receipt",
        "strict_fp32": strict_backend(),
        "accepted_hardware_class": accepted_class,
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "gpu_name": accepted_class["gpu_name"],
        "gpu_total_memory_bytes": accepted_class["total_memory_bytes"],
        "compute_capability": accepted_class["compute_capability"],
        "driver_version": accepted_class["driver_version"],
        "torch": accepted_class["torch"],
        "cuda_build": accepted_class["cuda_build"],
        "gpu_uuid": "GPU-H100-live",
        "attempts": [
            {
                "gpu_uuid": "GPU-H100-live",
                "exit_code": 0,
                "finished_utc": "2026-07-27T01:00:00+00:00",
            }
        ],
        "accumulated_active_seconds": 3600.0,
        "elapsed_hours": 1.0,
        "completed_utc": "2026-07-27T01:00:00+00:00",
        "epochs_run": 50,
        "best_dev_f1": 0.5,
        "test_f1": 0.4,
    }
    (run / "runtime_provenance.json").write_text(json.dumps(provenance))
    kwargs = {
        "runs_root": runs,
        "campaign_id": "campaign",
        "git_sha": "git",
        "detector_sha256": "detector",
        "sif_sha256": "sif",
        "base_oci": "oci",
        "package": package,
        "acceptance_uuid": "acceptance",
        "source_validation_sha256": "source-receipt",
        "cutover_ready_sha256": "cutover-receipt",
        "v100_core_archived_sha256": "v100-receipt",
        "strict_fp32": strict_backend(),
        "accepted_hardware_class": accepted_class,
    }
    assert campaign.existing_cell_state(cell, **kwargs) == "complete"
    marker["test_f1"] = float("nan")
    (run / "final_metrics.json").write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="fully train.test complete"):
        campaign.existing_cell_state(cell, **kwargs)


def test_training_only_and_unfinalized_scored_cells_resume(tmp_path):
    cell = contracts.load_cells(REPO)[0]
    runs = tmp_path / "runs"
    run = runs / cell.exp_id
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints/best.ckpt").write_bytes(b"best")
    (run / "checkpoints/last.ckpt").write_bytes(b"last")
    marker = completion_payload(cell)
    (run / "final_metrics.json").write_text(json.dumps(marker))
    provenance = runtime_provenance(cell)
    (run / "runtime_provenance.json").write_text(json.dumps(provenance))
    kwargs = existing_state_kwargs(runs)
    assert campaign.existing_cell_state(cell, **kwargs) == "resume"

    marker.update(
        {
            "test_inference_precision": "32-true",
            "test_f1": 0.4,
            "test_precision": 0.5,
            "test_recall": 0.3,
            "test_near_shore_f1": 0.2,
        }
    )
    (run / "final_metrics.json").write_text(json.dumps(marker))
    assert campaign.existing_cell_state(cell, **kwargs) == "resume"
    provenance["package"] = {**provenance["package"], "manifest_sha256": "bad"}
    (run / "runtime_provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError, match="runtime provenance"):
        campaign.existing_cell_state(cell, **kwargs)


def test_acceptance_probe_marker_requires_finite_fp32_batch16(tmp_path):
    marker = tmp_path / "final_metrics.json"
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "best.ckpt").write_bytes(b"best")
    (checkpoints / "last.ckpt").write_bytes(b"last")
    marker.write_text(
        json.dumps(
            {
                "exp_id": "probe",
                "precision": "32-true",
                "micro_batch": 16,
                "gradient_accumulation": 1,
                "effective_batch": 16,
                "best_dev_f1": 0.2,
                "train_loss": 1.5,
                "git_sha": "git",
                "detector_sha256": "detector",
            }
        )
    )
    acceptance.validate_probe_marker(
        marker,
        "probe",
        expected_git_sha="git",
        expected_detector_sha256="detector",
    )
    payload = json.loads(marker.read_text())
    payload["train_loss"] = float("nan")
    marker.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="invalid strict-FP32"):
        acceptance.validate_probe_marker(
            marker,
            "probe",
            expected_git_sha="git",
            expected_detector_sha256="detector",
        )


def test_acceptance_refuses_stale_ready_probe_and_nonfinite_projection_input(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    ready = runs / ".h100/H100_READY.json"
    acceptance.validate_fresh_acceptance_state(
        ready=ready,
        runs_root=runs,
        remaining_v100_wall_hours=12.0,
    )
    (runs / acceptance.PROBE_IDS[0]).mkdir()
    with pytest.raises(RuntimeError, match="probe namespaces"):
        acceptance.validate_fresh_acceptance_state(
            ready=ready,
            runs_root=runs,
            remaining_v100_wall_hours=12.0,
        )
    (runs / acceptance.PROBE_IDS[0]).rmdir()
    ready.parent.mkdir()
    ready.write_text("{}")
    with pytest.raises(RuntimeError, match="already exists"):
        acceptance.validate_fresh_acceptance_state(
            ready=ready,
            runs_root=runs,
            remaining_v100_wall_hours=12.0,
        )
    ready.unlink()
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(RuntimeError, match="finite and positive"):
            acceptance.validate_fresh_acceptance_state(
                ready=ready,
                runs_root=runs,
                remaining_v100_wall_hours=invalid,
            )


def test_projection_records_conservative_wall_clock():
    projection = contracts.estimate_grid_projection(
        probe_steps=200,
        probe_seconds=100,
        steps_per_epoch={"f10": 100, "f25": 250, "f50": 500, "f100": 1000},
    )
    assert projection["steps_per_second"] == 2
    assert projection["grid_steps_per_epoch"] == 8 * 1850
    assert projection["ceiling_gpu_hours"] > projection["expected_gpu_hours"]
    assert projection["ceiling_wall_hours_ideal"] == pytest.approx(
        projection["ceiling_gpu_hours"] / 8
    )


def test_staging_aware_projection_counts_each_required_allocation():
    projected = contracts.staging_aware_wall_clock(
        training_wall_hours=40.0,
        staging_seconds=7200.0,
    )
    assert projected["usable_training_hours_per_allocation"] == 34.25
    assert projected["projected_allocation_count"] == 2
    assert projected["conservative_h100_wall_hours"] == 44.0
    with pytest.raises(ValueError, match="no usable training"):
        contracts.staging_aware_wall_clock(
            training_wall_hours=1.0,
            staging_seconds=36.5 * 3600,
        )


def test_cutover_rechecks_current_v100_forecast():
    ready = {
        "projection": {
            "conservative_h100_wall_hours": 20.0,
            "remaining_v100_wall_hours": 30.0,
        }
    }
    assert cutover.validate_current_v100_advantage(ready, 25.0) == {
        "conservative_h100_wall_hours": 20.0,
        "acceptance_remaining_v100_wall_hours": 30.0,
        "current_remaining_v100_wall_hours": 25.0,
    }
    with pytest.raises(RuntimeError, match="no longer slower"):
        cutover.validate_current_v100_advantage(ready, 19.0)


def test_projection_uses_exact_pinned_fraction_workload_and_scratch_gate():
    steps = {
        label: int(item["steps_per_epoch"])
        for label, item in acceptance.EXPECTED_FRACTION_WORKLOAD.items()
    }
    assert steps == {"f10": 701, "f25": 1662, "f50": 3310, "f100": 6588}
    assert [
        item["chip_count"] for item in acceptance.EXPECTED_FRACTION_WORKLOAD.values()
    ] == [11_218, 26_603, 52_967, 105_408]
    projection = contracts.estimate_grid_projection(
        probe_steps=200,
        probe_seconds=100,
        steps_per_epoch=steps,
    )
    assert projection["grid_steps_per_epoch"] == 8 * sum(steps.values())
    assert acceptance.validate_scratch_free_before_extraction(500_000_000_000) == (
        500_000_000_000
    )
    with pytest.raises(RuntimeError, match="pre-extraction"):
        acceptance.validate_scratch_free_before_extraction(499_999_999_999)


def test_hpc_checkpoint_is_atomically_promoted(tmp_path):
    run = tmp_path / "cell"
    hpc = run / "hpc_ckpt_1.ckpt"
    hpc.parent.mkdir(parents=True)
    hpc.write_bytes(b"checkpoint")
    promoted = campaign.promote_hpc_checkpoint(run)
    assert promoted == run / "checkpoints/last.ckpt"
    assert promoted.read_bytes() == b"checkpoint"
    assert not promoted.with_suffix(".ckpt.promoting").exists()

    # A stale training HPC checkpoint must never replace the durable completed
    # training checkpoint when preemption arrives during restartable scoring.
    (run / "cell_wrapper.json").write_text(
        json.dumps({"phase": "score-test", "exp_id": "fixture"})
    )
    best = run / "checkpoints/best.ckpt"
    best.write_bytes(b"best")
    promoted.write_bytes(b"newer-last")
    assert campaign.checkpoint_for_preemption(run) == promoted
    assert promoted.read_bytes() == b"newer-last"


def test_slurm_smoke_interrupts_requeues_once_and_resumes_second_allocation(tmp_path):
    ready, bindings, receipt = completed_smoke(tmp_path)
    validated = slurm_smoke.validate_smoke_receipt(
        ready,
        expected_bindings=bindings,
    )
    assert validated == receipt
    assert receipt["real_requeue_count"] == 1
    assert receipt["signal"] == {
        "name": "SIGUSR1",
        "count": 1,
        "handled": True,
    }
    assert receipt["synthetic"] == {
        "cell_ids": [slurm_smoke.SYNTHETIC_CELL_ID],
        "launch_count": 2,
    }
    assert [item["restart_count"] for item in receipt["allocations"]] == [0, 1]
    assert receipt["checkpoint"]["hpc_sha256"] == receipt["checkpoint"]["last_sha256"]
    assert receipt["resume"]["final_step"] > receipt["resume"]["resumed_step"]
    smoke_run = ready.parent / "synthetic-cell"
    assert (smoke_run / "checkpoints/last.ckpt").is_file()
    assert not list(smoke_run.rglob("*.promoting"))
    assert not any((tmp_path / "runs" / cell.exp_id).exists() for cell in contracts.load_cells(REPO))


def test_slurm_smoke_requeue_authorization_is_one_shot_and_binding_locked(tmp_path):
    bindings = smoke_bindings()
    runs = tmp_path / "runs"
    assert slurm_smoke.run_allocation(
        runs_root=runs,
        bindings=bindings,
        job_id="4321",
        restart_count=0,
    ) == 75
    root = slurm_smoke.smoke_root(runs)
    slurm_smoke.authorize_host_requeue(
        root=root,
        bindings=bindings,
        job_id="4321",
        restart_count=0,
    )
    with pytest.raises(RuntimeError, match="one-shot"):
        slurm_smoke.authorize_host_requeue(
            root=root,
            bindings=bindings,
            job_id="4321",
            restart_count=0,
        )
    incompatible = json.loads(json.dumps(bindings))
    incompatible["package"]["manifest_sha256"] = "9" * 64
    with pytest.raises(RuntimeError, match="bindings mismatch"):
        slurm_smoke.run_allocation(
            runs_root=runs,
            bindings=incompatible,
            job_id="4321",
            restart_count=1,
        )


def test_container_is_digest_pinned_offline_and_exact():
    definition = DEFINITION.read_text()
    assert build_container.definition_base(DEFINITION) == build_container.PINNED_BASE
    assert "@sha256:28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941" in definition
    post = definition.split("%post", 1)[1].split("%environment", 1)[0]
    assert "--no-index" in post
    assert not any(token in post for token in ("apt ", "apt-get", "curl ", "wget "))
    build_source = (REPO / "scripts/h100/build_container.py").read_text()
    assert '"build",' in build_source and '"--fakeroot"' in build_source
    assert '"inspect", f"docker://{base}"' in build_source
    assert "temporary_sif" in build_source
    assert "os.replace(temporary_sif, output)" in build_source
    assert "assert_wheelhouse_unchanged(wheelhouse, wheels)" in build_source
    assert 'EXPECTED_PYTHON_VERSION = "3.11.15"' in build_source
    assert '"python_version": python_version' in build_source


def test_wheelhouse_rejects_nonwheel_and_symlink_entries(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "example-1-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    initial = build_container.wheelhouse_manifest(wheelhouse)
    assert set(initial) == {wheel.name}
    assert build_container.assert_wheelhouse_unchanged(wheelhouse, initial) == initial
    wheel.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="changed while the SIF was building"):
        build_container.assert_wheelhouse_unchanged(wheelhouse, initial)
    wheel.write_bytes(b"wheel")
    (wheelhouse / "README.txt").write_text("unexpected")
    with pytest.raises(RuntimeError, match="regular .whl"):
        build_container.wheelhouse_manifest(wheelhouse)
    (wheelhouse / "README.txt").unlink()
    (wheelhouse / "alias.whl").symlink_to(wheel)
    with pytest.raises(RuntimeError, match="regular .whl"):
        build_container.wheelhouse_manifest(wheelhouse)


def test_normalized_sif_freeze_allows_only_bootstrap_extras():
    lock = "Alpha_Pkg==1.0\nbeta==2\n"
    freeze = "alpha-pkg==1.0\nbeta==2\npip==99\nsetuptools==80\nwheel==1\n"
    assert build_container.assert_freeze_matches_lock(lock, freeze) == {
        "alpha-pkg": "1.0",
        "beta": "2",
    }
    with pytest.raises(RuntimeError, match="unexpected"):
        build_container.assert_freeze_matches_lock(lock, freeze + "mystery==1\n")


def test_slurm_defaults_hash_order_scratch_and_links_are_static():
    job = SBATCH.read_text()
    assert "#SBATCH --account=geofam" in job
    assert "#SBATCH --partition=minor-use-case" in job
    assert "#SBATCH --reservation=geofam" in job
    assert "#SBATCH --nodes=1" in job
    assert "#SBATCH --ntasks=1" in job
    assert "#SBATCH --gpus-per-node=8" in job
    assert "#SBATCH --cpus-per-task=48" in job
    assert "#SBATCH --mem=256G" in job
    assert "#SBATCH --time=1-12:30:00" in job
    assert "#SBATCH --signal=B:USR1@900" in job
    assert "#SBATCH --requeue" in job
    export_at = job.index("export NVIDIA_TF32_OVERRIDE=0")
    source_at = job.index('source "$site_env"')
    assert export_at < source_at
    assert job.index("scratch_free=") < job.index("scripts.handoff extract")
    assert 'git clone "$H100_REPO_BUNDLE" "$repo"' in job
    assert '--destination "$payload"' in job
    for relative in ("chips", "raw", "weights"):
        assert f'data/$relative' in job
    assert 'ln -s "$H100_RUNS_ROOT" "$repo/runs"' in job
    assert 'PYTHONPATH="$repo/scripts/h100:$repo' in job
    assert 'check_sha "$H100_SIF_BUILD_SHA256"' in job
    assert "while true; do" in job and 'kill -0 "$controller_pid"' in job
    assert "500000000000" in job
    assert "exec --cleanenv --containall --nv" in job
    assert 'mkdir "$payload"' not in job
    assert "unset BOX_JWT_CONFIG BOX_FOLDER_ID" in job
    assert "scripts.h100.strict_fp32_probe --expected-gpus 8" in job
    assert '"${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue' in job


def test_submit_and_site_interfaces_are_safe_and_untracked():
    submit = SUBMIT.read_text()
    example = SITE_EXAMPLE.read_text()
    assert '--account="${H100_ACCOUNT:-geofam}"' in submit
    assert '--partition="${H100_PARTITION:-minor-use-case}"' in submit
    assert '--reservation="${H100_RESERVATION:-geofam}"' in submit
    assert '--output="$H100_JOB_LOG_DIR/%x-%j.out"' in submit
    assert "realpath --relative-to" in submit
    assert 'env -u BOX_JWT_CONFIG -u BOX_FOLDER_ID sbatch' in submit
    assert '"$mode" != "smoke"' in submit
    for name in (
        "H100_PROJECT_ROOT",
        "H100_TRANSFER_PYTHON",
        "H100_JOB_LOG_DIR",
        "H100_MAIL_USER",
        "BOX_JWT_CONFIG",
        "BOX_FOLDER_ID",
    ):
        assert name in example
    assert "H100_MAIL_TYPE=ALL" in example
    source_guard = (REPO / "scripts/h100/source_validation.py").read_text()
    assert '"--untracked-files=all"' in source_guard
    assert (REPO / "slurm/h100/.gitignore").read_text().strip() == "site.env"
    assert "refusing tracked site.env" in submit


def test_shell_entrypoints_are_executable():
    for path in (SUBMIT, SBATCH, SMOKE_SBATCH, SHIM):
        assert path.stat().st_mode & stat.S_IXUSR, f"{path} is not executable"


def test_lightweight_smoke_batch_uses_host_only_one_shot_requeue():
    smoke = SMOKE_SBATCH.read_text()
    assert "#SBATCH --gpus-per-node=1" in smoke
    assert "#SBATCH --time=00:15:00" in smoke
    assert "#SBATCH --requeue" in smoke
    assert "scripts.h100.slurm_smoke run" in smoke
    assert "scripts.h100.slurm_smoke authorize-requeue" in smoke
    assert '"${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue' in smoke
    assert "exec --cleanenv --containall --nv" in smoke
    assert "unset BOX_JWT_CONFIG BOX_FOLDER_ID" in smoke
    assert "sha256sum --check" not in smoke
    shim = SHIM.read_text()
    assert "refusing real scontrol inside" in shim
    assert 'exec "${H100_REAL_SCONTROL' not in shim


def test_cutover_separates_h100_and_reference_shas(tmp_path):
    smoke_path, bindings, smoke_receipt = completed_smoke(tmp_path)
    package = dict(bindings["package"])
    frozen = {
        relative: (bindings["source"]["detector_sha256"] if relative == "configs/detector.yaml" else "a" * 64)
        for relative in contracts.FROZEN_PATHS
    }
    ready = {
        "status": "ready",
        "acceptance_uuid": str(uuid.uuid4()),
        "source": {
            "git_sha": bindings["source"]["git_sha"],
            "frozen_sha256": frozen,
        },
        "sif": {
            "sha256": bindings["sif"]["sha256"],
            "container_build_sha256": bindings["sif"]["container_build_sha256"],
        },
        "package": package,
        "strict_fp32": strict_backend(),
        "hardware": h100_hardware(),
        "gates": {
            "pytest_seconds": 1.0,
            "hardware_probe_seconds": 1.0,
            "vit_gate_seconds": 1.0,
            "cnn_200step_seconds": 1.0,
        },
        "projection": {
            "steps_per_second": 1.0,
            "expected_gpu_hours": 2.0,
            "ceiling_gpu_hours": 3.0,
            "expected_wall_hours_ideal": 1.0,
            "ceiling_wall_hours_ideal": 1.5,
            "conservative_h100_wall_hours": 2.0,
            "remaining_v100_wall_hours": 3.0,
            "staging_seconds": 360.0,
            "staging_hours_per_allocation": 0.1,
            "allocation_wall_hours": 36.5,
            "signal_lead_hours": 0.25,
            "usable_training_hours_per_allocation": 36.15,
            "projected_allocation_count": 1,
            "training_wall_hours_before_staging": 1.9,
        },
        "slurm_smoke": {
            "sha256": contracts.sha256_file(smoke_path),
            "receipt": smoke_receipt,
        },
    }
    ready_path = tmp_path / "H100_READY.json"
    add_host_acceptance_receipts(tmp_path, ready)
    contracts.atomic_write_json(ready_path, ready)
    cutover.validate_h100_ready(
        ready_path,
        expected_git_sha=bindings["source"]["git_sha"],
        expected_sif_sha256=bindings["sif"]["sha256"],
        expected_container_build_sha256=bindings["sif"]["container_build_sha256"],
        expected_package=package,
        expected_frozen_sha256=frozen,
        expected_smoke_receipt=smoke_receipt,
        expected_smoke_sha256=contracts.sha256_file(smoke_path),
    )
    provenance = reference_provenance("v100-sha")
    cutover.validate_reference_provenance(
        provenance,
        expected_git_sha="v100-sha",
        expected_campaign_id="fresh34-v100-fp32-20260726",
    )
    with pytest.raises(RuntimeError, match="git SHA"):
        cutover.validate_reference_provenance(
            provenance,
            expected_git_sha="h100-sha",
            expected_campaign_id="fresh34-v100-fp32-20260726",
        )


def test_reference_specific_r2_r3_validation(tmp_path):
    r2 = tmp_path / "yolo26-f100"
    r2.mkdir()
    (r2 / "final_metrics.json").write_text(
        json.dumps(
            {
                "exp_id": "yolo26-f100",
                "threshold": 0.1,
                "dev_f1": 0.2,
                "test_f1": 0.3,
                "test_precision": 0.4,
                "test_recall": 0.5,
                "test_near_shore_f1": 0.1,
            }
        )
    )
    (r2 / "runtime_provenance.json").write_text(
        json.dumps(reference_provenance())
    )
    kwargs = {
        "expected_git_sha": "v100-sha",
        "expected_campaign_id": "fresh34-v100-fp32-20260726",
    }
    cutover.validate_r2(r2, **kwargs)

    r3 = tmp_path / "locateanything-zs"
    r3.mkdir()
    (r3 / "final_metrics.json").write_text(
        json.dumps(
            {
                "exp_id": "locateanything-zs",
                "best_prompt": "boat",
                "per_prompt": {
                    "boat": {
                        "f1": 0.1,
                        "precision": 0.2,
                        "recall": 0.3,
                        "threshold": 0.4,
                    },
                    "ship": {
                        "f1": 0.1,
                        "precision": 0.2,
                        "recall": 0.3,
                        "threshold": 0.4,
                    },
                    "vessel": {
                        "f1": 0.1,
                        "precision": 0.2,
                        "recall": 0.3,
                        "threshold": 0.4,
                    },
                },
            }
        )
    )
    (r3 / "runtime_provenance.json").write_text(
        json.dumps(reference_provenance())
    )
    cutover.validate_r3(r3, **kwargs)


def test_cutover_contains_no_process_control():
    for relative in ("scripts/h100/cutover.py", "scripts/h100/operator_cutover.py"):
        source = (REPO / relative).read_text()
        assert "os.kill" not in source
        assert "terminate(" not in source
        assert "scontrol" not in source
    assert "v100_action" in (REPO / "scripts/h100/cutover.py").read_text()


def test_operator_cutover_persists_byte_identical_canonical_evidence(tmp_path):
    fixture = operator_evidence_fixture(tmp_path)
    validation_args = {
        key: value
        for key, value in fixture.items()
        if key != "meta_root"
    }
    validated = operator_cutover.validate_operator_archive(**validation_args)
    assert validated["status"] == "operator-cutover-validated"
    receipt_bytes = fixture["receipt"].read_bytes()
    manifest_bytes = fixture["archive_manifest"].read_bytes()
    evidence = operator_cutover.persist_operator_evidence(
        meta_root=fixture["meta_root"],
        cutover_ready=fixture["cutover_ready"],
        cutover_ready_sha256=fixture["cutover_ready_sha256"],
        receipt=fixture["receipt"],
        receipt_sha256=fixture["receipt_sha256"],
        archive_manifest=fixture["archive_manifest"],
        archive_manifest_sha256=fixture["archive_manifest_sha256"],
    )
    canonical_receipt = fixture["meta_root"] / "V100_CORE_ARCHIVED.json"
    canonical_manifest = fixture["meta_root"] / "V100_CORE_ARCHIVE_MANIFEST.json"
    assert canonical_receipt.read_bytes() == receipt_bytes
    assert canonical_manifest.read_bytes() == manifest_bytes
    assert evidence == {
        "cutover_ready": {
            "path": str(fixture["cutover_ready"]),
            "sha256": fixture["cutover_ready_sha256"],
        },
        "v100_core_archived": {
            "path": str(canonical_receipt),
            "sha256": fixture["receipt_sha256"],
        },
        "archive_manifest": {
            "path": str(canonical_manifest),
            "sha256": fixture["archive_manifest_sha256"],
        },
    }
    assert stat.S_IMODE(canonical_receipt.stat().st_mode) == 0o444
    assert stat.S_IMODE(canonical_manifest.stat().st_mode) == 0o444
    operator_cutover.validate_operator_archive(
        **{
            **validation_args,
            "receipt": canonical_receipt,
            "archive_manifest": canonical_manifest,
            "bound_archive_manifest": canonical_manifest,
        }
    )


def test_operator_cutover_rejects_symlink_and_nonzero_v100_processes(tmp_path):
    fixture = operator_evidence_fixture(tmp_path)
    receipt_link = tmp_path / "receipt-link.json"
    receipt_link.symlink_to(fixture["receipt"])
    args = {
        key: value
        for key, value in fixture.items()
        if key != "meta_root"
    }
    with pytest.raises(RuntimeError, match="symlink"):
        operator_cutover.validate_operator_archive(
            **{**args, "receipt": receipt_link}
        )
    receipt = json.loads(fixture["receipt"].read_text())
    receipt["v100"]["running_core_processes"] = 1
    contracts.atomic_write_json(fixture["receipt"], receipt)
    with pytest.raises(RuntimeError, match="V100 archive state mismatch"):
        operator_cutover.validate_operator_archive(
            **{
                **args,
                "receipt_sha256": contracts.sha256_file(fixture["receipt"]),
            }
        )


def test_operator_cutover_schemas_are_exact_and_closed():
    receipt = json.loads(V100_RECEIPT_SCHEMA.read_text())
    archive = json.loads(V100_ARCHIVE_SCHEMA.read_text())
    assert receipt["additionalProperties"] is False
    assert archive["additionalProperties"] is False
    assert receipt["properties"]["v100"]["properties"]["running_core_processes"] == {
        "const": 0
    }
    assert archive["properties"]["scope"] == {"const": "v100-core-diagnostics"}


def test_source_receipt_and_campaign_resume_bindings_are_hash_locked(tmp_path):
    frozen = {relative: "a" * 64 for relative in contracts.FROZEN_PATHS}
    package = dict(smoke_bindings()["package"])
    payload = source_validation.expected_receipt(
        git_sha="1" * 40,
        frozen_sha256=frozen,
        package=package,
    )
    receipt = tmp_path / source_validation.SOURCE_RECEIPT_NAME
    contracts.atomic_write_json(receipt, payload)
    digest = contracts.sha256_file(receipt)
    assert source_validation.validate_source_receipt(
        receipt,
        expected_sha256=digest,
        expected_git_sha="1" * 40,
        expected_hashes=frozen,
        expected_package=package,
    ) == payload
    bindings = {
        "campaign_id": "campaign",
        "acceptance_uuid": "acceptance",
        "source_validation_sha256": digest,
        "cutover_ready_sha256": "b" * 64,
        "v100_core_archived_sha256": "c" * 64,
        "archive_manifest_sha256": "d" * 64,
    }
    campaign.validate_campaign_resume_bindings(dict(bindings), bindings)
    with pytest.raises(RuntimeError, match="source_validation_sha256"):
        campaign.validate_campaign_resume_bindings(
            {**bindings, "source_validation_sha256": "e" * 64}, bindings
        )


def test_sbatch_supplies_every_required_acceptance_and_campaign_argument():
    job = SBATCH.read_text()
    acceptance_args = (
        "--scratch-free-before-extraction",
        "--source-validation-json",
        "--source-validation-sha256",
        "--host-test-receipt",
        "--host-test-receipt-sha256",
    )
    campaign_args = (
        "--expected-git-sha",
        "--source-validation-json",
        "--source-validation-sha256",
        "--cutover-ready-json",
        "--cutover-ready-sha256",
        "--v100-core-archived-json",
        "--v100-core-archived-sha256",
        "--v100-archive-manifest-json",
        "--v100-archive-manifest-sha256",
    )
    acceptance_source = (REPO / "scripts/h100/acceptance.py").read_text()
    campaign_source = (REPO / "scripts/h100/campaign.py").read_text()
    for option in acceptance_args:
        assert f'parser.add_argument("{option}"' in acceptance_source
        assert option in job
    for option in campaign_args:
        assert f'parser.add_argument("{option}"' in campaign_source
        assert option in job
    assert job.index("scratch_free=") < job.index("scripts.handoff extract")
    assert '--scratch-free-before-extraction "$scratch_free"' in job


def test_host_only_gates_precede_sif_pythonpath_and_campaign_resets_canonical_paths():
    job = SBATCH.read_text()
    source_gate = job.index("scripts.h100.source_validation")
    host_gate = job.index("scripts.h100.host_test_gate")
    strict_path = job.index('export PYTHONPATH="$repo/scripts/h100:$repo"')
    assert source_gate < host_gate < strict_path
    assert 'PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON"' in job
    assert "tests/test_h100_handoff.py" in host_test_gate.HOST_TESTS
    assert "tests/test_experiment_manifest.py" in host_test_gate.HOST_TESTS
    for name in (
        "V100_CORE_ARCHIVED.json",
        "V100_CORE_ARCHIVE_MANIFEST.json",
    ):
        reset = f'H100_{"V100_CORE_ARCHIVED" if name.startswith("V100_CORE_ARCHIVED") else "V100_ARCHIVE_MANIFEST"}='
        assert job.index(reset) > job.index('source "$site_env"')
        assert name in job


def test_submit_cutover_is_non_submitting_and_uses_transfer_python():
    submit = SUBMIT.read_text()
    cutover_block = submit.index('if [[ "$mode" == "cutover-check" ]]')
    campaign_block = submit.index('if [[ "$mode" == "campaign" ]]')
    sbatch_at = submit.index("sbatch \\")
    assert cutover_block < submit.index("exit 0", cutover_block) < sbatch_at
    assert campaign_block < sbatch_at
    assert submit.count('PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON"') >= 2
    assert "--persist-meta-root" in submit
    assert "--bound-archive-manifest" in submit


def test_reverse_results_delegates_shared_handoff_schema(tmp_path):
    argv = reverse_command(
        repo=tmp_path / "repo",
        runs_root=tmp_path / "runs",
        campaign_manifest=tmp_path / "runs/.h100/campaign_manifest.json",
        output=tmp_path / "out",
        max_part_bytes=123,
    )
    assert argv[1:5] == ["-m", "scripts.handoff", "build-results", "--repo"]
    assert argv[-2:] == ["--max-part-bytes", "123"]
    reverse_source = (REPO / "scripts/h100/reverse_results.py").read_text()
    assert 'runs_root / "summary/grid.csv"' in reverse_source
