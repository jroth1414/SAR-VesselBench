"""CPU-only fixture/static guards for the native strict-FP32 H100 lane."""

from __future__ import annotations

import csv
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.analysis import curves
from scripts.h100 import (
    acceptance,
    build_venv,
    campaign,
    cell as h100_cell,
    contracts,
    cutover,
    host_test_gate,
    lightning_contract,
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

RUNTIME_GIT_SHA = "1" * 40
BASE_GIT_SHA = source_validation.BASE_PAYLOAD_GIT_SHA
VENV_SHA256 = "3" * 64
VENV_BUILD_SHA256 = "4" * 64
BASE_PYTHON_SHA256 = "5" * 64
BASE_PYTHON_RUNTIME_SHA256 = "0" * 64
WHEELHOUSE_SHA256 = "1" * 64
BASE_EXTRACTION_RECEIPT_SHA256 = "2" * 64


def base_payload() -> dict[str, str]:
    return {
        "package_id": source_validation.BASE_PAYLOAD_PACKAGE_ID,
        "git_sha": BASE_GIT_SHA,
        "manifest_sha256": "6" * 64,
        "ready_sha256": "7" * 64,
        "sha256sums_sha256": "8" * 64,
        "repo_bundle_sha256": "9" * 64,
    }


def runtime_amendment() -> dict[str, str]:
    identity = "a" * 64
    return {
        "package_id": f"xview3-h100-runtime-{RUNTIME_GIT_SHA}-{identity}",
        "git_sha": RUNTIME_GIT_SHA,
        "manifest_sha256": "b" * 64,
        "ready_sha256": "c" * 64,
        "sha256sums_sha256": "d" * 64,
        "runtime_bundle_sha256": "e" * 64,
    }


def base_python() -> dict[str, object]:
    return {
        "version": "3.11.13",
        "implementation": "cpython",
        "resolved_path": "/opt/python-3.11.13/bin/python3.11",
        "executable_sha256": BASE_PYTHON_SHA256,
        "runtime": {
            "algorithm": build_venv.BASE_RUNTIME_DIGEST_ALGORITHM,
            "sha256": BASE_PYTHON_RUNTIME_SHA256,
        },
    }


def wheelhouse() -> dict[str, object]:
    identity = {
        "algorithm": "xview3-wheelhouse-tree-v1",
        "sha256": WHEELHOUSE_SHA256,
        "files": 1,
        "bytes": 1,
    }
    receipt = {
        "format_version": 1,
        "package_id": base_payload()["package_id"],
        "manifest_sha256": base_payload()["manifest_sha256"],
        "wheelhouse": identity,
    }
    return {
        "identity": identity,
        "artifacts": {"fixture.whl": {"sha256": "3" * 64, "bytes": 1}},
        "base_extraction": {
            "path": "/persistent/base/HANDOFF_EXTRACTED.json",
            "sha256": BASE_EXTRACTION_RECEIPT_SHA256,
            "receipt": receipt,
        },
        "reverified_after_build": True,
    }


def staged_base_extraction() -> dict[str, object]:
    return {
        "path": "/scratch/payload/HANDOFF_EXTRACTED.json",
        "sha256": BASE_EXTRACTION_RECEIPT_SHA256,
        "receipt": wheelhouse()["base_extraction"]["receipt"],
    }


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


def h100_runtime_contract():
    return {
        "schema": 1,
        "status": "verified",
        "pre_trainer": {
            "schema": 1,
            "status": "verified",
            "stage": "pre-trainer",
            "precision": "32-true",
            "devices": 1,
            "micro_batch": 16,
            "gradient_accumulation": 1,
            "effective_batch": 16,
            "strict_fp32": strict_backend(),
            "autocast": {"global": False, "cuda": False, "cpu": False},
            "process": {
                "WORLD_SIZE": "unset",
                "SLURM_NTASKS": "unset",
                "effective_world_size": 1,
            },
            "model": {
                "floating_parameter_count": 2,
                "floating_parameter_dtypes": ["torch.float32"],
            },
        },
        "resolved_trainer": {
            "accelerator": lightning_contract.CUDA_ACCELERATOR,
            "precision_plugin": lightning_contract.PRECISION_PLUGIN,
            "precision": "32-true",
            "gradient_scaler": None,
            "strategy": lightning_contract.SINGLE_DEVICE_STRATEGY,
            "root_device_type": "cuda",
            "root_device_index": 0,
            "num_devices": 1,
            "world_size": 1,
            "device_ids": [0],
            "gradient_accumulation": 1,
        },
    }


def h100_hardware():
    return {
        "torch": "2.11.0+cu126",
        "cuda_build": "12.6",
        "driver_version": "590.1",
        "backend": strict_backend(),
        "devices": h100_devices(),
        "child_probes": [{"runtime_contract": h100_runtime_contract()} for _ in range(8)],
    }


def frozen_hashes(detector_sha256: str = "2" * 64) -> dict[str, str]:
    return {
        relative: detector_sha256 if relative == "configs/detector.yaml" else "f" * 64
        for relative in contracts.FROZEN_PATHS
    }


def smoke_bindings():
    return slurm_smoke.make_bindings(
        git_sha=RUNTIME_GIT_SHA,
        detector_sha256="2" * 64,
        venv_sha256=VENV_SHA256,
        venv_build_sha256=VENV_BUILD_SHA256,
        base_python_sha256=BASE_PYTHON_SHA256,
        base_python_runtime_sha256=BASE_PYTHON_RUNTIME_SHA256,
        wheelhouse_sha256=WHEELHOUSE_SHA256,
        base_extraction_receipt_sha256=BASE_EXTRACTION_RECEIPT_SHA256,
        base_payload=base_payload(),
        runtime_amendment=runtime_amendment(),
    )


def smoke_cli_args(bindings: dict[str, object], runs: Path, signal_ready: Path) -> list[str]:
    source = bindings["source"]
    venv = bindings["venv"]
    base = bindings["base_payload"]
    runtime = bindings["runtime_amendment"]
    return [
        sys.executable,
        "-m",
        "scripts.h100.slurm_smoke",
        "run",
        "--runs-root",
        str(runs),
        "--git-sha",
        source["git_sha"],
        "--detector-sha256",
        source["detector_sha256"],
        "--venv-sha256",
        venv["sha256"],
        "--venv-build-sha256",
        venv["build_receipt_sha256"],
        "--base-python-sha256",
        venv["base_python_sha256"],
        "--base-python-runtime-sha256",
        venv["base_python_runtime_sha256"],
        "--wheelhouse-sha256",
        venv["wheelhouse_sha256"],
        "--base-extraction-receipt-sha256",
        venv["base_extraction_receipt_sha256"],
        "--base-payload-package-id",
        base["package_id"],
        "--base-payload-git-sha",
        base["git_sha"],
        "--base-payload-manifest-sha256",
        base["manifest_sha256"],
        "--base-payload-ready-sha256",
        base["ready_sha256"],
        "--base-payload-sha256sums-sha256",
        base["sha256sums_sha256"],
        "--base-payload-repo-bundle-sha256",
        base["repo_bundle_sha256"],
        "--runtime-amendment-package-id",
        runtime["package_id"],
        "--runtime-amendment-git-sha",
        runtime["git_sha"],
        "--runtime-amendment-manifest-sha256",
        runtime["manifest_sha256"],
        "--runtime-amendment-ready-sha256",
        runtime["ready_sha256"],
        "--runtime-amendment-sha256sums-sha256",
        runtime["sha256sums_sha256"],
        "--runtime-amendment-bundle-sha256",
        runtime["runtime_bundle_sha256"],
        "--job-id",
        "1234",
        "--restart-count",
        "0",
        "--external-signal-ready",
        str(signal_ready),
        "--signal-timeout-seconds",
        "5",
    ]


def completed_smoke(tmp_path: Path):
    bindings = smoke_bindings()
    runs = tmp_path / "runs"
    root = slurm_smoke.smoke_root(runs)
    signal_ready = root / slurm_smoke.SIGNAL_READY_NAME
    process = subprocess.Popen(
        smoke_cli_args(bindings, runs, signal_ready),
        cwd=REPO,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not signal_ready.is_file():
        if process.poll() is not None:
            break
        time.sleep(0.02)
    if not signal_ready.is_file():
        stdout, stderr = process.communicate(timeout=5)
        raise AssertionError(f"smoke did not publish PID: {stdout=} {stderr=}")
    pid = int(signal_ready.read_text().strip())
    assert pid == process.pid
    os.kill(pid, signal.SIGUSR1)
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == slurm_smoke.HOST_REQUEUE_EXIT_CODE, (stdout, stderr)
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
        "h100_runtime_contract": h100_runtime_contract(),
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
        "schema": 2,
        "campaign_id": "campaign",
        "exp_id": cell.exp_id,
        "git_sha": "git",
        "detector_sha256": "detector",
        "venv_sha256": VENV_SHA256,
        "venv_build_sha256": VENV_BUILD_SHA256,
        "base_python": base_python(),
        "wheelhouse": wheelhouse(),
        "base_payload": base_payload(),
        "runtime_amendment": runtime_amendment(),
        "acceptance_uuid": "acceptance",
        "source_validation_sha256": "source-receipt",
        "cutover_ready_sha256": "cutover-receipt",
        "v100_diagnostic_isolation_sha256": "v100-receipt",
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
                "h100_runtime_contract": h100_runtime_contract(),
                "best_dev_f1": 0.5,
                "test_f1": 0.4,
            }
        )
    return payload


def existing_state_kwargs(runs):
    return {
        "phase": "train",
        "repo": REPO,
        "runs_root": runs,
        "campaign_id": "campaign",
        "git_sha": "a" * 40,
        "detector_sha256": "b" * 64,
        "candidate_floor": 0.05,
        "venv_sha256": VENV_SHA256,
        "venv_build_sha256": VENV_BUILD_SHA256,
        "base_python": base_python(),
        "wheelhouse": wheelhouse(),
        "base_payload": base_payload(),
        "runtime_amendment": runtime_amendment(),
        "acceptance_uuid": "acceptance",
        "source_validation_sha256": "source-receipt",
        "evaluation_ground_truth_sha256": "ground-truth-receipt",
        "cutover_ready_sha256": "cutover-receipt",
        "v100_diagnostic_isolation_sha256": "v100-receipt",
        "strict_fp32": strict_backend(),
        "accepted_hardware_class": campaign.hardware_class(h100_hardware()),
        "cohort": None,
        "cohort_sha256": None,
        "test_scene_ids": tuple(f"scene-{index:02d}" for index in range(16)),
    }


def add_acceptance_receipts(root: Path, ready: dict) -> None:
    source_path = root / source_validation.SOURCE_RECEIPT_NAME
    source_receipt = source_validation.expected_receipt(
        git_sha=ready["source"]["git_sha"],
        frozen_sha256=ready["source"]["frozen_sha256"],
        base_payload=ready["base_payload"],
        runtime_amendment=ready["runtime_amendment"],
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
        "log": {"path": str(host_log), "sha256": contracts.sha256_file(host_log)},
    }
    contracts.atomic_write_json(host_receipt_path, host_receipt)

    venv_log = root / "acceptance-logs/pytest-venv-remaining.log"
    venv_log.write_text("venv tests passed\n")
    venv_command = [
        "-m",
        "pytest",
        "-q",
        *(f"--ignore={path}" for path in host_test_gate.HOST_TESTS),
    ]
    suite = {
        "schema": 2,
        "status": "passed",
        "source_validation_sha256": source_sha,
        "coverage": {
            "host": host_test_gate.HOST_TESTS,
            "venv": "all pytest collection except the two host-only files",
            "aggregate": "entire repository pytest suite",
        },
        "host_handoff": {
            "receipt_path": str(host_receipt_path),
            "receipt_sha256": contracts.sha256_file(host_receipt_path),
            "receipt": host_receipt,
        },
        "venv_remaining": {
            "command": venv_command,
            "duration_seconds": 2.0,
            "log": {"path": str(venv_log), "sha256": contracts.sha256_file(venv_log)},
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


def ready_payload(smoke_path: Path, smoke_receipt: dict, bindings: dict) -> dict:
    steps = {
        label: int(item["steps_per_epoch"])
        for label, item in acceptance.EXPECTED_FRACTION_WORKLOAD.items()
    }
    return {
        "schema": 2,
        "status": "ready",
        "acceptance_uuid": str(uuid.uuid4()),
        "source": {
            "git_sha": bindings["source"]["git_sha"],
            "frozen_sha256": frozen_hashes(bindings["source"]["detector_sha256"]),
        },
        "venv": {
            "path": "/persistent/venvs/xview3-h100-fp32",
            "sha256": bindings["venv"]["sha256"],
            "venv_build_sha256": bindings["venv"]["build_receipt_sha256"],
            "base_python": base_python(),
            "wheelhouse": wheelhouse(),
            "staged_base_extraction": staged_base_extraction(),
        },
        "base_payload": dict(bindings["base_payload"]),
        "runtime_amendment": dict(bindings["runtime_amendment"]),
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
            "expected_gpu_hours": 1.0,
            "ceiling_gpu_hours": 1.5,
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
            "fraction_workload": acceptance.EXPECTED_FRACTION_WORKLOAD,
            "steps_per_epoch": steps,
            "grid_steps_per_epoch": 8 * sum(steps.values()),
        },
        "slurm_smoke": {
            "path": str(smoke_path),
            "sha256": contracts.sha256_file(smoke_path),
            "receipt": smoke_receipt,
        },
    }




def test_first_campaign_start_requires_empty_core_and_absent_cohort(tmp_path):
    runs_root = tmp_path / "runs"
    meta_root = runs_root / ".h100"
    meta_root.mkdir(parents=True)
    manifest = meta_root / "campaign_manifest.json"
    cohort = meta_root / "TRAINING_COHORT.json"

    assert campaign.validate_campaign_start_state(
        repo=REPO,
        runs_root=runs_root,
        manifest_path=manifest,
        cohort_path=cohort,
    ) is False

    cell = contracts.load_cells(REPO)[0]
    occupied = runs_root / cell.exp_id
    occupied.mkdir()
    with pytest.raises(RuntimeError, match="canonical namespaces are not empty"):
        campaign.validate_campaign_start_state(
            repo=REPO,
            runs_root=runs_root,
            manifest_path=manifest,
            cohort_path=cohort,
        )
    occupied.rmdir()

    cohort.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cohort exists without a campaign manifest"):
        campaign.validate_campaign_start_state(
            repo=REPO,
            runs_root=runs_root,
            manifest_path=manifest,
            cohort_path=cohort,
        )
    cohort.unlink()

    manifest.write_text("{}\n", encoding="utf-8")
    occupied.mkdir()
    assert campaign.validate_campaign_start_state(
        repo=REPO,
        runs_root=runs_root,
        manifest_path=manifest,
        cohort_path=cohort,
    ) is True


def test_empty_core_gate_rejects_broken_symlink_namespace(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    cell = contracts.load_cells(REPO)[0]
    (runs_root / cell.exp_id).symlink_to(tmp_path / "absent")
    with pytest.raises(RuntimeError, match="canonical namespaces are not empty"):
        contracts.assert_empty_core_namespaces(REPO, runs_root)


def test_matrix_is_exactly_32_and_expensive_first():
    cells = contracts.load_cells(REPO)
    assert len(cells) == len({cell.exp_id for cell in cells}) == 32
    assert [cell.fraction for cell in cells] == [1.0] * 8 + [0.5] * 8 + [0.25] * 8 + [0.1] * 8
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
    launches = campaign.next_launches(**common, failure_seen=False, preemption_seen=False)
    assert [cell.exp_id for _gpu, cell in launches] == [cell.exp_id for cell in cells[:8]]
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
    prior = {"fail_stop": {"engaged": True, "failed": [cells[0].exp_id], "allowed_to_finish": sorted(allowed)}}
    assert campaign.restore_fail_stop(prior, cells) == (True, {cells[0].exp_id}, allowed)


class _ScriptedProcess:
    def __init__(self, polls: list[int | None], *, pid: int = 4321) -> None:
        self._polls = list(polls)
        self._last = polls[-1]
        self.pid = pid
        self.signals: list[int] = []
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        if self._polls:
            self._last = self._polls.pop(0)
        return self._last

    def wait(self, timeout: float | None = None) -> int | None:
        del timeout
        return self._last

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def _bare_preemption_controller(
    tmp_path: Path,
    *,
    process: _ScriptedProcess,
    cell: contracts.Cell,
    cells: list[contracts.Cell],
) -> campaign.Controller:
    controller = object.__new__(campaign.Controller)
    controller.args = SimpleNamespace(
        campaign_id="preemption-race-fixture",
        checkpoint_timeout=0.01,
    )
    controller.repo = REPO
    controller.runs_root = tmp_path / "runs"
    controller.cells = cells
    controller.phase = "score-test"
    controller.git_sha = RUNTIME_GIT_SHA
    controller.detector_sha256 = "f" * 64
    controller.venv_sha256 = VENV_SHA256
    controller.venv_build_sha256 = VENV_BUILD_SHA256
    controller.base_python = {}
    controller.wheelhouse = {}
    controller.base_payload = {}
    controller.runtime_amendment = {}
    controller.acceptance_uuid = "acceptance"
    controller.source_validation_sha256 = "a" * 64
    controller.evaluation_ground_truth_sha256 = "d" * 64
    controller.cutover_ready_sha256 = "b" * 64
    controller.v100_diagnostic_isolation_sha256 = "c" * 64
    controller.strict_fp32 = {}
    controller.accepted_hardware_class = {}
    controller.running = {0: (process, cell, time.monotonic())}
    controller.training_complete_ids = {item.exp_id for item in cells}
    controller.test_complete_ids = set()
    controller.complete_ids = controller.test_complete_ids
    controller.cohort_path = controller.runs_root / ".h100/TRAINING_COHORT.json"
    controller.cohort = {}
    controller.cohort_sha256 = "e" * 64
    controller.candidate_floor = 0.05
    controller.test_scene_ids = tuple(
        f"scene-{index:02d}" for index in range(16)
    )
    controller.failure_seen = False
    controller.failed_ids = set()
    controller.failure_allowed_ids = set()
    controller.preemption_seen = True
    controller.preemption_forwarded = False
    controller.cell_runtime = {}
    controller.cell_phase_runtime = {}
    controller.request_dir = controller.runs_root / ".h100/requeue-requests"
    controller.request_dir.mkdir(parents=True)
    run_dir = controller.runs_root / cell.exp_id
    run_dir.mkdir(parents=True)
    contracts.atomic_write_json(
        run_dir / "runtime_provenance.json",
        {"attempts": [{"phase": "score-test"}], "accumulated_active_seconds": 0.0},
    )
    controller.test_events = []
    controller.test_statuses = []
    controller.record = lambda event, **payload: controller.test_events.append(
        {"event": event, **payload}
    )
    controller.write_manifest = (
        lambda status=None: controller.test_statuses.append(status)
    )
    return controller


@pytest.mark.parametrize(
    ("pending_cell", "expected_code", "expected_status"),
    [
        (True, campaign.HOST_REQUEUE_EXIT_CODE, "host-requeue-required"),
        (False, 0, "complete"),
    ],
)
def test_preemption_completion_race_reaps_code_zero_and_finalizes_or_requeues(
    tmp_path,
    monkeypatch,
    pending_cell,
    expected_code,
    expected_status,
):
    all_cells = contracts.load_cells(REPO)
    cell = all_cells[0]
    cells = [cell, all_cells[1]] if pending_cell else [cell]
    process = _ScriptedProcess([None, None, 0])
    controller = _bare_preemption_controller(
        tmp_path,
        process=process,
        cell=cell,
        cells=cells,
    )
    monkeypatch.setenv("SLURM_JOB_ID", "job-completion-race")
    monkeypatch.setattr(
        campaign,
        "validate_scored_completion",
        lambda *_args, **_kwargs: {
            "epochs_run": 1,
            "best_dev_f1": 0.1,
            "test_f1": 0.2,
            "test_scored_at": "2026-07-30T00:00:00+00:00",
            "test_result_sha256": "1" * 64,
            "training_cohort_sha256": "e" * 64,
            "h100_runtime_contract": {},
        },
    )
    monkeypatch.setattr(
        campaign,
        "validate_runtime_provenance",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        campaign,
        "_validated_phase_runtime",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        campaign,
        "finalized_cell_runtime",
        lambda *_args, **_kwargs: {
            "elapsed_hours": 0.1,
            "attempts": 1,
            "epochs_run": 1,
            "test_f1": 0.2,
        },
    )
    monkeypatch.setattr(
        campaign,
        "validate_training_cohort",
        lambda **_kwargs: (controller.cohort, controller.cohort_sha256),
    )
    monkeypatch.setattr(
        campaign,
        "validate_complete_test_cohort",
        lambda **_kwargs: {},
    )
    kill_attempts = []

    def vanished_before_signal(pid, signum):
        kill_attempts.append((pid, signum))
        raise ProcessLookupError

    monkeypatch.setattr(campaign.os, "kill", vanished_before_signal)
    grid = tmp_path / "grid.csv"
    grid.write_text("validated-grid\n")
    grid_calls = []

    def fake_collect(**kwargs):
        grid_calls.append(kwargs)
        return grid

    monkeypatch.setattr(campaign, "collect_and_validate_grid", fake_collect)

    assert controller.preempt_and_requeue() == expected_code
    assert controller.running == {}
    assert controller.complete_ids == {cell.exp_id}
    assert not controller.failure_seen
    assert kill_attempts == [(process.pid, signal.SIGUSR1)]
    assert controller.test_statuses[-1] == expected_status
    assert any(item["event"] == "cell_phase_complete" for item in controller.test_events)
    if pending_cell:
        assert grid_calls == []
    else:
        assert len(grid_calls) == 1
        assert any(
            item["event"] == "grid_validated" for item in controller.test_events
        )


def test_preemption_valid_job_bound_request_promotes_checkpoint_and_returns_75(
    tmp_path,
    monkeypatch,
):
    cells = contracts.load_cells(REPO)
    cell = cells[0]
    process = _ScriptedProcess([campaign.HOST_REQUEUE_EXIT_CODE])
    controller = _bare_preemption_controller(
        tmp_path,
        process=process,
        cell=cell,
        cells=cells[:2],
    )
    monkeypatch.setenv("SLURM_JOB_ID", "job-valid-marker")
    (controller.request_dir / "gpu-0.request").write_text("job-valid-marker\n")
    run_dir = controller.runs_root / cell.exp_id
    contracts.atomic_write_json(
        run_dir / "cell_wrapper.json",
        {"phase": "score-test", "exp_id": cell.exp_id},
    )
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "best.ckpt").write_bytes(b"best")
    (checkpoints / "last.ckpt").write_bytes(b"last")

    assert controller.preempt_and_requeue() == campaign.HOST_REQUEUE_EXIT_CODE
    assert controller.running == {}
    assert controller.test_statuses[-1] == "host-requeue-required"
    assert any(
        item["event"] == "checkpoint_promoted"
        and item["exp_id"] == cell.exp_id
        for item in controller.test_events
    )
    provenance = json.loads((run_dir / "runtime_provenance.json").read_text())
    assert provenance["attempts"][-1]["exit"] == "preempted"


@pytest.mark.parametrize(
    ("exit_code", "request_job"),
    [(1, None), (campaign.HOST_REQUEUE_EXIT_CODE, "different-job")],
)
def test_preemption_invalid_or_wrong_job_exit_fails_closed(
    tmp_path,
    monkeypatch,
    exit_code,
    request_job,
):
    cells = contracts.load_cells(REPO)
    cell = cells[0]
    controller = _bare_preemption_controller(
        tmp_path,
        process=_ScriptedProcess([exit_code]),
        cell=cell,
        cells=cells[:2],
    )
    monkeypatch.setenv("SLURM_JOB_ID", "current-job")
    if request_job is not None:
        (controller.request_dir / "gpu-0.request").write_text(request_job + "\n")

    assert controller.preempt_and_requeue() == 1
    assert controller.running == {}
    assert controller.failure_seen
    assert controller.failed_ids == {cell.exp_id}
    assert any(
        item["event"] == "preemption_invalid_child_exit"
        for item in controller.test_events
    )


def test_cell_completion_signal_race_writes_job_bound_requeue_request(
    tmp_path,
    monkeypatch,
):
    runs = tmp_path / "runs"
    exp_id = "race-f100-s0"
    run_dir = runs / exp_id
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    monkeypatch.setenv("H100_REQUEUE_REQUEST_DIR", str(request_dir))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "job-cell-race")
    monkeypatch.setattr(h100_cell, "assert_sitecustomize_active", lambda: None)
    monkeypatch.setattr(h100_cell, "assert_launch_process_contract", lambda: None)
    monkeypatch.setattr(
        h100_cell,
        "_validate_training",
        lambda *_args, **_kwargs: {},
    )

    def missing_process(*_args):
        raise ProcessLookupError

    monkeypatch.setattr(h100_cell.os, "kill", missing_process)

    class CompletingProcess:
        pid = 7654

        def poll(self):
            return None

        def wait(self):
            signal.raise_signal(signal.SIGUSR1)
            checkpoints = run_dir / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "best.ckpt").write_bytes(b"best")
            (checkpoints / "last.ckpt").write_bytes(b"last")
            return 0

    monkeypatch.setattr(
        h100_cell.subprocess,
        "Popen",
        lambda *_args, **_kwargs: CompletingProcess(),
    )
    args = SimpleNamespace(
        repo=REPO,
        runs_root=runs,
        phase="train",
        exp_id=exp_id,
        init="convnext_random",
        label_frac=1.0,
        git_sha=RUNTIME_GIT_SHA,
        detector_sha256="f" * 64,
        candidate_floor=0.05,
        training_cohort=None,
        training_cohort_sha256=None,
        workers=1,
    )

    assert h100_cell.run_cell(args) == h100_cell.PREEMPTED_EXIT_CODE
    assert (request_dir / "gpu-0.request").read_text() == "job-cell-race\n"
    assert not (run_dir / "final_metrics.json").exists()
    wrapper = json.loads((run_dir / "cell_wrapper.json").read_text())
    assert wrapper == {"phase": "train-complete", "exp_id": exp_id}


def test_cell_scoring_signal_requeues_before_test_metrics_exist(
    tmp_path,
    monkeypatch,
):
    runs = tmp_path / "runs"
    exp_id = "score-race-f100-s0"
    run_dir = runs / exp_id
    request_dir = tmp_path / "requests"
    request_dir.mkdir()
    monkeypatch.setenv("H100_REQUEUE_REQUEST_DIR", str(request_dir))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "job-score-race")
    monkeypatch.setattr(h100_cell, "assert_sitecustomize_active", lambda: None)
    monkeypatch.setattr(h100_cell, "assert_launch_process_contract", lambda: None)
    monkeypatch.setattr(
        h100_cell,
        "_validated_cohort",
        lambda *_args, **_kwargs: ({}, "e" * 64),
    )
    monkeypatch.setattr(
        h100_cell,
        "_validate_training",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        h100_cell,
        "_validate_scored",
        lambda *_args, **_kwargs: {},
    )

    class ScoreProcess:
        pid = 7656

        def poll(self):
            return None

        def send_signal(self, _signum):
            raise ProcessLookupError

        def wait(self):
            signal.raise_signal(signal.SIGUSR1)
            return 0

    monkeypatch.setattr(
        h100_cell.subprocess,
        "Popen",
        lambda *_args, **_kwargs: ScoreProcess(),
    )
    args = SimpleNamespace(
        repo=REPO,
        runs_root=runs,
        phase="score-test",
        exp_id=exp_id,
        init="convnext_random",
        label_frac=1.0,
        git_sha=RUNTIME_GIT_SHA,
        detector_sha256="f" * 64,
        candidate_floor=0.05,
        training_cohort=runs / ".h100/TRAINING_COHORT.json",
        training_cohort_sha256="e" * 64,
        workers=1,
    )

    assert h100_cell.run_cell(args) == h100_cell.PREEMPTED_EXIT_CODE
    assert (request_dir / "gpu-0.request").read_text() == "job-score-race\n"
    assert not (run_dir / "test_metrics.json").exists()
    wrapper = json.loads((run_dir / "cell_wrapper.json").read_text())
    assert wrapper == {
        "phase": "score-test",
        "exp_id": exp_id,
        "training_cohort_sha256": "e" * 64,
    }


def test_repeated_controller_crashes_close_open_attempts_before_last_ckpt_resume(
    tmp_path,
    monkeypatch,
):
    cell = contracts.load_cells(REPO)[0]
    controller = object.__new__(campaign.Controller)
    hardware = h100_hardware()
    accepted_class = campaign.hardware_class(hardware)
    controller.args = SimpleNamespace(
        campaign_id="crash-recovery-fixture",
        hardware=hardware,
        workers_per_gpu=1,
    )
    controller.repo = REPO
    controller.runs_root = tmp_path / "runs"
    controller.phase = "train"
    controller.git_sha = RUNTIME_GIT_SHA
    controller.detector_sha256 = "f" * 64
    controller.candidate_floor = 0.05
    controller.venv_sha256 = VENV_SHA256
    controller.venv_build_sha256 = VENV_BUILD_SHA256
    controller.base_python = base_python()
    controller.wheelhouse = wheelhouse()
    controller.base_payload = base_payload()
    controller.runtime_amendment = runtime_amendment()
    controller.acceptance_uuid = "acceptance"
    controller.source_validation_sha256 = "a" * 64
    controller.evaluation_ground_truth_sha256 = "d" * 64
    controller.cutover_ready_sha256 = "b" * 64
    controller.v100_diagnostic_isolation_sha256 = "c" * 64
    controller.strict_fp32 = strict_backend()
    controller.accepted_hardware_class = accepted_class
    controller.allocation_hardware_class = accepted_class
    controller.cohort_path = controller.runs_root / ".h100/TRAINING_COHORT.json"
    controller.cohort = None
    controller.cohort_sha256 = None
    controller.request_dir = controller.runs_root / ".h100/requeue-requests"
    controller.running = {}
    events = []
    controller.record = lambda event, **payload: events.append(
        {"event": event, **payload}
    )

    run_dir = controller.runs_root / cell.exp_id
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "last.ckpt").write_bytes(b"durable-resume")
    device = hardware["devices"][0]
    contracts.atomic_write_json(
        run_dir / "runtime_provenance.json",
        {
            "phase": "train",
            "attempts": [
                {
                    "attempt": 1,
                    "phase": "train",
                    "started_utc": "2026-08-04T00:00:00+00:00",
                    "slurm_job_id": "job-before-crash",
                    "gpu_local_index": 0,
                    "gpu_uuid": device["uuid"],
                }
            ],
            "phase_runtime": {},
            "accumulated_active_seconds": 0.0,
        },
    )
    timestamps = iter(
        (
            "2026-08-04T00:00:10+00:00",
            "2026-08-04T00:00:11+00:00",
            "2026-08-04T00:00:21+00:00",
            "2026-08-04T00:00:22+00:00",
        )
    )
    monkeypatch.setattr(campaign, "utc_now", lambda: next(timestamps))
    monkeypatch.setenv("SLURM_JOB_ID", "job-after-crash")

    class RunningProcess:
        pid = 8800

        def poll(self):
            return None

    monkeypatch.setattr(
        campaign.subprocess,
        "Popen",
        lambda *_args, **_kwargs: RunningProcess(),
    )

    controller.launch(0, cell)
    controller.running.clear()  # model a second abrupt controller/node loss
    controller.launch(0, cell)

    provenance = json.loads(
        (run_dir / "runtime_provenance.json").read_text(encoding="utf-8")
    )
    attempts = provenance["attempts"]
    assert len(attempts) == 3
    assert attempts[0]["exit"] == "controller-or-node-crash-recovered"
    assert attempts[0]["active_seconds"] == 10.0
    assert attempts[1]["exit"] == "controller-or-node-crash-recovered"
    assert attempts[1]["active_seconds"] == 10.0
    assert attempts[2]["resumed_from_last_ckpt"] is True
    assert "active_seconds" not in attempts[2]
    assert provenance["accumulated_active_seconds"] == 20.0
    assert [event["recovered_open_attempt"] for event in events] == [True, True]


@pytest.mark.parametrize("phase", ("train", "score-test"))
def test_crash_reconciliation_closes_each_supported_phase(phase):
    provenance = {
        "phase": phase,
        "attempts": [
            {
                "attempt": 1,
                "phase": phase,
                "started_utc": "2026-08-04T00:00:00+00:00",
            }
        ],
        "accumulated_active_seconds": 0.0,
    }
    assert campaign.reconcile_stale_open_attempt(
        provenance,
        phase=phase,
        recovered_utc="2026-08-04T00:00:05+00:00",
    )
    assert provenance["accumulated_active_seconds"] == 5.0
    assert provenance["attempts"][0] == {
        "attempt": 1,
        "phase": phase,
        "started_utc": "2026-08-04T00:00:00+00:00",
        "finished_utc": "2026-08-04T00:00:05+00:00",
        "exit": "controller-or-node-crash-recovered",
        "active_seconds": 5.0,
        "active_seconds_basis": "started-to-recovery-observation",
        "recovered_from": "open-attempt-on-controller-start",
    }


def test_crash_reconciliation_rejects_ambiguous_open_history():
    provenance = {
        "phase": "train",
        "attempts": [
            {
                "attempt": 1,
                "phase": "train",
                "started_utc": "2026-08-04T00:00:00+00:00",
            },
            {
                "attempt": 2,
                "phase": "train",
                "started_utc": "2026-08-04T00:00:01+00:00",
            },
        ],
        "accumulated_active_seconds": 0.0,
    }
    with pytest.raises(RuntimeError, match="only the final"):
        campaign.reconcile_stale_open_attempt(
            provenance,
            phase="train",
            recovered_utc="2026-08-04T00:00:05+00:00",
        )


def test_score_phase_cannot_launch_before_canonical_cohort():
    controller = object.__new__(campaign.Controller)
    controller.phase = "score-test"
    controller.cohort = None
    controller.cohort_sha256 = None
    cell = contracts.load_cells(REPO)[0]
    with pytest.raises(RuntimeError, match="without frozen cohort"):
        controller.launch(0, cell)


def test_training_cohort_transition_requires_exact_all_32_ids(
    tmp_path,
    monkeypatch,
):
    controller = object.__new__(campaign.Controller)
    controller.phase = "train"
    controller.running = {}
    controller.failure_seen = False
    controller.cells = contracts.load_cells(REPO)
    expected_ids = {cell.exp_id for cell in controller.cells}
    controller.training_complete_ids = {
        *list(expected_ids)[:-1],
        "not-a-core-cell",
    }
    controller.test_complete_ids = set()
    controller.complete_ids = controller.training_complete_ids
    controller.runs_root = tmp_path / "runs"
    controller.cohort_path = controller.runs_root / ".h100/TRAINING_COHORT.json"
    controller.git_sha = "a" * 40
    controller.detector_sha256 = "b" * 64
    controller.candidate_floor = 0.05
    events = []
    controller.record = lambda event, **payload: events.append(
        {"event": event, **payload}
    )
    calls = []
    monkeypatch.setattr(
        campaign,
        "create_training_cohort",
        lambda **kwargs: calls.append(("create", kwargs)),
    )
    monkeypatch.setattr(
        campaign,
        "validate_training_cohort",
        lambda **kwargs: ({"cell_count": 32}, "c" * 64),
    )

    with pytest.raises(RuntimeError, match="all 32"):
        controller.freeze_training_cohort()
    assert calls == []

    controller.training_complete_ids = expected_ids
    controller.complete_ids = controller.training_complete_ids
    controller.freeze_training_cohort()
    assert controller.phase == "score-test"
    assert controller.complete_ids is controller.test_complete_ids
    assert [kind for kind, _kwargs in calls] == ["create"]
    assert events[-1]["event"] == "training_cohort_frozen"
    assert events[-1]["cells"] == 32


def test_training_fraction_counts_use_frozen_nested_scenes_and_canonical_gt(
    tmp_path,
):
    from src.data.datasets import nested_fraction_scenes

    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    labels_root = repo / "data/raw/xview3/labels"
    labels_root.mkdir(parents=True)
    scenes = [f"scene-{index:02d}" for index in range(10)]
    (repo / "data/splits.json").write_text(
        json.dumps(
            {
                "splits": {
                    "train": scenes,
                    "dev": [],
                    "test": [],
                    "eval_final": [],
                }
            }
        ),
        encoding="utf-8",
    )
    (repo / "configs/data.yaml").write_text(
        "seed: 0\n"
        "paths:\n"
        "  splits: data/splits.json\n"
        "  raw_xview3: data/raw/xview3\n",
        encoding="utf-8",
    )
    fields = (
        "scene_id",
        "confidence",
        "is_vessel",
        "source",
        "distance_from_shore_km",
        "detect_scene_column",
        "detect_scene_row",
    )
    with (labels_root / "train.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, scene_id in enumerate(scenes):
            writer.writerow(
                {
                    "scene_id": scene_id,
                    "confidence": "MEDIUM",
                    "is_vessel": "true",
                    "source": "manual" if index % 2 == 0 else "ais",
                    "distance_from_shore_km": 2.0 if index % 3 == 0 else 3.0,
                    "detect_scene_column": index,
                    "detect_scene_row": index + 1,
                }
            )
            writer.writerow(
                {
                    "scene_id": scene_id,
                    "confidence": "HIGH",
                    "is_vessel": "false",
                    "source": "ais",
                    "distance_from_shore_km": 1.0,
                    "detect_scene_column": index,
                    "detect_scene_row": index + 1,
                }
            )
            writer.writerow(
                {
                    "scene_id": scene_id,
                    "confidence": "LOW",
                    "is_vessel": "",
                    "source": "manual",
                    "distance_from_shore_km": 1.0,
                    "detect_scene_column": index,
                    "detect_scene_row": index + 1,
                }
            )

    fractions = (0.1, 0.25, 0.5, 1.0)
    observed = curves.training_fraction_counts(repo=repo, fractions=fractions)
    for fraction in fractions:
        selected = nested_fraction_scenes(
            scenes,
            fraction,
            frac_seed=0,
        )
        indexes = {scenes.index(scene_id) for scene_id in selected}
        assert observed[fraction] == {
            "train_scene_count": len(selected),
            "train_vessel_count": len(selected),
            "train_dark_vessel_count": sum(index % 2 == 0 for index in indexes),
            "train_near_shore_vessel_count": sum(
                index % 3 == 0 for index in indexes
            ),
        }


def test_final_grid_requires_exact_counts_no_nans_and_monotonicity(
    tmp_path,
    monkeypatch,
):
    cells = contracts.load_cells(REPO)
    runs = tmp_path / "runs"
    arms = yaml.safe_load((REPO / "configs/arms.yaml").read_text())["arms"]
    fraction_counts = {
        fraction: {
            "train_scene_count": int(round(fraction * 100)),
            "train_vessel_count": int(round(fraction * 1000)),
            "train_dark_vessel_count": 0,
            "train_near_shore_vessel_count": int(round(fraction * 10)),
        }
        for fraction in (0.1, 0.25, 0.5, 1.0)
    }
    rows = [
        {
            "exp_id": cell.exp_id,
            "init": cell.init,
            "track": cell.track,
            "role": arms[cell.init]["role"],
            "label_frac": str(cell.fraction),
            "seed": str(cell.seed),
            "precision": "32-true",
            "detector_sha256": "detector",
            "git_sha": "git",
            "dev_f1": "0.4",
            "dev_threshold": "0.3",
            "test_f1": "0.5",
            "epochs_run": "2",
            **{
                key: str(value)
                for key, value in fraction_counts[cell.fraction].items()
            },
            "monotonicity_ok": "True",
        }
        for cell in cells
    ]

    def fake_collect(_command, *, cwd, check):
        assert cwd == REPO and check is True
        grid = runs / "summary/grid.csv"
        grid.parent.mkdir(parents=True, exist_ok=True)
        with grid.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=curves.GRID_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(campaign.subprocess, "run", fake_collect)
    monkeypatch.setattr(
        campaign,
        "training_fraction_counts",
        lambda **_kwargs: fraction_counts,
    )
    assert campaign.collect_and_validate_grid(
        repo=REPO,
        runs_root=runs,
        cells=cells,
        git_sha="git",
        detector_sha256="detector",
    ) == runs / "summary/grid.csv"
    rows[0]["monotonicity_ok"] = "False"
    with pytest.raises(RuntimeError, match="monotonicity STOP"):
        campaign.collect_and_validate_grid(
            repo=REPO,
            runs_root=runs,
            cells=cells,
            git_sha="git",
            detector_sha256="detector",
        )
    rows[0]["monotonicity_ok"] = "True"
    rows[0]["train_vessel_count"] = str(
        int(rows[0]["train_vessel_count"]) + 1
    )
    with pytest.raises(RuntimeError, match="TRAIN fraction counts mismatch"):
        campaign.collect_and_validate_grid(
            repo=REPO,
            runs_root=runs,
            cells=cells,
            git_sha="git",
            detector_sha256="detector",
        )
    rows[0]["train_vessel_count"] = str(
        fraction_counts[cells[0].fraction]["train_vessel_count"]
    )
    rows[0]["dev_f1"] = "NaN"
    with pytest.raises(RuntimeError, match="absent/NaN"):
        campaign.collect_and_validate_grid(
            repo=REPO,
            runs_root=runs,
            cells=cells,
            git_sha="git",
            detector_sha256="detector",
        )


def test_campaign_wires_shared_wrapper_and_single_controller_lock():
    source = (REPO / "scripts/h100/campaign.py").read_text()
    wrapper = (REPO / "scripts/h100/cell.py").read_text()
    assert '"scripts.h100.cell"' in source
    assert '"scripts/score_test_split.py"' in wrapper
    assert "fcntl.LOCK_EX | fcntl.LOCK_NB" in source
    assert '"src.analysis.curves",\n            "collect"' in source
    assert '"schema": 2' in source
    for token in (
        "venv_sha256",
        "venv_build_sha256",
        "base_python",
        "wheelhouse",
        "base_payload",
        "runtime_amendment",
    ):
        assert token in source


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


def test_completion_marker_is_recipe_and_hash_checked(tmp_path):
    cell = contracts.load_cells(REPO)[0]
    marker = tmp_path / "final_metrics.json"
    payload = completion_payload(cell)
    marker.write_text(json.dumps(payload))
    assert contracts.validate_completion_marker(marker, cell=cell, git_sha="git", detector_sha256="detector") == payload
    payload["precision"] = "16-mixed"
    marker.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="recipe-matched"):
        contracts.validate_completion_marker(marker, cell=cell, git_sha="git", detector_sha256="detector")


def _write_valid_training_marker(runs, cell):
    import torch

    run = runs / cell.exp_id
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    best = checkpoints / "best.ckpt"
    torch.save({"epoch": 3, "callbacks": {}}, best)
    torch.save({"epoch": 3, "callbacks": {}}, checkpoints / "last.ckpt")
    precision = 3 / 4
    recall = 3 / 5
    f1 = 2 * precision * recall / (precision + recall)
    best_dev = {
        "epoch": 3,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": 3,
        "fp": 1,
        "fn": 2,
        "ignored_predictions": 1,
        "threshold": 0.4,
        "n_candidates": 8,
    }
    payload = {
        "result_schema": 2,
        "exp_id": cell.exp_id,
        "git_sha": "a" * 40,
        "detector_sha256": "b" * 64,
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "epochs_run": 4,
        "best_dev_f1": f1,
        "best_dev": best_dev,
        "best_checkpoint": {
            "relative_path": "checkpoints/best.ckpt",
            "sha256": contracts.sha256_file(best),
            "epoch": 3,
        },
        "last_dev": dict(best_dev),
        "train_loss": 1.25,
        "h100_runtime_contract": h100_runtime_contract(),
    }
    marker = run / "final_metrics.json"
    contracts.atomic_write_json(marker, payload)
    marker.chmod(0o644)
    return run, marker, payload


def test_existing_state_recovers_and_seals_valid_writable_training_marker(
    tmp_path,
    monkeypatch,
):
    cell = contracts.load_cells(REPO)[0]
    runs = tmp_path / "runs"
    run, marker, payload = _write_valid_training_marker(runs, cell)
    started = datetime.fromtimestamp(
        marker.stat().st_mtime - 10.0,
        timezone.utc,
    ).isoformat()
    contracts.atomic_write_json(
        run / "runtime_provenance.json",
        {
            "phase": "train",
            "run_dir": str(run),
            "attempts": [{"phase": "train", "started_utc": started}],
            "accumulated_active_seconds": 0.0,
            "phase_runtime": {},
        },
    )
    monkeypatch.setattr(
        campaign,
        "validate_runtime_provenance",
        lambda *_args, **_kwargs: None,
    )

    assert stat.S_IMODE(marker.stat().st_mode) & 0o222
    assert campaign.existing_cell_state(
        cell,
        **existing_state_kwargs(runs),
    ) == "complete"
    assert stat.S_IMODE(marker.stat().st_mode) == 0o444
    assert json.loads(marker.read_text()) == payload
    recovered = json.loads((run / "runtime_provenance.json").read_text())
    assert recovered["attempts"][-1]["exit_code"] == 0
    assert (
        recovered["attempts"][-1]["recovered_from"]
        == "validated-completion-marker"
    )
    assert recovered["phase_runtime"]["train"]["controller_crash_recovered"] is True


def test_existing_state_does_not_seal_invalid_training_marker(
    tmp_path,
    monkeypatch,
):
    cell = contracts.load_cells(REPO)[0]
    runs = tmp_path / "runs"
    run, marker, _payload = _write_valid_training_marker(runs, cell)
    (run / "checkpoints/best.ckpt").write_bytes(b"tampered")
    (run / "runtime_provenance.json").write_text("{}")
    monkeypatch.setattr(
        campaign,
        "validate_runtime_provenance",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="invalid schema-2 training completion"):
        campaign.existing_cell_state(cell, **existing_state_kwargs(runs))
    assert stat.S_IMODE(marker.stat().st_mode) & 0o222


def test_existing_cell_rejects_legacy_runtime_provenance(tmp_path):
    cell = contracts.load_cells(REPO)[0]
    runs = tmp_path / "runs"
    run = runs / cell.exp_id
    run.mkdir(parents=True)
    (run / "runtime_provenance.json").write_text(
        json.dumps({"schema": 1, "sif_sha256": "legacy"})
    )
    with pytest.raises(RuntimeError, match="runtime provenance"):
        campaign.existing_cell_state(cell, **existing_state_kwargs(runs))


def test_acceptance_probe_marker_requires_finite_fp32_batch16(tmp_path):
    marker = tmp_path / "final_metrics.json"
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "best.ckpt").write_bytes(b"best")
    (checkpoints / "last.ckpt").write_bytes(b"last")
    marker.write_text(json.dumps({
        "exp_id": "probe",
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "best_dev_f1": 0.2,
        "train_loss": 1.5,
        "git_sha": "git",
        "detector_sha256": "detector",
        "h100_runtime_contract": h100_runtime_contract(),
    }))
    acceptance.validate_probe_marker(marker, "probe", expected_git_sha="git", expected_detector_sha256="detector")
    payload = json.loads(marker.read_text())
    payload["train_loss"] = float("nan")
    marker.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="invalid strict-FP32"):
        acceptance.validate_probe_marker(marker, "probe", expected_git_sha="git", expected_detector_sha256="detector")


def test_projection_and_scratch_gates_are_exact():
    steps = {label: int(item["steps_per_epoch"]) for label, item in acceptance.EXPECTED_FRACTION_WORKLOAD.items()}
    assert steps == {"f10": 701, "f25": 1662, "f50": 3310, "f100": 6588}
    projection = contracts.estimate_grid_projection(probe_steps=200, probe_seconds=100, steps_per_epoch=steps)
    assert projection["steps_per_second"] == 2
    assert projection["grid_steps_per_epoch"] == 8 * sum(steps.values())
    assert acceptance.validate_scratch_free_before_extraction(500_000_000_000) == 500_000_000_000
    with pytest.raises(RuntimeError, match="pre-extraction"):
        acceptance.validate_scratch_free_before_extraction(499_999_999_999)
    staged = contracts.staging_aware_wall_clock(training_wall_hours=40.0, staging_seconds=7200.0)
    assert staged["projected_allocation_count"] == 2
    assert staged["conservative_h100_wall_hours"] == 44.0


def test_cutover_rechecks_current_v100_forecast():
    ready = {"projection": {"conservative_h100_wall_hours": 20.0, "remaining_v100_wall_hours": 30.0}}
    forecast = cutover.validate_current_v100_advantage(
        ready,
        25.0,
        current_v100_diagnostic_status=contracts.V100_DIAGNOSTIC_RUNNING,
    )
    assert forecast == {
        "conservative_h100_wall_hours": 20.0,
        "acceptance_remaining_v100_wall_hours": 30.0,
        "current_remaining_v100_wall_hours": 25.0,
        "v100_diagnostic_status": contracts.V100_DIAGNOSTIC_RUNNING,
        "h100_scientifically_mandatory": True,
    }
    with pytest.raises(RuntimeError, match="no longer slower"):
        cutover.validate_current_v100_advantage(
            ready,
            19.0,
            current_v100_diagnostic_status=contracts.V100_DIAGNOSTIC_RUNNING,
        )
    with pytest.raises(RuntimeError, match="no longer slower"):
        cutover.validate_current_v100_advantage(
            ready,
            25.0,
            current_v100_diagnostic_status=contracts.V100_DIAGNOSTIC_COMPLETE,
        )


def test_cutover_allows_zero_only_for_completed_nonreportable_v100_diagnostic():
    ready = {
        "projection": {
            "conservative_h100_wall_hours": 20.0,
            "remaining_v100_wall_hours": 30.0,
        }
    }
    forecast = cutover.validate_current_v100_advantage(
        ready,
        0.0,
        current_v100_diagnostic_status=contracts.V100_DIAGNOSTIC_COMPLETE,
    )
    assert forecast["v100_diagnostic_status"] == (
        "complete-non-reportable-diagnostic"
    )
    assert forecast["h100_scientifically_mandatory"] is True
    with pytest.raises(RuntimeError, match="explicit complete"):
        cutover.validate_current_v100_advantage(
            ready,
            0.0,
            current_v100_diagnostic_status=contracts.V100_DIAGNOSTIC_RUNNING,
        )
    with pytest.raises(RuntimeError, match="nonnegative"):
        cutover.validate_current_v100_advantage(
            ready,
            -0.1,
            current_v100_diagnostic_status=contracts.V100_DIAGNOSTIC_COMPLETE,
        )


def test_cutover_preserves_original_acceptance_comparison():
    ready = {
        "projection": {
            "conservative_h100_wall_hours": 30.0,
            "remaining_v100_wall_hours": 20.0,
        }
    }
    with pytest.raises(RuntimeError, match="original acceptance comparison"):
        cutover.validate_current_v100_advantage(
            ready,
            40.0,
            current_v100_diagnostic_status=contracts.V100_DIAGNOSTIC_RUNNING,
        )


def test_hpc_checkpoint_is_atomically_promoted(tmp_path):
    run = tmp_path / "cell"
    hpc = run / "hpc_ckpt_1.ckpt"
    hpc.parent.mkdir(parents=True)
    hpc.write_bytes(b"checkpoint")
    promoted = campaign.promote_hpc_checkpoint(run)
    assert promoted.read_bytes() == b"checkpoint"
    assert not promoted.with_suffix(".ckpt.promoting").exists()
    (run / "cell_wrapper.json").write_text(json.dumps({"phase": "score-test", "exp_id": "fixture"}))
    (run / "checkpoints/best.ckpt").write_bytes(b"best")
    promoted.write_bytes(b"newer-last")
    assert campaign.checkpoint_for_preemption(run) == promoted


def test_slurm_smoke_proves_external_sigusr1_requeue_and_resume(tmp_path):
    ready, bindings, receipt = completed_smoke(tmp_path)
    assert slurm_smoke.validate_smoke_receipt(ready, expected_bindings=bindings) == receipt
    assert receipt["schema"] == 2
    assert receipt["real_requeue_count"] == 1
    assert receipt["signal"] == {
        "origin": slurm_smoke.SIGNAL_ORIGIN,
        "name": "SIGUSR1",
        "count": 1,
        "handled": True,
    }
    assert [item["restart_count"] for item in receipt["allocations"]] == [0, 1]
    assert receipt["checkpoint"]["hpc_sha256"] == receipt["checkpoint"]["last_sha256"]
    assert (
        receipt["checkpoint"]["dev_scene_eval_state"]
        == slurm_smoke.SYNTHETIC_DEV_SCENE_EVAL_STATE
    )
    assert (
        receipt["resume"]["restored_dev_scene_eval_state"]
        == slurm_smoke.SYNTHETIC_DEV_SCENE_EVAL_STATE
    )
    assert receipt["resume"]["dev_scene_eval_state_preserved"] is True
    assert receipt["resume"]["final_step"] > receipt["resume"]["resumed_step"]
    assert not list((ready.parent / "synthetic-cell").rglob("*.promoting"))
    assert not any((tmp_path / "runs" / cell.exp_id).exists() for cell in contracts.load_cells(REPO))


def test_slurm_smoke_rejects_dev_scene_eval_state_drift(tmp_path):
    ready, bindings, receipt = completed_smoke(tmp_path)
    receipt["resume"]["restored_dev_scene_eval_state"]["best"] = 0.25
    contracts.atomic_write_json(ready, receipt)
    with pytest.raises(RuntimeError, match="restore exact DevSceneEval"):
        slurm_smoke.validate_smoke_receipt(
            ready,
            expected_bindings=bindings,
        )


def test_slurm_smoke_external_signal_timeout_is_fail_closed(tmp_path):
    bindings = smoke_bindings()
    runs = tmp_path / "runs"
    signal_ready = slurm_smoke.smoke_root(runs) / slurm_smoke.SIGNAL_READY_NAME
    command = smoke_cli_args(bindings, runs, signal_ready)
    command[-1] = "0.05"
    completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, timeout=5)
    assert completed.returncode != 0
    assert "timed out" in completed.stderr
    assert not signal_ready.exists()
    assert not (slurm_smoke.smoke_root(runs) / slurm_smoke.READY_NAME).exists()


def test_slurm_smoke_rejects_pid_symlink_and_legacy_sif_receipt(tmp_path):
    bindings = smoke_bindings()
    runs = tmp_path / "runs"
    root = slurm_smoke.smoke_root(runs)
    root.mkdir(parents=True)
    target = tmp_path / "pid-target"
    target.write_text("1\n")
    signal_ready = root / slurm_smoke.SIGNAL_READY_NAME
    signal_ready.symlink_to(target)
    with pytest.raises(RuntimeError, match="already exists"):
        slurm_smoke.run_allocation(
            runs_root=runs,
            bindings=bindings,
            job_id="1",
            restart_count=0,
            external_signal_ready=signal_ready,
            signal_timeout_seconds=0.1,
        )
    legacy = root / slurm_smoke.READY_NAME
    contracts.atomic_write_json(legacy, {
        "schema": 1,
        "status": "ready",
        "bindings": {
            "source": bindings["source"],
            "sif": {"sha256": "3" * 64, "container_build_sha256": "4" * 64},
            "package": base_payload(),
        },
    })
    with pytest.raises(RuntimeError, match="binding groups"):
        slurm_smoke.validate_smoke_receipt(legacy, expected_bindings=bindings)


def test_slurm_smoke_binding_lock_distinguishes_both_transfers(tmp_path):
    ready, bindings, _receipt = completed_smoke(tmp_path)
    incompatible = json.loads(json.dumps(bindings))
    incompatible["runtime_amendment"]["manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bindings mismatch"):
        slurm_smoke.validate_smoke_receipt(ready, expected_bindings=incompatible)
    incompatible = json.loads(json.dumps(bindings))
    incompatible["base_payload"]["manifest_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bindings mismatch"):
        slurm_smoke.validate_smoke_receipt(ready, expected_bindings=incompatible)


def test_native_venv_builder_is_final_path_offline_and_copies_python():
    source = (REPO / "scripts/h100/build_venv.py").read_text()
    assert contracts.EXPECTED_NATIVE_PYTHON_VERSION == "3.11.13"
    assert build_venv.EXPECTED_PYTHON_VERSION == (
        contracts.EXPECTED_NATIVE_PYTHON_VERSION
    )
    assert '"venv",\n                "--copies"' in source
    assert '"--no-index"' in source
    assert '"--only-binary=:all:"' in source
    assert "make_tree_readonly(output)" in source
    assert "tree_manifest(output)" in source
    assert "receipt.chmod(0o444)" in source
    assert "venv Python interpreter must be copied" in source
    assert "relocat" in source.lower()


def test_native_venv_wheelhouse_and_freeze_are_fail_closed(tmp_path):
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = wheelhouse / "example-1-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    initial = build_venv.wheelhouse_manifest(wheelhouse)
    assert build_venv.assert_wheelhouse_unchanged(wheelhouse, initial) == initial
    wheel.write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="changed"):
        build_venv.assert_wheelhouse_unchanged(wheelhouse, initial)
    wheel.write_bytes(b"wheel")
    (wheelhouse / "alias.whl").symlink_to(wheel)
    with pytest.raises(RuntimeError, match="regular .whl"):
        build_venv.wheelhouse_manifest(wheelhouse)
    lock = "Alpha_Pkg==1.0\nbeta==2\n"
    freeze = "alpha-pkg==1.0\nbeta==2\npip==99\nsetuptools==80\nwheel==1\n"
    assert build_venv.assert_freeze_matches_lock(lock, freeze) == {"alpha-pkg": "1.0", "beta": "2"}
    with pytest.raises(RuntimeError, match="unexpected"):
        build_venv.assert_freeze_matches_lock(lock, freeze + "mystery==1\n")


def test_slurm_native_defaults_identity_order_and_clean_runtime_are_static():
    job = SBATCH.read_text()
    assert "#SBATCH --account=geofam" in job
    assert "#SBATCH --partition=minor-use-case" in job
    assert "#SBATCH --reservation=" not in job
    assert "#SBATCH --gpus-per-node=8" in job
    assert "#SBATCH --cpus-per-task=48" in job
    assert "#SBATCH --mem=256G" in job
    assert "#SBATCH --time=1-12:30:00" in job
    assert "#SBATCH --signal=B:USR1@900" in job and "#SBATCH --requeue" in job
    assert job.index("export NVIDIA_TF32_OVERRIDE=0") < job.index('source "$compute_site"')
    loader_export = 'export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"'
    assert job.index('source "$compute_site"') < job.index(loader_export)
    assert job.index(loader_export) < job.index('"$H100_TRANSFER_PYTHON"')
    assert '"LD_LIBRARY_PATH=$H100_BASE_PYTHON_LIB_DIR"' in job
    scratch_ready = job.index("\ncreate_allocation_scratch\n")
    scratch_gate = job.index("scratch_free=")
    assert scratch_ready < scratch_gate < job.index("scripts.h100.data_staging")
    assert "scripts.handoff extract" not in job
    assert 'git clone "$H100_RUNTIME_BUNDLE" "$repo"' in job
    assert 'git clone "$base_repo_bundle"' not in job
    assert '"$repo/scripts/h100/build_venv.py" verify' in job
    assert '"$H100_VENV_ROOT/bin/python"' in job
    assert "/usr/bin/env -i" in job
    assert "source " not in job[job.index("native_env=(") :]
    assert 'ln -s "$H100_RUNS_ROOT" "$repo/runs"' in job
    assert "500000000000" in job
    assert "unset BOX_JWT_CONFIG BOX_FOLDER_ID" in job
    assert "scripts.h100.strict_fp32_probe --expected-gpus 8" in job
    assert '"${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue' in job
    assert "--base-payload-root" in job and "--runtime-amendment-root" in job


def test_allocation_scratch_contract_is_private_guarded_and_portable():
    for batch in (SBATCH, SMOKE_SBATCH):
        source = batch.read_text()
        assert 'h100_scratch_root="$(canonical_write_root H100_SCRATCH_ROOT)"' in source
        assert 'expected="$H100_SCRATCH_ROOT/xview3-${job_id}-r${restart_count}"' in source
        assert 'mkdir -m 700 -- "$allocation_scratch"' in source
        assert "\"$(stat -c '%a' \"$allocation_scratch\")\" != \"700\"" in source
        assert "\"$(stat -c '%u' \"$allocation_scratch\")\" != \"$(id -u)\"" in source
        assert 'trap cleanup_allocation_scratch EXIT' in source
        assert 'find "$allocation_scratch" -xdev -depth -mindepth 1 -delete' in source
        assert 'rmdir -- "$allocation_scratch"' in source
        assert 'allocation scratch destination must start absent' in source
        assert source.index("\ncreate_allocation_scratch\n") < source.index(
            'scratch="$allocation_scratch"'
        )
        assert '${SLURM_TMPDIR:?' not in source
        assert 'scratch="${SLURM_TMPDIR' not in source
        requeue_call = source.index(
            '"${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue'
        )
        cleanup_call = source.rfind(
            "  cleanup_allocation_scratch_before_requeue\n", 0, requeue_call
        )
        assert cleanup_call != -1
    campaign = SBATCH.read_text()
    assert campaign.index("\ncreate_allocation_scratch\n") < campaign.index(
        "scratch_free="
    ) < campaign.index("scripts.h100.data_staging")
    assert "500000000000" in campaign


def test_operational_slurm_has_no_container_runtime_or_sif_surface():
    operational = "\n".join(path.read_text() for path in (SBATCH, SMOKE_SBATCH, SUBMIT))
    assert re.search(r"\b(apptainer|enroot|sif)\b", operational, re.IGNORECASE) is None
    assert "activate" not in operational
    assert '${SLURM_TMPDIR:?' not in operational
    assert 'scratch="${SLURM_TMPDIR' not in operational
    assert "--export=NONE" in SUBMIT.read_text()


def test_submit_and_site_interfaces_are_safe_and_untracked():
    submit = SUBMIT.read_text()
    example = SITE_EXAMPLE.read_text()
    assert '--account="${H100_ACCOUNT:-geofam}"' in submit
    assert '--partition="${H100_PARTITION:-minor-use-case}"' in submit
    assert 'if [[ -n "${H100_RESERVATION:-}" ]]' in submit
    assert 'sbatch_args+=(--reservation="$H100_RESERVATION")' in submit
    assert '--reservation="${H100_RESERVATION:-geofam}"' not in submit
    assert '--output="$H100_JOB_LOG_DIR/%x-%j.out"' in submit
    assert 'env -u BOX_JWT_CONFIG -u BOX_FOLDER_ID sbatch' in submit
    assert '--export=NONE' in submit
    for name in (
        "H100_BASE_PACKAGE_ROOT",
        "H100_BASE_PYTHON_LIB_DIR",
        "H100_RUNTIME_PACKAGE_ROOT",
        "H100_RUNTIME_BUNDLE",
        "H100_BASE_PYTHON",
        "H100_VENV_ROOT",
        "H100_VENV_BUILD_JSON",
        "H100_PROJECT_ROOT",
        "H100_TRANSFER_PYTHON",
        "H100_JOB_LOG_DIR",
        "H100_SCRATCH_ROOT",
        "H100_V100_CONTROL_PLANE",
        "H100_REFERENCES_PACKAGE_ROOT",
        "H100_REFERENCES_PACKAGE_ID",
        "H100_REFERENCES_PRODUCER_GIT_SHA",
        "H100_REFERENCES_IDENTITY_SHA256",
        "H100_REFERENCES_MANIFEST_SHA256",
        "H100_REFERENCES_READY_SHA256",
        "H100_REFERENCES_SHA256SUMS_SHA256",
        "H100_DIAGNOSTIC_ISOLATION_PACKAGE_ROOT",
        "H100_DIAGNOSTIC_ISOLATION_PACKAGE_ID",
        "H100_DIAGNOSTIC_ISOLATION_PRODUCER_GIT_SHA",
        "H100_DIAGNOSTIC_ISOLATION_IDENTITY_SHA256",
        "H100_DIAGNOSTIC_ISOLATION_MANIFEST_SHA256",
        "H100_DIAGNOSTIC_ISOLATION_READY_SHA256",
        "H100_DIAGNOSTIC_ISOLATION_SHA256SUMS_SHA256",
    ):
        assert name in example
    assert re.search(
        r"(?m)^\s*(?:export\s+)?BOX_(?:JWT_CONFIG|FOLDER_ID)=", example
    ) is None
    assert "H100_MAIL_TYPE=ALL" in example
    operational = "\n".join(
        path.read_text() for path in (SUBMIT, SBATCH, SMOKE_SBATCH)
    )
    assert "H100_V100_RUNS_ROOT" not in example
    assert "H100_V100_RUNS_ROOT" not in operational
    assert "H100_V100_CONTROL_PLANE=box-transfer-v1" in example
    assert operational.count("box-transfer-v1") >= 3
    assert "leaving the live V100 campaign untouched" in example
    assert '"$mode" == "cutover-check"' in submit
    assert (REPO / "slurm/h100/.gitignore").read_text().strip() == "site.env"
    assert "refusing tracked site.env" in submit


def test_shell_entrypoints_are_executable():
    for path in (SUBMIT, SBATCH, SMOKE_SBATCH):
        assert path.stat().st_mode & stat.S_IXUSR, f"{path} is not executable"


def test_smoke_batch_signals_direct_native_child_and_requeues_externally():
    smoke = SMOKE_SBATCH.read_text()
    assert "#SBATCH --gpus-per-node=1" in smoke
    assert "#SBATCH --time=00:15:00" in smoke
    assert "#SBATCH --requeue" in smoke
    assert '"$H100_VENV_ROOT/bin/python" -m scripts.h100.slurm_smoke run' in smoke
    assert 'signal_ready="$smoke_root/SLURM_SMOKE_SIGNAL_READY.pid"' in smoke
    assert 'kill -USR1 "$signal_pid"' in smoke
    assert '"${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue' in smoke
    assert "unset BOX_JWT_CONFIG BOX_FOLDER_ID" in smoke
    assert 'export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"' in smoke
    assert '"LD_LIBRARY_PATH=$H100_BASE_PYTHON_LIB_DIR"' in smoke
    assert 'h100_scratch_root="$(canonical_write_root H100_SCRATCH_ROOT)"' in smoke
    assert "\ncreate_allocation_scratch\n" in smoke


def test_cutover_validates_schema2_native_ready_and_rejects_v1_sif(tmp_path):
    smoke_path, bindings, smoke_receipt = completed_smoke(tmp_path)
    ready = ready_payload(smoke_path, smoke_receipt, bindings)
    add_acceptance_receipts(tmp_path, ready)
    ready_path = tmp_path / "H100_READY.json"
    contracts.atomic_write_json(ready_path, ready)
    assert cutover.validate_h100_ready(
        ready_path,
        expected_git_sha=RUNTIME_GIT_SHA,
        expected_venv_sha256=VENV_SHA256,
        expected_venv_build_sha256=VENV_BUILD_SHA256,
        expected_base_python_sha256=BASE_PYTHON_SHA256,
        expected_base_python_runtime_sha256=BASE_PYTHON_RUNTIME_SHA256,
        expected_wheelhouse_sha256=WHEELHOUSE_SHA256,
        expected_base_extraction_receipt_sha256=BASE_EXTRACTION_RECEIPT_SHA256,
        expected_base_payload=base_payload(),
        expected_runtime_amendment=runtime_amendment(),
        expected_frozen_sha256=frozen_hashes(),
        expected_smoke_receipt=smoke_receipt,
        expected_smoke_sha256=contracts.sha256_file(smoke_path),
    ) == ready
    legacy = dict(ready)
    legacy["schema"] = 1
    legacy["sif"] = {"sha256": "0" * 64}
    contracts.atomic_write_json(ready_path, legacy)
    with pytest.raises(RuntimeError, match="schema-2"):
        cutover.validate_h100_ready(
            ready_path,
            expected_git_sha=RUNTIME_GIT_SHA,
            expected_venv_sha256=VENV_SHA256,
            expected_venv_build_sha256=VENV_BUILD_SHA256,
            expected_base_python_sha256=BASE_PYTHON_SHA256,
            expected_base_python_runtime_sha256=BASE_PYTHON_RUNTIME_SHA256,
            expected_wheelhouse_sha256=WHEELHOUSE_SHA256,
            expected_base_extraction_receipt_sha256=(
                BASE_EXTRACTION_RECEIPT_SHA256
            ),
            expected_base_payload=base_payload(),
            expected_runtime_amendment=runtime_amendment(),
            expected_frozen_sha256=frozen_hashes(),
            expected_smoke_receipt=smoke_receipt,
            expected_smoke_sha256=contracts.sha256_file(smoke_path),
        )




def test_cutover_contains_no_process_control():
    for relative in ("scripts/h100/cutover.py", "scripts/h100/operator_cutover.py"):
        source = (REPO / relative).read_text()
        assert "os.kill" not in source
        assert "terminate(" not in source
        assert "scontrol" not in source
    assert "v100_action" in (REPO / "scripts/h100/cutover.py").read_text()




def test_source_receipt_is_dual_identity_schema2_and_rejects_v1_sif(tmp_path):
    frozen = frozen_hashes()
    payload = source_validation.expected_receipt(
        git_sha=RUNTIME_GIT_SHA,
        frozen_sha256=frozen,
        base_payload=base_payload(),
        runtime_amendment=runtime_amendment(),
    )
    assert payload["schema"] == 2
    receipt = tmp_path / source_validation.SOURCE_RECEIPT_NAME
    contracts.atomic_write_json(receipt, payload)
    digest = contracts.sha256_file(receipt)
    assert source_validation.validate_source_receipt(
        receipt,
        expected_sha256=digest,
        expected_git_sha=RUNTIME_GIT_SHA,
        expected_hashes=frozen,
        expected_base_payload=base_payload(),
        expected_runtime_amendment=runtime_amendment(),
    ) == payload
    legacy = {
        "schema": 1,
        "status": "source-validated",
        "git_sha": RUNTIME_GIT_SHA,
        "sif_sha256": "0" * 64,
        "package": base_payload(),
    }
    contracts.atomic_write_json(receipt, legacy)
    with pytest.raises(RuntimeError, match="keys"):
        source_validation.validate_source_receipt(
            receipt,
            expected_sha256=contracts.sha256_file(receipt),
            expected_git_sha=RUNTIME_GIT_SHA,
            expected_hashes=frozen,
            expected_base_payload=base_payload(),
            expected_runtime_amendment=runtime_amendment(),
        )


def test_campaign_resume_bindings_are_hash_locked():
    bindings = {
        "campaign_id": "campaign",
        "acceptance_uuid": "acceptance",
        "source_validation_sha256": "a" * 64,
        "evaluation_ground_truth_sha256": "d" * 64,
        "cutover_ready_sha256": "b" * 64,
        "v100_diagnostic_isolation_sha256": "c" * 64,
    }
    campaign.validate_campaign_resume_bindings(dict(bindings), bindings)
    with pytest.raises(RuntimeError, match="source_validation_sha256"):
        campaign.validate_campaign_resume_bindings({**bindings, "source_validation_sha256": "e" * 64}, bindings)


def test_sbatch_supplies_required_native_acceptance_and_campaign_arguments():
    job = SBATCH.read_text()
    acceptance_args = (
        "--venv-root",
        "--venv-sha256",
        "--venv-build-sha256",
        "--base-python",
        "--base-python-sha256",
        "--scratch-free-before-extraction",
        "--source-validation-json",
        "--source-validation-sha256",
        "--host-test-receipt",
        "--host-test-receipt-sha256",
        "--base-payload-package-id",
        "--runtime-amendment-package-id",
    )
    campaign_args = (
        "--expected-git-sha",
        "--venv-root",
        "--venv-sha256",
        "--venv-build-sha256",
        "--source-validation-json",
        "--source-validation-sha256",
        "--cutover-ready-json",
        "--cutover-ready-sha256",
        "--v100-diagnostic-isolation-json",
        "--v100-diagnostic-isolation-sha256",
    )
    acceptance_source = (REPO / "scripts/h100/acceptance.py").read_text()
    campaign_source = (REPO / "scripts/h100/campaign.py").read_text()
    for option in acceptance_args:
        assert f'parser.add_argument("{option}"' in acceptance_source
        assert option in job
    for option in campaign_args:
        assert f'parser.add_argument("{option}"' in campaign_source
        assert option in job
    assert '--scratch-free-before-extraction "$scratch_free"' in job


def test_source_validation_cli_verifies_both_package_roots_before_receipt():
    source = (REPO / "scripts/h100/source_validation.py").read_text()
    for option in ("--base-payload-root", "--runtime-amendment-root"):
        assert f'parser.add_argument("{option}"' in source
        assert option in SBATCH.read_text()
    assert "prepare_runtime_control_verifier" in source
    assert "prepare_runtime_verifier" not in source
    assert "verify_transfer_bindings(" in source


def test_host_gates_precede_native_pythonpath_and_campaign_resets_canonical_paths():
    job = SBATCH.read_text()
    source_gate = job.index("scripts.h100.source_validation")
    host_gate = job.index("scripts.h100.host_test_gate")
    native_path = job.index('"PYTHONPATH=$repo/scripts/h100:$repo"')
    assert source_gate < host_gate < native_path
    assert 'PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON"' in job
    assert "tests/test_h100_handoff.py" in host_test_gate.HOST_TESTS
    assert "tests/test_experiment_manifest.py" in host_test_gate.HOST_TESTS
    assert "tests/test_h100_submit_isolation.py" in host_test_gate.HOST_TESTS
    assert "V100_DIAGNOSTIC_ISOLATION.json" in job
    assert "V100_CORE_ARCHIVED.json" not in job
    assert "V100_CORE_ARCHIVE_MANIFEST.json" not in job


def test_submit_cutover_is_non_submitting_and_uses_transfer_python():
    submit = SUBMIT.read_text()
    cutover_block = submit.index('if [[ "$mode" == "cutover-check" ]]')
    sbatch_at = submit.index("sbatch", cutover_block)
    cutover_end = submit.index("CUTOVER_READY written", cutover_block)
    cutover_invocation = submit[cutover_block:cutover_end]
    assert cutover_block < submit.index("exit 0", cutover_block) < sbatch_at
    assert submit.count('PYTHONNOUSERSITE=1 PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON"') >= 2
    assert "--persist-meta-root" in submit
    cutover_source = (REPO / "scripts/h100/cutover.py").read_text()
    for option in (
        "--expected-h100-campaign-id",
        "--expected-venv-sha256",
        "--expected-base-payload-package-id",
        "--expected-base-payload-git-sha",
        "--expected-base-payload-manifest-sha256",
        "--expected-base-payload-ready-sha256",
        "--expected-base-payload-sha256sums-sha256",
        "--expected-base-payload-repo-bundle-sha256",
        "--expected-runtime-amendment-package-id",
        "--expected-runtime-amendment-git-sha",
        "--expected-runtime-amendment-manifest-sha256",
        "--expected-runtime-amendment-ready-sha256",
        "--expected-runtime-amendment-sha256sums-sha256",
        "--expected-runtime-amendment-bundle-sha256",
    ):
        assert option in cutover_source
        assert option in cutover_invocation
    for option in (
        "--references-package-root",
        "--expected-references-package-id",
        "--expected-references-producer-git-sha",
        "--expected-references-identity-sha256",
        "--expected-references-manifest-sha256",
        "--expected-references-ready-sha256",
        "--expected-references-sha256sums-sha256",
        "--diagnostic-isolation-package-root",
        "--expected-diagnostic-isolation-package-id",
        "--expected-diagnostic-isolation-producer-git-sha",
        "--expected-diagnostic-isolation-identity-sha256",
        "--expected-diagnostic-isolation-manifest-sha256",
        "--expected-diagnostic-isolation-ready-sha256",
        "--expected-diagnostic-isolation-sha256sums-sha256",
    ):
        assert option in submit
    assert "V100 remains untouched" in submit
    assert "--bound-archive-manifest" not in submit


def test_reverse_results_delegates_shared_handoff_schema(tmp_path):
    argv = reverse_command(
        repo=tmp_path / "repo",
        runs_root=tmp_path / "runs",
        campaign_manifest=tmp_path / "runs/.h100/campaign_manifest.json",
        output_dir=tmp_path / "out",
        max_part_bytes=123,
    )
    assert argv[1:5] == ["-m", "scripts.handoff", "build-results", "--repo"]
    assert argv[-2:] == ["--max-part-bytes", "123"]
    assert "--output-dir" in argv
    reverse_source = (REPO / "scripts/h100/reverse_results.py").read_text()
    assert 'runs_root / "summary/grid.csv"' in reverse_source
    cli_source = (REPO / "scripts/handoff/__main__.py").read_text()
    build_results_block = cli_source.split(
        'elif args.command == "build-results":', 1
    )[1].split("else:  # pragma: no cover", 1)[0]
    assert "_client(" not in build_results_block
    assert "preflight_box(" not in build_results_block
