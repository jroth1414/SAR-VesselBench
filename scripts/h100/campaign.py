"""Eight-GPU strict-FP32 H100 queue around the unchanged Lightning trainer."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from scripts.h100.build_venv import EXPECTED_PYTHON_VERSION
from scripts.h100.contracts import (
    EFFECTIVE_BATCH,
    EXPECTED_GPU_COUNT,
    EXPECTED_PRECISION,
    GRADIENT_ACCUMULATION,
    MICRO_BATCH,
    Cell,
    atomic_write_json,
    cutover_acceptance_bindings,
    load_cells,
    sha256_file,
    validate_completion_marker,
    validate_bound_cutover_forecast,
    validate_gpu_inventory,
)
from scripts.h100.precision import assert_sitecustomize_active
from scripts.h100.source_validation import validate_source_receipt

HOST_REQUEUE_EXIT_CODE = 75
SCORED_FIELDS = {
    "test_inference_precision",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_near_shore_f1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def promote_hpc_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.rglob("hpc_ckpt*.ckpt"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise RuntimeError(f"no Lightning HPC checkpoint found under {run_dir}")
    source = candidates[-1]
    destination = run_dir / "checkpoints/last.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".ckpt.promoting")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return destination


def marker_for(cell: Cell, runs_root: Path) -> Path:
    return runs_root / cell.exp_id / "final_metrics.json"


def hardware_class(payload: Mapping[str, object]) -> dict:
    devices = validate_gpu_inventory(payload.get("devices", []))
    result = {
        "gpu_name": devices[0]["name"],
        "total_memory_bytes": int(devices[0]["total_memory_bytes"]),
        "compute_capability": list(devices[0]["compute_capability"]),
        "driver_version": str(payload.get("driver_version", "")),
        "torch": str(payload.get("torch", "")),
        "cuda_build": str(payload.get("cuda_build", "")),
    }
    if any(not str(result[key]).strip() for key in ("driver_version", "torch", "cuda_build")):
        raise RuntimeError("H100 hardware class lacks driver/torch/CUDA provenance")
    return result


def validate_runtime_provenance(
    provenance: Mapping[str, object],
    *,
    cell: Cell,
    campaign_id: str,
    git_sha: str,
    detector_sha256: str,
    venv_sha256: str,
    venv_build_sha256: str,
    base_python: Mapping[str, object],
    base_payload: Mapping[str, str],
    runtime_amendment: Mapping[str, str],
    acceptance_uuid: str,
    source_validation_sha256: str,
    cutover_ready_sha256: str,
    v100_core_archived_sha256: str,
    strict_fp32: Mapping[str, str],
    accepted_hardware_class: Mapping[str, object],
) -> None:
    expected = {
        "schema": 2,
        "campaign_id": campaign_id,
        "exp_id": cell.exp_id,
        "git_sha": git_sha,
        "detector_sha256": detector_sha256,
        "venv_sha256": venv_sha256,
        "venv_build_sha256": venv_build_sha256,
        "base_python": dict(base_python),
        "base_payload": dict(base_payload),
        "runtime_amendment": dict(runtime_amendment),
        "acceptance_uuid": acceptance_uuid,
        "source_validation_sha256": source_validation_sha256,
        "cutover_ready_sha256": cutover_ready_sha256,
        "v100_core_archived_sha256": v100_core_archived_sha256,
        "strict_fp32": dict(strict_fp32),
        "accepted_hardware_class": dict(accepted_hardware_class),
        "precision": EXPECTED_PRECISION,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch": EFFECTIVE_BATCH,
    }
    mismatches = {
        key: (value, provenance.get(key))
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    gpu_expected = {
        "gpu_name": accepted_hardware_class["gpu_name"],
        "gpu_total_memory_bytes": accepted_hardware_class["total_memory_bytes"],
        "compute_capability": accepted_hardware_class["compute_capability"],
        "driver_version": accepted_hardware_class["driver_version"],
        "torch": accepted_hardware_class["torch"],
        "cuda_build": accepted_hardware_class["cuda_build"],
    }
    mismatches.update(
        {
            key: (value, provenance.get(key))
            for key, value in gpu_expected.items()
            if provenance.get(key) != value
        }
    )
    attempts = provenance.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        mismatches["attempts"] = ("nonempty list", attempts)
    elif any(not str(item.get("gpu_uuid", "")).strip() for item in attempts):
        mismatches["attempts.gpu_uuid"] = ("nonempty", attempts)
    if not str(provenance.get("gpu_uuid", "")).strip():
        mismatches["gpu_uuid"] = ("nonempty", provenance.get("gpu_uuid"))
    if mismatches:
        raise RuntimeError(f"recipe-mismatched runtime provenance for {cell.exp_id}: {mismatches}")


def validate_scored_completion(
    cell: Cell,
    *,
    runs_root: Path,
    git_sha: str,
    detector_sha256: str,
) -> dict:
    run_dir = runs_root / cell.exp_id
    payload = validate_completion_marker(
        marker_for(cell, runs_root),
        cell=cell,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
    )
    mismatches = {}
    if payload.get("test_inference_precision") != EXPECTED_PRECISION:
        mismatches["test_inference_precision"] = (
            EXPECTED_PRECISION,
            payload.get("test_inference_precision"),
        )
    for key in ("test_f1", "test_precision", "test_recall", "test_near_shore_f1"):
        try:
            finite = math.isfinite(float(payload[key]))
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite:
            mismatches[key] = ("finite", payload.get(key))
    for name in ("best.ckpt", "last.ckpt"):
        checkpoint = run_dir / "checkpoints" / name
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            mismatches[name] = ("nonempty checkpoint", None)
    if mismatches:
        raise RuntimeError(f"{cell.exp_id} is not fully train+test complete: {mismatches}")
    return payload


def checkpoint_for_preemption(run_dir: Path) -> Path:
    wrapper_path = run_dir / "cell_wrapper.json"
    phase = None
    if wrapper_path.exists():
        phase = json.loads(wrapper_path.read_text()).get("phase")
    if phase == "score-test":
        last = run_dir / "checkpoints/last.ckpt"
        best = run_dir / "checkpoints/best.ckpt"
        if not last.is_file() or not best.is_file():
            raise RuntimeError(f"scoring preemption lacks durable checkpoints: {run_dir}")
        return last
    if phase != "train":
        raise RuntimeError(f"preemption phase is absent or invalid: {run_dir}")
    return promote_hpc_checkpoint(run_dir)


def existing_cell_state(
    cell: Cell,
    *,
    runs_root: Path,
    campaign_id: str,
    git_sha: str,
    detector_sha256: str,
    venv_sha256: str,
    venv_build_sha256: str,
    base_python: Mapping[str, object],
    base_payload: Mapping[str, str],
    runtime_amendment: Mapping[str, str],
    acceptance_uuid: str,
    source_validation_sha256: str,
    cutover_ready_sha256: str,
    v100_core_archived_sha256: str,
    strict_fp32: Mapping[str, str],
    accepted_hardware_class: Mapping[str, object],
) -> str:
    run_dir = runs_root / cell.exp_id
    final = marker_for(cell, runs_root)
    scored_payload: dict | None = None
    if final.exists():
        training_payload = validate_completion_marker(
            final,
            cell=cell,
            git_sha=git_sha,
            detector_sha256=detector_sha256,
        )
        present_scored = SCORED_FIELDS & set(training_payload)
        if present_scored and present_scored != SCORED_FIELDS:
            raise RuntimeError(
                f"partially written test completion marker is invalid: {final}"
            )
        for name in ("best.ckpt", "last.ckpt"):
            checkpoint = run_dir / "checkpoints" / name
            if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
                raise RuntimeError(f"training-complete cell lacks {checkpoint}")
        if present_scored == SCORED_FIELDS:
            scored_payload = validate_scored_completion(
                cell,
                runs_root=runs_root,
                git_sha=git_sha,
                detector_sha256=detector_sha256,
            )
    if not run_dir.exists():
        return "pending"
    provenance_path = run_dir / "runtime_provenance.json"
    if not provenance_path.exists():
        raise RuntimeError(f"occupied namespace lacks provenance: {run_dir}")
    provenance = json.loads(provenance_path.read_text())
    validate_runtime_provenance(
        provenance,
        cell=cell,
        campaign_id=campaign_id,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        venv_sha256=venv_sha256,
        venv_build_sha256=venv_build_sha256,
        base_python=base_python,
        base_payload=base_payload,
        runtime_amendment=runtime_amendment,
        acceptance_uuid=acceptance_uuid,
        source_validation_sha256=source_validation_sha256,
        cutover_ready_sha256=cutover_ready_sha256,
        v100_core_archived_sha256=v100_core_archived_sha256,
        strict_fp32=strict_fp32,
        accepted_hardware_class=accepted_hardware_class,
    )
    if scored_payload is not None:
        try:
            finalized_cell_runtime(provenance, scored_payload, cell.exp_id)
        except RuntimeError:
            # The scorer may have completed immediately before allocation loss.
            # Relaunching the wrapper is safe: scoring sees test_f1 and no-ops,
            # then controller poll finalizes the new provenance attempt.
            return "resume"
        return "complete"
    return "resume"


def finalized_cell_runtime(
    provenance: Mapping[str, object], marker: Mapping[str, object], exp_id: str
) -> dict[str, object]:
    attempts = provenance.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError(f"{exp_id}: finalized provenance has no attempts")
    last = attempts[-1]
    if (
        not isinstance(last, Mapping)
        or last.get("exit_code") != 0
        or not str(last.get("finished_utc", "")).strip()
    ):
        raise RuntimeError(f"{exp_id}: final attempt is not durably complete")
    try:
        active_seconds = float(provenance["accumulated_active_seconds"])
        elapsed_hours = float(provenance["elapsed_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{exp_id}: finalized runtime is invalid") from exc
    if (
        not math.isfinite(active_seconds)
        or active_seconds <= 0
        or not math.isfinite(elapsed_hours)
        or not math.isclose(elapsed_hours, active_seconds / 3600.0)
        or not str(provenance.get("completed_utc", "")).strip()
        or provenance.get("epochs_run") != marker.get("epochs_run")
        or provenance.get("best_dev_f1") != marker.get("best_dev_f1")
        or provenance.get("test_f1") != marker.get("test_f1")
    ):
        raise RuntimeError(f"{exp_id}: finalized runtime/marker binding mismatch")
    return {
        "elapsed_hours": elapsed_hours,
        "attempts": len(attempts),
        "epochs_run": marker.get("epochs_run"),
        "test_f1": marker.get("test_f1"),
    }


def validate_failed_namespace(
    cell: Cell,
    *,
    runs_root: Path,
    campaign_id: str,
    git_sha: str,
    detector_sha256: str,
    venv_sha256: str,
    venv_build_sha256: str,
    base_python: Mapping[str, object],
    base_payload: Mapping[str, str],
    runtime_amendment: Mapping[str, str],
    acceptance_uuid: str,
    source_validation_sha256: str,
    cutover_ready_sha256: str,
    v100_core_archived_sha256: str,
    strict_fp32: Mapping[str, str],
    accepted_hardware_class: Mapping[str, object],
) -> None:
    run_dir = runs_root / cell.exp_id
    provenance_path = run_dir / "runtime_provenance.json"
    if not run_dir.is_dir() or not provenance_path.is_file():
        raise RuntimeError(f"known failed cell lacks its bound namespace: {run_dir}")
    provenance = json.loads(provenance_path.read_text())
    validate_runtime_provenance(
        provenance,
        cell=cell,
        campaign_id=campaign_id,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        venv_sha256=venv_sha256,
        venv_build_sha256=venv_build_sha256,
        base_python=base_python,
        base_payload=base_payload,
        runtime_amendment=runtime_amendment,
        acceptance_uuid=acceptance_uuid,
        source_validation_sha256=source_validation_sha256,
        cutover_ready_sha256=cutover_ready_sha256,
        v100_core_archived_sha256=v100_core_archived_sha256,
        strict_fp32=strict_fp32,
        accepted_hardware_class=accepted_hardware_class,
    )


def next_launches(
    cells: list[Cell],
    *,
    running_ids: set[str],
    complete_ids: set[str],
    free_gpus: list[int],
    failure_seen: bool,
    preemption_seen: bool,
) -> list[tuple[int, Cell]]:
    if failure_seen or preemption_seen:
        return []
    pending = [
        cell
        for cell in cells
        if cell.exp_id not in running_ids and cell.exp_id not in complete_ids
    ]
    return list(zip(free_gpus, pending[: len(free_gpus)], strict=False))


def fail_stop_launches(
    cells: list[Cell],
    *,
    allowed_ids: set[str],
    failed_ids: set[str],
    running_ids: set[str],
    complete_ids: set[str],
    free_gpus: list[int],
    preemption_seen: bool,
) -> list[tuple[int, Cell]]:
    """Resume only cells that were already running when fail-stop engaged."""

    if preemption_seen:
        return []
    pending = [
        cell
        for cell in cells
        if cell.exp_id in allowed_ids
        and cell.exp_id not in failed_ids
        and cell.exp_id not in running_ids
        and cell.exp_id not in complete_ids
    ]
    return list(zip(free_gpus, pending[: len(free_gpus)], strict=False))


def restore_fail_stop(
    prior: Mapping[str, object], cells: list[Cell]
) -> tuple[bool, set[str], set[str]]:
    record = prior.get("fail_stop")
    if not isinstance(record, Mapping) or set(record) != {
        "engaged",
        "failed",
        "allowed_to_finish",
    }:
        raise RuntimeError("existing campaign fail-stop record is absent or invalid")
    engaged = record.get("engaged")
    failed_raw = record.get("failed")
    allowed_raw = record.get("allowed_to_finish")
    if type(engaged) is not bool or not isinstance(failed_raw, list) or not isinstance(
        allowed_raw, list
    ):
        raise RuntimeError("existing campaign fail-stop record has invalid types")
    if any(not isinstance(value, str) for value in [*failed_raw, *allowed_raw]):
        raise RuntimeError("existing campaign fail-stop IDs must be strings")
    failed = set(failed_raw)
    allowed = set(allowed_raw)
    valid_ids = {cell.exp_id for cell in cells}
    if (
        len(failed) != len(failed_raw)
        or len(allowed) != len(allowed_raw)
        or not failed <= valid_ids
        or not allowed <= valid_ids
        or failed & allowed
        or engaged != bool(failed)
    ):
        raise RuntimeError("existing campaign fail-stop IDs/state are inconsistent")
    return engaged, failed, allowed


def validate_campaign_resume_bindings(
    prior: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    mismatches = {
        key: (value, prior.get(key))
        for key, value in expected.items()
        if prior.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"existing H100 campaign manifest binding mismatch: {mismatches}"
        )


def collect_and_validate_grid(
    *,
    repo: Path,
    runs_root: Path,
    cells: list[Cell],
    git_sha: str,
    detector_sha256: str,
) -> Path:
    grid = runs_root / "summary/grid.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.curves",
            "collect",
            "--runs-root",
            str(runs_root),
            "--out-csv",
            str(grid),
        ],
        cwd=repo,
        check=True,
    )
    with grid.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_ids = {cell.exp_id for cell in cells}
    actual_ids = {row.get("exp_id") for row in rows}
    if len(rows) != 32 or actual_ids != expected_ids:
        raise RuntimeError("grid.csv is not the exact 32-cell core matrix")
    for row in rows:
        try:
            finite = math.isfinite(float(row["test_f1"]))
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite:
            raise RuntimeError(f"grid.csv lacks finite test_f1 for {row.get('exp_id')}")
        if row.get("monotonicity_ok", "").lower() != "true":
            raise RuntimeError(
                f"grid.csv monotonicity STOP for {row.get('exp_id')}"
            )
        if row.get("git_sha") != git_sha or row.get("detector_sha256") != detector_sha256:
            raise RuntimeError(f"grid.csv provenance mismatch for {row.get('exp_id')}")
    return grid


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        assert_sitecustomize_active()
        self.args = args
        self.repo = args.repo.resolve()
        self.runs_root = args.runs_root.resolve()
        self.cells = load_cells(self.repo)
        self.git_sha = args.expected_git_sha
        self.detector_sha256 = sha256_file(self.repo / "configs/detector.yaml")
        self.venv_sha256 = args.venv_sha256
        self.venv_build_sha256 = args.venv_build_sha256
        self.venv_root = args.venv_root.absolute()
        expected_python = self.venv_root / "bin/python"
        if (
            Path(sys.prefix).resolve() != self.venv_root.resolve()
            or Path(sys.executable).resolve() != expected_python.resolve()
        ):
            raise RuntimeError("campaign is not running under the accepted native venv")
        self.acceptance = json.loads(args.acceptance_json.read_text())
        if self.acceptance.get("schema") != 2 or self.acceptance.get("status") != "ready":
            raise RuntimeError("campaign requires a schema-2 ready H100 acceptance receipt")
        accepted_venv = self.acceptance.get("venv")
        if not isinstance(accepted_venv, Mapping) or (
            accepted_venv.get("path") != str(self.venv_root)
            or accepted_venv.get("sha256") != self.venv_sha256
            or accepted_venv.get("venv_build_sha256") != self.venv_build_sha256
        ):
            raise RuntimeError("campaign native venv differs from acceptance")
        self.base_python = dict(accepted_venv.get("base_python", {}))
        if (
            not self.base_python
            or self.base_python.get("version") != EXPECTED_PYTHON_VERSION
        ):
            raise RuntimeError("campaign accepted base-Python identity is invalid")
        self.base_payload = dict(self.acceptance.get("base_payload", {}))
        self.runtime_amendment = dict(self.acceptance.get("runtime_amendment", {}))
        if not self.base_payload or not self.runtime_amendment:
            raise RuntimeError("campaign transfer identities are absent")
        self.acceptance_uuid = str(self.acceptance.get("acceptance_uuid", ""))
        if not self.acceptance_uuid:
            raise RuntimeError("campaign acceptance UUID is absent")
        accepted_source = self.acceptance.get("source")
        if not isinstance(accepted_source, Mapping):
            raise RuntimeError("campaign acceptance source binding is absent")
        if (
            accepted_source.get("git_sha") != self.git_sha
            or accepted_source.get("frozen_sha256", {}).get("configs/detector.yaml")
            != self.detector_sha256
        ):
            raise RuntimeError("campaign source differs from H100 acceptance")
        self.source_validation_sha256 = args.source_validation_sha256
        self.source_validation = validate_source_receipt(
            args.source_validation_json,
            expected_sha256=self.source_validation_sha256,
            expected_git_sha=self.git_sha,
            expected_hashes=accepted_source["frozen_sha256"],
            expected_base_payload=self.base_payload,
            expected_runtime_amendment=self.runtime_amendment,
        )
        if self.acceptance.get("source_validation", {}).get("sha256") != (
            self.source_validation_sha256
        ):
            raise RuntimeError("campaign source receipt differs from H100 acceptance")
        meta_root = self.runs_root / ".h100"
        canonical_paths = {
            "source validation": (
                args.source_validation_json.absolute(),
                meta_root / "SOURCE_VALIDATED.json",
            ),
            "CUTOVER_READY": (
                args.cutover_ready_json.absolute(),
                meta_root / "CUTOVER_READY.json",
            ),
            "V100_CORE_ARCHIVED": (
                args.v100_core_archived_json.absolute(),
                meta_root / "V100_CORE_ARCHIVED.json",
            ),
            "V100 archive manifest": (
                args.v100_archive_manifest_json.absolute(),
                meta_root / "V100_CORE_ARCHIVE_MANIFEST.json",
            ),
        }
        for label, (actual, expected) in canonical_paths.items():
            if actual != expected or actual.is_symlink() or not actual.is_file():
                raise RuntimeError(f"controller {label} is not canonical: {expected}")
        self.cutover_ready_sha256 = args.cutover_ready_sha256
        if sha256_file(args.cutover_ready_json) != self.cutover_ready_sha256:
            raise RuntimeError("controller CUTOVER_READY SHA-256 mismatch")
        self.cutover_ready = json.loads(args.cutover_ready_json.read_text())
        if self.cutover_ready.get("status") != "cutover-ready":
            raise RuntimeError("controller CUTOVER_READY status is invalid")
        if self.cutover_ready.get("h100_ready") != self.acceptance:
            raise RuntimeError("controller H100_READY differs from canonical CUTOVER_READY")
        if self.cutover_ready.get("acceptance") != cutover_acceptance_bindings(
            self.acceptance
        ):
            raise RuntimeError("controller cutover acceptance subset is inconsistent")
        validate_bound_cutover_forecast(self.cutover_ready)
        self.v100_core_archived_sha256 = args.v100_core_archived_sha256
        if (
            sha256_file(args.v100_core_archived_json)
            != self.v100_core_archived_sha256
        ):
            raise RuntimeError("controller V100_CORE_ARCHIVED SHA-256 mismatch")
        self.v100_core_archived = json.loads(
            args.v100_core_archived_json.read_text()
        )
        if self.v100_core_archived.get("cutover_ready_sha256") != (
            self.cutover_ready_sha256
        ):
            raise RuntimeError("operator archive receipt does not bind CUTOVER_READY")
        expected_h100 = {
            "acceptance_uuid": self.acceptance_uuid,
            "git_sha": self.git_sha,
            "venv_sha256": self.venv_sha256,
            "base_payload": self.base_payload,
            "runtime_amendment": self.runtime_amendment,
        }
        if self.v100_core_archived.get("h100") != expected_h100:
            raise RuntimeError("operator archive receipt differs from accepted H100 identity")
        self.archive_manifest_sha256 = args.v100_archive_manifest_sha256
        if sha256_file(args.v100_archive_manifest_json) != (
            self.archive_manifest_sha256
        ):
            raise RuntimeError("controller V100 archive manifest SHA-256 mismatch")
        self.archive_manifest = json.loads(
            args.v100_archive_manifest_json.read_text()
        )
        archive_binding = self.v100_core_archived.get("archive", {})
        if archive_binding != {
            "manifest_path": str(args.v100_archive_manifest_json.absolute()),
            "manifest_sha256": self.archive_manifest_sha256,
        }:
            raise RuntimeError("operator receipt archive-manifest binding mismatch")
        self.strict_fp32 = dict(self.acceptance.get("strict_fp32", {}))
        if set(self.strict_fp32.values()) != {"ieee"}:
            raise RuntimeError("campaign acceptance is not strict IEEE FP32")
        if args.hardware.get("backend") != self.strict_fp32:
            raise RuntimeError("current allocation strict backend differs from acceptance")
        self.accepted_hardware_class = hardware_class(self.acceptance["hardware"])
        self.allocation_hardware_class = hardware_class(args.hardware)
        if self.allocation_hardware_class != self.accepted_hardware_class:
            raise RuntimeError(
                "current allocation hardware class differs from H100 acceptance: "
                f"{self.allocation_hardware_class} != {self.accepted_hardware_class}"
            )
        meta_root.mkdir(parents=True, exist_ok=True)
        self.lock_handle = (meta_root / "campaign.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another H100 campaign controller holds campaign.lock") from exc
        self.running: dict[int, tuple[subprocess.Popen, Cell, float]] = {}
        self.complete_ids: set[str] = set()
        self.failure_seen = False
        self.failed_ids: set[str] = set()
        self.failure_allowed_ids: set[str] = set()
        self.preemption_seen = False
        self.preemption_forwarded = False
        self.events: list[dict] = []
        self.cell_runtime: dict[str, dict] = {}
        self.request_dir = self.runs_root / ".h100/requeue-requests"
        self.manifest_path = self.runs_root / ".h100/campaign_manifest.json"
        signal.signal(signal.SIGUSR1, self._on_usr1)

    def _on_usr1(self, _signum, _frame) -> None:
        self.preemption_seen = True

    def record(self, event: str, **payload: object) -> None:
        item = {"utc": utc_now(), "event": event, **payload}
        self.events.append(item)
        print(json.dumps(item, sort_keys=True), flush=True)
        self.write_manifest()

    def write_manifest(self, status: str | None = None) -> None:
        payload = {
            "schema": 2,
            "campaign_id": self.args.campaign_id,
            "status": status or ("failed" if self.failure_seen else "running"),
            "git_sha": self.git_sha,
            "detector_sha256": self.detector_sha256,
            "precision": EXPECTED_PRECISION,
            "micro_batch": MICRO_BATCH,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch": EFFECTIVE_BATCH,
            "venv_sha256": self.venv_sha256,
            "venv_build_sha256": self.venv_build_sha256,
            "base_python": self.base_python,
            "base_payload": self.base_payload,
            "runtime_amendment": self.runtime_amendment,
            "acceptance_uuid": self.acceptance_uuid,
            "source_validation_sha256": self.source_validation_sha256,
            "cutover_ready_sha256": self.cutover_ready_sha256,
            "v100_core_archived_sha256": self.v100_core_archived_sha256,
            "archive_manifest_sha256": self.archive_manifest_sha256,
            "hardware": self.args.hardware,
            "accepted_hardware_class": self.accepted_hardware_class,
            "allocation_hardware_class": self.allocation_hardware_class,
            "strict_fp32": self.strict_fp32,
            "acceptance": {
                "uuid": self.acceptance_uuid,
                "created_utc": self.acceptance.get("created_utc"),
                "gates": self.acceptance.get("gates"),
                "frozen_sha256": self.acceptance.get("source", {}).get(
                    "frozen_sha256"
                ),
                "slurm_smoke": self.acceptance.get("slurm_smoke"),
                "scratch_free_bytes": self.acceptance.get("scratch_free_bytes"),
            },
            "operator_cutover": {
                "cutover_ready_sha256": self.cutover_ready_sha256,
                "v100_core_archived_sha256": self.v100_core_archived_sha256,
                "archive_manifest_sha256": self.archive_manifest_sha256,
                "v100_core_archived_path": str(
                    self.args.v100_core_archived_json.absolute()
                ),
                "archive_manifest_path": str(
                    self.args.v100_archive_manifest_json.absolute()
                ),
                "v100_core_archived": self.v100_core_archived,
                "archive_manifest": self.archive_manifest,
            },
            "source_validation": {
                "path": str(self.args.source_validation_json.absolute()),
                "sha256": self.source_validation_sha256,
                "receipt": self.source_validation,
            },
            "throughput_projection": self.acceptance.get("projection"),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cell_order": [cell.exp_id for cell in self.cells],
            "complete": sorted(self.complete_ids),
            "running": {
                str(gpu): cell.exp_id
                for gpu, (_process, cell, _started) in self.running.items()
            },
            "events": self.events,
            "cell_runtime": self.cell_runtime,
            "fail_stop": {
                "engaged": self.failure_seen,
                "failed": sorted(self.failed_ids),
                "allowed_to_finish": sorted(self.failure_allowed_ids),
            },
            "updated_utc": utc_now(),
        }
        atomic_write_json(self.manifest_path, payload)

    def initialize(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        resumed = self.manifest_path.exists()
        if self.manifest_path.exists():
            prior = json.loads(self.manifest_path.read_text())
            bindings = {
                "campaign_id": self.args.campaign_id,
                "git_sha": self.git_sha,
                "detector_sha256": self.detector_sha256,
                "venv_sha256": self.venv_sha256,
                "venv_build_sha256": self.venv_build_sha256,
                "base_python": self.base_python,
                "base_payload": self.base_payload,
                "runtime_amendment": self.runtime_amendment,
                "acceptance_uuid": self.acceptance_uuid,
                "source_validation_sha256": self.source_validation_sha256,
                "cutover_ready_sha256": self.cutover_ready_sha256,
                "v100_core_archived_sha256": self.v100_core_archived_sha256,
                "archive_manifest_sha256": self.archive_manifest_sha256,
                "strict_fp32": self.strict_fp32,
                "accepted_hardware_class": self.accepted_hardware_class,
            }
            validate_campaign_resume_bindings(prior, bindings)
            self.events = list(prior.get("events", []))
            self.cell_runtime = dict(prior.get("cell_runtime", {}))
            (
                self.failure_seen,
                self.failed_ids,
                self.failure_allowed_ids,
            ) = restore_fail_stop(prior, self.cells)
        for cell in self.cells:
            if cell.exp_id in self.failed_ids:
                validate_failed_namespace(
                    cell,
                    runs_root=self.runs_root,
                    campaign_id=self.args.campaign_id,
                    git_sha=self.git_sha,
                    detector_sha256=self.detector_sha256,
                    venv_sha256=self.venv_sha256,
                    venv_build_sha256=self.venv_build_sha256,
                    base_python=self.base_python,
                    base_payload=self.base_payload,
                    runtime_amendment=self.runtime_amendment,
                    acceptance_uuid=self.acceptance_uuid,
                    source_validation_sha256=self.source_validation_sha256,
                    cutover_ready_sha256=self.cutover_ready_sha256,
                    v100_core_archived_sha256=self.v100_core_archived_sha256,
                    strict_fp32=self.strict_fp32,
                    accepted_hardware_class=self.accepted_hardware_class,
                )
                continue
            state = existing_cell_state(
                cell,
                runs_root=self.runs_root,
                campaign_id=self.args.campaign_id,
                git_sha=self.git_sha,
                detector_sha256=self.detector_sha256,
                venv_sha256=self.venv_sha256,
                venv_build_sha256=self.venv_build_sha256,
                base_python=self.base_python,
                base_payload=self.base_payload,
                runtime_amendment=self.runtime_amendment,
                acceptance_uuid=self.acceptance_uuid,
                source_validation_sha256=self.source_validation_sha256,
                cutover_ready_sha256=self.cutover_ready_sha256,
                v100_core_archived_sha256=self.v100_core_archived_sha256,
                strict_fp32=self.strict_fp32,
                accepted_hardware_class=self.accepted_hardware_class,
            )
            if state == "complete":
                self.complete_ids.add(cell.exp_id)
                marker = validate_scored_completion(
                    cell,
                    runs_root=self.runs_root,
                    git_sha=self.git_sha,
                    detector_sha256=self.detector_sha256,
                )
                provenance = json.loads(
                    (self.runs_root / cell.exp_id / "runtime_provenance.json").read_text()
                )
                self.cell_runtime[cell.exp_id] = finalized_cell_runtime(
                    provenance, marker, cell.exp_id
                )
        self.request_dir.mkdir(parents=True, exist_ok=True)
        for request in self.request_dir.glob("gpu-*.request"):
            request.unlink()
        self.record("controller_started", resumed=resumed)

    def launch(self, gpu: int, cell: Cell) -> None:
        run_dir = self.runs_root / cell.exp_id
        run_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt = run_dir / "checkpoints/last.ckpt"
        hardware = self.args.hardware["devices"][gpu]
        provenance_path = run_dir / "runtime_provenance.json"
        prior = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        attempts = list(prior.get("attempts", []))
        attempts.append(
            {
                "attempt": len(attempts) + 1,
                "started_utc": utc_now(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "gpu_local_index": gpu,
                "gpu_uuid": hardware["uuid"],
                "gpu_name": hardware["name"],
                "gpu_total_memory_bytes": hardware["total_memory_bytes"],
                "compute_capability": hardware["compute_capability"],
                "driver_version": self.allocation_hardware_class["driver_version"],
                "torch": self.allocation_hardware_class["torch"],
                "cuda_build": self.allocation_hardware_class["cuda_build"],
                "resumed_from_last_ckpt": last_ckpt.exists(),
            }
        )
        provenance = {
            **prior,
            "schema": 2,
            "campaign_id": self.args.campaign_id,
            "exp_id": cell.exp_id,
            "git_sha": self.git_sha,
            "detector_sha256": self.detector_sha256,
            "precision": EXPECTED_PRECISION,
            "micro_batch": MICRO_BATCH,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch": EFFECTIVE_BATCH,
            "venv_sha256": self.venv_sha256,
            "venv_build_sha256": self.venv_build_sha256,
            "base_python": self.base_python,
            "base_payload": self.base_payload,
            "runtime_amendment": self.runtime_amendment,
            "acceptance_uuid": self.acceptance_uuid,
            "source_validation_sha256": self.source_validation_sha256,
            "cutover_ready_sha256": self.cutover_ready_sha256,
            "v100_core_archived_sha256": self.v100_core_archived_sha256,
            "strict_fp32": self.strict_fp32,
            "accepted_hardware_class": self.accepted_hardware_class,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
            "gpu_local_index": gpu,
            "gpu_name": hardware["name"],
            "gpu_uuid": hardware["uuid"],
            "gpu_total_memory_bytes": hardware["total_memory_bytes"],
            "compute_capability": hardware["compute_capability"],
            "driver_version": self.allocation_hardware_class["driver_version"],
            "torch": self.allocation_hardware_class["torch"],
            "cuda_build": self.allocation_hardware_class["cuda_build"],
            "started_utc": prior.get("started_utc", attempts[0]["started_utc"]),
            "attempts": attempts,
            "accumulated_active_seconds": float(
                prior.get("accumulated_active_seconds", 0.0)
            ),
        }
        atomic_write_json(provenance_path, provenance)
        log_path = self.runs_root / "logs/h100" / f"{cell.exp_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "scripts.h100.cell",
            "--repo",
            str(self.repo),
            "--runs-root",
            str(self.runs_root),
            "--exp-id",
            cell.exp_id,
            "--init",
            cell.init,
            "--label-frac",
            str(cell.fraction),
            "--git-sha",
            self.git_sha,
            "--workers",
            str(self.args.workers_per_gpu),
        ]
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "H100_REQUEUE_REQUEST_DIR": str(self.request_dir),
            "PATH": f"{self.repo / 'slurm/h100/shims'}:{os.environ['PATH']}",
        }
        output = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=self.repo,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        output.close()
        self.running[gpu] = (process, cell, time.monotonic())
        self.record("cell_launched", gpu=gpu, exp_id=cell.exp_id)

    def poll(self) -> None:
        for gpu in list(self.running):
            process, cell, started = self.running[gpu]
            code = process.poll()
            if code is None:
                continue
            elapsed = time.monotonic() - started
            del self.running[gpu]
            valid = False
            if code == 0:
                try:
                    payload = validate_scored_completion(
                        cell,
                        runs_root=self.runs_root,
                        git_sha=self.git_sha,
                        detector_sha256=self.detector_sha256,
                    )
                    provenance_path = (
                        self.runs_root / cell.exp_id / "runtime_provenance.json"
                    )
                    provenance = json.loads(provenance_path.read_text())
                    validate_runtime_provenance(
                        provenance,
                        cell=cell,
                        campaign_id=self.args.campaign_id,
                        git_sha=self.git_sha,
                        detector_sha256=self.detector_sha256,
                        venv_sha256=self.venv_sha256,
                        venv_build_sha256=self.venv_build_sha256,
                        base_python=self.base_python,
                        base_payload=self.base_payload,
                        runtime_amendment=self.runtime_amendment,
                        acceptance_uuid=self.acceptance_uuid,
                        source_validation_sha256=self.source_validation_sha256,
                        cutover_ready_sha256=self.cutover_ready_sha256,
                        v100_core_archived_sha256=self.v100_core_archived_sha256,
                        strict_fp32=self.strict_fp32,
                        accepted_hardware_class=self.accepted_hardware_class,
                    )
                    valid = True
                except RuntimeError as exc:
                    self.record("invalid_completion_marker", exp_id=cell.exp_id, error=str(exc))
            if valid:
                self.complete_ids.add(cell.exp_id)
                provenance_path = self.runs_root / cell.exp_id / "runtime_provenance.json"
                provenance = json.loads(provenance_path.read_text())
                total_seconds = float(provenance.get("accumulated_active_seconds", 0.0)) + elapsed
                provenance["attempts"][-1].update(
                    {"finished_utc": utc_now(), "exit_code": code, "active_seconds": elapsed}
                )
                provenance.update(
                    {
                        "completed_utc": utc_now(),
                        "accumulated_active_seconds": total_seconds,
                        "elapsed_hours": total_seconds / 3600.0,
                        "epochs_run": payload.get("epochs_run"),
                        "best_dev_f1": payload.get("best_dev_f1"),
                        "test_f1": payload.get("test_f1"),
                        "test_scored_at": payload.get("test_scored_at"),
                    }
                )
                atomic_write_json(provenance_path, provenance)
                self.cell_runtime[cell.exp_id] = {
                    "elapsed_hours": total_seconds / 3600.0,
                    "attempts": len(provenance["attempts"]),
                    "epochs_run": payload.get("epochs_run"),
                    "test_f1": payload.get("test_f1"),
                }
                self.record(
                    "cell_complete",
                    gpu=gpu,
                    exp_id=cell.exp_id,
                    elapsed_hours=total_seconds / 3600.0,
                )
            else:
                first_failure = not self.failure_seen
                self.failure_seen = True
                self.failed_ids.add(cell.exp_id)
                self.failure_allowed_ids.discard(cell.exp_id)
                if first_failure:
                    self.failure_allowed_ids.update(
                        running_cell.exp_id
                        for _process, running_cell, _started in self.running.values()
                    )
                    self.record(
                        "fail_stop_engaged",
                        failed_exp_id=cell.exp_id,
                        allowed_to_finish=sorted(self.failure_allowed_ids),
                    )
                self.record("cell_failed", gpu=gpu, exp_id=cell.exp_id, exit_code=code)

    def _stop_children(self) -> None:
        for process, _cell, _started in self.running.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and any(
            process.poll() is None for process, _cell, _started in self.running.values()
        ):
            time.sleep(0.5)
        for process, _cell, _started in self.running.values():
            if process.poll() is None:
                process.terminate()
        for process, _cell, _started in self.running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def preempt_and_requeue(self) -> int:
        if not self.running:
            if self.failure_seen:
                self.write_manifest(status="failed")
                return 1
            self.write_manifest(status="host-requeue-required")
            if len(self.complete_ids) != len(self.cells):
                return HOST_REQUEUE_EXIT_CODE
            return 0
        if not self.preemption_forwarded:
            for process, _cell, _started in self.running.values():
                os.kill(process.pid, signal.SIGUSR1)
            self.preemption_forwarded = True
            self.record("usr1_forwarded", count=len(self.running))

        deadline = time.monotonic() + self.args.checkpoint_timeout
        expected = {
            gpu: self.request_dir / f"gpu-{gpu}.request" for gpu in self.running
        }
        while time.monotonic() < deadline and not all(path.exists() for path in expected.values()):
            time.sleep(1)
        missing = [gpu for gpu, path in expected.items() if not path.exists()]
        if missing:
            self.failure_seen = True
            self._stop_children()
            self.running.clear()
            self.record("preemption_checkpoint_timeout", missing_gpus=missing)
            return 1

        for _gpu, (_process, cell, started) in self.running.items():
            checkpoint = checkpoint_for_preemption(self.runs_root / cell.exp_id)
            provenance_path = self.runs_root / cell.exp_id / "runtime_provenance.json"
            provenance = json.loads(provenance_path.read_text())
            elapsed = time.monotonic() - started
            provenance["accumulated_active_seconds"] = float(
                provenance.get("accumulated_active_seconds", 0.0)
            ) + elapsed
            provenance["attempts"][-1].update(
                {"finished_utc": utc_now(), "exit": "preempted", "active_seconds": elapsed}
            )
            atomic_write_json(provenance_path, provenance)
            self.record(
                "checkpoint_promoted",
                exp_id=cell.exp_id,
                last_ckpt=str(checkpoint),
                sha256=sha256_file(checkpoint),
            )
        self._stop_children()
        self.running.clear()
        self.write_manifest(status="host-requeue-required")
        return HOST_REQUEUE_EXIT_CODE

    def run(self) -> int:
        self.initialize()
        while True:
            self.poll()
            if self.preemption_seen:
                return self.preempt_and_requeue()
            running_ids = {cell.exp_id for _process, cell, _started in self.running.values()}
            free = [gpu for gpu in range(EXPECTED_GPU_COUNT) if gpu not in self.running]
            if self.failure_seen:
                launches = fail_stop_launches(
                    self.cells,
                    allowed_ids=self.failure_allowed_ids,
                    failed_ids=self.failed_ids,
                    running_ids=running_ids,
                    complete_ids=self.complete_ids,
                    free_gpus=free,
                    preemption_seen=self.preemption_seen,
                )
            else:
                launches = next_launches(
                    self.cells,
                    running_ids=running_ids,
                    complete_ids=self.complete_ids,
                    free_gpus=free,
                    failure_seen=False,
                    preemption_seen=self.preemption_seen,
                )
            for gpu, cell in launches:
                self.launch(gpu, cell)
            if self.failure_seen and not self.running:
                self.write_manifest(status="failed")
                return 1
            if len(self.complete_ids) == len(self.cells) and not self.running:
                try:
                    grid = collect_and_validate_grid(
                        repo=self.repo,
                        runs_root=self.runs_root,
                        cells=self.cells,
                        git_sha=self.git_sha,
                        detector_sha256=self.detector_sha256,
                    )
                except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                    self.failure_seen = True
                    self.record("grid_validation_failed", error=str(exc))
                    self.write_manifest(status="failed")
                    return 1
                self.record(
                    "grid_validated",
                    path=str(grid),
                    sha256=sha256_file(grid),
                    rows=32,
                )
                self.write_manifest(status="complete")
                return 0
            time.sleep(self.args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--venv-sha256", required=True)
    parser.add_argument("--venv-build-sha256", required=True)
    parser.add_argument("--hardware-json", type=Path, required=True)
    parser.add_argument("--acceptance-json", type=Path, required=True)
    parser.add_argument("--source-validation-json", type=Path, required=True)
    parser.add_argument("--source-validation-sha256", required=True)
    parser.add_argument("--cutover-ready-json", type=Path, required=True)
    parser.add_argument("--cutover-ready-sha256", required=True)
    parser.add_argument("--v100-core-archived-json", type=Path, required=True)
    parser.add_argument("--v100-core-archived-sha256", required=True)
    parser.add_argument("--v100-archive-manifest-json", type=Path, required=True)
    parser.add_argument("--v100-archive-manifest-sha256", required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--checkpoint-timeout", type=float, default=240.0)
    args = parser.parse_args()
    if args.workers_per_gpu * EXPECTED_GPU_COUNT > 48:
        parser.error("workers-per-gpu exceeds the 48-CPU allocation")
    args.hardware = json.loads(args.hardware_json.read_text())
    return Controller(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
