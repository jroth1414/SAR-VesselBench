"""Deterministic reverse Box bundle for the completed 32-cell H100 grid."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import yaml

from scripts.h100.build_venv import verify as verify_native_venv
from scripts.h100.campaign import (
    hardware_class,
    validate_runtime_provenance,
    validate_scored_completion,
)
from scripts.h100.contracts import (
    EFFECTIVE_BATCH,
    EXPECTED_PRECISION,
    FROZEN_PATHS,
    GRADIENT_ACCUMULATION,
    MICRO_BATCH,
    Cell,
    cutover_acceptance_bindings,
    frozen_hashes,
    load_cells,
    sha256_file,
    validate_bound_cutover_forecast,
    validate_gpu_inventory,
)
from scripts.h100.cutover import validate_h100_ready
from scripts.h100.host_test_gate import validate_host_gate
from scripts.h100.slurm_smoke import (
    READY_NAME as SMOKE_READY_NAME,
    STATE_NAME as SMOKE_STATE_NAME,
    make_bindings as make_smoke_bindings,
    validate_smoke_receipt,
)
from scripts.h100.source_validation import (
    BASE_PAYLOAD_KEYS,
    RUNTIME_AMENDMENT_KEYS,
    validate_source_receipt,
)

from .package import (
    EXPECTED_TORCH,
    RUNTIME_BRANCH,
    FORMAT_VERSION,
    PackageError,
    _archive_artifact,
    _canonical_json,
    _git_value,
    _hash_file,
    _inside,
    _inside_repository_worktrees,
    _physical_paths,
    _physical_record,
    _require_no_symlink_components,
    _source_entries,
    _write_bytes,
    verify_package,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ALLOCATION_NAME = re.compile(
    r"^h100_runtime-(?P<job_id>[^/]+)-r(?P<restart>[0-9]+)\.json$"
)
_STRICT_FP32_KEYS = frozenset(
    {
        "cuda_matmul_fp32_precision",
        "cudnn_conv_fp32_precision",
        "cudnn_rnn_fp32_precision",
    }
)
_GRID_FIELDS = (
    "exp_id",
    "init",
    "track",
    "role",
    "label_frac",
    "seed",
    "precision",
    "detector_sha256",
    "git_sha",
    "dev_f1",
    "dev_threshold",
    "test_f1",
    "epochs_run",
    "monotonicity_ok",
)
_ACCEPTANCE_LOGS = (
    "pytest-venv-remaining.log",
    "vit-fp32.log",
    "cnn-200step-fp32.log",
)
_TEST_METRICS = (
    "test_f1",
    "test_precision",
    "test_recall",
    "test_near_shore_f1",
)


def _mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise PackageError(f"{description} is absent or not an object")
    return dict(value)


def _result_identity(campaign: Mapping[str, object]) -> tuple[dict[str, object], str]:
    """Bind the scientific recipe and accepted/live H100 hardware classes."""

    acceptance = _mapping(campaign.get("acceptance"), "campaign acceptance")
    hardware = _mapping(campaign.get("hardware"), "campaign allocation hardware")
    devices = hardware.get("devices")
    allocation_gpu_uuids = (
        [str(item.get("uuid", "")) for item in devices if isinstance(item, Mapping)]
        if isinstance(devices, list)
        else []
    )
    identity = {
        "campaign_id": campaign.get("campaign_id"),
        "git_commit": campaign.get("git_sha"),
        "detector_sha256": campaign.get("detector_sha256"),
        "venv_sha256": campaign.get("venv_sha256"),
        "venv_build_sha256": campaign.get("venv_build_sha256"),
        "base_python": campaign.get("base_python"),
        "base_payload": campaign.get("base_payload"),
        "runtime_amendment": campaign.get("runtime_amendment"),
        "acceptance_uuid": acceptance.get("uuid"),
        "source_validation_sha256": campaign.get(
            "source_validation_sha256"
        ),
        "strict_fp32": campaign.get("strict_fp32"),
        "accepted_hardware_class": campaign.get("accepted_hardware_class"),
        "allocation_hardware_class": campaign.get("allocation_hardware_class"),
        "allocation_gpu_uuids": allocation_gpu_uuids,
        "operator_cutover": campaign.get("operator_cutover"),
        "cell_runtime": campaign.get("cell_runtime"),
        "precision": campaign.get("precision"),
        "micro_batch": campaign.get("micro_batch"),
        "gradient_accumulation": campaign.get("gradient_accumulation"),
        "effective_batch": campaign.get("effective_batch"),
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return identity, digest


def _result_package_id(campaign: Mapping[str, object]) -> str:
    _identity, digest = _result_identity(campaign)
    return f"xview3-h100-results-{campaign['git_sha']}-{digest}"


def _json(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise PackageError(f"symlinked result evidence is forbidden: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"invalid required result JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PackageError(f"result JSON is not an object: {path}")
    return value


def _exact_bindings(
    payload: Mapping[str, object],
    expected: Mapping[str, object],
    description: str,
) -> None:
    mismatches = {
        key: (value, payload.get(key))
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise PackageError(f"{description} binding mismatch: {mismatches}")


def _strict_backend(value: object, description: str) -> dict[str, object]:
    backend = _mapping(value, description)
    if set(backend) != _STRICT_FP32_KEYS or set(backend.values()) != {"ieee"}:
        raise PackageError(f"{description} is not the exact strict IEEE FP32 backend")
    return backend


def _transfer_identity(
    value: object,
    *,
    keys: set[str],
    description: str,
) -> dict[str, str]:
    identity = _mapping(value, description)
    if set(identity) != keys:
        raise PackageError(f"{description} keys do not match the closed contract")
    normalized = {key: str(identity[key]) for key in keys}
    if not re.fullmatch(r"[0-9a-f]{40}", normalized["git_sha"]):
        raise PackageError(f"{description} git SHA is invalid")
    if any(
        not _HEX64.fullmatch(value)
        for key, value in normalized.items()
        if key not in {"package_id", "git_sha"}
    ):
        raise PackageError(f"{description} contains an invalid SHA-256")
    return normalized


def _finite(value: object, description: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PackageError(f"{description} is not finite") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise PackageError(f"{description} is not finite and positive")
    return result


def _allocation_files(
    meta_root: Path,
    *,
    strict_fp32: Mapping[str, object],
    accepted_hardware_class: Mapping[str, object],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for path in sorted(meta_root.glob("h100_runtime-*.json"), key=lambda item: item.name):
        match = _ALLOCATION_NAME.fullmatch(path.name)
        if match is None:
            raise PackageError(f"invalid allocation-inventory filename: {path.name}")
        identity = (match.group("job_id"), int(match.group("restart")))
        if identity in seen:
            raise PackageError(f"duplicate allocation-inventory identity: {identity}")
        seen.add(identity)
        payload = _json(path)
        validate_gpu_inventory(payload.get("devices", []))
        if _strict_backend(
            payload.get("backend"), f"{path.name} strict backend"
        ) != dict(strict_fp32):
            raise PackageError(f"{path.name} strict backend differs from acceptance")
        observed_class = hardware_class(payload)
        if observed_class != dict(accepted_hardware_class):
            raise PackageError(
                f"{path.name} hardware class differs from H100 acceptance"
            )
        records.append(
            {
                "path": path,
                "payload": payload,
                "job_id": identity[0],
                "restart": identity[1],
            }
        )
    if not records:
        raise PackageError("no h100_runtime-*.json allocation inventories are present")
    return records


def _smoke_state_matches(
    state: Mapping[str, object],
    receipt: Mapping[str, object],
    ready_sha256: str,
) -> None:
    if state.get("status") != "complete" or state.get("ready_sha256") != ready_sha256:
        raise PackageError("Slurm smoke state is not complete or READY-bound")
    for key in (
        "bindings",
        "allocations",
        "signal",
        "checkpoint",
        "resume",
        "synthetic",
    ):
        if state.get(key) != receipt.get(key):
            raise PackageError(f"Slurm smoke STATE/READY mismatch for {key}")


def _validate_campaign(
    repo: Path,
    runs_root: Path,
    campaign_manifest: Path,
) -> tuple[dict[str, object], list[Cell], dict[str, object]]:
    campaign = _json(campaign_manifest)
    cells = load_cells(repo)
    if len(cells) != 32 or len({cell.exp_id for cell in cells}) != 32:
        raise PackageError("active repository does not resolve to 32 unique core cells")
    expected_order = [cell.exp_id for cell in cells]
    expected_ids = set(expected_order)
    git_sha = _git_value(repo, "rev-parse", "HEAD")
    detector_sha256 = sha256_file(repo / "configs/detector.yaml")

    _exact_bindings(
        campaign,
        {
            "schema": 2,
            "status": "complete",
            "git_sha": git_sha,
            "detector_sha256": detector_sha256,
            "precision": EXPECTED_PRECISION,
            "micro_batch": MICRO_BATCH,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch": EFFECTIVE_BATCH,
        },
        "campaign",
    )
    complete = campaign.get("complete")
    if (
        not isinstance(complete, list)
        or len(complete) != 32
        or len(set(complete)) != 32
        or set(complete) != expected_ids
    ):
        raise PackageError("campaign manifest does not complete the exact unique grid")
    if campaign.get("running") != {}:
        raise PackageError("campaign manifest still has running cells")
    if campaign.get("cell_order") != expected_order:
        raise PackageError("campaign cell_order is not exact expensive-first order")

    meta_root = runs_root / ".h100"
    ready_path = meta_root / "H100_READY.json"
    runtime_path = meta_root / "h100_runtime.json"
    projection_path = meta_root / "throughput_projection.json"
    build_path = meta_root / "venv_build.json"
    cutover_path = meta_root / "CUTOVER_READY.json"
    source_validation_path = meta_root / "SOURCE_VALIDATED.json"
    host_test_receipt_path = meta_root / "HOST_HANDOFF_TESTS.json"
    host_test_log_path = (
        meta_root / "acceptance-logs" / "pytest-handoff-host.log"
    )
    test_suite_path = meta_root / "PYTEST_ACCEPTANCE.json"
    smoke_root = meta_root / "slurm-smoke"
    smoke_ready_path = smoke_root / SMOKE_READY_NAME
    smoke_state_path = smoke_root / SMOKE_STATE_NAME

    ready = _json(ready_path)
    runtime = _json(runtime_path)
    projection = _json(projection_path)
    venv_build = _json(build_path)
    cutover = _json(cutover_path)
    source_validation = _json(source_validation_path)
    host_test_receipt = _json(host_test_receipt_path)
    test_suite = _json(test_suite_path)
    smoke_receipt_raw = _json(smoke_ready_path)
    smoke_state = _json(smoke_state_path)

    base_payload = _transfer_identity(
        ready.get("base_payload"),
        keys=BASE_PAYLOAD_KEYS,
        description="base-payload identity",
    )
    runtime_amendment = _transfer_identity(
        ready.get("runtime_amendment"),
        keys=RUNTIME_AMENDMENT_KEYS,
        description="runtime-amendment identity",
    )
    strict_fp32 = _strict_backend(ready.get("strict_fp32"), "H100 acceptance backend")
    accepted_venv = _mapping(ready.get("venv"), "H100 acceptance native venv")
    venv_sha256 = str(accepted_venv.get("sha256", ""))
    venv_build_sha256 = sha256_file(build_path)
    base_python = _mapping(accepted_venv.get("base_python"), "base-Python identity")
    base_python_sha256 = str(base_python.get("executable_sha256", ""))
    if not _HEX64.fullmatch(venv_sha256) or not _HEX64.fullmatch(base_python_sha256):
        raise PackageError("H100 acceptance native-venv identity is invalid")
    if accepted_venv.get("venv_build_sha256") != venv_build_sha256:
        raise PackageError("H100 acceptance native-venv receipt hash mismatch")
    try:
        verified_venv = verify_native_venv(
            repo=repo,
            venv_root=Path(str(accepted_venv.get("path", ""))),
            base_python=Path(str(base_python.get("requested_path", ""))),
            expected_venv_sha256=venv_sha256,
            expected_receipt_sha256=venv_build_sha256,
            expected_base_python_sha256=base_python_sha256,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise PackageError(f"native venv failed reverse-package verification: {exc}") from exc
    if verified_venv != venv_build:
        raise PackageError("persisted native-venv receipt differs from verified sidecar")

    expected_frozen = frozen_hashes(repo)
    smoke_bindings = make_smoke_bindings(
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        venv_sha256=venv_sha256,
        venv_build_sha256=venv_build_sha256,
        base_python_sha256=base_python_sha256,
        base_payload=base_payload,
        runtime_amendment=runtime_amendment,
    )
    smoke_receipt = validate_smoke_receipt(
        smoke_ready_path, expected_bindings=smoke_bindings
    )
    if smoke_receipt != smoke_receipt_raw:
        raise PackageError("Slurm smoke receipt changed while validating")
    smoke_sha256 = sha256_file(smoke_ready_path)

    ready = validate_h100_ready(
        ready_path,
        expected_git_sha=git_sha,
        expected_venv_sha256=venv_sha256,
        expected_venv_build_sha256=venv_build_sha256,
        expected_base_python_sha256=base_python_sha256,
        expected_base_payload=base_payload,
        expected_runtime_amendment=runtime_amendment,
        expected_frozen_sha256=expected_frozen,
        expected_smoke_receipt=smoke_receipt,
        expected_smoke_sha256=smoke_sha256,
    )
    source_validation_sha256 = sha256_file(source_validation_path)
    source_validation = validate_source_receipt(
        source_validation_path,
        expected_sha256=source_validation_sha256,
        expected_git_sha=git_sha,
        expected_hashes=expected_frozen,
        expected_base_payload=base_payload,
        expected_runtime_amendment=runtime_amendment,
    )
    expected_source_validation = {
        "path": str(source_validation_path),
        "sha256": source_validation_sha256,
        "receipt": source_validation,
    }
    if (
        ready.get("source_validation") != expected_source_validation
        or campaign.get("source_validation_sha256")
        != source_validation_sha256
        or campaign.get("source_validation") != expected_source_validation
    ):
        raise PackageError("source-validation receipt bindings mismatch")

    host_test_receipt_sha256 = sha256_file(host_test_receipt_path)
    host_test_receipt = validate_host_gate(
        host_test_receipt_path,
        expected_sha256=host_test_receipt_sha256,
        expected_source_validation_sha256=source_validation_sha256,
    )
    if host_test_receipt.get("log") != {
        "path": str(host_test_log_path),
        "sha256": sha256_file(host_test_log_path),
    }:
        raise PackageError("host handoff-test log binding mismatch")
    test_suite_sha256 = sha256_file(test_suite_path)
    expected_test_suite = {
        "path": str(test_suite_path),
        "sha256": test_suite_sha256,
        "receipt": test_suite,
    }
    if ready.get("test_suite") != expected_test_suite:
        raise PackageError("H100_READY aggregate pytest receipt binding mismatch")
    if (
        test_suite.get("schema") != 2
        or test_suite.get("status") != "passed"
        or test_suite.get("source_validation_sha256")
        != source_validation_sha256
    ):
        raise PackageError("aggregate pytest receipt is invalid")
    host_handoff = _mapping(
        test_suite.get("host_handoff"), "aggregate host-handoff evidence"
    )
    if host_handoff != {
        "receipt_path": str(host_test_receipt_path),
        "receipt_sha256": host_test_receipt_sha256,
        "receipt": host_test_receipt,
    }:
        raise PackageError("aggregate host-handoff receipt binding mismatch")
    venv_remaining = _mapping(
        test_suite.get("venv_remaining"), "aggregate venv pytest evidence"
    )
    venv_test_log_path = (
        meta_root / "acceptance-logs" / "pytest-venv-remaining.log"
    )
    if venv_remaining.get("log") != {
        "path": str(venv_test_log_path),
        "sha256": sha256_file(venv_test_log_path),
    }:
        raise PackageError("aggregate venv pytest log binding mismatch")
    _finite(
        test_suite.get("aggregate_duration_seconds"),
        "aggregate pytest duration",
        positive=True,
    )
    if runtime != ready.get("hardware"):
        raise PackageError("h100_runtime.json differs from accepted hardware inventory")
    validate_gpu_inventory(runtime.get("devices", []))
    if _strict_backend(runtime.get("backend"), "accepted hardware backend") != strict_fp32:
        raise PackageError("accepted hardware backend differs from H100_READY")
    accepted_hardware_class = hardware_class(runtime)
    if accepted_hardware_class.get("torch") != EXPECTED_TORCH:
        raise PackageError(
            f"accepted H100 torch is not the locked {EXPECTED_TORCH}"
        )

    campaign_hardware = _mapping(
        campaign.get("hardware"), "campaign allocation hardware"
    )
    validate_gpu_inventory(campaign_hardware.get("devices", []))
    if _strict_backend(
        campaign_hardware.get("backend"), "campaign allocation backend"
    ) != strict_fp32:
        raise PackageError("campaign allocation backend differs from acceptance")
    allocation_hardware_class = hardware_class(campaign_hardware)
    if allocation_hardware_class != accepted_hardware_class:
        raise PackageError("campaign allocation hardware class differs from acceptance")

    _exact_bindings(
        campaign,
        {
            "venv_sha256": venv_sha256,
            "venv_build_sha256": venv_build_sha256,
            "base_python": base_python,
            "base_payload": base_payload,
            "runtime_amendment": runtime_amendment,
            "strict_fp32": strict_fp32,
            "accepted_hardware_class": accepted_hardware_class,
            "allocation_hardware_class": allocation_hardware_class,
        },
        "campaign hardware/package",
    )
    acceptance = _mapping(campaign.get("acceptance"), "campaign acceptance")
    _exact_bindings(
        acceptance,
        {
            "uuid": ready.get("acceptance_uuid"),
            "created_utc": ready.get("created_utc"),
            "gates": ready.get("gates"),
            "frozen_sha256": expected_frozen,
            "slurm_smoke": ready.get("slurm_smoke"),
            "scratch_free_bytes": ready.get("scratch_free_bytes"),
        },
        "campaign acceptance",
    )
    if campaign.get("throughput_projection") != projection or projection != ready.get(
        "projection"
    ):
        raise PackageError("campaign/acceptance throughput projection mismatch")

    if cutover.get("status") != "cutover-ready" or cutover.get(
        "h100_ready"
    ) != ready:
        raise PackageError("CUTOVER_READY does not embed the accepted H100 receipt")
    cutover_sha256 = sha256_file(cutover_path)
    operator_cutover = _mapping(
        campaign.get("operator_cutover"), "campaign operator cutover"
    )
    archived_receipt_path = meta_root / "V100_CORE_ARCHIVED.json"
    archive_manifest_path = meta_root / "V100_CORE_ARCHIVE_MANIFEST.json"
    archived_receipt_canonical = _json(archived_receipt_path)
    archive_manifest_canonical = _json(archive_manifest_path)
    v100_core_archived_sha256 = str(
        operator_cutover.get("v100_core_archived_sha256", "")
    )
    archive_manifest_sha256 = str(
        operator_cutover.get("archive_manifest_sha256", "")
    )
    if (
        operator_cutover.get("cutover_ready_sha256") != cutover_sha256
        or not _HEX64.fullmatch(v100_core_archived_sha256)
        or not _HEX64.fullmatch(archive_manifest_sha256)
    ):
        raise PackageError("campaign operator-cutover hashes are invalid")
    _exact_bindings(
        campaign,
        {
            "acceptance_uuid": ready.get("acceptance_uuid"),
            "source_validation_sha256": source_validation_sha256,
            "cutover_ready_sha256": cutover_sha256,
            "v100_core_archived_sha256": v100_core_archived_sha256,
            "archive_manifest_sha256": archive_manifest_sha256,
        },
        "campaign acceptance/cutover receipts",
    )
    if (
        sha256_file(archived_receipt_path) != v100_core_archived_sha256
        or sha256_file(archive_manifest_path) != archive_manifest_sha256
        or operator_cutover.get("v100_core_archived_path")
        != str(archived_receipt_path)
        or operator_cutover.get("archive_manifest_path")
        != str(archive_manifest_path)
        or operator_cutover.get("v100_core_archived")
        != archived_receipt_canonical
        or operator_cutover.get("archive_manifest")
        != archive_manifest_canonical
    ):
        raise PackageError("canonical operator evidence/campaign binding mismatch")
    archived_receipt = _mapping(
        operator_cutover.get("v100_core_archived"),
        "campaign V100 archive attestation",
    )
    if (
        archived_receipt.get("status") != "v100-core-archived"
        or archived_receipt.get("attestation") != "external-human-operator"
        or archived_receipt.get("cutover_ready_sha256") != cutover_sha256
    ):
        raise PackageError("campaign V100 archive attestation is invalid")
    if archived_receipt != archived_receipt_canonical:
        raise PackageError("campaign/operator receipt payload mismatch")
    if archived_receipt.get("archive") != {
        "manifest_path": str(archive_manifest_path),
        "manifest_sha256": archive_manifest_sha256,
    }:
        raise PackageError("operator receipt archive-manifest binding mismatch")
    if archive_manifest_canonical.get("status") != (
        "v100-core-diagnostics-archived"
    ):
        raise PackageError("canonical V100 archive manifest is not complete")
    try:
        expected_cutover_acceptance = cutover_acceptance_bindings(ready)
    except RuntimeError as exc:
        raise PackageError(str(exc)) from exc
    if cutover.get("acceptance") != expected_cutover_acceptance:
        raise PackageError("CUTOVER_READY acceptance bindings are incomplete")
    try:
        validate_bound_cutover_forecast(cutover)
    except RuntimeError as exc:
        raise PackageError(str(exc)) from exc

    _smoke_state_matches(smoke_state, smoke_receipt, smoke_sha256)
    allocation_records = _allocation_files(
        meta_root,
        strict_fp32=strict_fp32,
        accepted_hardware_class=accepted_hardware_class,
    )
    campaign_job = str(campaign.get("slurm_job_id", "")).strip()
    if not campaign_job or not any(
        record["job_id"] == campaign_job
        and record["payload"] == campaign_hardware
        for record in allocation_records
    ):
        raise PackageError(
            "campaign Slurm job/hardware pair is absent from allocation inventories"
        )

    acceptance_logs = [
        meta_root / "acceptance-logs" / name for name in _ACCEPTANCE_LOGS
    ]
    for path in acceptance_logs:
        if not path.is_file() or path.is_symlink():
            raise PackageError(f"required acceptance log is absent or symlinked: {path}")
    slurm_root = meta_root / "slurm"
    if slurm_root.is_symlink():
        raise PackageError(f"symlinked Slurm log root is forbidden: {slurm_root}")
    slurm_logs: list[Path] = []
    if slurm_root.exists():
        if not slurm_root.is_dir():
            raise PackageError(f"Slurm log root is not a directory: {slurm_root}")
        slurm_logs = sorted(slurm_root.glob("*.out"), key=lambda path: path.name)
        if any(not path.is_file() or path.is_symlink() for path in slurm_logs):
            raise PackageError("Slurm *.out evidence contains a non-file or symlink")

    context: dict[str, object] = {
        "meta_root": meta_root,
        "ready_path": ready_path,
        "runtime_path": runtime_path,
        "projection_path": projection_path,
        "build_path": build_path,
        "cutover_path": cutover_path,
        "source_validation_path": source_validation_path,
        "source_validation_sha256": source_validation_sha256,
        "host_test_receipt_path": host_test_receipt_path,
        "host_test_log_path": host_test_log_path,
        "test_suite_path": test_suite_path,
        "test_suite_sha256": test_suite_sha256,
        "archived_receipt_path": archived_receipt_path,
        "archive_manifest_path": archive_manifest_path,
        "smoke_ready_path": smoke_ready_path,
        "smoke_state_path": smoke_state_path,
        "venv_sha256": venv_sha256,
        "venv_build_sha256": venv_build_sha256,
        "base_python": base_python,
        "base_payload": base_payload,
        "runtime_amendment": runtime_amendment,
        "strict_fp32": strict_fp32,
        "accepted_hardware_class": accepted_hardware_class,
        "allocation_hardware_class": allocation_hardware_class,
        "operator_cutover": operator_cutover,
        "cutover_ready_sha256": cutover_sha256,
        "v100_core_archived_sha256": v100_core_archived_sha256,
        "allocation_records": allocation_records,
        "acceptance_logs": acceptance_logs,
        "slurm_logs": slurm_logs,
    }
    return campaign, cells, context


def _attempt_inventory(
    allocation_records: Sequence[Mapping[str, object]],
) -> dict[str, list[tuple[dict[str, object], dict[str, object]]]]:
    by_job: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = {}
    for record in allocation_records:
        payload = _mapping(record.get("payload"), "allocation payload")
        devices = validate_gpu_inventory(payload.get("devices", []))
        for device in devices:
            by_job.setdefault(str(record["job_id"]), []).append((device, payload))
    return by_job


def _validate_attempts(
    provenance: Mapping[str, object],
    *,
    exp_id: str,
    allocation_records: Sequence[Mapping[str, object]],
    accepted_hardware_class: Mapping[str, object],
) -> list[str]:
    attempts = provenance.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise PackageError(f"{exp_id} has no runtime attempts")
    inventory = _attempt_inventory(allocation_records)
    observed_uuids: list[str] = []
    for index, raw in enumerate(attempts, start=1):
        if not isinstance(raw, Mapping) or raw.get("attempt") != index:
            raise PackageError(f"{exp_id} runtime attempts are not sequential")
        attempt = dict(raw)
        job_id = str(attempt.get("slurm_job_id", "")).strip()
        gpu_uuid = str(attempt.get("gpu_uuid", "")).strip()
        candidates = [
            (device, hardware)
            for device, hardware in inventory.get(job_id, [])
            if str(device.get("uuid", "")) == gpu_uuid
        ]
        if not candidates:
            raise PackageError(
                f"{exp_id} attempt {index} GPU UUID is absent from its allocation"
            )
        try:
            local_index = int(attempt.get("gpu_local_index", -1))
        except (TypeError, ValueError) as exc:
            raise PackageError(f"{exp_id} attempt {index} local GPU is invalid") from exc
        matching_candidates = []
        for device, hardware in candidates:
            devices = hardware.get("devices")
            if (
                isinstance(devices, list)
                and 0 <= local_index < len(devices)
                and devices[local_index].get("uuid") == gpu_uuid
            ):
                matching_candidates.append((device, hardware))
        if not matching_candidates:
            raise PackageError(
                f"{exp_id} attempt {index} local GPU index/UUID mismatch"
            )
        device, _hardware = matching_candidates[0]
        expected_device = {
            "gpu_uuid": device["uuid"],
            "gpu_name": device["name"],
            "gpu_total_memory_bytes": device["total_memory_bytes"],
            "compute_capability": device["compute_capability"],
            "driver_version": accepted_hardware_class["driver_version"],
            "torch": accepted_hardware_class["torch"],
            "cuda_build": accepted_hardware_class["cuda_build"],
        }
        _exact_bindings(
            attempt, expected_device, f"{exp_id} attempt {index} hardware"
        )
        if not str(attempt.get("started_utc", "")).strip() or not str(
            attempt.get("finished_utc", "")
        ).strip():
            raise PackageError(f"{exp_id} attempt {index} timestamps are incomplete")
        _finite(
            attempt.get("active_seconds"),
            f"{exp_id} attempt {index} active_seconds",
            positive=True,
        )
        observed_uuids.append(gpu_uuid)
    if attempts[-1].get("exit_code") != 0:
        raise PackageError(f"{exp_id} final attempt did not exit successfully")
    if provenance.get("gpu_uuid") != observed_uuids[-1]:
        raise PackageError(f"{exp_id} final GPU UUID differs from the final attempt")
    return observed_uuids


def _run_entries(
    runs_root: Path,
    cell: Cell,
    campaign: Mapping[str, object],
    context: Mapping[str, object],
) -> tuple[list[tuple[Path, PurePosixPath]], dict[str, object], dict[str, object]]:
    exp_id = cell.exp_id
    run_dir = runs_root / exp_id
    marker = validate_scored_completion(
        cell,
        runs_root=runs_root,
        git_sha=str(campaign["git_sha"]),
        detector_sha256=str(campaign["detector_sha256"]),
    )
    provenance_path = run_dir / "runtime_provenance.json"
    provenance = _json(provenance_path)
    validate_runtime_provenance(
        provenance,
        cell=cell,
        campaign_id=str(campaign["campaign_id"]),
        git_sha=str(campaign["git_sha"]),
        detector_sha256=str(campaign["detector_sha256"]),
        venv_sha256=str(campaign["venv_sha256"]),
        venv_build_sha256=str(campaign["venv_build_sha256"]),
        base_python=context["base_python"],  # type: ignore[arg-type]
        base_payload=context["base_payload"],  # type: ignore[arg-type]
        runtime_amendment=context["runtime_amendment"],  # type: ignore[arg-type]
        acceptance_uuid=str(
            _mapping(campaign["acceptance"], "campaign acceptance")["uuid"]
        ),
        source_validation_sha256=str(
            context["source_validation_sha256"]
        ),
        cutover_ready_sha256=str(context["cutover_ready_sha256"]),
        v100_core_archived_sha256=str(
            context["v100_core_archived_sha256"]
        ),
        strict_fp32=context["strict_fp32"],  # type: ignore[arg-type]
        accepted_hardware_class=context["accepted_hardware_class"],  # type: ignore[arg-type]
    )
    attempt_uuids = _validate_attempts(
        provenance,
        exp_id=exp_id,
        allocation_records=context["allocation_records"],  # type: ignore[arg-type]
        accepted_hardware_class=context["accepted_hardware_class"],  # type: ignore[arg-type]
    )
    for key in _TEST_METRICS:
        _finite(marker.get(key), f"{exp_id} {key}")
    elapsed = _finite(
        provenance.get("elapsed_hours"), f"{exp_id} elapsed_hours", positive=True
    )
    accumulated = _finite(
        provenance.get("accumulated_active_seconds"),
        f"{exp_id} accumulated_active_seconds",
        positive=True,
    )
    if not math.isclose(elapsed, accumulated / 3600.0, rel_tol=1e-12, abs_tol=1e-12):
        raise PackageError(f"{exp_id} elapsed-hours provenance is inconsistent")
    _exact_bindings(
        provenance,
        {
            "completed_utc": provenance.get("completed_utc"),
            "epochs_run": marker.get("epochs_run"),
            "best_dev_f1": marker.get("best_dev_f1"),
            "test_f1": marker.get("test_f1"),
        },
        f"{exp_id} completion provenance",
    )
    if not str(provenance.get("completed_utc", "")).strip():
        raise PackageError(f"{exp_id} completion timestamp is absent")
    cell_runtime = _mapping(campaign.get("cell_runtime"), "campaign cell_runtime")
    expected_runtime = {
        "elapsed_hours": elapsed,
        "attempts": len(attempt_uuids),
        "epochs_run": marker.get("epochs_run"),
        "test_f1": marker.get("test_f1"),
    }
    if cell_runtime.get(exp_id) != expected_runtime:
        raise PackageError(f"{exp_id} campaign cell-runtime summary mismatch")

    allowlist = {
        "final_metrics.json": run_dir / "final_metrics.json",
        "config.yaml": run_dir / "config.yaml",
        "metrics/metrics.csv": run_dir / "metrics" / "metrics.csv",
        "runtime_provenance.json": provenance_path,
        "checkpoints/best.ckpt": run_dir / "checkpoints" / "best.ckpt",
        "checkpoints/last.ckpt": run_dir / "checkpoints" / "last.ckpt",
        "log.txt": runs_root / "logs" / "h100" / f"{exp_id}.log",
    }
    entries: list[tuple[Path, PurePosixPath]] = []
    for relative, source in allowlist.items():
        if not source.is_file() or source.is_symlink():
            raise PackageError(f"required {exp_id} result artifact is absent: {source}")
        entries.extend(
            _source_entries(
                source, PurePosixPath(f"results/core/{exp_id}/{relative}")
            )
        )
    return entries, marker, provenance


def _grid_expectations(
    repo: Path, cells: Sequence[Cell]
) -> tuple[list[str], dict[str, dict[str, object]]]:
    try:
        config = yaml.safe_load((repo / "configs/arms.yaml").read_text())
        arms = config["arms"]
        fractions = config["label_fracs"]
        seeds = config["seeds"]["core"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise PackageError("cannot derive the exact grid.csv order") from exc
    expected: list[str] = []
    metadata: dict[str, dict[str, object]] = {}
    for init, raw in arms.items():
        for fraction in fractions:
            for seed in seeds:
                exp_id = (
                    f"{raw['short']}-f{int(round(float(fraction) * 100))}-s{seed}"
                )
                expected.append(exp_id)
                metadata[exp_id] = {
                    "init": init,
                    "track": raw["track"],
                    "role": raw["role"],
                    "label_frac": float(fraction),
                    "seed": int(seed),
                }
    if len(expected) != 32 or set(expected) != {cell.exp_id for cell in cells}:
        raise PackageError("grid.csv expectations are not the exact core matrix")
    return expected, metadata


def _same_number(raw: object, expected: object, description: str) -> None:
    actual = _finite(raw, description)
    wanted = _finite(expected, description)
    if actual != wanted:
        raise PackageError(f"{description} differs from final_metrics.json")


def _validate_grid(
    *,
    repo: Path,
    runs_root: Path,
    cells: Sequence[Cell],
    campaign: Mapping[str, object],
    markers: Mapping[str, Mapping[str, object]],
) -> Path:
    grid = runs_root / "summary/grid.csv"
    if grid.is_symlink():
        raise PackageError("summary/grid.csv may not be a symlink")
    try:
        with grid.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise PackageError("reverse handoff requires a readable summary/grid.csv") from exc
    if fields != _GRID_FIELDS:
        raise PackageError(f"summary/grid.csv columns mismatch: {fields}")
    expected_order, metadata = _grid_expectations(repo, cells)
    actual_order = [str(row.get("exp_id", "")) for row in rows]
    if (
        len(rows) != 32
        or len(set(actual_order)) != 32
        or actual_order != expected_order
    ):
        raise PackageError("summary/grid.csv IDs/order are not the exact 32-cell grid")
    for row in rows:
        exp_id = row["exp_id"]
        expected = metadata[exp_id]
        marker = markers[exp_id]
        _exact_bindings(
            row,
            {
                "init": str(expected["init"]),
                "track": str(expected["track"]),
                "role": str(expected["role"]),
                "precision": EXPECTED_PRECISION,
                "detector_sha256": campaign["detector_sha256"],
                "git_sha": campaign["git_sha"],
            },
            f"{exp_id} grid row",
        )
        _same_number(row["label_frac"], expected["label_frac"], f"{exp_id} label_frac")
        _same_number(row["seed"], expected["seed"], f"{exp_id} seed")
        _same_number(row["test_f1"], marker.get("test_f1"), f"{exp_id} test_f1")
        _same_number(row["dev_f1"], marker.get("best_dev_f1"), f"{exp_id} dev_f1")
        last_dev = _mapping(marker.get("last_dev"), f"{exp_id} last_dev")
        _same_number(
            row["dev_threshold"],
            last_dev.get("threshold"),
            f"{exp_id} dev_threshold",
        )
        _same_number(
            row["epochs_run"], marker.get("epochs_run"), f"{exp_id} epochs_run"
        )
        if row["monotonicity_ok"].lower() != "true":
            raise PackageError(f"summary/grid.csv monotonicity STOP for {exp_id}")

    grid_sha256 = sha256_file(grid)
    events = campaign.get("events")
    matching_events = [
        event
        for event in events
        if isinstance(event, Mapping)
        and event.get("event") == "grid_validated"
        and event.get("sha256") == grid_sha256
        and event.get("rows") == 32
    ] if isinstance(events, list) else []
    if len(matching_events) != 1:
        raise PackageError("campaign manifest does not bind the exact summary/grid.csv")
    return grid


def _member_sha256(
    entries: Sequence[tuple[Path, PurePosixPath]]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for source, archive_path in entries:
        if not source.is_file():
            continue
        name = archive_path.as_posix()
        if name in result:
            raise PackageError(f"duplicate archive member path: {name}")
        result[name] = _hash_file(source)
    if not result:
        raise PackageError("archive has no regular-file SHA bindings")
    return result


def _bound_archive(**kwargs: object) -> dict[str, object]:
    entries = kwargs["entries"]
    if not isinstance(entries, Sequence):
        raise PackageError("archive entries are invalid")
    artifact = _archive_artifact(**kwargs)  # type: ignore[arg-type]
    artifact["member_sha256"] = _member_sha256(entries)  # type: ignore[arg-type]
    return artifact


def _provenance_entries(
    *,
    campaign_manifest: Path,
    grid: Path,
    context: Mapping[str, object],
) -> list[tuple[Path, PurePosixPath]]:
    sources: list[tuple[Path, PurePosixPath]] = [
        (
            campaign_manifest,
            PurePosixPath("results/provenance/campaign_manifest.json"),
        ),
        (
            context["ready_path"],  # type: ignore[arg-type]
            PurePosixPath("results/provenance/H100_READY.json"),
        ),
        (
            context["runtime_path"],  # type: ignore[arg-type]
            PurePosixPath("results/provenance/h100_runtime.json"),
        ),
        (
            context["projection_path"],  # type: ignore[arg-type]
            PurePosixPath("results/provenance/throughput_projection.json"),
        ),
        (
            context["build_path"],  # type: ignore[arg-type]
            PurePosixPath("results/provenance/venv_build.json"),
        ),
        (
            context["cutover_path"],  # type: ignore[arg-type]
            PurePosixPath("results/provenance/CUTOVER_READY.json"),
        ),
        (
            context["source_validation_path"],  # type: ignore[arg-type]
            PurePosixPath("results/provenance/SOURCE_VALIDATED.json"),
        ),
        (
            context["host_test_receipt_path"],  # type: ignore[arg-type]
            PurePosixPath("results/provenance/HOST_HANDOFF_TESTS.json"),
        ),
        (
            context["host_test_log_path"],  # type: ignore[arg-type]
            PurePosixPath(
                "results/provenance/acceptance-logs/pytest-handoff-host.log"
            ),
        ),
        (
            context["test_suite_path"],  # type: ignore[arg-type]
            PurePosixPath("results/provenance/PYTEST_ACCEPTANCE.json"),
        ),
        (
            context["archived_receipt_path"],  # type: ignore[arg-type]
            PurePosixPath("results/provenance/V100_CORE_ARCHIVED.json"),
        ),
        (
            context["archive_manifest_path"],  # type: ignore[arg-type]
            PurePosixPath(
                "results/provenance/V100_CORE_ARCHIVE_MANIFEST.json"
            ),
        ),
        (
            context["smoke_ready_path"],  # type: ignore[arg-type]
            PurePosixPath(f"results/provenance/slurm-smoke/{SMOKE_READY_NAME}"),
        ),
        (
            context["smoke_state_path"],  # type: ignore[arg-type]
            PurePosixPath(f"results/provenance/slurm-smoke/{SMOKE_STATE_NAME}"),
        ),
        (grid, PurePosixPath("results/provenance/summary/grid.csv")),
    ]
    for record in context["allocation_records"]:  # type: ignore[union-attr]
        path = record["path"]
        sources.append(
            (
                path,
                PurePosixPath(f"results/provenance/allocations/{path.name}"),
            )
        )
    for path in context["acceptance_logs"]:  # type: ignore[union-attr]
        sources.append(
            (
                path,
                PurePosixPath(f"results/provenance/acceptance-logs/{path.name}"),
            )
        )
    for path in context["slurm_logs"]:  # type: ignore[union-attr]
        sources.append(
            (path, PurePosixPath(f"results/provenance/slurm/{path.name}"))
        )

    entries: list[tuple[Path, PurePosixPath]] = []
    seen: set[str] = set()
    for source, archive_path in sources:
        name = archive_path.as_posix()
        if name in seen:
            raise PackageError(f"duplicate campaign provenance member: {name}")
        seen.add(name)
        entries.extend(_source_entries(source, archive_path))
    return entries


def build_results_package(
    *,
    repo: Path,
    runs_root: Path,
    campaign_manifest: Path,
    output: Path,
    max_part_bytes: int,
) -> Path:
    """Require 32 scored H100 cells, then publish a provenance-bound package."""

    original_campaign_manifest = campaign_manifest.absolute()
    if original_campaign_manifest.is_symlink():
        raise PackageError("campaign manifest may not be a symlink")
    repo = repo.resolve()
    runs_root = _require_no_symlink_components(runs_root, leaf="directory")
    campaign_manifest = original_campaign_manifest.resolve()
    output = output.resolve()
    if campaign_manifest != runs_root / ".h100/campaign_manifest.json":
        raise PackageError("campaign manifest must be runs/.h100/campaign_manifest.json")
    if _inside_repository_worktrees(output, repo):
        raise PackageError("result package output must be outside the repository")
    if output.exists():
        raise PackageError(f"result package destination already exists: {output}")
    if max_part_bytes <= 0:
        raise PackageError("max_part_bytes must be positive")
    if not campaign_manifest.is_file():
        raise PackageError(f"campaign manifest is absent: {campaign_manifest}")
    branch = _git_value(repo, "branch", "--show-current")
    if branch != RUNTIME_BRANCH:
        raise PackageError(
            f"result package requires clean branch {RUNTIME_BRANCH}, found {branch!r}"
        )
    if _git_value(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise PackageError("result package requires a clean source worktree")

    campaign, cells, context = _validate_campaign(
        repo, runs_root, campaign_manifest
    )
    run_entries: dict[str, list[tuple[Path, PurePosixPath]]] = {}
    markers: dict[str, dict[str, object]] = {}
    runtime_provenance_sha256: dict[str, str] = {}
    for cell in cells:
        entries, marker, _provenance = _run_entries(
            runs_root, cell, campaign, context
        )
        run_entries[cell.exp_id] = entries
        markers[cell.exp_id] = marker
        runtime_provenance_sha256[cell.exp_id] = sha256_file(
            runs_root / cell.exp_id / "runtime_provenance.json"
        )
    grid = _validate_grid(
        repo=repo,
        runs_root=runs_root,
        cells=cells,
        campaign=campaign,
        markers=markers,
    )
    provenance_entries = _provenance_entries(
        campaign_manifest=campaign_manifest,
        grid=grid,
        context=context,
    )

    git_sha = str(campaign["git_sha"])
    result_identity, result_identity_sha256 = _result_identity(campaign)
    package_id = _result_package_id(campaign)
    if output.name != package_id:
        raise PackageError(f"result output directory must be named {package_id}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    artifacts: list[dict[str, object]] = []
    try:
        for cell in cells:
            exp_id = cell.exp_id
            artifacts.append(
                _bound_archive(
                    staging=staging,
                    output_relative=PurePosixPath(
                        f"results/core/{exp_id}.tar.zst"
                    ),
                    entries=run_entries[exp_id],
                    kind="core_result",
                    name=exp_id,
                    extraction_root=PurePosixPath(f"results/core/{exp_id}"),
                    max_part_bytes=max_part_bytes,
                )
            )

        provenance_artifact = _bound_archive(
            staging=staging,
            output_relative=PurePosixPath(
                "results/provenance/campaign.tar.zst"
            ),
            entries=provenance_entries,
            kind="campaign_provenance",
            name=str(campaign.get("campaign_id", "")),
            extraction_root=PurePosixPath("results/provenance"),
            max_part_bytes=max_part_bytes,
        )
        artifacts.append(provenance_artifact)

        created = str(campaign.get("updated_utc") or "")
        if not created:
            created = datetime.fromtimestamp(0, tz=timezone.utc).isoformat()
        manifest: dict[str, object] = {
            "format_version": FORMAT_VERSION,
            "package_type": "h100-core-results",
            "package_id": package_id,
            "created_at": created,
            "source": {
                "branch": RUNTIME_BRANCH,
                "git_commit": git_sha,
                "campaign_id": campaign.get("campaign_id"),
                "detector_sha256": campaign.get("detector_sha256"),
                "venv_sha256": campaign.get("venv_sha256"),
                "venv_build_sha256": campaign.get("venv_build_sha256"),
                "base_python": campaign.get("base_python"),
                "base_payload": context["base_payload"],
                "runtime_amendment": context["runtime_amendment"],
                "acceptance_uuid": _mapping(
                    campaign["acceptance"], "campaign acceptance"
                )["uuid"],
                "strict_fp32": context["strict_fp32"],
                "accepted_hardware_class": context[
                    "accepted_hardware_class"
                ],
                "allocation_hardware_class": context[
                    "allocation_hardware_class"
                ],
                "operator_cutover": context["operator_cutover"],
                "source_validation_sha256": context[
                    "source_validation_sha256"
                ],
                "acceptance_test_suite_sha256": context[
                    "test_suite_sha256"
                ],
                "campaign_manifest_sha256": sha256_file(campaign_manifest),
                "summary_grid_sha256": sha256_file(grid),
                "runtime_provenance_sha256": runtime_provenance_sha256,
                "campaign_provenance_member_sha256": provenance_artifact[
                    "member_sha256"
                ],
                "result_identity": result_identity,
                "result_identity_sha256": result_identity_sha256,
            },
            "contract": {
                "production": True,
                "core_only": True,
                "strict_fp32": True,
                "tf32": False,
                "maximum_physical_file_bytes": max_part_bytes,
                "excluded": [
                    "data",
                    "weights",
                    "environment/wheelhouse",
                    "secrets",
                    "V100 core diagnostic payloads",
                    "R2/R3 reference artifacts",
                    "superseded runs",
                ],
            },
            "counts": {
                "core_result_archives": 32,
                "provenance_archives": 1,
            },
            "cells": [cell.exp_id for cell in cells],
            "artifacts": artifacts,
        }
        _write_bytes(staging / "manifest.json", _canonical_json(manifest))
        checksums = "".join(
            f"{_hash_file(staging / relative)}  {relative}\n"
            for relative in _physical_paths(artifacts)
        )
        _write_bytes(staging / "SHA256SUMS", checksums.encode("utf-8"))
        ready = {
            "format_version": FORMAT_VERSION,
            "status": "READY",
            "package_id": package_id,
            "git_commit": git_sha,
            "manifest": _physical_record(staging / "manifest.json", staging),
            "checksums": _physical_record(staging / "SHA256SUMS", staging),
        }
        _write_bytes(staging / "READY.json", _canonical_json(ready))
        verify_package(staging)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
