"""Validate H100 readiness plus fresh R2/R3; never signal the V100 campaign."""

from __future__ import annotations

import argparse
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from scripts.h100.acceptance import (
    EXPECTED_FRACTION_WORKLOAD,
    validate_hardware_runtime_contracts,
)
from scripts.h100.build_venv import EXPECTED_PYTHON_VERSION
from scripts.h100.contracts import (
    FROZEN_PATHS,
    atomic_write_json,
    cutover_acceptance_bindings,
    sha256_file,
    staging_aware_wall_clock,
    validate_bound_cutover_forecast,
    validate_gpu_inventory,
)
from scripts.h100.host_test_gate import HOST_TESTS, validate_host_gate
from scripts.h100.slurm_smoke import make_bindings as make_smoke_bindings
from scripts.h100.slurm_smoke import validate_smoke_receipt
from scripts.h100.source_validation import validate_source_receipt

HEX64 = re.compile(r"^[0-9a-f]{64}$")
REFERENCE_HASH_FIELDS = (
    "environment_sha256",
    "environment_lock_sha256",
    "campaign_manifest_sha256",
    "runtime_launcher_sha256",
)
REFERENCE_PROVENANCE_KEYS = {
    "campaign_id",
    "git_sha",
    *REFERENCE_HASH_FIELDS,
    "hardware",
    "container_local_gpu",
    "gpu_uuid",
    "started_utc",
    "finished_utc",
    "elapsed_hours",
    "gpu_hours",
    "reference_precision",
}
REFERENCE_RESULT_KEYS = {
    "metrics",
    "metrics_sha256",
    "provenance",
    "provenance_sha256",
}


def write_new_cutover_ready(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise RuntimeError(
            "CUTOVER_READY already exists; archive or remove it only through the "
            "operator cutover procedure"
        )
    atomic_write_json(path, payload)
    path.chmod(0o444)


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is not finite")
    return number


def validate_reference_provenance(
    payload: Mapping[str, object],
    *,
    expected_git_sha: str,
    expected_campaign_id: str,
) -> None:
    if set(payload) != REFERENCE_PROVENANCE_KEYS:
        raise RuntimeError("reference provenance keys do not match fresh34")
    if payload.get("git_sha") != expected_git_sha:
        raise RuntimeError("reference provenance git SHA mismatch")
    if payload.get("campaign_id") != expected_campaign_id:
        raise RuntimeError("reference provenance campaign mismatch")
    if payload.get("hardware") != "Tesla V100-SXM2-32GB":
        raise RuntimeError("reference provenance is not from Tesla V100-SXM2-32GB")
    if not str(payload.get("gpu_uuid", "")).strip():
        raise RuntimeError("reference provenance lacks a GPU UUID")
    gpu_index = payload.get("container_local_gpu")
    if type(gpu_index) is not int or not 0 <= gpu_index < 8:
        raise RuntimeError("reference provenance has an invalid container-local GPU")
    if not payload.get("reference_precision"):
        raise RuntimeError("reference precision is absent")
    elapsed = _finite(payload.get("elapsed_hours"), "reference elapsed_hours")
    gpu_hours = _finite(payload.get("gpu_hours"), "reference gpu_hours")
    if elapsed <= 0 or not math.isclose(elapsed, gpu_hours):
        raise RuntimeError("reference elapsed_hours must be positive")
    finished = payload.get("finished_utc")
    if not payload.get("started_utc") or not finished:
        raise RuntimeError("reference timestamps are absent")
    invalid_hashes = [
        name
        for name in REFERENCE_HASH_FIELDS
        if not HEX64.fullmatch(str(payload.get(name, "")))
    ]
    if invalid_hashes:
        raise RuntimeError(
            "reference provenance lacks explicit 64-hex hashes: "
            + ", ".join(invalid_hashes)
        )


def _reference_json(path: Path, label: str) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular non-symlink file")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload, sha256_file(path)


def _reference_result(
    *,
    metrics_path: Path,
    provenance_path: Path,
    metrics: Mapping[str, object],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    return {
        "metrics": dict(metrics),
        "metrics_sha256": sha256_file(metrics_path),
        "provenance": dict(provenance),
        "provenance_sha256": sha256_file(provenance_path),
    }


def validate_r2(run_dir: Path, **provenance_kwargs) -> dict:
    metrics_path = run_dir / "final_metrics.json"
    provenance_path = run_dir / "runtime_provenance.json"
    metrics, _metrics_sha256 = _reference_json(metrics_path, "R2 final metrics")
    if metrics.get("exp_id") != "yolo26-f100":
        raise RuntimeError("R2 exp_id mismatch")
    for key in (
        "threshold",
        "dev_f1",
        "test_f1",
        "test_precision",
        "test_recall",
        "test_near_shore_f1",
    ):
        _finite(metrics.get(key), f"R2 {key}")
    provenance, _provenance_sha256 = _reference_json(
        provenance_path, "R2 runtime provenance"
    )
    validate_reference_provenance(provenance, **provenance_kwargs)
    return _reference_result(
        metrics_path=metrics_path,
        provenance_path=provenance_path,
        metrics=metrics,
        provenance=provenance,
    )


def validate_r3(run_dir: Path, **provenance_kwargs) -> dict:
    metrics_path = run_dir / "final_metrics.json"
    provenance_path = run_dir / "runtime_provenance.json"
    metrics, _metrics_sha256 = _reference_json(metrics_path, "R3 final metrics")
    if metrics.get("exp_id") != "locateanything-zs":
        raise RuntimeError("R3 exp_id mismatch")
    prompts = metrics.get("per_prompt")
    best = metrics.get("best_prompt")
    if not isinstance(prompts, dict) or set(prompts) != {"ship", "vessel", "boat"}:
        raise RuntimeError("R3 prompt set must be exactly ship/vessel/boat")
    if best not in prompts:
        raise RuntimeError("R3 prompt results/best_prompt are invalid")
    for prompt, result in prompts.items():
        if not isinstance(result, dict):
            raise RuntimeError(f"R3 prompt {prompt} result is invalid")
        for key in ("f1", "precision", "recall", "threshold"):
            _finite(result.get(key), f"R3 {prompt}.{key}")
    provenance, _provenance_sha256 = _reference_json(
        provenance_path, "R3 runtime provenance"
    )
    validate_reference_provenance(provenance, **provenance_kwargs)
    return _reference_result(
        metrics_path=metrics_path,
        provenance_path=provenance_path,
        metrics=metrics,
        provenance=provenance,
    )


def validate_h100_ready(
    path: Path,
    *,
    expected_git_sha: str,
    expected_venv_sha256: str,
    expected_venv_build_sha256: str,
    expected_base_python_sha256: str,
    expected_base_python_runtime_sha256: str,
    expected_wheelhouse_sha256: str,
    expected_base_extraction_receipt_sha256: str,
    expected_base_payload: Mapping[str, str],
    expected_runtime_amendment: Mapping[str, str],
    expected_frozen_sha256: Mapping[str, str],
    expected_smoke_receipt: Mapping[str, object],
    expected_smoke_sha256: str,
) -> dict:
    path = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("H100_READY must be a regular non-symlink")
    payload = json.loads(path.read_text())
    if payload.get("schema") != 2 or payload.get("status") != "ready":
        raise RuntimeError("H100 acceptance marker is not schema-2 ready")
    if payload.get("source", {}).get("git_sha") != expected_git_sha:
        raise RuntimeError("H100-ready git SHA mismatch")
    accepted_venv = payload.get("venv")
    if not isinstance(accepted_venv, Mapping):
        raise RuntimeError("H100-ready native-venv binding is absent")
    if accepted_venv.get("sha256") != expected_venv_sha256:
        raise RuntimeError("H100-ready native-venv tree digest mismatch")
    if accepted_venv.get("venv_build_sha256") != expected_venv_build_sha256:
        raise RuntimeError("H100-ready native-venv build receipt mismatch")
    base_python = accepted_venv.get("base_python")
    base_python_runtime = (
        base_python.get("runtime") if isinstance(base_python, Mapping) else None
    )
    if (
        not isinstance(base_python, Mapping)
        or base_python.get("version") != EXPECTED_PYTHON_VERSION
        or base_python.get("executable_sha256") != expected_base_python_sha256
        or not isinstance(base_python_runtime, Mapping)
        or base_python_runtime.get("sha256")
        != expected_base_python_runtime_sha256
    ):
        raise RuntimeError("H100-ready base-Python identity mismatch")
    wheelhouse = accepted_venv.get("wheelhouse")
    wheelhouse_identity = (
        wheelhouse.get("identity") if isinstance(wheelhouse, Mapping) else None
    )
    base_extraction = (
        wheelhouse.get("base_extraction")
        if isinstance(wheelhouse, Mapping)
        else None
    )
    extraction_receipt = (
        base_extraction.get("receipt")
        if isinstance(base_extraction, Mapping)
        else None
    )
    staged_base_extraction = accepted_venv.get("staged_base_extraction")
    staged_extraction_receipt = (
        staged_base_extraction.get("receipt")
        if isinstance(staged_base_extraction, Mapping)
        else None
    )
    if (
        not isinstance(wheelhouse_identity, Mapping)
        or wheelhouse_identity.get("sha256") != expected_wheelhouse_sha256
        or not isinstance(base_extraction, Mapping)
        or base_extraction.get("sha256")
        != expected_base_extraction_receipt_sha256
        or not isinstance(extraction_receipt, Mapping)
        or extraction_receipt.get("package_id")
        != expected_base_payload.get("package_id")
        or extraction_receipt.get("manifest_sha256")
        != expected_base_payload.get("manifest_sha256")
        or extraction_receipt.get("wheelhouse") != wheelhouse_identity
        or not isinstance(staged_base_extraction, Mapping)
        or staged_base_extraction.get("sha256")
        != expected_base_extraction_receipt_sha256
        or not isinstance(staged_extraction_receipt, Mapping)
        or staged_extraction_receipt.get("package_id")
        != expected_base_payload.get("package_id")
        or staged_extraction_receipt.get("manifest_sha256")
        != expected_base_payload.get("manifest_sha256")
        or staged_extraction_receipt.get("wheelhouse") != wheelhouse_identity
    ):
        raise RuntimeError("H100-ready wheelhouse/base-extraction identity mismatch")
    if payload.get("base_payload") != dict(expected_base_payload):
        raise RuntimeError("H100-ready base-payload bindings mismatch")
    if payload.get("runtime_amendment") != dict(expected_runtime_amendment):
        raise RuntimeError("H100-ready runtime-amendment bindings mismatch")
    if payload.get("source", {}).get("frozen_sha256") != dict(expected_frozen_sha256):
        raise RuntimeError("H100-ready frozen-file hash bindings mismatch")
    source_binding = payload.get("source_validation")
    expected_source_path = path.parent / "SOURCE_VALIDATED.json"
    if (
        not isinstance(source_binding, Mapping)
        or set(source_binding) != {"path", "sha256", "receipt"}
        or source_binding.get("path") != str(expected_source_path)
        or not HEX64.fullmatch(str(source_binding.get("sha256", "")))
    ):
        raise RuntimeError("H100-ready source-validation binding is invalid")
    source_receipt = validate_source_receipt(
        expected_source_path,
        expected_sha256=str(source_binding["sha256"]),
        expected_git_sha=expected_git_sha,
        expected_hashes=expected_frozen_sha256,
        expected_base_payload=expected_base_payload,
        expected_runtime_amendment=expected_runtime_amendment,
    )
    if source_binding.get("receipt") != source_receipt:
        raise RuntimeError("H100-ready embedded source-validation receipt differs")
    try:
        uuid.UUID(str(payload.get("acceptance_uuid")))
    except (ValueError, AttributeError) as exc:
        raise RuntimeError("H100-ready acceptance UUID is absent or invalid") from exc
    if set(payload.get("strict_fp32", {}).values()) != {"ieee"}:
        raise RuntimeError("H100-ready marker does not assert strict IEEE FP32")
    validate_gpu_inventory(payload.get("hardware", {}).get("devices", []))
    validate_hardware_runtime_contracts(payload.get("hardware"))
    for key in ("torch", "cuda_build", "driver_version"):
        if not str(payload.get("hardware", {}).get(key, "")).strip():
            raise RuntimeError(f"H100-ready hardware lacks {key}")
    gates = payload.get("gates")
    required_gates = {
        "pytest_seconds",
        "hardware_probe_seconds",
        "vit_gate_seconds",
        "cnn_200step_seconds",
    }
    if not isinstance(gates, Mapping) or set(gates) != required_gates:
        raise RuntimeError("H100-ready acceptance gate evidence is incomplete")
    if any(_finite(gates[key], f"H100 gate {key}") < 0 for key in required_gates):
        raise RuntimeError("H100-ready gate durations cannot be negative")
    test_binding = payload.get("test_suite")
    expected_test_path = path.parent / "PYTEST_ACCEPTANCE.json"
    if (
        not isinstance(test_binding, Mapping)
        or set(test_binding) != {"path", "sha256", "receipt"}
        or test_binding.get("path") != str(expected_test_path)
        or not HEX64.fullmatch(str(test_binding.get("sha256", "")))
        or expected_test_path.is_symlink()
        or not expected_test_path.is_file()
        or sha256_file(expected_test_path) != test_binding.get("sha256")
    ):
        raise RuntimeError("H100-ready aggregate test-suite binding is invalid")
    test_suite = json.loads(expected_test_path.read_text())
    if test_binding.get("receipt") != test_suite or set(test_suite) != {
        "schema",
        "status",
        "source_validation_sha256",
        "coverage",
        "host_handoff",
        "venv_remaining",
        "aggregate_duration_seconds",
    }:
        raise RuntimeError("H100-ready aggregate test-suite receipt differs")
    if (
        test_suite.get("schema") != 2
        or test_suite.get("status") != "passed"
        or test_suite.get("source_validation_sha256") != source_binding["sha256"]
        or test_suite.get("coverage")
        != {
            "host": HOST_TESTS,
            "venv": "all pytest collection except the two host-only files",
            "aggregate": "entire repository pytest suite",
        }
    ):
        raise RuntimeError("aggregate test-suite scope/source binding is invalid")
    host_slice = test_suite.get("host_handoff")
    expected_host_path = path.parent / "HOST_HANDOFF_TESTS.json"
    if (
        not isinstance(host_slice, Mapping)
        or set(host_slice) != {"receipt_path", "receipt_sha256", "receipt"}
        or host_slice.get("receipt_path") != str(expected_host_path)
    ):
        raise RuntimeError("aggregate host-test slice binding is invalid")
    host_receipt = validate_host_gate(
        expected_host_path,
        expected_sha256=str(host_slice.get("receipt_sha256", "")),
        expected_source_validation_sha256=str(source_binding["sha256"]),
    )
    if host_slice.get("receipt") != host_receipt:
        raise RuntimeError("aggregate embedded host-test receipt differs")
    venv_slice = test_suite.get("venv_remaining")
    expected_venv_log = path.parent / "acceptance-logs/pytest-venv-remaining.log"
    if not isinstance(venv_slice, Mapping) or set(venv_slice) != {
        "command",
        "duration_seconds",
        "log",
    }:
        raise RuntimeError("aggregate venv-test slice is invalid")
    venv_log = venv_slice.get("log")
    if (
        venv_slice.get("command")
        != ["-m", "pytest", "-q", *(f"--ignore={item}" for item in HOST_TESTS)]
        or not isinstance(venv_log, Mapping)
        or set(venv_log) != {"path", "sha256"}
        or venv_log.get("path") != str(expected_venv_log)
        or expected_venv_log.is_symlink()
        or not expected_venv_log.is_file()
        or not HEX64.fullmatch(str(venv_log.get("sha256", "")))
        or sha256_file(expected_venv_log) != venv_log.get("sha256")
    ):
        raise RuntimeError("aggregate venv-test log/scope binding is invalid")
    aggregate_seconds = _finite(
        test_suite.get("aggregate_duration_seconds"), "aggregate pytest duration"
    )
    expected_seconds = _finite(
        host_receipt.get("duration_seconds"), "host pytest duration"
    ) + _finite(venv_slice.get("duration_seconds"), "venv pytest duration")
    if (
        aggregate_seconds <= 0
        or not math.isclose(aggregate_seconds, expected_seconds)
        or not math.isclose(float(gates["pytest_seconds"]), aggregate_seconds)
    ):
        raise RuntimeError("aggregate pytest durations are inconsistent")
    projection = payload.get("projection")
    projection_fields = (
        "steps_per_second",
        "expected_gpu_hours",
        "ceiling_gpu_hours",
        "expected_wall_hours_ideal",
        "ceiling_wall_hours_ideal",
        "conservative_h100_wall_hours",
        "remaining_v100_wall_hours",
        "staging_seconds",
        "staging_hours_per_allocation",
        "allocation_wall_hours",
        "signal_lead_hours",
        "usable_training_hours_per_allocation",
        "training_wall_hours_before_staging",
    )
    if not isinstance(projection, Mapping):
        raise RuntimeError("H100-ready throughput projection is absent")
    expected_steps = {
        label: int(item["steps_per_epoch"])
        for label, item in EXPECTED_FRACTION_WORKLOAD.items()
    }
    if (
        projection.get("fraction_workload") != EXPECTED_FRACTION_WORKLOAD
        or projection.get("steps_per_epoch") != expected_steps
        or projection.get("grid_steps_per_epoch") != 8 * sum(expected_steps.values())
    ):
        raise RuntimeError("H100-ready projection does not bind the exact workload")
    values = {key: _finite(projection.get(key), f"projection {key}") for key in projection_fields}
    if any(value <= 0 for value in values.values()):
        raise RuntimeError("H100-ready projection values must be positive")
    try:
        recomputed_wall = staging_aware_wall_clock(
            training_wall_hours=values["training_wall_hours_before_staging"],
            staging_seconds=values["staging_seconds"],
            allocation_wall_hours=values["allocation_wall_hours"],
            signal_lead_hours=values["signal_lead_hours"],
        )
    except ValueError as exc:
        raise RuntimeError("H100-ready staging projection is invalid") from exc
    projected_allocations = projection.get("projected_allocation_count")
    if (
        type(projected_allocations) is not int
        or projected_allocations != recomputed_wall["projected_allocation_count"]
        or any(
            not math.isclose(values[key], float(recomputed_wall[key]))
            for key in (
                "staging_hours_per_allocation",
                "usable_training_hours_per_allocation",
                "training_wall_hours_before_staging",
                "conservative_h100_wall_hours",
            )
        )
    ):
        raise RuntimeError("H100-ready staging/wall-clock projection is inconsistent")
    if values["conservative_h100_wall_hours"] >= values["remaining_v100_wall_hours"]:
        raise RuntimeError("H100-ready projection no longer proves a faster cutover")
    smoke = payload.get("slurm_smoke")
    if not isinstance(smoke, Mapping):
        raise RuntimeError("H100-ready Slurm smoke binding is absent")
    if smoke.get("sha256") != expected_smoke_sha256:
        raise RuntimeError("H100-ready Slurm smoke receipt hash mismatch")
    if smoke.get("receipt") != expected_smoke_receipt:
        raise RuntimeError("H100-ready Slurm smoke receipt payload mismatch")
    return payload


def validate_current_v100_advantage(
    ready: Mapping[str, object], current_remaining_v100_wall_hours: float
) -> dict[str, float]:
    """Recheck the H100 advantage against the V100 forecast at cutover time."""

    projection = ready.get("projection")
    if not isinstance(projection, Mapping):
        raise RuntimeError("H100-ready throughput projection is absent")
    current = _finite(
        current_remaining_v100_wall_hours, "current remaining V100 wall hours"
    )
    conservative = _finite(
        projection.get("conservative_h100_wall_hours"),
        "conservative H100 wall hours",
    )
    accepted_v100 = _finite(
        projection.get("remaining_v100_wall_hours"),
        "acceptance remaining V100 wall hours",
    )
    if current <= 0 or conservative <= 0 or accepted_v100 <= 0:
        raise RuntimeError("cutover wall-clock forecasts must be positive")
    if conservative >= current:
        raise RuntimeError(
            "H100 cutover rejected: the current V100 forecast is no longer slower"
        )
    return {
        "conservative_h100_wall_hours": conservative,
        "acceptance_remaining_v100_wall_hours": accepted_v100,
        "current_remaining_v100_wall_hours": current,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h100-ready", type=Path, required=True)
    parser.add_argument("--r2-run-dir", type=Path, required=True)
    parser.add_argument("--r3-run-dir", type=Path, required=True)
    parser.add_argument("--expected-h100-git-sha", required=True)
    parser.add_argument("--expected-reference-git-sha", required=True)
    parser.add_argument("--expected-venv-sha256", required=True)
    parser.add_argument("--expected-venv-build-sha256", required=True)
    parser.add_argument("--expected-base-python-sha256", required=True)
    parser.add_argument("--expected-base-python-runtime-sha256", required=True)
    parser.add_argument("--expected-wheelhouse-sha256", required=True)
    parser.add_argument(
        "--expected-base-extraction-receipt-sha256", required=True
    )
    parser.add_argument("--expected-base-payload-package-id", required=True)
    parser.add_argument("--expected-base-payload-git-sha", required=True)
    parser.add_argument("--expected-base-payload-manifest-sha256", required=True)
    parser.add_argument("--expected-base-payload-ready-sha256", required=True)
    parser.add_argument("--expected-base-payload-sha256sums-sha256", required=True)
    parser.add_argument("--expected-base-payload-repo-bundle-sha256", required=True)
    parser.add_argument("--expected-runtime-amendment-package-id", required=True)
    parser.add_argument("--expected-runtime-amendment-git-sha", required=True)
    parser.add_argument("--expected-runtime-amendment-manifest-sha256", required=True)
    parser.add_argument("--expected-runtime-amendment-ready-sha256", required=True)
    parser.add_argument("--expected-runtime-amendment-sha256sums-sha256", required=True)
    parser.add_argument("--expected-runtime-amendment-bundle-sha256", required=True)
    parser.add_argument(
        "--expected-frozen-sha256",
        action="append",
        required=True,
        help="repeat in FROZEN_PATHS order",
    )
    parser.add_argument("--smoke-ready", type=Path, required=True)
    parser.add_argument("--expected-reference-campaign-id", required=True)
    parser.add_argument(
        "--current-remaining-v100-wall-hours", type=float, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.expected_frozen_sha256) != len(FROZEN_PATHS):
        parser.error(
            f"--expected-frozen-sha256 must be repeated {len(FROZEN_PATHS)} times"
        )

    expected_frozen = dict(
        zip(FROZEN_PATHS, args.expected_frozen_sha256, strict=True)
    )
    expected_base_payload = {
        "package_id": args.expected_base_payload_package_id,
        "git_sha": args.expected_base_payload_git_sha,
        "manifest_sha256": args.expected_base_payload_manifest_sha256,
        "ready_sha256": args.expected_base_payload_ready_sha256,
        "sha256sums_sha256": args.expected_base_payload_sha256sums_sha256,
        "repo_bundle_sha256": args.expected_base_payload_repo_bundle_sha256,
    }
    expected_runtime_amendment = {
        "package_id": args.expected_runtime_amendment_package_id,
        "git_sha": args.expected_runtime_amendment_git_sha,
        "manifest_sha256": args.expected_runtime_amendment_manifest_sha256,
        "ready_sha256": args.expected_runtime_amendment_ready_sha256,
        "sha256sums_sha256": args.expected_runtime_amendment_sha256sums_sha256,
        "runtime_bundle_sha256": args.expected_runtime_amendment_bundle_sha256,
    }
    smoke_bindings = make_smoke_bindings(
        git_sha=args.expected_h100_git_sha,
        detector_sha256=expected_frozen["configs/detector.yaml"],
        venv_sha256=args.expected_venv_sha256,
        venv_build_sha256=args.expected_venv_build_sha256,
        base_python_sha256=args.expected_base_python_sha256,
        base_python_runtime_sha256=(
            args.expected_base_python_runtime_sha256
        ),
        wheelhouse_sha256=args.expected_wheelhouse_sha256,
        base_extraction_receipt_sha256=(
            args.expected_base_extraction_receipt_sha256
        ),
        base_payload=expected_base_payload,
        runtime_amendment=expected_runtime_amendment,
    )
    smoke = validate_smoke_receipt(
        args.smoke_ready,
        expected_bindings=smoke_bindings,
    )
    smoke_sha256 = sha256_file(args.smoke_ready)

    provenance_kwargs = {
        "expected_git_sha": args.expected_reference_git_sha,
        "expected_campaign_id": args.expected_reference_campaign_id,
    }
    ready = validate_h100_ready(
        args.h100_ready,
        expected_git_sha=args.expected_h100_git_sha,
        expected_venv_sha256=args.expected_venv_sha256,
        expected_venv_build_sha256=args.expected_venv_build_sha256,
        expected_base_python_sha256=args.expected_base_python_sha256,
        expected_base_python_runtime_sha256=(
            args.expected_base_python_runtime_sha256
        ),
        expected_wheelhouse_sha256=args.expected_wheelhouse_sha256,
        expected_base_extraction_receipt_sha256=(
            args.expected_base_extraction_receipt_sha256
        ),
        expected_base_payload=expected_base_payload,
        expected_runtime_amendment=expected_runtime_amendment,
        expected_frozen_sha256=expected_frozen,
        expected_smoke_receipt=smoke,
        expected_smoke_sha256=smoke_sha256,
    )
    r2 = validate_r2(args.r2_run_dir, **provenance_kwargs)
    r3 = validate_r3(args.r3_run_dir, **provenance_kwargs)
    cutover_forecast = validate_current_v100_advantage(
        ready, args.current_remaining_v100_wall_hours
    )
    marker = {
        "schema": 2,
        "status": "cutover-ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "h100_ready": ready,
        "acceptance": cutover_acceptance_bindings(ready),
        "cutover_forecast": cutover_forecast,
        "references": {
            "r2": r2,
            "r3": r3,
        },
        "v100_action": "none; this guard never stops or signals V100 processes",
    }
    validate_bound_cutover_forecast(marker)
    write_new_cutover_ready(args.output, marker)
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
