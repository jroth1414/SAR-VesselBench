"""Run source, strict-FP32 H100, and 200-step cutover acceptance gates."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts.h100.contracts import (
    EXPECTED_PRECISION,
    FROZEN_PATHS,
    MIN_SCRATCH_BYTES,
    assert_empty_core_namespaces,
    atomic_write_json,
    estimate_grid_projection,
    sha256_file,
    staging_aware_wall_clock,
    verify_expected_hashes,
)
from scripts.h100.build_venv import verify as verify_native_venv
from scripts.h100.host_test_gate import HOST_TESTS, validate_host_gate
from scripts.h100.lightning_contract import validate_trainer_contract_evidence
from scripts.h100.precision import assert_sitecustomize_active
from scripts.h100.slurm_smoke import (
    make_bindings as make_smoke_bindings,
    validate_smoke_receipt,
)
from scripts.h100.source_validation import validate_source_receipt
from scripts.h100.wheelhouse import validate_base_extraction_receipt

WEIGHT_SUBDIRS = (
    "satdino",
    "sarmae",
    "bigearthnet_s1",
    "bigearthnet_s2",
    "imagenet_vit_augreg_in1k",
    "imagenet_cnn_fcmae_ft_in1k",
)
PROBE_IDS = (
    "vitin1k-f100-s0-h100-acceptance",
    "cnnin1k-f100-s0-h100-200step",
)
EXPECTED_FRACTION_WORKLOAD = {
    "f10": {"fraction": 0.1, "scene_count": 12, "chip_count": 11_218, "steps_per_epoch": 701},
    "f25": {"fraction": 0.25, "scene_count": 28, "chip_count": 26_603, "steps_per_epoch": 1_662},
    "f50": {"fraction": 0.5, "scene_count": 56, "chip_count": 52_967, "steps_per_epoch": 3_310},
    "f100": {"fraction": 1.0, "scene_count": 111, "chip_count": 105_408, "steps_per_epoch": 6_588},
}


def validate_scratch_free_before_extraction(value: int) -> int:
    if type(value) is not int or value < MIN_SCRATCH_BYTES:
        raise RuntimeError(
            "pre-extraction scratch free space was "
            f"{value / 1e9:.1f} GB; at least 500 GB required"
        )
    return value


def validate_fresh_acceptance_state(
    *, ready: Path, runs_root: Path, remaining_v100_wall_hours: float
) -> None:
    if ready.exists():
        raise RuntimeError("H100_READY.json already exists; acceptance must be fresh")
    if (
        not math.isfinite(remaining_v100_wall_hours)
        or remaining_v100_wall_hours <= 0
    ):
        raise RuntimeError("remaining V100 wall hours must be finite and positive")
    occupied = [exp_id for exp_id in PROBE_IDS if (runs_root / exp_id).exists()]
    if occupied:
        raise RuntimeError(
            "acceptance probe namespaces must start absent; STOP and inspect: "
            + ", ".join(occupied)
        )


def derive_fraction_workload(*, chips_root: Path, splits_path: Path) -> dict:
    """Count the exact frozen nested-scene workload received on the H100."""

    from src.data.datasets import nested_fraction_scenes

    train_scenes = json.loads(splits_path.read_text())["splits"]["train"]
    workload: dict[str, dict[str, object]] = {}
    for label, expected in EXPECTED_FRACTION_WORKLOAD.items():
        scenes = nested_fraction_scenes(
            train_scenes,
            float(expected["fraction"]),
            frac_seed=0,
        )
        missing = [scene for scene in scenes if not (chips_root / scene).is_dir()]
        if missing:
            raise RuntimeError(
                f"fraction workload is missing received chip scenes: {missing[:3]}"
            )
        chip_count = sum(
            1 for scene in scenes for _path in (chips_root / scene).glob("*.npy")
        )
        workload[label] = {
            "fraction": expected["fraction"],
            "scene_count": len(scenes),
            "chip_count": chip_count,
            "steps_per_epoch": chip_count // 16,
        }
    if workload != EXPECTED_FRACTION_WORKLOAD:
        raise RuntimeError(
            "received chip/frozen split workload differs from the pinned projection: "
            f"{workload} != {EXPECTED_FRACTION_WORKLOAD}"
        )
    return workload


def validate_runtime_source(
    *,
    repo: Path,
    receipt_path: Path,
    receipt_sha256: str,
    expected_git_sha: str,
    expected_hashes: dict[str, str],
    expected_base_payload: dict[str, str],
    expected_runtime_amendment: dict[str, str],
) -> dict:
    receipt = validate_source_receipt(
        receipt_path,
        expected_sha256=receipt_sha256,
        expected_git_sha=expected_git_sha,
        expected_hashes=expected_hashes,
        expected_base_payload=expected_base_payload,
        expected_runtime_amendment=expected_runtime_amendment,
    )
    # Rehash frozen files in the detached runtime checkout. Git cleanliness
    # itself is the host-only check recorded by SOURCE_VALIDATED.json.
    actual_hashes = verify_expected_hashes(repo, expected_hashes)
    detector = yaml.safe_load((repo / "configs/detector.yaml").read_text())
    if detector["schedule"]["precision"] != EXPECTED_PRECISION:
        raise RuntimeError("detector precision is not shared 32-true")

    required = (
        repo / "data/chips",
        repo / "data/raw/xview3/GRD",
        repo / "data/raw/xview3/labels/train.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    missing.extend(
        str(repo / "data/weights" / subdir)
        for subdir in WEIGHT_SUBDIRS
        if not (repo / "data/weights" / subdir).is_dir()
    )
    if missing:
        raise RuntimeError("target payload is incomplete: " + ", ".join(missing))
    return {
        "git_sha": receipt["git_sha"],
        "frozen_sha256": actual_hashes,
    }


def run_logged(command: list[str], *, cwd: Path, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(
            command,
            cwd=cwd,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.monotonic() - started


def guarded_python(*args: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.h100.child_exec",
        "--",
        sys.executable,
        *args,
    ]


def validate_hardware_runtime_contracts(hardware: object) -> list[dict]:
    """Require one verified Lightning contract from every H100 child probe."""

    if not isinstance(hardware, Mapping):
        raise RuntimeError("H100 hardware probe is not a JSON object")
    children = hardware.get("child_probes")
    if not isinstance(children, list) or len(children) != 8:
        raise RuntimeError("H100 hardware probe lacks eight child runtime contracts")
    validated: list[dict] = []
    for index, child in enumerate(children):
        if not isinstance(child, Mapping):
            raise RuntimeError(f"H100 child probe {index} is not a JSON object")
        try:
            contract = validate_trainer_contract_evidence(
                child.get("runtime_contract")
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"H100 child probe {index} lacks a valid Lightning contract"
            ) from exc
        validated.append(contract)
    return validated


def validate_probe_marker(
    path: Path,
    expected_exp_id: str,
    *,
    expected_git_sha: str,
    expected_detector_sha256: str,
) -> dict:
    payload = json.loads(path.read_text())
    expected = {
        "exp_id": expected_exp_id,
        "precision": EXPECTED_PRECISION,
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "git_sha": expected_git_sha,
        "detector_sha256": expected_detector_sha256,
    }
    mismatches = {
        key: (value, payload.get(key))
        for key, value in expected.items()
        if payload.get(key) != value
    }
    for key in ("best_dev_f1", "train_loss"):
        try:
            finite = math.isfinite(float(payload[key]))
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite:
            mismatches[key] = ("finite", payload.get(key))
    try:
        validate_trainer_contract_evidence(payload.get("h100_runtime_contract"))
    except RuntimeError as exc:
        mismatches["h100_runtime_contract"] = ("valid", str(exc))
    if mismatches:
        raise RuntimeError(f"invalid strict-FP32 acceptance result {path}: {mismatches}")
    for name in ("best.ckpt", "last.ckpt"):
        checkpoint = path.parent / "checkpoints" / name
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"strict-FP32 acceptance result lacks {checkpoint}")
    return payload


def run_acceptance(args: argparse.Namespace) -> dict:
    assert_sitecustomize_active()
    repo = args.repo.resolve()
    runs_root = args.runs_root.resolve()
    scratch = args.scratch.resolve()
    ready = args.ready.resolve()

    validate_fresh_acceptance_state(
        ready=ready,
        runs_root=runs_root,
        remaining_v100_wall_hours=args.remaining_v100_wall_hours,
    )

    free_bytes = validate_scratch_free_before_extraction(
        args.scratch_free_before_extraction
    )
    expected_hashes = {
        relative: digest
        for relative, digest in zip(FROZEN_PATHS, args.frozen_sha256, strict=True)
    }
    expected_base_payload = {
        "package_id": args.base_payload_package_id,
        "git_sha": args.base_payload_git_sha,
        "manifest_sha256": args.base_payload_manifest_sha256,
        "ready_sha256": args.base_payload_ready_sha256,
        "sha256sums_sha256": args.base_payload_sha256sums_sha256,
        "repo_bundle_sha256": args.base_payload_repo_bundle_sha256,
    }
    expected_runtime_amendment = {
        "package_id": args.runtime_amendment_package_id,
        "git_sha": args.runtime_amendment_git_sha,
        "manifest_sha256": args.runtime_amendment_manifest_sha256,
        "ready_sha256": args.runtime_amendment_ready_sha256,
        "sha256sums_sha256": args.runtime_amendment_sha256sums_sha256,
        "runtime_bundle_sha256": args.runtime_amendment_bundle_sha256,
    }
    source_validation = validate_source_receipt(
        args.source_validation_json,
        expected_sha256=args.source_validation_sha256,
        expected_git_sha=args.expected_git_sha,
        expected_hashes=expected_hashes,
        expected_base_payload=expected_base_payload,
        expected_runtime_amendment=expected_runtime_amendment,
    )
    source = validate_runtime_source(
        repo=repo,
        receipt_path=args.source_validation_json,
        receipt_sha256=args.source_validation_sha256,
        expected_git_sha=args.expected_git_sha,
        expected_hashes=expected_hashes,
        expected_base_payload=expected_base_payload,
        expected_runtime_amendment=expected_runtime_amendment,
    )
    smoke_bindings = make_smoke_bindings(
        git_sha=args.expected_git_sha,
        detector_sha256=source["frozen_sha256"]["configs/detector.yaml"],
        venv_sha256=args.venv_sha256,
        venv_build_sha256=args.venv_build_sha256,
        base_python_sha256=args.base_python_sha256,
        base_python_runtime_sha256=args.base_python_runtime_sha256,
        wheelhouse_sha256=args.wheelhouse_sha256,
        base_extraction_receipt_sha256=(
            args.base_extraction_receipt_sha256
        ),
        base_payload=expected_base_payload,
        runtime_amendment=expected_runtime_amendment,
    )
    smoke = validate_smoke_receipt(
        args.smoke_ready.resolve(),
        expected_bindings=smoke_bindings,
    )
    smoke_sha256 = sha256_file(args.smoke_ready)
    assert_empty_core_namespaces(repo, runs_root)
    venv_receipt = verify_native_venv(
        repo=repo,
        venv_root=args.venv_root,
        base_python=args.base_python,
        wheelhouse=args.wheelhouse,
        base_extraction_receipt=args.base_extraction_receipt,
        expected_venv_sha256=args.venv_sha256,
        expected_receipt_sha256=args.venv_build_sha256,
        expected_base_python_sha256=args.base_python_sha256,
        expected_base_python_runtime_sha256=args.base_python_runtime_sha256,
        expected_wheelhouse_sha256=args.wheelhouse_sha256,
        expected_base_extraction_receipt_sha256=(
            args.base_extraction_receipt_sha256
        ),
        expected_base_payload_package_id=expected_base_payload["package_id"],
        expected_base_payload_manifest_sha256=(
            expected_base_payload["manifest_sha256"]
        ),
    )
    staged_payload = scratch / "payload"
    staged_base_extraction = validate_base_extraction_receipt(
        staged_payload / "HANDOFF_EXTRACTED.json",
        wheelhouse=staged_payload / "environment/wheelhouse",
        expected_package_id=expected_base_payload["package_id"],
        expected_manifest_sha256=expected_base_payload["manifest_sha256"],
        expected_receipt_sha256=args.base_extraction_receipt_sha256,
        expected_wheelhouse_sha256=args.wheelhouse_sha256,
    )
    if (
        staged_base_extraction["receipt"].get("wheelhouse")
        != venv_receipt.get("wheelhouse", {}).get("identity")
    ):
        raise RuntimeError(
            "fresh scratch extraction wheelhouse differs from native-venv build"
        )
    meta_root = runs_root / ".h100"
    persisted_receipt = meta_root / "venv_build.json"
    if (
        persisted_receipt.is_symlink()
        or not persisted_receipt.is_file()
        or sha256_file(persisted_receipt) != args.venv_build_sha256
        or json.loads(persisted_receipt.read_text()) != venv_receipt
    ):
        raise RuntimeError("persisted native-venv build receipt binding mismatch")
    logs = meta_root / "acceptance-logs"
    host_test = validate_host_gate(
        args.host_test_receipt,
        expected_sha256=args.host_test_receipt_sha256,
        expected_source_validation_sha256=args.source_validation_sha256,
    )
    venv_test_command = [
        "-m",
        "pytest",
        "-q",
        *(f"--ignore={path}" for path in HOST_TESTS),
    ]
    venv_test_seconds = run_logged(
        guarded_python(*venv_test_command),
        cwd=repo,
        log_path=logs / "pytest-venv-remaining.log",
    )
    venv_test_log = logs / "pytest-venv-remaining.log"
    test_seconds = float(host_test["duration_seconds"]) + venv_test_seconds
    test_suite = {
        "schema": 2,
        "status": "passed",
        "source_validation_sha256": args.source_validation_sha256,
        "coverage": {
            "host": HOST_TESTS,
            "venv": "all pytest collection except the two host-only files",
            "aggregate": "entire repository pytest suite",
        },
        "host_handoff": {
            "receipt_path": str(args.host_test_receipt.absolute()),
            "receipt_sha256": args.host_test_receipt_sha256,
            "receipt": host_test,
        },
        "venv_remaining": {
            "command": venv_test_command,
            "duration_seconds": venv_test_seconds,
            "log": {
                "path": str(venv_test_log),
                "sha256": sha256_file(venv_test_log),
            },
        },
        "aggregate_duration_seconds": test_seconds,
    }
    test_suite_path = meta_root / "PYTEST_ACCEPTANCE.json"
    atomic_write_json(test_suite_path, test_suite)
    hardware_seconds = time.monotonic()
    completed = subprocess.run(
        guarded_python(
            "-m",
            "scripts.h100.strict_fp32_probe",
            "--expected-gpus",
            "8",
        ),
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    hardware_seconds = time.monotonic() - hardware_seconds
    hardware = json.loads(completed.stdout)
    validate_hardware_runtime_contracts(hardware)
    atomic_write_json(meta_root / "h100_runtime.json", hardware)

    # Both families cross a real forward/backward/dev-inference path.  The
    # CNN f100 gate is exactly 200 optimizer steps and supplies the projection.
    vit_seconds = run_logged(
        guarded_python(
            "-m",
            "src.train.finetune",
            "--init",
            "vit_imagenet",
            "--label_frac",
            "1.0",
            "--seed",
            "0",
            "--git-sha",
            args.expected_git_sha,
            "--epochs",
            "1",
            "--samples-per-epoch",
            "16",
            "--dev-every",
            "1",
            "--n-dev-scenes",
            "1",
            "--workers",
            "6",
            "--micro-batch",
            "16",
            "--exp-suffix",
            "h100-acceptance",
        ),
        cwd=repo,
        log_path=logs / "vit-fp32.log",
    )
    validate_probe_marker(
        runs_root / "vitin1k-f100-s0-h100-acceptance/final_metrics.json",
        "vitin1k-f100-s0-h100-acceptance",
        expected_git_sha=args.expected_git_sha,
        expected_detector_sha256=source["frozen_sha256"]["configs/detector.yaml"],
    )
    probe_seconds = run_logged(
        guarded_python(
            "-m",
            "src.train.finetune",
            "--init",
            "cnn_imagenet",
            "--label_frac",
            "1.0",
            "--seed",
            "0",
            "--git-sha",
            args.expected_git_sha,
            "--epochs",
            "1",
            "--samples-per-epoch",
            "3200",
            "--dev-every",
            "1",
            "--n-dev-scenes",
            "1",
            "--workers",
            "6",
            "--micro-batch",
            "16",
            "--exp-suffix",
            "h100-200step",
        ),
        cwd=repo,
        log_path=logs / "cnn-200step-fp32.log",
    )
    validate_probe_marker(
        runs_root / "cnnin1k-f100-s0-h100-200step/final_metrics.json",
        "cnnin1k-f100-s0-h100-200step",
        expected_git_sha=args.expected_git_sha,
        expected_detector_sha256=source["frozen_sha256"]["configs/detector.yaml"],
    )
    fraction_workload = derive_fraction_workload(
        chips_root=repo / "data/chips",
        splits_path=repo / "data/splits.json",
    )
    exact_steps = {
        label: int(item["steps_per_epoch"])
        for label, item in fraction_workload.items()
    }
    projection = estimate_grid_projection(
        probe_steps=200,
        probe_seconds=probe_seconds,
        steps_per_epoch=exact_steps,
    )
    projection.update(
        {
            "probe_steps": 200,
            "probe_wall_seconds": probe_seconds,
            "fraction_workload": fraction_workload,
            "note": (
                "Conservative end-to-end timing includes the probe's one-scene "
                "dev evaluation; live telemetry supersedes this projection."
            ),
        }
    )
    longest_cell_hours = (
        exact_steps["f100"]
        * 50
        / projection["steps_per_second"]
        / 3600.0
    )
    training_wall_hours = max(
        projection["ceiling_wall_hours_ideal"], longest_cell_hours
    )
    wall_clock = staging_aware_wall_clock(
        training_wall_hours=training_wall_hours,
        staging_seconds=args.staging_seconds,
    )
    conservative_h100_wall_hours = float(
        wall_clock["conservative_h100_wall_hours"]
    )
    projection.update(
        {
            "longest_f100_ceiling_hours": longest_cell_hours,
            "remaining_v100_wall_hours": args.remaining_v100_wall_hours,
            **wall_clock,
        }
    )
    if conservative_h100_wall_hours >= args.remaining_v100_wall_hours:
        raise RuntimeError(
            "H100 cutover rejected: conservative projection "
            f"{conservative_h100_wall_hours:.2f}h does not beat remaining V100 "
            f"{args.remaining_v100_wall_hours:.2f}h"
        )
    atomic_write_json(meta_root / "throughput_projection.json", projection)

    payload = {
        "schema": 2,
        "status": "ready",
        "acceptance_uuid": str(uuid.uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        },
        "source": source,
        "source_validation": {
            "path": str(args.source_validation_json.absolute()),
            "sha256": args.source_validation_sha256,
            "receipt": source_validation,
        },
        "strict_fp32": hardware["backend"],
        "hardware": hardware,
        "venv": {
            "path": str(args.venv_root.absolute()),
            "sha256": args.venv_sha256,
            "venv_build_sha256": args.venv_build_sha256,
            "base_python": venv_receipt["base_python"],
            "wheelhouse": venv_receipt["wheelhouse"],
            "staged_base_extraction": staged_base_extraction,
        },
        "base_payload": expected_base_payload,
        "runtime_amendment": expected_runtime_amendment,
        "slurm_smoke": {
            "path": str(args.smoke_ready.resolve()),
            "sha256": smoke_sha256,
            "receipt": smoke,
        },
        "scratch_free_bytes": free_bytes,
        "test_suite": {
            "path": str(test_suite_path),
            "sha256": sha256_file(test_suite_path),
            "receipt": test_suite,
        },
        "gates": {
            "pytest_seconds": test_seconds,
            "hardware_probe_seconds": hardware_seconds,
            "vit_gate_seconds": vit_seconds,
            "cnn_200step_seconds": probe_seconds,
        },
        "projection": projection,
    }
    if ready.exists() or ready.is_symlink():
        raise RuntimeError("H100_READY.json appeared during acceptance; refusing overwrite")
    atomic_write_json(ready, payload)
    ready.chmod(0o444)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--scratch-free-before-extraction", type=int, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--smoke-ready", type=Path, required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--venv-sha256", required=True)
    parser.add_argument("--venv-build-sha256", required=True)
    parser.add_argument("--base-python", type=Path, required=True)
    parser.add_argument("--base-python-sha256", required=True)
    parser.add_argument("--base-python-runtime-sha256", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--wheelhouse-sha256", required=True)
    parser.add_argument("--base-extraction-receipt", type=Path, required=True)
    parser.add_argument("--base-extraction-receipt-sha256", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--source-validation-json", type=Path, required=True)
    parser.add_argument("--source-validation-sha256", required=True)
    parser.add_argument("--host-test-receipt", type=Path, required=True)
    parser.add_argument("--host-test-receipt-sha256", required=True)
    parser.add_argument(
        "--frozen-sha256",
        action="append",
        required=True,
        help="repeat in FROZEN_PATHS order: detector, scorer, splits, stats, lsssdd",
    )
    parser.add_argument("--base-payload-package-id", required=True)
    parser.add_argument("--base-payload-git-sha", required=True)
    parser.add_argument("--base-payload-manifest-sha256", required=True)
    parser.add_argument("--base-payload-ready-sha256", required=True)
    parser.add_argument("--base-payload-sha256sums-sha256", required=True)
    parser.add_argument("--base-payload-repo-bundle-sha256", required=True)
    parser.add_argument("--runtime-amendment-package-id", required=True)
    parser.add_argument("--runtime-amendment-git-sha", required=True)
    parser.add_argument("--runtime-amendment-manifest-sha256", required=True)
    parser.add_argument("--runtime-amendment-ready-sha256", required=True)
    parser.add_argument("--runtime-amendment-sha256sums-sha256", required=True)
    parser.add_argument("--runtime-amendment-bundle-sha256", required=True)
    parser.add_argument("--remaining-v100-wall-hours", type=float, required=True)
    parser.add_argument("--staging-seconds", type=float, required=True)
    args = parser.parse_args()
    if len(args.frozen_sha256) != len(FROZEN_PATHS):
        parser.error(f"--frozen-sha256 must be repeated {len(FROZEN_PATHS)} times")
    print(json.dumps(run_acceptance(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
