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
from scripts.h100.host_test_gate import HOST_TESTS, validate_host_gate
from scripts.h100.precision import assert_sitecustomize_active
from scripts.h100.slurm_smoke import (
    make_bindings as make_smoke_bindings,
    validate_smoke_receipt,
)
from scripts.h100.source_validation import validate_source_receipt

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
    expected_package: dict[str, str],
) -> dict:
    receipt = validate_source_receipt(
        receipt_path,
        expected_sha256=receipt_sha256,
        expected_git_sha=expected_git_sha,
        expected_hashes=expected_hashes,
        expected_package=expected_package,
    )
    # Rehash the frozen files inside the SIF.  Git cleanliness itself is the
    # host-only check recorded by SOURCE_VALIDATED.json.
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
    expected_package = {
        "manifest_sha256": args.package_manifest_sha256,
        "ready_sha256": args.package_ready_sha256,
        "sha256sums_sha256": args.package_sha256sums_sha256,
        "repo_bundle_sha256": args.package_repo_bundle_sha256,
    }
    source_validation = validate_source_receipt(
        args.source_validation_json,
        expected_sha256=args.source_validation_sha256,
        expected_git_sha=args.expected_git_sha,
        expected_hashes=expected_hashes,
        expected_package=expected_package,
    )
    source = validate_runtime_source(
        repo=repo,
        receipt_path=args.source_validation_json,
        receipt_sha256=args.source_validation_sha256,
        expected_git_sha=args.expected_git_sha,
        expected_hashes=expected_hashes,
        expected_package=expected_package,
    )
    smoke_bindings = make_smoke_bindings(
        git_sha=args.expected_git_sha,
        detector_sha256=source["frozen_sha256"]["configs/detector.yaml"],
        sif_sha256=args.sif_sha256,
        container_build_sha256=args.container_build_sha256,
        package_manifest_sha256=args.package_manifest_sha256,
        package_ready_sha256=args.package_ready_sha256,
        package_sha256sums_sha256=args.package_sha256sums_sha256,
        package_repo_bundle_sha256=args.package_repo_bundle_sha256,
    )
    smoke = validate_smoke_receipt(
        args.smoke_ready.resolve(),
        expected_bindings=smoke_bindings,
    )
    smoke_sha256 = sha256_file(args.smoke_ready)
    assert_empty_core_namespaces(repo, runs_root)
    if sha256_file(args.sif) != args.sif_sha256:
        raise RuntimeError("persistent SIF hash does not match the approved receipt")
    container_receipt = runs_root / ".h100/container_build.json"
    if sha256_file(container_receipt) != args.container_build_sha256:
        raise RuntimeError("persisted container build receipt hash mismatch")
    receipt = json.loads(container_receipt.read_text())
    receipt_expected = {
        "sif_sha256": (receipt.get("sif") or {}).get("sha256"),
        "base_oci": receipt.get("base_oci"),
        "environment_lock_sha256": receipt.get("environment_lock_sha256"),
    }
    receipt_actual = {
        "sif_sha256": args.sif_sha256,
        "base_oci": args.base_oci,
        "environment_lock_sha256": sha256_file(repo / "locks/env-v100node.txt"),
    }
    if receipt_expected != receipt_actual:
        raise RuntimeError(
            f"container build receipt bindings mismatch: {receipt_expected} != {receipt_actual}"
        )

    meta_root = runs_root / ".h100"
    logs = meta_root / "acceptance-logs"
    host_test = validate_host_gate(
        args.host_test_receipt,
        expected_sha256=args.host_test_receipt_sha256,
        expected_source_validation_sha256=args.source_validation_sha256,
    )
    sif_test_command = [
        "-m",
        "pytest",
        "-q",
        *(f"--ignore={path}" for path in HOST_TESTS),
    ]
    sif_test_seconds = run_logged(
        guarded_python(*sif_test_command),
        cwd=repo,
        log_path=logs / "pytest-sif-remaining.log",
    )
    sif_test_log = logs / "pytest-sif-remaining.log"
    test_seconds = float(host_test["duration_seconds"]) + sif_test_seconds
    test_suite = {
        "schema": 1,
        "status": "passed",
        "source_validation_sha256": args.source_validation_sha256,
        "coverage": {
            "host": HOST_TESTS,
            "sif": "all pytest collection except the two host-only files",
            "aggregate": "entire repository pytest suite",
        },
        "host_handoff": {
            "receipt_path": str(args.host_test_receipt.absolute()),
            "receipt_sha256": args.host_test_receipt_sha256,
            "receipt": host_test,
        },
        "sif_remaining": {
            "command": sif_test_command,
            "duration_seconds": sif_test_seconds,
            "log": {
                "path": str(sif_test_log),
                "sha256": sha256_file(sif_test_log),
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
        "schema": 1,
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
        "sif": {
            "path": str(args.sif.resolve()),
            "sha256": args.sif_sha256,
            "container_build_sha256": args.container_build_sha256,
            "base_oci": args.base_oci,
        },
        "package": {
            "manifest_sha256": args.package_manifest_sha256,
            "ready_sha256": args.package_ready_sha256,
            "sha256sums_sha256": args.package_sha256sums_sha256,
            "repo_bundle_sha256": args.package_repo_bundle_sha256,
        },
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
    parser.add_argument("--sif", type=Path, required=True)
    parser.add_argument("--sif-sha256", required=True)
    parser.add_argument("--container-build-sha256", required=True)
    parser.add_argument("--base-oci", required=True)
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
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--package-ready-sha256", required=True)
    parser.add_argument("--package-sha256sums-sha256", required=True)
    parser.add_argument("--package-repo-bundle-sha256", required=True)
    parser.add_argument("--remaining-v100-wall-hours", type=float, required=True)
    parser.add_argument("--staging-seconds", type=float, required=True)
    args = parser.parse_args()
    if len(args.frozen_sha256) != len(FROZEN_PATHS):
        parser.error(f"--frozen-sha256 must be repeated {len(FROZEN_PATHS)} times")
    print(json.dumps(run_acceptance(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
