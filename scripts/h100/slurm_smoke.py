"""Exercise the Slurm SIGUSR1/checkpoint/requeue/resume contract cheaply.

The synthetic cell never uses a reportable experiment namespace.  Its first
allocation takes a real SIGUSR1, writes a Lightning-shaped HPC checkpoint,
and atomically promotes it to ``last.ckpt``.  It then exits with a reserved
code so the *host* batch script can perform exactly one real ``scontrol
requeue``.  The requeued allocation resumes the same synthetic cell and
writes a persistent, provenance-bound readiness receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from scripts.h100.campaign import promote_hpc_checkpoint
from scripts.h100.contracts import atomic_write_json, sha256_file

SCHEMA = 1
SYNTHETIC_CELL_ID = "h100-slurm-smoke-synthetic-v1"
HOST_REQUEUE_EXIT_CODE = 75
STATE_NAME = "SLURM_SMOKE_STATE.json"
READY_NAME = "SLURM_SMOKE_READY.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
PACKAGE_HASH_KEYS = (
    "manifest_sha256",
    "ready_sha256",
    "sha256sums_sha256",
    "repo_bundle_sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def smoke_root(runs_root: Path) -> Path:
    return runs_root / ".h100/slurm-smoke"


def make_bindings(
    *,
    git_sha: str,
    detector_sha256: str,
    sif_sha256: str,
    container_build_sha256: str,
    package_manifest_sha256: str,
    package_ready_sha256: str,
    package_sha256sums_sha256: str,
    package_repo_bundle_sha256: str,
) -> dict:
    bindings = {
        "source": {
            "git_sha": git_sha,
            "detector_sha256": detector_sha256,
        },
        "sif": {
            "sha256": sif_sha256,
            "container_build_sha256": container_build_sha256,
        },
        "package": {
            "manifest_sha256": package_manifest_sha256,
            "ready_sha256": package_ready_sha256,
            "sha256sums_sha256": package_sha256sums_sha256,
            "repo_bundle_sha256": package_repo_bundle_sha256,
        },
    }
    if not HEX40.fullmatch(git_sha):
        raise RuntimeError("Slurm smoke requires the full 40-hex source git SHA")
    hash_values = [
        detector_sha256,
        sif_sha256,
        container_build_sha256,
        package_manifest_sha256,
        package_ready_sha256,
        package_sha256sums_sha256,
        package_repo_bundle_sha256,
    ]
    if any(not HEX64.fullmatch(value) for value in hash_values):
        raise RuntimeError("Slurm smoke bindings require lowercase 64-hex SHA-256 values")
    return bindings


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Slurm smoke JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Slurm smoke JSON is not an object: {path}")
    return payload


def _require_bindings(payload: Mapping[str, object], expected: Mapping[str, object]) -> None:
    actual = payload.get("bindings")
    if actual != expected:
        raise RuntimeError("Slurm smoke provenance bindings mismatch")


def validate_smoke_receipt(path: Path, *, expected_bindings: Mapping[str, object]) -> dict:
    """Fail closed unless ``path`` proves one interruption and one requeue."""

    payload = _read_json(path)
    _require_bindings(payload, expected_bindings)
    if payload.get("schema") != SCHEMA or payload.get("status") != "ready":
        raise RuntimeError("Slurm smoke receipt is not ready")
    if payload.get("signal") != {
        "name": "SIGUSR1",
        "count": 1,
        "handled": True,
    }:
        raise RuntimeError("Slurm smoke did not prove exactly one handled SIGUSR1")
    if payload.get("real_requeue_count") != 1:
        raise RuntimeError("Slurm smoke did not prove exactly one real requeue")

    allocations = payload.get("allocations")
    if not isinstance(allocations, list) or len(allocations) != 2:
        raise RuntimeError("Slurm smoke must span exactly two allocations")
    allocation_keys = {
        (item.get("job_id"), item.get("restart_count"))
        for item in allocations
        if isinstance(item, dict)
    }
    if len(allocation_keys) != 2:
        raise RuntimeError("Slurm smoke allocation identities are not unique")
    first, second = allocations
    if first.get("job_id") != second.get("job_id"):
        raise RuntimeError("Slurm requeue unexpectedly changed the Slurm job ID")
    if int(second.get("restart_count", -1)) <= int(first.get("restart_count", -1)):
        raise RuntimeError("second Slurm allocation did not advance restart_count")

    synthetic = payload.get("synthetic")
    if not isinstance(synthetic, dict):
        raise RuntimeError("Slurm smoke synthetic-cell evidence is absent")
    if synthetic.get("cell_ids") != [SYNTHETIC_CELL_ID]:
        raise RuntimeError("Slurm smoke created duplicate or unexpected synthetic cells")
    if synthetic.get("launch_count") != 2:
        raise RuntimeError("synthetic cell did not launch exactly once per allocation")

    checkpoint = payload.get("checkpoint")
    resume = payload.get("resume")
    if not isinstance(checkpoint, dict) or not isinstance(resume, dict):
        raise RuntimeError("Slurm smoke checkpoint/resume evidence is absent")
    if checkpoint.get("atomic_promotion") is not True:
        raise RuntimeError("Slurm smoke checkpoint was not atomically promoted")
    if checkpoint.get("hpc_sha256") != checkpoint.get("last_sha256"):
        raise RuntimeError("promoted checkpoint bytes differ from the HPC checkpoint")
    if resume.get("from_last_ckpt") is not True:
        raise RuntimeError("second Slurm allocation did not resume from last.ckpt")
    if int(resume.get("resumed_step", -1)) != int(checkpoint.get("step", -2)):
        raise RuntimeError("Slurm smoke resumed from the wrong checkpoint step")
    if int(resume.get("final_step", -1)) <= int(resume.get("resumed_step", -1)):
        raise RuntimeError("Slurm smoke made no progress after resume")
    return payload


def _allocation(job_id: str, restart_count: int) -> dict:
    if not job_id:
        raise RuntimeError("SLURM_JOB_ID is required for the Slurm smoke")
    if restart_count < 0:
        raise RuntimeError("SLURM_RESTART_COUNT cannot be negative")
    return {
        "job_id": job_id,
        "restart_count": restart_count,
        "started_utc": utc_now(),
    }


def _write_synthetic_checkpoint(run_dir: Path, step: int) -> Path:
    checkpoint = run_dir / "lightning/hpc_ckpt_1.ckpt"
    atomic_write_json(
        checkpoint,
        {
            "schema": SCHEMA,
            "cell_id": SYNTHETIC_CELL_ID,
            "step": step,
            "signal": "SIGUSR1",
            "created_utc": utc_now(),
        },
    )
    return checkpoint


def _first_allocation(
    *,
    root: Path,
    bindings: Mapping[str, object],
    job_id: str,
    restart_count: int,
    trigger_step: int,
) -> int:
    state_path = root / STATE_NAME
    if state_path.exists():
        raise RuntimeError(
            "Slurm smoke state already exists; refusing a duplicate first allocation"
        )
    run_dir = root / "synthetic-cell"
    allocation = _allocation(job_id, restart_count)
    state = {
        "schema": SCHEMA,
        "status": "first-allocation-running",
        "created_utc": utc_now(),
        "bindings": bindings,
        "allocations": [allocation],
        "real_requeue_count": 0,
        "synthetic": {
            "cell_ids": [SYNTHETIC_CELL_ID],
            "launch_count": 1,
        },
    }
    atomic_write_json(state_path, state)

    handled: list[int] = []
    prior_handler = signal.getsignal(signal.SIGUSR1)

    def on_usr1(signum, _frame) -> None:
        handled.append(signum)
        _write_synthetic_checkpoint(run_dir, trigger_step)

    signal.signal(signal.SIGUSR1, on_usr1)
    try:
        # This is a real POSIX signal and the checkpoint is written by its
        # handler, matching the interruption path rather than a direct call.
        os.kill(os.getpid(), signal.SIGUSR1)
    finally:
        signal.signal(signal.SIGUSR1, prior_handler)
    if handled != [signal.SIGUSR1]:
        raise RuntimeError("synthetic cell did not handle exactly one SIGUSR1")

    hpc = run_dir / "lightning/hpc_ckpt_1.ckpt"
    promoted = promote_hpc_checkpoint(run_dir)
    hpc_sha = sha256_file(hpc)
    last_sha = sha256_file(promoted)
    if hpc_sha != last_sha or promoted.with_suffix(".ckpt.promoting").exists():
        raise RuntimeError("atomic checkpoint promotion verification failed")

    state.update(
        {
            "status": "host-requeue-required",
            "signal": {"name": "SIGUSR1", "count": 1, "handled": True},
            "checkpoint": {
                "step": trigger_step,
                "hpc_sha256": hpc_sha,
                "last_sha256": last_sha,
                "atomic_promotion": True,
            },
            "updated_utc": utc_now(),
        }
    )
    atomic_write_json(state_path, state)
    return HOST_REQUEUE_EXIT_CODE


def authorize_host_requeue(
    *,
    root: Path,
    bindings: Mapping[str, object],
    job_id: str,
    restart_count: int,
) -> dict:
    """Persist a one-shot token before the host invokes real ``scontrol``."""

    state_path = root / STATE_NAME
    state = _read_json(state_path)
    _require_bindings(state, bindings)
    if state.get("status") != "host-requeue-required":
        raise RuntimeError("host requeue is not in the one-shot required state")
    allocation = state.get("allocations", [{}])[0]
    if (allocation.get("job_id"), allocation.get("restart_count")) != (
        job_id,
        restart_count,
    ):
        raise RuntimeError("host requeue authorization allocation mismatch")
    if state.get("real_requeue_count") != 0:
        raise RuntimeError("real Slurm requeue was already authorized")
    state.update(
        {
            "status": "host-requeue-authorized",
            "real_requeue_count": 1,
            "requeue_authorized_utc": utc_now(),
        }
    )
    atomic_write_json(state_path, state)
    return state


def _second_allocation(
    *,
    root: Path,
    bindings: Mapping[str, object],
    job_id: str,
    restart_count: int,
    finish_step: int,
) -> int:
    state_path = root / STATE_NAME
    ready_path = root / READY_NAME
    state = _read_json(state_path)
    _require_bindings(state, bindings)
    if state.get("status") != "host-requeue-authorized":
        raise RuntimeError("second allocation arrived without one host requeue authorization")
    if state.get("real_requeue_count") != 1:
        raise RuntimeError("second allocation does not have exactly one requeue")
    first = state.get("allocations", [{}])[0]
    if first.get("job_id") != job_id:
        raise RuntimeError("requeued smoke allocation changed Slurm job ID")
    if restart_count <= int(first.get("restart_count", -1)):
        raise RuntimeError("requeued smoke allocation did not advance restart_count")

    run_dir = root / "synthetic-cell"
    last = run_dir / "checkpoints/last.ckpt"
    checkpoint = _read_json(last)
    if checkpoint.get("cell_id") != SYNTHETIC_CELL_ID:
        raise RuntimeError("last.ckpt belongs to the wrong synthetic cell")
    resumed_step = int(checkpoint.get("step", -1))
    if resumed_step < 0 or finish_step <= resumed_step:
        raise RuntimeError("synthetic resume target must advance beyond last.ckpt")

    second = _allocation(job_id, restart_count)
    state["allocations"].append(second)
    state["synthetic"]["launch_count"] = 2
    state.update(
        {
            "status": "second-allocation-running",
            "resume": {
                "from_last_ckpt": True,
                "resumed_step": resumed_step,
                "final_step": finish_step,
            },
            "updated_utc": utc_now(),
        }
    )
    atomic_write_json(state_path, state)

    final_marker = run_dir / "synthetic_final.json"
    atomic_write_json(
        final_marker,
        {
            "schema": SCHEMA,
            "cell_id": SYNTHETIC_CELL_ID,
            "resumed_step": resumed_step,
            "final_step": finish_step,
            "completed_utc": utc_now(),
        },
    )
    receipt = {
        "schema": SCHEMA,
        "status": "ready",
        "created_utc": utc_now(),
        "bindings": bindings,
        "allocations": state["allocations"],
        "real_requeue_count": 1,
        "signal": state["signal"],
        "checkpoint": state["checkpoint"],
        "resume": state["resume"],
        "synthetic": state["synthetic"],
        "final_marker_sha256": sha256_file(final_marker),
    }
    atomic_write_json(ready_path, receipt)
    validate_smoke_receipt(ready_path, expected_bindings=bindings)
    state.update(
        {
            "status": "complete",
            "ready_sha256": sha256_file(ready_path),
            "completed_utc": utc_now(),
        }
    )
    atomic_write_json(state_path, state)
    return 0


def run_allocation(
    *,
    runs_root: Path,
    bindings: Mapping[str, object],
    job_id: str,
    restart_count: int,
    trigger_step: int = 3,
    finish_step: int = 5,
) -> int:
    root = smoke_root(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    ready = root / READY_NAME
    if ready.exists():
        validate_smoke_receipt(ready, expected_bindings=bindings)
        return 0
    state_path = root / STATE_NAME
    if not state_path.exists():
        return _first_allocation(
            root=root,
            bindings=bindings,
            job_id=job_id,
            restart_count=restart_count,
            trigger_step=trigger_step,
        )
    return _second_allocation(
        root=root,
        bindings=bindings,
        job_id=job_id,
        restart_count=restart_count,
        finish_step=finish_step,
    )


def _binding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--sif-sha256", required=True)
    parser.add_argument("--container-build-sha256", required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--package-ready-sha256", required=True)
    parser.add_argument("--package-sha256sums-sha256", required=True)
    parser.add_argument("--package-repo-bundle-sha256", required=True)


def _bindings_from_args(args: argparse.Namespace) -> dict:
    return make_bindings(
        git_sha=args.git_sha,
        detector_sha256=args.detector_sha256,
        sif_sha256=args.sif_sha256,
        container_build_sha256=args.container_build_sha256,
        package_manifest_sha256=args.package_manifest_sha256,
        package_ready_sha256=args.package_ready_sha256,
        package_sha256sums_sha256=args.package_sha256sums_sha256,
        package_repo_bundle_sha256=args.package_repo_bundle_sha256,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "authorize-requeue", "validate"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--runs-root", type=Path, required=True)
        _binding_arguments(subparser)
    run_parser = subparsers.choices["run"]
    run_parser.add_argument("--job-id", required=True)
    run_parser.add_argument("--restart-count", type=int, required=True)
    run_parser.add_argument("--trigger-step", type=int, default=3)
    run_parser.add_argument("--finish-step", type=int, default=5)
    authorize = subparsers.choices["authorize-requeue"]
    authorize.add_argument("--job-id", required=True)
    authorize.add_argument("--restart-count", type=int, required=True)
    args = parser.parse_args()
    bindings = _bindings_from_args(args)

    if args.command == "run":
        return run_allocation(
            runs_root=args.runs_root,
            bindings=bindings,
            job_id=args.job_id,
            restart_count=args.restart_count,
            trigger_step=args.trigger_step,
            finish_step=args.finish_step,
        )
    root = smoke_root(args.runs_root)
    if args.command == "authorize-requeue":
        payload = authorize_host_requeue(
            root=root,
            bindings=bindings,
            job_id=args.job_id,
            restart_count=args.restart_count,
        )
    else:
        payload = validate_smoke_receipt(
            root / READY_NAME,
            expected_bindings=bindings,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
