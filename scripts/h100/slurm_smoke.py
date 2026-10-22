"""Exercise the native-venv Slurm SIGUSR1/requeue/resume contract cheaply.

The synthetic cell never uses a reportable experiment namespace.  Its first
allocation publishes its PID and waits boundedly for a real SIGUSR1 from the
host batch script.  The signal handler writes a Lightning-shaped HPC
checkpoint and atomically promotes it to ``last.ckpt``.  The process then
exits with a reserved code so the host can perform exactly one real
``scontrol requeue``.  The requeued allocation resumes the same synthetic
cell and writes a persistent, provenance-bound readiness receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from scripts.h100.campaign import promote_hpc_checkpoint
from scripts.h100.contracts import atomic_write_json, atomic_write_text, sha256_file

SCHEMA = 2
SYNTHETIC_CELL_ID = "h100-slurm-smoke-native-venv-synthetic-v2"
HOST_REQUEUE_EXIT_CODE = 75
STATE_NAME = "SLURM_SMOKE_STATE.json"
READY_NAME = "SLURM_SMOKE_READY.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
SIGNAL_READY_NAME = "SLURM_SMOKE_SIGNAL_READY.pid"
SIGNAL_ORIGIN = "slurm-batch-to-native-venv-child"
DEFAULT_SIGNAL_TIMEOUT_SECONDS = 30.0
SYNTHETIC_DEV_SCENE_EVAL_STATE = {
    "best": 0.5,
    "best_result": {
        "epoch": 4,
        "f1": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "ignored_predictions": 2,
        "threshold": 0.4,
        "n_candidates": 4,
    },
    "last_result": {
        "epoch": 9,
        "f1": 0.4,
        "precision": 1.0 / 3.0,
        "recall": 0.5,
        "tp": 1,
        "fp": 2,
        "fn": 1,
        "ignored_predictions": 1,
        "threshold": 0.6,
        "n_candidates": 4,
    },
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
BASE_TRANSFER_KEYS = {
    "package_id",
    "git_sha",
    "manifest_sha256",
    "ready_sha256",
    "sha256sums_sha256",
    "repo_bundle_sha256",
}
RUNTIME_TRANSFER_KEYS = {
    "package_id",
    "git_sha",
    "manifest_sha256",
    "ready_sha256",
    "sha256sums_sha256",
    "runtime_bundle_sha256",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def smoke_root(runs_root: Path) -> Path:
    return runs_root / ".h100/slurm-smoke"


def make_bindings(
    *,
    git_sha: str,
    detector_sha256: str,
    venv_sha256: str,
    venv_build_sha256: str,
    base_python_sha256: str,
    base_python_runtime_sha256: str,
    wheelhouse_sha256: str,
    base_extraction_receipt_sha256: str,
    base_payload: Mapping[str, str],
    runtime_amendment: Mapping[str, str],
) -> dict:
    bindings = {
        "source": {
            "git_sha": git_sha,
            "detector_sha256": detector_sha256,
        },
        "venv": {
            "sha256": venv_sha256,
            "build_receipt_sha256": venv_build_sha256,
            "base_python_sha256": base_python_sha256,
            "base_python_runtime_sha256": base_python_runtime_sha256,
            "wheelhouse_sha256": wheelhouse_sha256,
            "base_extraction_receipt_sha256": base_extraction_receipt_sha256,
        },
        "base_payload": dict(base_payload),
        "runtime_amendment": dict(runtime_amendment),
    }
    _validate_bindings(bindings)
    return bindings


def _validate_transfer_binding(
    value: object,
    *,
    label: str,
    keys: set[str],
    bundle_key: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"Slurm smoke {label} binding keys are invalid")
    git_sha = value["git_sha"]
    package_id = value["package_id"]
    if not isinstance(git_sha, str) or not HEX40.fullmatch(git_sha):
        raise RuntimeError(f"Slurm smoke {label} Git SHA is invalid")
    expected_id = (
        f"xview3-h100-fp32-{git_sha}"
        if label == "base-payload"
        else rf"xview3-h100-runtime-{git_sha}-[0-9a-f]{{64}}"
    )
    if not isinstance(package_id, str) or (
        package_id != expected_id
        if label == "base-payload"
        else re.fullmatch(expected_id, package_id) is None
    ):
        raise RuntimeError(f"Slurm smoke {label} package ID is invalid")
    hash_keys = keys - {"package_id", "git_sha"}
    if bundle_key not in hash_keys or any(
        not isinstance(value[key], str) or not HEX64.fullmatch(value[key])
        for key in hash_keys
    ):
        raise RuntimeError(
            f"Slurm smoke {label} bindings require lowercase 64-hex SHA-256 values"
        )


def _validate_bindings(bindings: object) -> None:
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source",
        "venv",
        "base_payload",
        "runtime_amendment",
    }:
        raise RuntimeError("Slurm smoke binding groups are invalid")
    source = bindings["source"]
    venv = bindings["venv"]
    if not isinstance(source, Mapping) or set(source) != {
        "git_sha",
        "detector_sha256",
    }:
        raise RuntimeError("Slurm smoke source binding keys are invalid")
    if not isinstance(venv, Mapping) or set(venv) != {
        "sha256",
        "build_receipt_sha256",
        "base_python_sha256",
        "base_python_runtime_sha256",
        "wheelhouse_sha256",
        "base_extraction_receipt_sha256",
    }:
        raise RuntimeError("Slurm smoke native-venv binding keys are invalid")
    git_sha = source["git_sha"]
    detector_sha256 = source["detector_sha256"]
    if not isinstance(git_sha, str) or not HEX40.fullmatch(git_sha):
        raise RuntimeError("Slurm smoke requires the full 40-hex source Git SHA")
    hash_values = [detector_sha256, *venv.values()]
    if any(
        not isinstance(value, str) or not HEX64.fullmatch(value)
        for value in hash_values
    ):
        raise RuntimeError(
            "Slurm smoke bindings require lowercase 64-hex SHA-256 values"
        )
    _validate_transfer_binding(
        bindings["base_payload"],
        label="base-payload",
        keys=BASE_TRANSFER_KEYS,
        bundle_key="repo_bundle_sha256",
    )
    _validate_transfer_binding(
        bindings["runtime_amendment"],
        label="runtime-amendment",
        keys=RUNTIME_TRANSFER_KEYS,
        bundle_key="runtime_bundle_sha256",
    )
    if bindings["base_payload"]["git_sha"] == git_sha:
        raise RuntimeError("base payload and runtime amendment must have distinct Git SHAs")
    if bindings["runtime_amendment"]["git_sha"] != git_sha:
        raise RuntimeError("runtime amendment must bind the executing source Git SHA")


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Slurm smoke JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Slurm smoke JSON is not an object: {path}")
    return payload


def _require_bindings(payload: Mapping[str, object], expected: Mapping[str, object]) -> None:
    _validate_bindings(expected)
    actual = payload.get("bindings")
    _validate_bindings(actual)
    if actual != expected:
        raise RuntimeError("Slurm smoke provenance bindings mismatch")


def validate_smoke_receipt(path: Path, *, expected_bindings: Mapping[str, object]) -> dict:
    """Fail closed unless ``path`` proves one interruption and one requeue."""

    payload = _read_json(path)
    _require_bindings(payload, expected_bindings)
    if payload.get("schema") != SCHEMA or payload.get("status") != "ready":
        raise RuntimeError("Slurm smoke receipt is not ready")
    if payload.get("signal") != {
        "origin": SIGNAL_ORIGIN,
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
    if checkpoint.get("dev_scene_eval_state") != SYNTHETIC_DEV_SCENE_EVAL_STATE:
        raise RuntimeError("Slurm smoke checkpoint lacks exact DevSceneEval state")
    if resume.get("restored_dev_scene_eval_state") != SYNTHETIC_DEV_SCENE_EVAL_STATE:
        raise RuntimeError("Slurm smoke did not restore exact DevSceneEval state")
    if resume.get("dev_scene_eval_state_preserved") is not True:
        raise RuntimeError("Slurm smoke did not prove callback-state preservation")
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
            "callbacks": {"DevSceneEval": SYNTHETIC_DEV_SCENE_EVAL_STATE},
        },
    )
    return checkpoint


def _signal_ready_path(root: Path, requested: Path) -> Path:
    expected = Path(os.path.abspath(root / SIGNAL_READY_NAME))
    path = Path(os.path.abspath(requested))
    if path != expected:
        raise RuntimeError(f"external signal PID file must be exactly {expected}")
    if os.path.lexists(path):
        raise RuntimeError(
            f"external signal PID file already exists; refusing to replace it: {path}"
        )
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(
            "external signal PID file parent must be an existing non-symlink directory"
        )
    return path


def _wait_for_external_usr1(
    *,
    signal_ready: Path,
    run_dir: Path,
    trigger_step: int,
    timeout_seconds: float,
) -> dict[str, object]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise RuntimeError("external SIGUSR1 timeout must be finite and positive")

    handled: list[int] = []
    prior_usr1 = signal.getsignal(signal.SIGUSR1)
    prior_alarm = signal.getsignal(signal.SIGALRM)

    def on_usr1(signum, _frame) -> None:
        handled.append(signum)
        _write_synthetic_checkpoint(run_dir, trigger_step)

    def on_alarm(_signum, _frame) -> None:
        raise TimeoutError(
            f"timed out after {timeout_seconds:g}s waiting for external SIGUSR1"
        )

    signal.signal(signal.SIGUSR1, on_usr1)
    signal.signal(signal.SIGALRM, on_alarm)
    try:
        atomic_write_text(signal_ready, f"{os.getpid()}\n")
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        while not handled:
            signal.pause()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior_alarm)
        signal.signal(signal.SIGUSR1, prior_usr1)
        signal_ready.unlink(missing_ok=True)

    if handled != [signal.SIGUSR1]:
        raise RuntimeError("synthetic cell did not handle exactly one external SIGUSR1")
    return {
        "name": "SIGUSR1",
        "count": 1,
        "handled": True,
        "origin": SIGNAL_ORIGIN,
    }


def _first_allocation(
    *,
    root: Path,
    bindings: Mapping[str, object],
    job_id: str,
    restart_count: int,
    trigger_step: int,
    external_signal_ready: Path,
    signal_timeout_seconds: float,
) -> int:
    state_path = root / STATE_NAME
    if state_path.exists():
        raise RuntimeError(
            "Slurm smoke state already exists; refusing a duplicate first allocation"
        )
    signal_ready = _signal_ready_path(root, external_signal_ready)
    if not math.isfinite(signal_timeout_seconds) or signal_timeout_seconds <= 0:
        raise RuntimeError("external SIGUSR1 timeout must be finite and positive")

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

    signal_record = _wait_for_external_usr1(
        signal_ready=signal_ready,
        run_dir=run_dir,
        trigger_step=trigger_step,
        timeout_seconds=signal_timeout_seconds,
    )

    hpc = run_dir / "lightning/hpc_ckpt_1.ckpt"
    promoted = promote_hpc_checkpoint(run_dir)
    hpc_sha = sha256_file(hpc)
    last_sha = sha256_file(promoted)
    if hpc_sha != last_sha or promoted.with_suffix(".ckpt.promoting").exists():
        raise RuntimeError("atomic checkpoint promotion verification failed")
    checkpoint_payload = _read_json(promoted)
    callbacks = checkpoint_payload.get("callbacks")
    dev_scene_eval = (
        callbacks.get("DevSceneEval") if isinstance(callbacks, Mapping) else None
    )
    if dev_scene_eval != SYNTHETIC_DEV_SCENE_EVAL_STATE:
        raise RuntimeError("synthetic checkpoint lost DevSceneEval callback state")

    state.update(
        {
            "status": "host-requeue-required",
            "signal": signal_record,
            "checkpoint": {
                "step": trigger_step,
                "hpc_sha256": hpc_sha,
                "last_sha256": last_sha,
                "atomic_promotion": True,
                "dev_scene_eval_state": dev_scene_eval,
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
    if state.get("schema") != SCHEMA:
        raise RuntimeError("Slurm smoke state schema is unsupported")
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
    if state.get("schema") != SCHEMA:
        raise RuntimeError("Slurm smoke state schema is unsupported")
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
    callbacks = checkpoint.get("callbacks")
    restored_dev_scene_eval = (
        callbacks.get("DevSceneEval") if isinstance(callbacks, Mapping) else None
    )
    if (
        restored_dev_scene_eval != SYNTHETIC_DEV_SCENE_EVAL_STATE
        or state.get("checkpoint", {}).get("dev_scene_eval_state")
        != SYNTHETIC_DEV_SCENE_EVAL_STATE
    ):
        raise RuntimeError("DevSceneEval callback state drifted across requeue")

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
                "restored_dev_scene_eval_state": restored_dev_scene_eval,
                "dev_scene_eval_state_preserved": True,
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
    external_signal_ready: Path | None = None,
    signal_timeout_seconds: float = DEFAULT_SIGNAL_TIMEOUT_SECONDS,
) -> int:
    _validate_bindings(bindings)
    root = smoke_root(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    ready = root / READY_NAME
    if ready.exists():
        validate_smoke_receipt(ready, expected_bindings=bindings)
        return 0
    state_path = root / STATE_NAME
    if not state_path.exists():
        if external_signal_ready is None:
            raise RuntimeError(
                "first Slurm smoke allocation requires an external signal PID file"
            )
        return _first_allocation(
            root=root,
            bindings=bindings,
            job_id=job_id,
            restart_count=restart_count,
            trigger_step=trigger_step,
            external_signal_ready=external_signal_ready,
            signal_timeout_seconds=signal_timeout_seconds,
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
    parser.add_argument("--venv-sha256", required=True)
    parser.add_argument("--venv-build-sha256", required=True)
    parser.add_argument("--base-python-sha256", required=True)
    parser.add_argument("--base-python-runtime-sha256", required=True)
    parser.add_argument("--wheelhouse-sha256", required=True)
    parser.add_argument("--base-extraction-receipt-sha256", required=True)
    for prefix in ("base-payload", "runtime-amendment"):
        parser.add_argument(f"--{prefix}-package-id", required=True)
        parser.add_argument(f"--{prefix}-git-sha", required=True)
        parser.add_argument(f"--{prefix}-manifest-sha256", required=True)
        parser.add_argument(f"--{prefix}-ready-sha256", required=True)
        parser.add_argument(f"--{prefix}-sha256sums-sha256", required=True)
    parser.add_argument("--base-payload-repo-bundle-sha256", required=True)
    parser.add_argument("--runtime-amendment-bundle-sha256", required=True)


def _bindings_from_args(args: argparse.Namespace) -> dict:
    base_payload = {
        "package_id": args.base_payload_package_id,
        "git_sha": args.base_payload_git_sha,
        "manifest_sha256": args.base_payload_manifest_sha256,
        "ready_sha256": args.base_payload_ready_sha256,
        "sha256sums_sha256": args.base_payload_sha256sums_sha256,
        "repo_bundle_sha256": args.base_payload_repo_bundle_sha256,
    }
    runtime_amendment = {
        "package_id": args.runtime_amendment_package_id,
        "git_sha": args.runtime_amendment_git_sha,
        "manifest_sha256": args.runtime_amendment_manifest_sha256,
        "ready_sha256": args.runtime_amendment_ready_sha256,
        "sha256sums_sha256": args.runtime_amendment_sha256sums_sha256,
        "runtime_bundle_sha256": args.runtime_amendment_bundle_sha256,
    }
    return make_bindings(
        git_sha=args.git_sha,
        detector_sha256=args.detector_sha256,
        venv_sha256=args.venv_sha256,
        venv_build_sha256=args.venv_build_sha256,
        base_python_sha256=args.base_python_sha256,
        base_python_runtime_sha256=args.base_python_runtime_sha256,
        wheelhouse_sha256=args.wheelhouse_sha256,
        base_extraction_receipt_sha256=args.base_extraction_receipt_sha256,
        base_payload=base_payload,
        runtime_amendment=runtime_amendment,
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
    run_parser.add_argument("--external-signal-ready", type=Path, required=True)
    run_parser.add_argument(
        "--signal-timeout-seconds",
        type=float,
        default=DEFAULT_SIGNAL_TIMEOUT_SECONDS,
    )
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
            external_signal_ready=args.external_signal_ready,
            signal_timeout_seconds=args.signal_timeout_seconds,
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
