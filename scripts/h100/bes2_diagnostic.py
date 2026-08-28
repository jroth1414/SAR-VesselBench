"""Judy-only staging, test gate, and requeue-safe BES2 pair controller."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.h100.campaign import promote_hpc_checkpoint
from scripts.h100.contracts import atomic_write_json, atomic_write_text
from scripts.h100.data_staging import (
    DATA_VIEW_RECEIPT,
    EXPECTED_WEIGHT_DIRS,
    TRAINING_VIEW_CONTRACT,
    validate_data_view,
)
from scripts.h100.lightning_contract import assert_launch_process_contract
from scripts.h100.precision import assert_sitecustomize_active
from src.analysis.bes2_contract import (
    BES2ContractError,
    DIAGNOSTIC_FRACTIONS,
    DIAGNOSTIC_VARIANTS,
    H100_CAMPAIGN_GIT_SHA,
    assert_no_final_consumption,
    diagnostic_run_id,
    fraction_tag,
    hash_binding,
    payload_sha256,
    read_regular_json,
    sha256_file,
    stage_fractions,
    utc_now,
    validate_diagnostic_metrics,
    write_new_immutable,
)
from src.analysis.bes2_evidence import (
    validate_data_scope,
    validate_h100_ready,
)
from src.analysis.bes2_root_cause import (
    READY_FILENAME,
    _scope_identity,
    main as root_cause_main,
    validate_readiness,
)

PREEMPTED_EXIT_CODE = 75
DATA_VIEW_READY = "BES2_DATA_VIEW_READY.json"
TEST_RECEIPT = "BES2_TEST_GATE.json"
EXECUTION_STATE = "BES2_EXECUTION_STATE.json"
PROBE_COMPLETE = "BES2_PROBES_COMPLETE.json"
EXECUTION_COMPLETE = "BES2_EXECUTION_COMPLETE.json"
REQUEST_TIMEOUT_SECONDS = 600.0
_PASSED = re.compile(r"(\d+) passed")


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BES2ContractError("could not resolve BES2 checkout SHA") from exc


def _require_source(repo: Path, expected_git_sha: str) -> str:
    observed = _git_sha(repo)
    if observed != expected_git_sha:
        raise BES2ContractError(
            f"BES2 checkout SHA mismatch: {observed} != {expected_git_sha}"
        )
    return observed


def _require_h100_devices(expected: int) -> list[str]:
    import torch

    count = int(torch.cuda.device_count())
    names = [str(torch.cuda.get_device_name(index)) for index in range(count)]
    if count != expected or any("H100" not in name for name in names):
        raise BES2ContractError(
            f"BES2 runtime requires {expected} visible H100(s), found {names}"
        )
    return names


def _visible_device_tokens(expected: int) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return [str(index) for index in range(expected)]
    tokens = [token.strip() for token in raw.split(",")]
    if len(tokens) != expected or any(not token for token in tokens):
        raise BES2ContractError(
            "CUDA_VISIBLE_DEVICES does not describe the exact BES2 allocation"
        )
    return tokens


def diagnostic_view_payload(
    *,
    source_git_sha: str,
    h100_ready_sha256: str,
    data_config_sha256: str,
    staging_receipt: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": 1,
        "status": "ready",
        "purpose": "bes2-diagnostic-train-dev-view",
        "source_git_sha": source_git_sha,
        "h100_ready_sha256": h100_ready_sha256,
        "data_config_sha256": data_config_sha256,
        "authorized_splits": ["train", "dev8"],
        "forbidden_splits": ["test", "eval_final"],
        "staging_receipt_sha256": payload_sha256(staging_receipt),
        "staging_receipt": dict(staging_receipt),
    }


def prepare_view(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    source_sha = _require_source(repo, args.expected_git_sha)
    assert_no_final_consumption(args.original_runs_root)
    h100 = validate_h100_ready(args.h100_ready)
    if h100["sha256"] != args.h100_ready_sha256:
        raise BES2ContractError("BES2 data view H100_READY binding mismatch")

    requested_root = args.data_view_root
    if not requested_root.is_absolute() or requested_root.is_symlink():
        raise BES2ContractError("BES2 data-view root must be absolute/non-symlink")
    root = requested_root.resolve(strict=True)
    if not root.is_dir():
        raise BES2ContractError("BES2 data-view root must be a non-symlink directory")
    staging_path = root / DATA_VIEW_RECEIPT
    if sha256_file(staging_path) != args.staging_receipt_sha256:
        raise BES2ContractError("H100 staged data-view receipt SHA-256 mismatch")
    staging = validate_data_view(
        root,
        repo=repo,
        runs_root=args.staging_runs_root,
        expected_sha256=args.staging_receipt_sha256,
        expected_git_sha=H100_CAMPAIGN_GIT_SHA,
        expected_phase="train",
        expected_purpose="campaign",
        expected_base_package_id=args.expected_base_package_id,
        expected_base_manifest_sha256=args.expected_base_manifest_sha256,
        expected_runtime_package_id=args.expected_runtime_package_id,
        expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
    )
    data_config = args.data_config.resolve(strict=True)
    config_payload = read_regular_json(
        repo / "data/splits.json", "frozen diagnostic splits"
    )
    if not isinstance(config_payload.get("splits"), Mapping):
        raise BES2ContractError("frozen diagnostic split mapping is absent")
    import yaml

    loaded_config = yaml.safe_load(data_config.read_text(encoding="utf-8"))
    if not isinstance(loaded_config, Mapping):
        raise BES2ContractError("diagnostic data config is not a mapping")
    scope = validate_data_scope(
        repo=repo,
        data_view_root=root,
        data_config=loaded_config,
    )
    if (
        staging.get("chips") != scope["train_scene_ids"]
        or staging.get("rasters") != scope["dev_scene_ids"]
        or staging.get("weights") != list(EXPECTED_WEIGHT_DIRS)
        or staging.get("training_cohort") is not None
        or staging.get("contract") != TRAINING_VIEW_CONTRACT
    ):
        raise BES2ContractError("staged view differs from exact TRAIN/DEV scope")

    output = args.output.absolute()
    if output != root / DATA_VIEW_READY:
        raise BES2ContractError(
            f"BES2 data-view receipt must be {root / DATA_VIEW_READY}"
        )
    payload = diagnostic_view_payload(
        source_git_sha=source_sha,
        h100_ready_sha256=h100["sha256"],
        data_config_sha256=sha256_file(data_config),
        staging_receipt=staging,
    )
    write_new_immutable(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _test_receipt_valid(path: Path, source_sha: str) -> dict[str, object]:
    payload = read_regular_json(path, "BES2 complete-test receipt")
    if (
        set(payload)
        != {
            "schema",
            "status",
            "purpose",
            "source_git_sha",
            "command",
            "exit_code",
            "passed",
            "duration_seconds",
        }
        or payload.get("schema") != 1
        or payload.get("status") != "passed"
        or payload.get("purpose") != "bes2-diagnostic-complete-tests"
        or payload.get("source_git_sha") != source_sha
        or payload.get("exit_code") != 0
        or type(payload.get("passed")) is not int
        or not payload.get("command")
        or int(payload["passed"]) <= 0
        or not isinstance(payload.get("command"), list)
        or isinstance(payload.get("duration_seconds"), bool)
        or not isinstance(payload.get("duration_seconds"), (int, float))
        or not math.isfinite(float(payload["duration_seconds"]))
        or float(payload["duration_seconds"]) <= 0.0
    ):
        raise BES2ContractError("BES2 complete-test receipt is invalid")
    return payload


def run_test_gate(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    source_sha = _require_source(repo, args.expected_git_sha)
    receipt = args.receipt.absolute()
    if receipt.exists() or receipt.is_symlink():
        payload = _test_receipt_valid(receipt, source_sha)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    log = args.log.absolute()
    if log.exists() or log.is_symlink():
        raise BES2ContractError(
            "BES2 test log exists without a valid immutable receipt"
        )
    command = [sys.executable, "-B", "-m", "pytest", "-q"]
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("xb") as output:
        result = subprocess.run(
            command,
            cwd=repo,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = max(time.monotonic() - started, 1.0e-9)
    if result.returncode != 0:
        raise BES2ContractError(
            f"complete BES2 test gate failed with exit {result.returncode}"
        )
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = _PASSED.findall(text)
    if not matches or int(matches[-1]) <= 0:
        raise BES2ContractError("could not parse a positive BES2 pytest pass count")
    log.chmod(0o444)
    payload = {
        "schema": 1,
        "status": "passed",
        "purpose": "bes2-diagnostic-complete-tests",
        "source_git_sha": source_sha,
        "command": command,
        "exit_code": 0,
        "passed": int(matches[-1]),
        "duration_seconds": duration,
    }
    write_new_immutable(receipt, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_dir(
    root: Path, variant: str, stage: str, fraction: float
) -> Path:
    return root / diagnostic_run_id(variant, stage, fraction)


def _metrics_path(
    root: Path, variant: str, stage: str, fraction: float
) -> Path:
    return (
        _run_dir(root, variant, stage, fraction)
        / "diagnostic_metrics.json"
    )


def _validated_completion(
    root: Path,
    variant: str,
    stage: str,
    fraction: float,
    *,
    readiness_sha256: str | None = None,
) -> dict[str, object]:
    payload = validate_diagnostic_metrics(
        _metrics_path(root, variant, stage, fraction),
        expected_variant=variant,
        expected_fraction=fraction,
        expected_stage=stage,
    )
    binding = payload.get("readiness")
    if readiness_sha256 is not None and (
        not isinstance(binding, Mapping)
        or binding.get("sha256") != readiness_sha256
    ):
        raise BES2ContractError(
            f"{fraction_tag(fraction)}/{variant}/{stage} is not bound to "
            "current BES2 readiness"
        )
    return payload


def _request_requeue() -> None:
    request_dir = Path(os.environ["H100_REQUEUE_REQUEST_DIR"])
    gpu = os.environ["H100_DIAGNOSTIC_GPU_SLOT"]
    job_id = os.environ["SLURM_JOB_ID"]
    if (
        request_dir.is_symlink()
        or not request_dir.is_dir()
        or not gpu.isdigit()
        or not job_id.isdigit()
    ):
        raise BES2ContractError("unsafe BES2 requeue-request directory")
    atomic_write_text(request_dir / f"gpu-{gpu}.request", job_id + "\n")


def run_worker(args: argparse.Namespace) -> int:
    assert_sitecustomize_active()
    assert_launch_process_contract()
    _require_h100_devices(1)
    repo = args.repo.resolve()
    _require_source(repo, args.expected_git_sha)
    root = args.diagnostic_root.resolve()
    diagnostic_run_id(args.variant, args.stage, args.fraction)
    preempted = False
    process: subprocess.Popen | None = None

    def request_completion_race() -> None:
        _validated_completion(
            root, args.variant, args.stage, args.fraction
        )
        _request_requeue()

    def on_usr1(_signum, _frame) -> None:
        nonlocal preempted
        preempted = True
        if process is not None and process.poll() is None:
            try:
                os.kill(process.pid, signal.SIGUSR1)
                return
            except ProcessLookupError:
                pass
        request_completion_race()

    def on_stop(signum, _frame) -> None:
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signum)
            except ProcessLookupError:
                pass

    old_usr1 = signal.signal(signal.SIGUSR1, on_usr1)
    old_int = signal.signal(signal.SIGINT, on_stop)
    old_term = signal.signal(signal.SIGTERM, on_stop)
    try:
        command = [
            sys.executable,
            "-B",
            "-m",
            "src.analysis.bes2_root_cause",
            "run",
            "--repo",
            str(repo),
            "--diagnostic-root",
            str(root),
            "--original-runs-root",
            str(args.original_runs_root),
            "--variant",
            args.variant,
            "--stage",
            args.stage,
            "--fraction",
            str(args.fraction),
            "--data-view-root",
            str(args.data_view_root),
            "--data-view-receipt",
            str(args.data_view_receipt),
            "--data-config",
            str(args.data_config),
            "--weights-root",
            str(args.weights_root),
            "--workers",
            str(args.workers),
        ]
        process = subprocess.Popen(command, cwd=repo)
        code = process.wait()
        if preempted:
            if code == 0:
                request_completion_race()
            return PREEMPTED_EXIT_CODE
        if code != 0:
            return code
        _validated_completion(
            root, args.variant, args.stage, args.fraction
        )
        return 0
    finally:
        signal.signal(signal.SIGUSR1, old_usr1)
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)


def _request_ready(request_dir: Path, gpu: int, job_id: str) -> bool:
    path = request_dir / f"gpu-{gpu}.request"
    try:
        return (
            path.is_file()
            and not path.is_symlink()
            and path.read_text(encoding="utf-8").strip() == job_id
        )
    except (OSError, UnicodeError):
        return False


def _clear_request(request_dir: Path, gpu: int) -> None:
    path = request_dir / f"gpu-{gpu}.request"
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise BES2ContractError("unsafe stale BES2 requeue request")
    path.unlink(missing_ok=True)


def _execution_state(path: Path, source_sha: str) -> dict[str, object]:
    if path.exists() or path.is_symlink():
        payload = read_regular_json(path, "BES2 execution state")
        if (
            payload.get("schema") != 2
            or payload.get("purpose") != "bes2-fraction-matrix-h100-execution"
            or payload.get("source_git_sha") != source_sha
            or not isinstance(payload.get("events"), list)
        ):
            raise BES2ContractError("BES2 execution state is invalid")
        return payload
    return {
        "schema": 2,
        "purpose": "bes2-fraction-matrix-h100-execution",
        "source_git_sha": source_sha,
        "events": [],
    }


def _record(
    path: Path,
    state: dict[str, object],
    event: str,
    **values: object,
) -> None:
    state["events"].append(
        {
            "event": event,
            "created_utc": utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
            **values,
        }
    )
    atomic_write_json(path, state)


def _stage_receipt(
    root: Path,
    stage: str,
    readiness_sha256: str,
) -> dict[str, object]:
    runs: dict[str, dict[str, object]] = {}
    gpu_hours = 0.0
    for fraction in stage_fractions(stage):
        tag = fraction_tag(fraction)
        runs[tag] = {}
        for variant in DIAGNOSTIC_VARIANTS:
            metrics = _validated_completion(
                root,
                variant,
                stage,
                fraction,
                readiness_sha256=readiness_sha256,
            )
            runs[tag][variant] = hash_binding(
                _metrics_path(root, variant, stage, fraction),
                relative_to=root,
            )
            gpu_hours += float(metrics["gpu_hours"])
    payload: dict[str, object] = {
        "schema": 2,
        "status": "complete",
        "purpose": f"bes2-{stage}-fraction-matrix",
        "created_utc": utc_now(),
        "readiness_sha256": readiness_sha256,
        "fractions": list(stage_fractions(stage)),
        "runs": runs,
        "run_count": len(stage_fractions(stage)) * len(DIAGNOSTIC_VARIANTS),
        "gpu_hours": gpu_hours,
    }
    return payload


def _positive_hours(value: object, description: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise BES2ContractError(f"{description} must be finite and positive")
    return float(value)


def _validate_stage_marker(
    path: Path,
    *,
    root: Path,
    stage: str,
    readiness_sha256: str,
) -> dict[str, object]:
    payload = read_regular_json(path, f"BES2 {stage} completion")
    required = {
        "schema",
        "status",
        "purpose",
        "created_utc",
        "readiness_sha256",
        "fractions",
        "runs",
        "run_count",
        "gpu_hours",
    }
    if stage == "full":
        required |= {
            "probe_matrix",
            "root_cause",
            "report",
            "total_diagnostic_gpu_hours",
        }
    expected = _stage_receipt(root, stage, readiness_sha256)
    observed_hours = _positive_hours(
        payload.get("gpu_hours"),
        f"BES2 {stage} completion GPU-hours",
    )
    if (
        set(payload) != required
        or payload.get("schema") != 2
        or payload.get("status") != "complete"
        or payload.get("purpose") != f"bes2-{stage}-fraction-matrix"
        or payload.get("readiness_sha256") != readiness_sha256
        or payload.get("fractions") != expected["fractions"]
        or payload.get("runs") != expected["runs"]
        or payload.get("run_count") != expected["run_count"]
        or not math.isclose(
            observed_hours,
            float(expected["gpu_hours"]),
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise BES2ContractError(f"BES2 {stage} completion marker is invalid")
    if stage == "full":
        probe_path = root / ".control" / PROBE_COMPLETE
        probe = _validate_stage_marker(
            probe_path,
            root=root,
            stage="probe",
            readiness_sha256=readiness_sha256,
        )
        expected_bindings = {
            "probe_matrix": hash_binding(probe_path, relative_to=root),
            "root_cause": hash_binding(
                root / ".control" / "BES2_ROOT_CAUSE.json",
                relative_to=root,
            ),
            "report": hash_binding(
                root / ".control" / "BES2_ROOT_CAUSE.md",
                relative_to=root,
            ),
        }
        expected_total = float(expected["gpu_hours"]) + float(
            probe["gpu_hours"]
        )
        observed_total = _positive_hours(
            payload.get("total_diagnostic_gpu_hours"),
            "BES2 diagnostic completion total GPU-hours",
        )
        bindings_differ = any(
            payload.get(name) != binding
            for name, binding in expected_bindings.items()
        )
        if bindings_differ or not math.isclose(
            observed_total,
            expected_total,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise BES2ContractError(
                "BES2 full completion evidence binding is invalid"
            )
    return payload


def _finalize_stage(
    *,
    root: Path,
    stage: str,
    readiness_sha256: str,
) -> dict[str, object]:
    marker = root / ".control" / (
        PROBE_COMPLETE if stage == "probe" else EXECUTION_COMPLETE
    )
    if marker.exists() or marker.is_symlink():
        return _validate_stage_marker(
            marker,
            root=root,
            stage=stage,
            readiness_sha256=readiness_sha256,
        )
    if stage == "probe":
        payload = _stage_receipt(root, stage, readiness_sha256)
    else:
        if root_cause_main(["summarize", "--diagnostic-root", str(root)]) != 0:
            raise BES2ContractError("BES2 root-cause summarize returned nonzero")
        probe = read_regular_json(
            root / ".control" / PROBE_COMPLETE,
            "BES2 probe-matrix completion",
        )
        payload = _stage_receipt(root, stage, readiness_sha256)
        payload["probe_matrix"] = hash_binding(
            root / ".control" / PROBE_COMPLETE,
            relative_to=root,
        )
        payload["root_cause"] = hash_binding(
            root / ".control" / "BES2_ROOT_CAUSE.json",
            relative_to=root,
        )
        payload["report"] = hash_binding(
            root / ".control" / "BES2_ROOT_CAUSE.md",
            relative_to=root,
        )
        payload["total_diagnostic_gpu_hours"] = (
            float(payload["gpu_hours"]) + float(probe["gpu_hours"])
        )
        if not math.isfinite(float(payload["total_diagnostic_gpu_hours"])):
            raise BES2ContractError("BES2 total GPU-hours are nonfinite")
    write_new_immutable(marker, payload)
    return _validate_stage_marker(
        marker,
        root=root,
        stage=stage,
        readiness_sha256=readiness_sha256,
    )


def _stage_is_complete(
    *,
    root: Path,
    stage: str,
    readiness_sha256: str,
) -> bool:
    for fraction in stage_fractions(stage):
        for variant in DIAGNOSTIC_VARIANTS:
            path = _metrics_path(root, variant, stage, fraction)
            if path.is_symlink():
                raise BES2ContractError(
                    "diagnostic metrics must not be a symlink"
                )
            if not path.is_file():
                return False
            _validated_completion(
                root,
                variant,
                stage,
                fraction,
                readiness_sha256=readiness_sha256,
            )
    return True


def _finish_fraction_or_stage(
    *,
    root: Path,
    stage: str,
    fraction: float,
    readiness_sha256: str,
) -> dict[str, object]:
    if _stage_is_complete(
        root=root,
        stage=stage,
        readiness_sha256=readiness_sha256,
    ):
        return _finalize_stage(
            root=root,
            stage=stage,
            readiness_sha256=readiness_sha256,
        )
    tag = fraction_tag(fraction)
    metrics = {
        variant: _validated_completion(
            root,
            variant,
            stage,
            fraction,
            readiness_sha256=readiness_sha256,
        )
        for variant in DIAGNOSTIC_VARIANTS
    }
    return {
        "schema": 1,
        "status": "fraction-complete",
        "purpose": "bes2-fraction-wave",
        "stage": stage,
        "fraction": fraction,
        "fraction_tag": tag,
        "readiness_sha256": readiness_sha256,
        "runs": {
            variant: hash_binding(
                _metrics_path(root, variant, stage, fraction),
                relative_to=root,
            )
            for variant in DIAGNOSTIC_VARIANTS
        },
        "gpu_hours": sum(
            float(metrics[variant]["gpu_hours"])
            for variant in DIAGNOSTIC_VARIANTS
        ),
    }


def _worker_command(
    args: argparse.Namespace, variant: str, fraction: float
) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "scripts.h100.bes2_diagnostic",
        "worker",
        "--repo",
        str(args.repo),
        "--diagnostic-root",
        str(args.diagnostic_root),
        "--original-runs-root",
        str(args.original_runs_root),
        "--expected-git-sha",
        args.expected_git_sha,
        "--variant",
        variant,
        "--stage",
        args.stage,
        "--fraction",
        str(fraction),
        "--data-view-root",
        str(args.data_view_root),
        "--data-view-receipt",
        str(args.data_view_receipt),
        "--data-config",
        str(args.data_config),
        "--weights-root",
        str(args.weights_root),
        "--workers",
        str(args.workers),
    ]


def run_controller(args: argparse.Namespace) -> int:
    assert_sitecustomize_active()
    _require_h100_devices(2)
    if (
        isinstance(args.checkpoint_timeout, bool)
        or not isinstance(args.checkpoint_timeout, (int, float))
        or not math.isfinite(float(args.checkpoint_timeout))
        or not 0.0 < float(args.checkpoint_timeout) <= REQUEST_TIMEOUT_SECONDS
    ):
        raise BES2ContractError("BES2 checkpoint timeout must be within (0, 600]")

    repo = args.repo.resolve()
    source_sha = _require_source(repo, args.expected_git_sha)
    root = args.diagnostic_root.resolve()
    diagnostic_run_id(DIAGNOSTIC_VARIANTS[0], args.stage, args.fraction)
    assert_no_final_consumption(args.original_runs_root)
    readiness_path = root / ".control" / READY_FILENAME
    validate_readiness(
        readiness_path,
        expected_source_sha=source_sha,
        diagnostic_root=root,
    )
    readiness_sha256 = sha256_file(readiness_path)
    complete_name = PROBE_COMPLETE if args.stage == "probe" else EXECUTION_COMPLETE
    complete_path = root / ".control" / complete_name
    if complete_path.exists() or complete_path.is_symlink():
        payload = _validate_stage_marker(
            complete_path,
            root=root,
            stage=args.stage,
            readiness_sha256=readiness_sha256,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    request_dir = root / ".control" / "requeue-requests"
    request_dir.mkdir(mode=0o700, exist_ok=True)
    if request_dir.is_symlink() or not request_dir.is_dir():
        raise BES2ContractError("unsafe BES2 requeue request root")
    state_path = root / ".control" / EXECUTION_STATE
    state = _execution_state(state_path, source_sha)

    incomplete = []
    for variant in DIAGNOSTIC_VARIANTS:
        metrics = _metrics_path(root, variant, args.stage, args.fraction)
        if metrics.is_symlink():
            raise BES2ContractError("diagnostic metrics must not be a symlink")
        if metrics.is_file():
            _validated_completion(
                root,
                variant,
                args.stage,
                args.fraction,
                readiness_sha256=readiness_sha256,
            )
        else:
            incomplete.append(variant)
    if not incomplete:
        payload = _finish_fraction_or_stage(
            root=root,
            stage=args.stage,
            fraction=args.fraction,
            readiness_sha256=readiness_sha256,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    preemption_seen = False

    def on_usr1(_signum, _frame) -> None:
        nonlocal preemption_seen
        preemption_seen = True

    prior = signal.signal(signal.SIGUSR1, on_usr1)
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        raise BES2ContractError("BES2 controller requires SLURM_JOB_ID")
    device_tokens = _visible_device_tokens(2)
    running: dict[int, tuple[subprocess.Popen, str]] = {}
    for gpu, variant in enumerate(incomplete):
        _clear_request(request_dir, gpu)
        log = _run_dir(root, variant, args.stage, args.fraction) / "controller.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        output = log.open("a", encoding="utf-8")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": device_tokens[gpu],
            "H100_REQUEUE_REQUEST_DIR": str(request_dir),
            "H100_DIAGNOSTIC_GPU_SLOT": str(gpu),
            "PATH": f"{repo / 'slurm/h100/shims'}:{os.environ['PATH']}",
        }
        process = subprocess.Popen(
            _worker_command(args, variant, args.fraction),
            cwd=repo,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        output.close()
        running[gpu] = (process, variant)
        _record(
            state_path,
            state,
            "worker-launched",
            stage=args.stage,
            fraction=args.fraction,
            variant=variant,
            gpu=gpu,
            pid=process.pid,
        )

    try:
        while running and not preemption_seen:
            for gpu in list(running):
                process, variant = running[gpu]
                code = process.poll()
                if code is None:
                    continue
                if code != 0:
                    for other, _name in running.values():
                        if other.poll() is None:
                            other.terminate()
                    _record(
                        state_path,
                        state,
                        "worker-failed",
                        stage=args.stage,
                        fraction=args.fraction,
                        variant=variant,
                        gpu=gpu,
                        exit_code=code,
                    )
                    return code
                _validated_completion(root, variant, args.stage, args.fraction)
                _record(
                    state_path,
                    state,
                    "worker-complete",
                    stage=args.stage,
                    fraction=args.fraction,
                    variant=variant,
                    gpu=gpu,
                )
                del running[gpu]
            if running:
                time.sleep(1.0)

        if preemption_seen:
            for process, _variant in running.values():
                if process.poll() is None:
                    try:
                        os.kill(process.pid, signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
            deadline = time.monotonic() + args.checkpoint_timeout
            while time.monotonic() < deadline:
                if all(
                    _request_ready(request_dir, gpu, job_id)
                    for gpu in running
                ):
                    break
                failed = [
                    (gpu, process.poll())
                    for gpu, (process, _variant) in running.items()
                    if process.poll() not in (None, 0, PREEMPTED_EXIT_CODE)
                ]
                if failed:
                    raise BES2ContractError(
                        f"BES2 worker failed during checkpoint barrier: {failed}"
                    )
                time.sleep(1.0)
            missing = [
                gpu
                for gpu in running
                if not _request_ready(request_dir, gpu, job_id)
            ]
            if missing:
                raise BES2ContractError(
                    f"BES2 checkpoint barrier timed out for GPUs {missing}"
                )
            for gpu, (process, variant) in running.items():
                metrics = _metrics_path(root, variant, args.stage, args.fraction)
                if metrics.is_file() and not metrics.is_symlink():
                    _validated_completion(root, variant, args.stage, args.fraction)
                else:
                    checkpoint = promote_hpc_checkpoint(
                        _run_dir(root, variant, args.stage, args.fraction)
                    )
                    _record(
                        state_path,
                        state,
                        "checkpoint-promoted",
                        stage=args.stage,
                        fraction=args.fraction,
                        variant=variant,
                        gpu=gpu,
                        checkpoint_sha256=sha256_file(checkpoint),
                    )
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired as exc:
                    process.kill()
                    process.wait()
                    raise BES2ContractError(
                        "BES2 worker did not stop after checkpoint promotion"
                    ) from exc
            _record(
                state_path,
                state,
                "host-requeue-required",
                stage=args.stage,
                fraction=args.fraction,
            )
            return PREEMPTED_EXIT_CODE

        payload = _finish_fraction_or_stage(
            root=root,
            stage=args.stage,
            fraction=args.fraction,
            readiness_sha256=readiness_sha256,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        signal.signal(signal.SIGUSR1, prior)


def _common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--original-runs-root", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--stage", choices=("probe", "full"), required=True)
    parser.add_argument("--data-view-root", type=Path, required=True)
    parser.add_argument("--data-view-receipt", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    view = commands.add_parser("prepare-view")
    view.add_argument("--repo", type=Path, required=True)
    view.add_argument("--diagnostic-root", type=Path, required=True)
    view.add_argument("--original-runs-root", type=Path, required=True)
    view.add_argument("--staging-runs-root", type=Path, required=True)
    view.add_argument("--data-view-root", type=Path, required=True)
    view.add_argument("--data-config", type=Path, required=True)
    view.add_argument("--output", type=Path, required=True)
    view.add_argument("--h100-ready", type=Path, required=True)
    view.add_argument("--h100-ready-sha256", required=True)
    view.add_argument("--staging-receipt-sha256", required=True)
    view.add_argument("--expected-git-sha", required=True)
    view.add_argument("--expected-base-package-id", required=True)
    view.add_argument("--expected-base-manifest-sha256", required=True)
    view.add_argument("--expected-runtime-package-id", required=True)
    view.add_argument("--expected-runtime-manifest-sha256", required=True)

    tests = commands.add_parser("test-gate")
    tests.add_argument("--repo", type=Path, required=True)
    tests.add_argument("--expected-git-sha", required=True)
    tests.add_argument("--receipt", type=Path, required=True)
    tests.add_argument("--log", type=Path, required=True)

    worker = commands.add_parser("worker")
    _common_run_arguments(worker)
    worker.add_argument("--variant", choices=DIAGNOSTIC_VARIANTS, required=True)
    worker.add_argument(
        "--fraction", type=float, choices=DIAGNOSTIC_FRACTIONS, required=True
    )

    controller = commands.add_parser("controller")
    _common_run_arguments(controller)
    controller.add_argument(
        "--fraction", type=float, choices=DIAGNOSTIC_FRACTIONS, required=True
    )
    controller.add_argument(
        "--checkpoint-timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-view":
        return prepare_view(args)
    if args.command == "test-gate":
        return run_test_gate(args)
    if args.command == "worker":
        return run_worker(args)
    if args.command == "controller":
        return run_controller(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
