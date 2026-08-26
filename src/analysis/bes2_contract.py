"""Pure contracts and predeclared decisions for the BES2 root-cause study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval.result_contract import (
    ResultContractError,
    validate_completion_payload,
    validate_dev_result,
)

DIAGNOSTIC_READY_SCHEMA = 1
DIAGNOSTIC_METRICS_SCHEMA = 1
ROOT_CAUSE_SCHEMA = 1
H100_CAMPAIGN_GIT_SHA = "1a82d508fbeb9fdf6868a9637611e9018952fb43"
DEV_BASE_GIT_SHA = "322dea060a37ec793f1df3d49a7513dfd90b324f"
S2_FROZEN_DEV_F1 = 0.8635917566
RANDOM_FROZEN_DEV_F1 = 0.8919449902
REPLAY_TOLERANCE = 0.02
STEM_DOMINANT_RECOVERY = 0.50
POST_STEM_RECOVERY = 0.20
MAX_APPROVED_GPU_HOURS = 125.0
MANDATORY_STOP_GPU_HOURS = 250.0
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")

TRAINING_PATH_SCOPES = (
    "configs/data.yaml",
    "configs/detector.yaml",
    "scripts/h100",
    "src/data",
    "src/eval",
    "src/models",
    "src/train",
)

FINAL_CONSUMPTION_RELATIVE_PATHS = (
    "final_eval.lock",
    ".h100/FINAL_DATA_VIEW.json",
    ".h100/FINAL_GROUND_TRUTH_CONSUMED.json",
    ".h100/FINAL_NORMALIZED_GROUND_TRUTH.json",
    ".h100/FINAL_EVAL_COMPLETE.json",
)


class BES2ContractError(RuntimeError):
    """A diagnostic artifact is unsafe, incomplete, or not decision-valid."""


def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise BES2ContractError(
            f"artifact must be a regular non-symlink file: {source}"
        )
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def read_regular_json(path: str | Path, description: str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise BES2ContractError(
            f"{description} must be a regular non-symlink file: {source}"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BES2ContractError(f"invalid {description}: {source}") from exc
    if not isinstance(payload, dict):
        raise BES2ContractError(f"{description} root must be a JSON object")
    return payload


def write_new_immutable(path: str | Path, payload: Mapping[str, object]) -> Path:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise BES2ContractError(
            f"immutable diagnostic artifact already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
        temporary = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, TypeError, ValueError) as exc:
        raise BES2ContractError(
            f"could not publish immutable artifact: {destination}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def hash_binding(path: str | Path, *, relative_to: str | Path | None = None) -> dict[str, str]:
    source = Path(path)
    if relative_to is None:
        relative = str(source.absolute())
    else:
        try:
            relative = source.absolute().relative_to(
                Path(relative_to).absolute()
            ).as_posix()
        except ValueError as exc:
            raise BES2ContractError(
                f"artifact is outside its binding root: {source}"
            ) from exc
    return {"path": relative, "sha256": sha256_file(source)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_forecast_hours(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BES2ContractError("forecast GPU-hours must be a finite number")
    forecast = float(value)
    if not math.isfinite(forecast) or forecast <= 0.0:
        raise BES2ContractError("forecast GPU-hours must be finite and positive")
    if forecast > MAX_APPROVED_GPU_HOURS:
        suffix = (
            " and exceeds the mandatory roughly-2x STOP"
            if forecast > MANDATORY_STOP_GPU_HOURS
            else ""
        )
        raise BES2ContractError(
            f"forecast {forecast:.3f} GPU-hours exceeds the approved "
            f"{MAX_APPROVED_GPU_HOURS:.0f}-hour diagnostic budget{suffix}"
        )
    return forecast


def assert_no_final_consumption(runs_root: str | Path) -> dict[str, object]:
    root = Path(runs_root).absolute()
    present = [
        relative
        for relative in FINAL_CONSUMPTION_RELATIVE_PATHS
        if (root / relative).exists() or (root / relative).is_symlink()
    ]
    per_cell = sorted(root.glob("*/final_verified_metrics.json"))
    if present or per_cell:
        details = present + [
            path.relative_to(root).as_posix() for path in per_cell[:8]
        ]
        raise BES2ContractError(
            "verified-final data may already have been staged or consumed: "
            + ", ".join(details)
        )
    authorization = root / ".h100/FINAL_EVAL_OWNER_AMENDMENT.json"
    return {
        "status": "unconsumed",
        "checked_paths": list(FINAL_CONSUMPTION_RELATIVE_PATHS),
        "owner_authorization": (
            hash_binding(authorization, relative_to=root)
            if authorization.is_file() and not authorization.is_symlink()
            else None
        ),
    }


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo}", *args],
            cwd=repo,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BES2ContractError(
            f"could not inspect Git training-path identity: {' '.join(args)}"
        ) from exc


def verify_training_path_identity(
    repo: str | Path,
    *,
    require_dev_ref: bool = True,
) -> dict[str, object]:
    """Prove dev's production path equals the completed H100 campaign source."""

    root = Path(repo).resolve()
    if require_dev_ref:
        observed_dev = _git(root, "rev-parse", "dev").strip()
        if observed_dev != DEV_BASE_GIT_SHA:
            raise BES2ContractError(
                f"dev moved from the approved base: "
                f"{observed_dev} != {DEV_BASE_GIT_SHA}"
            )
    else:
        observed_dev = _git(
            root, "rev-parse", f"{DEV_BASE_GIT_SHA}^{{commit}}"
        ).strip()
        if observed_dev != DEV_BASE_GIT_SHA:
            raise BES2ContractError("approved dev base object is unavailable")
    changed = _git(
        root,
        "diff",
        "--name-only",
        f"{H100_CAMPAIGN_GIT_SHA}..{DEV_BASE_GIT_SHA}",
        "--",
        *TRAINING_PATH_SCOPES,
    ).splitlines()
    if changed:
        raise BES2ContractError(
            "dev training path differs from the H100 campaign revision: "
            + ", ".join(changed)
        )

    listing = _git(
        root,
        "ls-tree",
        "-r",
        H100_CAMPAIGN_GIT_SHA,
        "--",
        *TRAINING_PATH_SCOPES,
    ).splitlines()
    blobs: dict[str, str] = {}
    for line in listing:
        metadata, path = line.split("\t", 1)
        _mode, object_type, digest = metadata.split()
        if object_type == "blob":
            blobs[path] = digest
    if not blobs:
        raise BES2ContractError("training-path Git tree is unexpectedly empty")
    return {
        "status": "byte-identical",
        "h100_campaign_git_sha": H100_CAMPAIGN_GIT_SHA,
        "dev_base_git_sha": DEV_BASE_GIT_SHA,
        "scopes": list(TRAINING_PATH_SCOPES),
        "blob_count": len(blobs),
        "blob_manifest_sha256": payload_sha256(blobs),
    }


def _finite_unit(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BES2ContractError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise BES2ContractError(f"{field} must be within [0, 1]")
    return result


def classify_cause(
    replay_best_dev_f1: object,
    reset_best_dev_f1: object,
) -> dict[str, object]:
    """Apply the owner-predeclared replay and recovery decision exactly."""

    replay = _finite_unit(replay_best_dev_f1, "current_replay best DEV F1")
    reset = _finite_unit(reset_best_dev_f1, "first_conv_reset best DEV F1")
    replay_delta = replay - S2_FROZEN_DEV_F1
    denominator = RANDOM_FROZEN_DEV_F1 - replay

    common = {
        "current_replay_best_dev_f1": replay,
        "first_conv_reset_best_dev_f1": reset,
        "frozen_s2_best_dev_f1": S2_FROZEN_DEV_F1,
        "frozen_random_best_dev_f1": RANDOM_FROZEN_DEV_F1,
        "replay_delta": replay_delta,
        "replay_tolerance": REPLAY_TOLERANCE,
    }
    if abs(replay_delta) > REPLAY_TOLERANCE and not math.isclose(
        abs(replay_delta),
        REPLAY_TOLERANCE,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        return {
            **common,
            "status": "indeterminate",
            "classification": "replay-not-reproducible",
            "recovery": None,
            "replacement_eligibility": "blocked",
            "required_next_step": "investigate replay mismatch before replacement",
        }
    if replay >= RANDOM_FROZEN_DEV_F1:
        return {
            **common,
            "status": "determinate",
            "classification": "deficit-did-not-reproduce",
            "recovery": None,
            "replacement_eligibility": "not-established",
            "required_next_step": "retain original arm pending owner review",
        }

    if denominator <= 0.0:
        raise BES2ContractError("recovery denominator must be positive")
    recovery = (reset - replay) / denominator
    if recovery < 0.0:
        classification = "stem-conversion-not-performance-cause"
        eligibility = "blocked"
        next_step = (
            "native-three-channel optical-RS checkpoint required for replacement"
        )
    elif recovery >= STEM_DOMINANT_RECOVERY or math.isclose(
        recovery,
        STEM_DOMINANT_RECOVERY,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        classification = "stem-conversion-explains-most-deficit"
        eligibility = "conditional-owner-amendment-required"
        next_step = (
            "owner may approve documented timm first-convolution fallback semantics"
        )
    elif recovery <= POST_STEM_RECOVERY or math.isclose(
        recovery,
        POST_STEM_RECOVERY,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        classification = "first-convolution-reset-insufficient"
        eligibility = "blocked"
        next_step = (
            "native-three-channel optical-RS checkpoint required for replacement"
        )
    else:
        classification = "mixed-mechanism"
        eligibility = "blocked"
        next_step = (
            "native-three-channel optical-RS checkpoint required for replacement"
        )
    return {
        **common,
        "status": "determinate",
        "classification": classification,
        "recovery": recovery,
        "replacement_eligibility": eligibility,
        "required_next_step": next_step,
    }


def validate_training_metrics(
    path: str | Path,
    *,
    expected_variant: str,
    expected_stage: str,
    candidate_floor: float,
) -> tuple[dict[str, object], Path]:
    marker_path = Path(path)
    payload = read_regular_json(marker_path, "diagnostic training metrics")
    expected_top = {
        "diagnostic_run_schema",
        "purpose",
        "variant",
        "stage",
        "trainer_max_epochs",
        "schedule_horizon_epochs",
        "initialization",
        "dev_history",
        "runtime",
        "training_result",
    }
    if (
        set(payload) != expected_top
        or payload.get("diagnostic_run_schema") != 1
        or payload.get("purpose") != "bes2-root-cause-training"
        or payload.get("variant") != expected_variant
        or payload.get("stage") != expected_stage
    ):
        raise BES2ContractError("diagnostic training marker identity is invalid")
    expected_epochs = 5 if expected_stage == "probe" else 50
    if (
        payload.get("trainer_max_epochs") != expected_epochs
        or payload.get("schedule_horizon_epochs") != 50
    ):
        raise BES2ContractError("diagnostic 5/50 epoch contract is invalid")

    run_dir = marker_path.parent
    if (run_dir / "final_metrics.json").exists() or (
        run_dir / "final_metrics.json"
    ).is_symlink():
        raise BES2ContractError(
            "diagnostic run contains a forbidden core final_metrics marker"
        )
    training = payload.get("training_result")
    if not isinstance(training, Mapping):
        raise BES2ContractError("diagnostic training result is absent")
    try:
        checkpoint = validate_completion_payload(
            training,
            run_dir=run_dir,
            candidate_floor=candidate_floor,
            expected_recipe={
                "exp_id": run_dir.name,
                "git_sha": training.get("git_sha"),
                "detector_sha256": training.get("detector_sha256"),
                "precision": "32-true",
                "micro_batch": 16,
                "gradient_accumulation": 1,
                "effective_batch": 16,
            },
        )
    except ResultContractError as exc:
        raise BES2ContractError(
            f"diagnostic schema-2 training result is invalid: {exc}"
        ) from exc

    history = payload.get("dev_history")
    if not isinstance(history, list) or not history:
        raise BES2ContractError("diagnostic DEV history is absent")
    validated_history = [
        validate_dev_result(
            item,
            candidate_floor=candidate_floor,
            description=f"dev_history[{index}]",
        )
        for index, item in enumerate(history)
    ]
    if (
        validated_history[-1] != training.get("last_dev")
        or max(validated_history, key=lambda item: float(item["f1"]))
        != training.get("best_dev")
    ):
        raise BES2ContractError(
            "diagnostic DEV history does not reproduce best/last results"
        )
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "gpu_count",
        "seconds",
        "gpu_hours",
    }:
        raise BES2ContractError("diagnostic runtime evidence is invalid")
    seconds = runtime.get("seconds")
    hours = runtime.get("gpu_hours")
    if (
        runtime.get("gpu_count") != 1
        or isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or isinstance(hours, bool)
        or not isinstance(hours, (int, float))
        or not math.isfinite(float(seconds))
        or not math.isfinite(float(hours))
        or float(seconds) <= 0.0
        or not math.isclose(
            float(hours), float(seconds) / 3600.0, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise BES2ContractError("diagnostic GPU-hour evidence is invalid")
    return dict(payload), checkpoint


def validate_diagnostic_metrics(
    path: str | Path,
    *,
    expected_variant: str,
    expected_stage: str = "full",
) -> dict[str, object]:
    payload = read_regular_json(path, "diagnostic metrics")
    return validate_diagnostic_metrics_payload(
        payload,
        expected_variant=expected_variant,
        expected_stage=expected_stage,
    )


def validate_diagnostic_metrics_payload(
    payload: Mapping[str, object],
    *,
    expected_variant: str,
    expected_stage: str = "full",
) -> dict[str, object]:
    """Validate diagnostic evidence without requiring its original Judy path."""

    required = {
        "diagnostic_metrics_schema",
        "status",
        "purpose",
        "variant",
        "stage",
        "created_utc",
        "readiness",
        "training",
        "dev_evidence",
        "activation_statistics",
        "layer_drift",
        "runtime_provenance",
        "gpu_hours",
    }
    if (
        set(payload) != required
        or payload.get("diagnostic_metrics_schema")
        != DIAGNOSTIC_METRICS_SCHEMA
        or payload.get("status") != "complete"
        or payload.get("purpose") != "bes2-root-cause"
        or payload.get("variant") != expected_variant
        or payload.get("stage") != expected_stage
    ):
        raise BES2ContractError("diagnostic metrics identity is invalid")
    training = payload.get("training")
    dev = payload.get("dev_evidence")
    if not isinstance(training, Mapping) or not isinstance(dev, Mapping):
        raise BES2ContractError("diagnostic metrics evidence is incomplete")
    best = training.get("training_result")
    if (
        not isinstance(best, Mapping)
        or dev.get("selected_operating_point") != best.get("best_dev")
    ):
        raise BES2ContractError(
            "diagnostic DEV evidence is not bound to best_dev"
        )
    gpu_hours = payload.get("gpu_hours")
    if (
        isinstance(gpu_hours, bool)
        or not isinstance(gpu_hours, (int, float))
        or not math.isfinite(float(gpu_hours))
        or float(gpu_hours) <= 0.0
    ):
        raise BES2ContractError("diagnostic metrics GPU-hours are invalid")
    return dict(payload)


def summarize_payload(
    replay: Mapping[str, object],
    reset: Mapping[str, object],
    *,
    replay_binding: Mapping[str, str],
    reset_binding: Mapping[str, str],
) -> dict[str, object]:
    replay_training = replay["training"]
    reset_training = reset["training"]
    if not isinstance(replay_training, Mapping) or not isinstance(
        reset_training, Mapping
    ):
        raise BES2ContractError("summary training evidence is absent")
    replay_result = replay_training.get("training_result")
    reset_result = reset_training.get("training_result")
    if not isinstance(replay_result, Mapping) or not isinstance(
        reset_result, Mapping
    ):
        raise BES2ContractError("summary schema-2 training results are absent")

    decision = classify_cause(
        replay_result.get("best_dev_f1"),
        reset_result.get("best_dev_f1"),
    )
    return {
        "root_cause_schema": ROOT_CAUSE_SCHEMA,
        "status": decision["status"],
        "purpose": "bes2-root-cause-decision",
        "created_utc": utc_now(),
        "scope": {
            "split": "TRAIN+fixed-DEV8-only",
            "fraction": 1.0,
            "seed": 0,
            "single_seed_point_estimates": True,
            "significance_claims": False,
            "seed_variance_claims": False,
            "bigearthnet_s1": (
                "out-of-scope; shared-helper use remains an unresolved "
                "compliance issue"
            ),
        },
        "runs": {
            "current_replay": dict(replay_binding),
            "first_conv_reset": dict(reset_binding),
        },
        "decision": decision,
        "supporting_evidence": {
            "current_replay": {
                "best_dev": replay_result.get("best_dev"),
                "dev_evidence": replay.get("dev_evidence"),
                "activation_statistics": replay.get("activation_statistics"),
                "layer_drift": replay.get("layer_drift"),
            },
            "first_conv_reset": {
                "best_dev": reset_result.get("best_dev"),
                "dev_evidence": reset.get("dev_evidence"),
                "activation_statistics": reset.get("activation_statistics"),
                "layer_drift": reset.get("layer_drift"),
            },
        },
    }
