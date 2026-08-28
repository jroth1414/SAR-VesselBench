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

DIAGNOSTIC_READY_SCHEMA = 2
DIAGNOSTIC_METRICS_SCHEMA = 2
ROOT_CAUSE_SCHEMA = 2
H100_CAMPAIGN_GIT_SHA = "1a82d508fbeb9fdf6868a9637611e9018952fb43"
DEV_BASE_GIT_SHA = "322dea060a37ec793f1df3d49a7513dfd90b324f"
DIAGNOSTIC_FRACTIONS = (0.1, 0.5)
PROBE_FRACTIONS = (0.5,)
DIAGNOSTIC_VARIANTS = ("current_replay", "first_conv_reset")
FROZEN_COMPARATOR_EXPERIMENTS = {
    "f10": {
        "fraction": 0.1,
        "bigearthnet_s2": "beS2-f10-s0",
        "cnn_random": "cnnrand-f10-s0",
    },
    "f50": {
        "fraction": 0.5,
        "bigearthnet_s2": "beS2-f50-s0",
        "cnn_random": "cnnrand-f50-s0",
    },
}
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


def fraction_tag(fraction: object) -> str:
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise BES2ContractError("diagnostic fraction must be numeric")
    value = float(fraction)
    for allowed in DIAGNOSTIC_FRACTIONS:
        if math.isclose(value, allowed, rel_tol=0.0, abs_tol=1.0e-12):
            return f"f{int(round(allowed * 100))}"
    raise BES2ContractError(
        f"diagnostic fraction {value!r} is not one of {DIAGNOSTIC_FRACTIONS}"
    )


def stage_fractions(stage: str) -> tuple[float, ...]:
    if stage == "probe":
        return PROBE_FRACTIONS
    if stage == "full":
        return DIAGNOSTIC_FRACTIONS
    raise BES2ContractError(f"invalid diagnostic stage: {stage!r}")


def diagnostic_run_id(variant: str, stage: str, fraction: object) -> str:
    if variant not in DIAGNOSTIC_VARIANTS:
        raise BES2ContractError(f"invalid diagnostic variant: {variant!r}")
    tag = fraction_tag(fraction)
    allowed = {fraction_tag(value) for value in stage_fractions(stage)}
    if tag not in allowed:
        raise BES2ContractError(
            f"{stage} does not authorize diagnostic fraction {tag}"
        )
    return f"bes2-{variant}-{stage}-{tag}-s0"


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


def _strict_h100_result(
    path: Path,
    *,
    runs_root: Path,
    expected_exp_id: str,
) -> dict[str, object]:
    if path.parent.is_symlink() or path.is_symlink() or not path.is_file():
        raise BES2ContractError(
            f"frozen comparator must be a regular non-symlink file: {path}"
        )
    if path.stat().st_mode & 0o222:
        raise BES2ContractError(
            f"frozen comparator result remains writable: {path}"
        )
    payload = read_regular_json(path, f"frozen comparator {expected_exp_id}")
    runtime = payload.get("h100_runtime_contract")
    pre = runtime.get("pre_trainer") if isinstance(runtime, Mapping) else None
    resolved = (
        runtime.get("resolved_trainer") if isinstance(runtime, Mapping) else None
    )
    strict = pre.get("strict_fp32") if isinstance(pre, Mapping) else None
    expected_strict = {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }
    if (
        payload.get("result_schema") != 2
        or payload.get("exp_id") != expected_exp_id
        or payload.get("git_sha") != H100_CAMPAIGN_GIT_SHA
        or payload.get("precision") != "32-true"
        or payload.get("micro_batch") != 16
        or payload.get("gradient_accumulation") != 1
        or payload.get("effective_batch") != 16
        or not isinstance(runtime, Mapping)
        or runtime.get("schema") != 1
        or runtime.get("status") != "verified"
        or not isinstance(pre, Mapping)
        or pre.get("devices") != 1
        or pre.get("precision") != "32-true"
        or pre.get("micro_batch") != 16
        or pre.get("gradient_accumulation") != 1
        or pre.get("effective_batch") != 16
        or strict != expected_strict
        or not isinstance(resolved, Mapping)
        or resolved.get("num_devices") != 1
        or resolved.get("world_size") != 1
        or resolved.get("gradient_accumulation") != 1
        or any(str(key).startswith(("test_", "final_")) for key in payload)
    ):
        raise BES2ContractError(
            f"{expected_exp_id} is not a canonical strict-FP32 H100 result"
        )
    best = validate_dev_result(
        payload.get("best_dev"),
        candidate_floor=0.05,
        description=f"{expected_exp_id}.best_dev",
    )
    best_f1 = _finite_unit(
        payload.get("best_dev_f1"), f"{expected_exp_id}.best_dev_f1"
    )
    if best_f1 != float(best["f1"]):
        raise BES2ContractError(
            f"{expected_exp_id} best_dev_f1 differs from best_dev"
        )
    return {
        "exp_id": expected_exp_id,
        "result": hash_binding(path, relative_to=runs_root),
        "best_dev_f1": best_f1,
        "best_dev": best,
    }


def frozen_comparator_payload(runs_root: str | Path) -> dict[str, object]:
    """Snapshot only the four approved H100 best-DEV numerical comparators."""

    root = Path(runs_root).resolve(strict=True)
    fractions: dict[str, object] = {}
    detector_sha256: str | None = None
    for tag, specification in FROZEN_COMPARATOR_EXPERIMENTS.items():
        roles: dict[str, object] = {}
        for role in ("bigearthnet_s2", "cnn_random"):
            exp_id = str(specification[role])
            path = root / exp_id / "final_metrics.json"
            roles[role] = _strict_h100_result(
                path,
                runs_root=root,
                expected_exp_id=exp_id,
            )
            result = read_regular_json(path, f"frozen comparator {exp_id}")
            observed_detector = str(result.get("detector_sha256"))
            if not _HEX64.fullmatch(observed_detector):
                raise BES2ContractError(
                    f"{exp_id} detector SHA-256 is malformed"
                )
            if detector_sha256 is None:
                detector_sha256 = observed_detector
            elif detector_sha256 != observed_detector:
                raise BES2ContractError(
                    "frozen comparator results bind different detector bytes"
                )
        s2 = float(roles["bigearthnet_s2"]["best_dev_f1"])
        random = float(roles["cnn_random"]["best_dev_f1"])
        if not s2 < random:
            raise BES2ContractError(
                f"{tag} does not contain the approved S2-below-random deficit"
            )
        fractions[tag] = {
            "fraction": specification["fraction"],
            **roles,
            "frozen_random_minus_s2_f1": random - s2,
        }
    payload = {
        "schema": 1,
        "status": "frozen",
        "purpose": "bes2-h100-best-dev-numerical-comparators",
        "created_utc": utc_now(),
        "campaign_git_sha": H100_CAMPAIGN_GIT_SHA,
        "detector_sha256": detector_sha256,
        "read_scope": {
            "final_metrics_json": sorted(
                f"{specification[role]}/final_metrics.json"
                for specification in FROZEN_COMPARATOR_EXPERIMENTS.values()
                for role in ("bigearthnet_s2", "cnn_random")
            ),
            "checkpoint_bytes": False,
            "training_cohort": False,
            "test_metrics": False,
            "verified_final": False,
        },
        "fractions": fractions,
    }
    return validate_frozen_comparator_payload(payload)


def validate_frozen_comparator_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    required = {
        "schema",
        "status",
        "purpose",
        "created_utc",
        "campaign_git_sha",
        "detector_sha256",
        "read_scope",
        "fractions",
    }
    fractions = payload.get("fractions")
    if (
        set(payload) != required
        or payload.get("schema") != 1
        or payload.get("status") != "frozen"
        or payload.get("purpose")
        != "bes2-h100-best-dev-numerical-comparators"
        or payload.get("campaign_git_sha") != H100_CAMPAIGN_GIT_SHA
        or not _HEX64.fullmatch(str(payload.get("detector_sha256")))
        or not isinstance(fractions, Mapping)
        or set(fractions) != set(FROZEN_COMPARATOR_EXPERIMENTS)
    ):
        raise BES2ContractError("frozen comparator receipt identity is invalid")
    expected_scope = {
        "final_metrics_json": sorted(
            f"{specification[role]}/final_metrics.json"
            for specification in FROZEN_COMPARATOR_EXPERIMENTS.values()
            for role in ("bigearthnet_s2", "cnn_random")
        ),
        "checkpoint_bytes": False,
        "training_cohort": False,
        "test_metrics": False,
        "verified_final": False,
    }
    if payload.get("read_scope") != expected_scope:
        raise BES2ContractError("frozen comparator read scope is invalid")
    for tag, specification in FROZEN_COMPARATOR_EXPERIMENTS.items():
        item = fractions[tag]
        if not isinstance(item, Mapping) or set(item) != {
            "fraction",
            "bigearthnet_s2",
            "cnn_random",
            "frozen_random_minus_s2_f1",
        }:
            raise BES2ContractError(f"{tag} comparator schema is invalid")
        if fraction_tag(item.get("fraction")) != tag:
            raise BES2ContractError(f"{tag} comparator fraction is invalid")
        for role in ("bigearthnet_s2", "cnn_random"):
            record = item.get(role)
            if not isinstance(record, Mapping) or set(record) != {
                "exp_id",
                "result",
                "best_dev_f1",
                "best_dev",
            }:
                raise BES2ContractError(f"{tag}/{role} comparator is invalid")
            if record.get("exp_id") != specification[role]:
                raise BES2ContractError(f"{tag}/{role} exp_id is invalid")
            binding = record.get("result")
            if (
                not isinstance(binding, Mapping)
                or set(binding) != {"path", "sha256"}
                or binding.get("path")
                != f"{specification[role]}/final_metrics.json"
                or not _HEX64.fullmatch(str(binding.get("sha256")))
            ):
                raise BES2ContractError(f"{tag}/{role} result binding is invalid")
            best = validate_dev_result(
                record.get("best_dev"),
                candidate_floor=0.05,
                description=f"{tag}/{role}.best_dev",
            )
            value = _finite_unit(
                record.get("best_dev_f1"), f"{tag}/{role}.best_dev_f1"
            )
            if value != float(best["f1"]):
                raise BES2ContractError(f"{tag}/{role} best DEV values differ")
        gap = float(item["cnn_random"]["best_dev_f1"]) - float(
            item["bigearthnet_s2"]["best_dev_f1"]
        )
        if gap <= 0.0 or item.get("frozen_random_minus_s2_f1") != gap:
            raise BES2ContractError(f"{tag} frozen deficit is invalid")
    return dict(payload)


def classify_cause(
    replay_best_dev_f1: object,
    reset_best_dev_f1: object,
    *,
    fraction: object,
    comparator: Mapping[str, object],
) -> dict[str, object]:
    """Apply the owner-predeclared replay and recovery decision per fraction."""

    replay = _finite_unit(replay_best_dev_f1, "current_replay best DEV F1")
    reset = _finite_unit(reset_best_dev_f1, "first_conv_reset best DEV F1")
    tag = fraction_tag(fraction)
    if fraction_tag(comparator.get("fraction")) != tag:
        raise BES2ContractError("cause comparator fraction differs from run")
    s2_record = comparator.get("bigearthnet_s2")
    random_record = comparator.get("cnn_random")
    if not isinstance(s2_record, Mapping) or not isinstance(random_record, Mapping):
        raise BES2ContractError("cause comparator records are absent")
    frozen_s2 = _finite_unit(
        s2_record.get("best_dev_f1"), f"{tag} frozen S2 DEV F1"
    )
    frozen_random = _finite_unit(
        random_record.get("best_dev_f1"), f"{tag} frozen random DEV F1"
    )
    if frozen_s2 >= frozen_random:
        raise BES2ContractError(f"{tag} frozen comparator has no S2 deficit")
    replay_delta = replay - frozen_s2
    denominator = frozen_random - replay

    common = {
        "fraction": float(comparator["fraction"]),
        "fraction_tag": tag,
        "current_replay_best_dev_f1": replay,
        "first_conv_reset_best_dev_f1": reset,
        "frozen_s2_best_dev_f1": frozen_s2,
        "frozen_random_best_dev_f1": frozen_random,
        "frozen_random_minus_s2_f1": frozen_random - frozen_s2,
        "replay_delta": replay_delta,
        "replay_tolerance": REPLAY_TOLERANCE,
        "recovery_denominator": denominator if denominator > 0.0 else None,
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
    if replay >= frozen_random:
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


def aggregate_fraction_decisions(
    decisions: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if set(decisions) != set(FROZEN_COMPARATOR_EXPERIMENTS):
        raise BES2ContractError("fraction decisions do not cover f10 and f50")
    classifications = {
        tag: str(decisions[tag].get("classification"))
        for tag in FROZEN_COMPARATOR_EXPERIMENTS
    }
    statuses = {
        tag: str(decisions[tag].get("status"))
        for tag in FROZEN_COMPARATOR_EXPERIMENTS
    }
    if any(status != "determinate" for status in statuses.values()):
        affected = sorted(
            tag for tag, status in statuses.items() if status != "determinate"
        )
        return {
            "status": "indeterminate",
            "classification": "fraction-replay-not-reproducible",
            "fraction_classifications": classifications,
            "concordant": False,
            "replacement_eligibility": "blocked",
            "required_next_step": (
                "review replay mismatch; diagnostic-only random replay may be "
                f"requested for {', '.join(affected)}"
            ),
        }
    unique = set(classifications.values())
    if len(unique) == 1:
        representative = decisions[next(iter(FROZEN_COMPARATOR_EXPERIMENTS))]
        return {
            "status": "determinate",
            "classification": next(iter(unique)),
            "fraction_classifications": classifications,
            "concordant": True,
            "replacement_eligibility": representative[
                "replacement_eligibility"
            ],
            "required_next_step": representative["required_next_step"],
        }
    return {
        "status": "determinate",
        "classification": "fraction-dependent-mixed-mechanism",
        "fraction_classifications": classifications,
        "concordant": False,
        "replacement_eligibility": "blocked",
        "required_next_step": (
            "review fraction-dependent activation and optimization evidence; "
            "do not approve a stem-reset replacement"
        ),
    }


OPTIMIZATION_TRACE_STEPS = (1, 10, 100, 500)


def validate_optimization_trace(
    payload: object,
    *,
    require_complete: bool = True,
) -> dict[str, object]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "milestones",
        "records",
    }:
        raise BES2ContractError("diagnostic optimization trace schema is invalid")
    records = payload.get("records")
    observed_steps = (
        [
            record.get("optimizer_step")
            for record in records
            if isinstance(record, Mapping)
        ]
        if isinstance(records, list)
        else None
    )
    expected_steps = list(OPTIMIZATION_TRACE_STEPS)
    if (
        payload.get("schema") != 1
        or payload.get("milestones") != expected_steps
        or not isinstance(records, list)
        or observed_steps
        != (expected_steps if require_complete else expected_steps[: len(records)])
    ):
        raise BES2ContractError(
            "diagnostic optimization milestones are invalid or incomplete"
        )
    required_groups = {
        "stem",
        "stage_0",
        "stage_1",
        "stage_2",
        "stage_3",
        "detector_head",
    }
    group_fields = {
        "parameter_count",
        "parameter_norm_before",
        "gradient_norm",
        "gradient_to_parameter_norm",
        "lr_min",
        "lr_max",
        "lr_scale_min",
        "lr_scale_max",
        "update_norm",
        "relative_update_norm",
    }
    first_fields = {
        "name",
        "parameter_count",
        "parameter_norm_before",
        "gradient_norm",
        "gradient_to_parameter_norm",
        "lr",
        "lr_scale",
        "update_norm",
        "relative_update_norm",
    }
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "optimizer_step",
            "groups",
            "first_convolution",
        }:
            raise BES2ContractError("optimization trace record is invalid")
        groups = record.get("groups")
        first = record.get("first_convolution")
        if (
            not isinstance(groups, Mapping)
            or not required_groups <= set(groups)
            or not isinstance(first, Mapping)
            or set(first) != first_fields
            or first.get("name") != "backbone.model.stem.0.weight"
        ):
            raise BES2ContractError("optimization trace groups are invalid")
        for name, item in [*groups.items(), ("first_convolution", first)]:
            expected = first_fields if name == "first_convolution" else group_fields
            if not isinstance(item, Mapping) or set(item) != expected:
                raise BES2ContractError(f"optimization trace {name} schema is invalid")
            count = item.get("parameter_count")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise BES2ContractError(
                    f"optimization trace {name} parameter count is invalid"
                )
            for field, value in item.items():
                if field in {"name", "parameter_count"}:
                    continue
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise BES2ContractError(
                        f"optimization trace {name}.{field} is invalid"
                    )
    return dict(payload)


def validate_training_metrics(
    path: str | Path,
    *,
    expected_variant: str,
    expected_stage: str,
    expected_fraction: float,
    candidate_floor: float,
) -> tuple[dict[str, object], Path]:
    marker_path = Path(path)
    payload = read_regular_json(marker_path, "diagnostic training metrics")
    expected_top = {
        "diagnostic_run_schema",
        "purpose",
        "variant",
        "stage",
        "fraction",
        "trainer_max_epochs",
        "schedule_horizon_epochs",
        "initialization",
        "dev_history",
        "optimization_trace",
        "runtime",
        "training_result",
    }
    if (
        set(payload) != expected_top
        or payload.get("diagnostic_run_schema") != 2
        or payload.get("purpose") != "bes2-root-cause-training"
        or payload.get("variant") != expected_variant
        or payload.get("stage") != expected_stage
        or fraction_tag(payload.get("fraction"))
        != fraction_tag(expected_fraction)
        or marker_path.parent.name
        != diagnostic_run_id(expected_variant, expected_stage, expected_fraction)
    ):
        raise BES2ContractError("diagnostic training marker identity is invalid")
    expected_epochs = 5 if expected_stage == "probe" else 50
    if (
        payload.get("trainer_max_epochs") != expected_epochs
        or payload.get("schedule_horizon_epochs") != 50
    ):
        raise BES2ContractError("diagnostic 5/50 epoch contract is invalid")
    validate_optimization_trace(payload.get("optimization_trace"))

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
    expected_fraction: float,
    expected_stage: str = "full",
) -> dict[str, object]:
    payload = read_regular_json(path, "diagnostic metrics")
    return validate_diagnostic_metrics_payload(
        payload,
        expected_variant=expected_variant,
        expected_fraction=expected_fraction,
        expected_stage=expected_stage,
    )


def validate_diagnostic_metrics_payload(
    payload: Mapping[str, object],
    *,
    expected_variant: str,
    expected_fraction: float,
    expected_stage: str = "full",
) -> dict[str, object]:
    """Validate diagnostic evidence without requiring its original Judy path."""

    required = {
        "diagnostic_metrics_schema",
        "status",
        "purpose",
        "variant",
        "stage",
        "fraction",
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
        or fraction_tag(payload.get("fraction"))
        != fraction_tag(expected_fraction)
    ):
        raise BES2ContractError("diagnostic metrics identity is invalid")
    training = payload.get("training")
    dev = payload.get("dev_evidence")
    if not isinstance(training, Mapping) or not isinstance(dev, Mapping):
        raise BES2ContractError("diagnostic metrics evidence is incomplete")
    if (
        training.get("variant") != expected_variant
        or training.get("stage") != expected_stage
        or fraction_tag(training.get("fraction"))
        != fraction_tag(expected_fraction)
    ):
        raise BES2ContractError(
            "diagnostic metrics training identity is inconsistent"
        )
    validate_optimization_trace(training.get("optimization_trace"))
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


def _validate_summary_binding(
    binding: Mapping[str, object], description: str
) -> dict[str, str]:
    if (
        set(binding) != {"path", "sha256"}
        or not isinstance(binding.get("path"), str)
        or not binding.get("path")
        or not _HEX64.fullmatch(str(binding.get("sha256")))
    ):
        raise BES2ContractError(f"{description} binding is invalid")
    return {"path": str(binding["path"]), "sha256": str(binding["sha256"])}


def summarize_payload(
    metrics: Mapping[str, Mapping[str, Mapping[str, object]]],
    *,
    run_bindings: Mapping[str, Mapping[str, Mapping[str, object]]],
    comparator_receipt: Mapping[str, object],
    comparator_binding: Mapping[str, object],
) -> dict[str, object]:
    """Build the no-pooling f10/f50 root-cause decision."""

    comparator = validate_frozen_comparator_payload(comparator_receipt)
    expected_tags = set(FROZEN_COMPARATOR_EXPERIMENTS)
    if set(metrics) != expected_tags or set(run_bindings) != expected_tags:
        raise BES2ContractError("summary inputs do not cover f10 and f50")

    decisions: dict[str, dict[str, object]] = {}
    runs: dict[str, dict[str, dict[str, str]]] = {}
    supporting: dict[str, dict[str, object]] = {}
    for tag, specification in FROZEN_COMPARATOR_EXPERIMENTS.items():
        fraction = float(specification["fraction"])
        fraction_metrics = metrics[tag]
        fraction_bindings = run_bindings[tag]
        if (
            set(fraction_metrics) != set(DIAGNOSTIC_VARIANTS)
            or set(fraction_bindings) != set(DIAGNOSTIC_VARIANTS)
        ):
            raise BES2ContractError(f"{tag} summary variants are incomplete")

        validated: dict[str, dict[str, object]] = {}
        runs[tag] = {}
        for variant in DIAGNOSTIC_VARIANTS:
            validated[variant] = validate_diagnostic_metrics_payload(
                fraction_metrics[variant],
                expected_variant=variant,
                expected_stage="full",
                expected_fraction=fraction,
            )
            runs[tag][variant] = _validate_summary_binding(
                fraction_bindings[variant], f"{tag}/{variant}"
            )

        replay_training = validated["current_replay"].get("training")
        reset_training = validated["first_conv_reset"].get("training")
        if not isinstance(replay_training, Mapping) or not isinstance(
            reset_training, Mapping
        ):
            raise BES2ContractError(f"{tag} summary training evidence is absent")
        replay_result = replay_training.get("training_result")
        reset_result = reset_training.get("training_result")
        if not isinstance(replay_result, Mapping) or not isinstance(
            reset_result, Mapping
        ):
            raise BES2ContractError(
                f"{tag} summary schema-2 training results are absent"
            )

        decisions[tag] = classify_cause(
            replay_result.get("best_dev_f1"),
            reset_result.get("best_dev_f1"),
            fraction=fraction,
            comparator=comparator["fractions"][tag],
        )
        supporting[tag] = {}
        for variant, training, result in (
            ("current_replay", replay_training, replay_result),
            ("first_conv_reset", reset_training, reset_result),
        ):
            metric = validated[variant]
            supporting[tag][variant] = {
                "best_dev": result.get("best_dev"),
                "dev_evidence": metric.get("dev_evidence"),
                "activation_statistics": metric.get("activation_statistics"),
                "layer_drift": metric.get("layer_drift"),
                "optimization_trace": training.get("optimization_trace"),
            }

    aggregate = aggregate_fraction_decisions(decisions)
    return {
        "root_cause_schema": ROOT_CAUSE_SCHEMA,
        "status": aggregate["status"],
        "purpose": "bes2-root-cause-decision",
        "created_utc": utc_now(),
        "scope": {
            "split": "TRAIN+fixed-DEV8-only",
            "fractions": list(DIAGNOSTIC_FRACTIONS),
            "probe_fractions": list(PROBE_FRACTIONS),
            "analysis_unit": "per-fraction-no-pooling",
            "seed": 0,
            "single_seed_point_estimates": True,
            "significance_claims": False,
            "seed_variance_claims": False,
            "bigearthnet_s1": (
                "out-of-scope; shared-helper use remains an unresolved "
                "compliance issue"
            ),
        },
        "comparators": {
            "receipt": _validate_summary_binding(
                comparator_binding, "frozen comparator"
            ),
            "campaign_git_sha": comparator["campaign_git_sha"],
            "fractions": {
                tag: {
                    "fraction": comparator["fractions"][tag]["fraction"],
                    "bigearthnet_s2": comparator["fractions"][tag][
                        "bigearthnet_s2"
                    ],
                    "cnn_random": comparator["fractions"][tag]["cnn_random"],
                }
                for tag in FROZEN_COMPARATOR_EXPERIMENTS
            },
        },
        "runs": runs,
        "decision": {
            "per_fraction": decisions,
            "aggregate": aggregate,
        },
        "supporting_evidence": supporting,
    }
