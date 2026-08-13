"""Hash-bound owner authorization for the post-TEST final-eval amendment.

This receipt does not make a failed monotonicity diagnostic pass.  It records
the owner's decision to consume the verified-final set once, for all 32 frozen
cells, after the exact failed TEST grid has been reviewed.  Every scientific
artifact remains bound to the original campaign source; the clean evaluator
source is recorded separately.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.eval.heldout_contract import HeldoutContractError
from src.eval.result_contract import sha256_file


AUTHORIZATION_SCHEMA = 1
AUTHORIZATION_STATUS = "owner-authorized"
AUTHORIZATION_POLICY = "post-test-monotonicity-stop-all-32-final-once-v1"
AUTHORIZATION_FILENAME = "FINAL_EVAL_OWNER_AMENDMENT.json"
AUTHORIZED_OWNER = "johnroth"
OWNER_DECISION = (
    "evaluate-all-32-core-cells-on-the-50-human-verified-scenes-exactly-once;"
    "preserve-and-disclose-the-failed-test-monotonicity-diagnostic"
)
CONSTRAINTS = {
    "selected_cell_count": 32,
    "all_arms_and_fractions": True,
    "no_retraining": True,
    "no_checkpoint_changes": True,
    "no_threshold_changes": True,
    "no_final_tuning": True,
    "no_selective_retry": True,
    "no_requeue": True,
    "single_seed_point_estimates": True,
    "post_test_amendment_disclosed": True,
}

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _regular_json(path: Path, description: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise HeldoutContractError(
            f"{description} must be a regular non-symlink: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeldoutContractError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise HeldoutContractError(f"{description} root must be a JSON object")
    return payload


def _utc_timestamp(value: object) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise HeldoutContractError("owner authorization created_utc is absent")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HeldoutContractError(
            "owner authorization created_utc is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise HeldoutContractError(
            "owner authorization timestamps must be canonical UTC"
        )
    return parsed


def _hash_binding(value: object, *, description: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "relative_path",
        "sha256",
    }:
        raise HeldoutContractError(f"{description} binding is invalid")
    relative_path = value.get("relative_path")
    digest = value.get("sha256")
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(relative_path).parts)
        or not isinstance(digest, str)
        or not _HEX64.fullmatch(digest)
    ):
        raise HeldoutContractError(f"{description} binding is malformed")
    return {"relative_path": relative_path, "sha256": digest}


def build_authorization(
    *,
    owner: str,
    campaign: Mapping[str, object],
    evaluator_git_sha: str,
    cohort: Mapping[str, str],
    phase5_controls: Mapping[str, object],
    grid: Mapping[str, object],
    test_results: Sequence[Mapping[str, str]],
    selected_cells: Sequence[str],
    created_utc: str | None = None,
) -> dict[str, object]:
    """Build the exact schema-1 authorization payload."""

    if owner != AUTHORIZED_OWNER:
        raise HeldoutContractError(
            f"owner authorization identity must be {AUTHORIZED_OWNER!r}"
        )
    if not _HEX40.fullmatch(evaluator_git_sha):
        raise HeldoutContractError("evaluator Git SHA is malformed")
    payload = {
        "authorization_schema": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZATION_STATUS,
        "created_utc": created_utc
        or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "owner": owner.strip(),
        "decision": OWNER_DECISION,
        "policy": AUTHORIZATION_POLICY,
        "campaign": dict(campaign),
        "evaluator_git_sha": evaluator_git_sha,
        "cohort": dict(cohort),
        "phase5_controls": dict(phase5_controls),
        "grid": dict(grid),
        "test_results": [dict(item) for item in test_results],
        "selected_cells": list(selected_cells),
        "constraints": dict(CONSTRAINTS),
    }
    try:
        normalized = json.loads(
            json.dumps(payload, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise HeldoutContractError(
            "owner authorization cannot be normalized as strict JSON"
        ) from exc
    # Apply the same strict validation used at consumption time.
    validate_authorization_payload(normalized, expected=normalized)
    return normalized


def validate_authorization_payload(
    payload: Mapping[str, object],
    *,
    expected: Mapping[str, object],
) -> dict[str, object]:
    """Require an exact receipt apart from human identity and decision time."""

    keys = {
        "authorization_schema",
        "status",
        "created_utc",
        "owner",
        "decision",
        "policy",
        "campaign",
        "evaluator_git_sha",
        "cohort",
        "phase5_controls",
        "grid",
        "test_results",
        "selected_cells",
        "constraints",
    }
    if set(payload) != keys or set(expected) != keys:
        raise HeldoutContractError("owner authorization keys do not match schema 1")
    if (
        payload.get("authorization_schema") != AUTHORIZATION_SCHEMA
        or payload.get("status") != AUTHORIZATION_STATUS
        or payload.get("decision") != OWNER_DECISION
        or payload.get("policy") != AUTHORIZATION_POLICY
        or payload.get("constraints") != CONSTRAINTS
    ):
        raise HeldoutContractError("owner authorization policy is invalid")
    created_utc = _utc_timestamp(payload.get("created_utc"))
    owner = payload.get("owner")
    if owner != AUTHORIZED_OWNER:
        raise HeldoutContractError("owner authorization identity is invalid")

    evaluator_git_sha = payload.get("evaluator_git_sha")
    if not isinstance(evaluator_git_sha, str) or not _HEX40.fullmatch(
        evaluator_git_sha
    ):
        raise HeldoutContractError("owner authorization evaluator SHA is malformed")

    campaign = payload.get("campaign")
    if not isinstance(campaign, Mapping) or set(campaign) != {
        "campaign_id",
        "git_sha",
        "runs_root",
        "manifest",
        "terminal_event",
    }:
        raise HeldoutContractError("owner authorization campaign binding is invalid")
    campaign_git_sha = campaign.get("git_sha")
    if not isinstance(campaign_git_sha, str) or not _HEX40.fullmatch(
        campaign_git_sha
    ):
        raise HeldoutContractError("owner authorization campaign SHA is malformed")
    campaign_id = campaign.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id.strip():
        raise HeldoutContractError("owner authorization campaign ID is absent")
    runs_root = campaign.get("runs_root")
    if (
        not isinstance(runs_root, str)
        or not runs_root.startswith("/")
        or os.path.normpath(runs_root) != runs_root
        or runs_root == "/"
    ):
        raise HeldoutContractError("owner authorization runs root is invalid")
    _hash_binding(campaign.get("manifest"), description="campaign manifest")
    if campaign["manifest"].get("relative_path") != ".h100/campaign_manifest.json":
        raise HeldoutContractError("owner authorization campaign path is invalid")
    terminal = campaign.get("terminal_event")
    if (
        not isinstance(terminal, Mapping)
        or set(terminal) != {"event", "error", "utc"}
        or terminal.get("event") != "grid_validation_failed"
        or not isinstance(terminal.get("error"), str)
        or "monotonicity STOP" not in str(terminal.get("error"))
    ):
        raise HeldoutContractError(
            "owner authorization lacks the monotonicity terminal event"
        )
    terminal_utc = _utc_timestamp(terminal.get("utc"))
    if created_utc < terminal_utc:
        raise HeldoutContractError(
            "owner authorization predates the monotonicity STOP"
        )

    cohort = _hash_binding(payload.get("cohort"), description="training cohort")
    if cohort["relative_path"] != ".h100/TRAINING_COHORT.json":
        raise HeldoutContractError("owner authorization cohort path is invalid")
    controls = payload.get("phase5_controls")
    if not isinstance(controls, Mapping) or set(controls) != {
        "status",
        "h100_ready",
        "cutover_ready",
        "v100_diagnostic_isolation",
    }:
        raise HeldoutContractError("owner authorization Phase-5 controls are invalid")
    if controls.get("status") != "reporting-controls-persisted":
        raise HeldoutContractError("owner authorization Phase-5 status is invalid")
    h100_ready = _hash_binding(
        controls.get("h100_ready"), description="H100_READY"
    )
    cutover = _hash_binding(
        controls.get("cutover_ready"), description="CUTOVER_READY"
    )
    isolation = _hash_binding(
        controls.get("v100_diagnostic_isolation"),
        description="V100 diagnostic-isolation",
    )
    if h100_ready["relative_path"] != ".h100/H100_READY.json":
        raise HeldoutContractError("owner authorization H100_READY path is invalid")
    if cutover["relative_path"] != ".h100/CUTOVER_READY.json":
        raise HeldoutContractError("owner authorization CUTOVER_READY path is invalid")
    if isolation["relative_path"] != ".h100/V100_DIAGNOSTIC_ISOLATION.json":
        raise HeldoutContractError(
            "owner authorization diagnostic-isolation path is invalid"
        )
    grid = payload.get("grid")
    if not isinstance(grid, Mapping) or set(grid) != {
        "relative_path",
        "sha256",
        "monotonicity_tolerance",
        "monotonicity_ok",
        "violations",
    }:
        raise HeldoutContractError("owner authorization grid binding is invalid")
    _hash_binding(
        {"relative_path": grid.get("relative_path"), "sha256": grid.get("sha256")},
        description="TEST grid",
    )
    if grid.get("relative_path") != "summary/grid.csv":
        raise HeldoutContractError("owner authorization grid path is invalid")
    if grid.get("monotonicity_ok") is not False:
        raise HeldoutContractError(
            "owner amendment requires a truthfully failed monotonicity diagnostic"
        )
    tolerance = grid.get("monotonicity_tolerance")
    if type(tolerance) is not float or tolerance != 0.02:
        raise HeldoutContractError("owner authorization tolerance is not exactly 0.02")
    violations = grid.get("violations")
    if not isinstance(violations, list) or not violations:
        raise HeldoutContractError("owner authorization has no bound violations")
    violation_keys = {
        "init",
        "from_fraction",
        "to_fraction",
        "from_test_f1",
        "to_test_f1",
        "drop",
    }
    observed_transitions: set[tuple[str, float, float]] = set()
    for item in violations:
        if not isinstance(item, Mapping) or set(item) != violation_keys:
            raise HeldoutContractError(
                "owner authorization monotonicity violation is invalid"
            )
        init_name = item.get("init")
        if not isinstance(init_name, str) or not init_name:
            raise HeldoutContractError(
                "owner authorization monotonicity init is invalid"
            )
        numeric: dict[str, float] = {}
        for field in violation_keys - {"init"}:
            value = item.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HeldoutContractError(
                    f"owner authorization monotonicity {field} is invalid"
                )
            numeric[field] = float(value)
            if not math.isfinite(numeric[field]):
                raise HeldoutContractError(
                    f"owner authorization monotonicity {field} is not finite"
                )
        from_fraction = numeric["from_fraction"]
        to_fraction = numeric["to_fraction"]
        from_f1 = numeric["from_test_f1"]
        to_f1 = numeric["to_test_f1"]
        drop = numeric["drop"]
        if (
            (from_fraction, to_fraction)
            not in {(0.1, 0.25), (0.25, 0.5), (0.5, 1.0)}
            or not 0.0 <= from_f1 <= 1.0
            or not 0.0 <= to_f1 <= 1.0
            or drop <= tolerance
            or not math.isclose(drop, from_f1 - to_f1, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise HeldoutContractError(
                "owner authorization monotonicity violation is inconsistent"
            )
        transition = (init_name, from_fraction, to_fraction)
        if transition in observed_transitions:
            raise HeldoutContractError(
                "owner authorization repeats a monotonicity violation"
            )
        observed_transitions.add(transition)

    tests = payload.get("test_results")
    selected = payload.get("selected_cells")
    if (
        not isinstance(tests, list)
        or len(tests) != 32
        or not isinstance(selected, list)
        or len(selected) != 32
        or any(not isinstance(exp_id, str) or not exp_id for exp_id in selected)
        or len(set(selected)) != 32
    ):
        raise HeldoutContractError(
            "owner authorization must bind all 32 TEST results and selected cells"
        )
    observed_ids: list[str] = []
    for item in tests:
        if not isinstance(item, Mapping) or set(item) != {
            "exp_id",
            "relative_path",
            "sha256",
        }:
            raise HeldoutContractError("owner authorization TEST binding is invalid")
        exp_id = item.get("exp_id")
        if not isinstance(exp_id, str) or not exp_id:
            raise HeldoutContractError("owner authorization TEST ID is invalid")
        _hash_binding(
            {
                "relative_path": item.get("relative_path"),
                "sha256": item.get("sha256"),
            },
            description=f"{exp_id} TEST result",
        )
        if item.get("relative_path") != f"{exp_id}/test_metrics.json":
            raise HeldoutContractError(
                f"owner authorization TEST path is invalid for {exp_id}"
            )
        observed_ids.append(exp_id)
    if observed_ids != selected or len(set(observed_ids)) != 32:
        raise HeldoutContractError(
            "owner authorization TEST bindings differ from selected-cell order"
        )

    # Everything except human identity/time is derived from immutable inputs and
    # must equal the independently rebuilt expected payload exactly.
    for key in keys - {"owner", "created_utc"}:
        if payload.get(key) != expected.get(key):
            raise HeldoutContractError(
                f"owner authorization {key} differs from current immutable evidence"
            )
    return dict(payload)


def validate_authorization(
    path: Path,
    *,
    expected: Mapping[str, object],
) -> tuple[dict[str, object], str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise HeldoutContractError(
            "owner final-eval authorization cannot be opened safely"
        ) from exc
    try:
        with os.fdopen(fd, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise HeldoutContractError(
                    "owner final-eval authorization is not a regular file"
                )
            if metadata.st_mode & 0o222:
                raise HeldoutContractError(
                    "owner final-eval authorization is writable"
                )
            raw = stream.read()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise HeldoutContractError(
                "owner final-eval authorization root must be an object"
            )
    except json.JSONDecodeError as exc:
        raise HeldoutContractError(
            "owner final-eval authorization is invalid JSON"
        ) from exc
    except OSError as exc:
        raise HeldoutContractError(
            "owner final-eval authorization cannot be inspected"
        ) from exc
    if path.is_symlink():
        raise HeldoutContractError(
            "owner final-eval authorization cannot be a symlink"
        )
    return (
        validate_authorization_payload(payload, expected=expected),
        hashlib.sha256(raw).hexdigest(),
    )


def write_authorization(path: Path, payload: Mapping[str, object]) -> str:
    """Publish a new immutable authorization receipt exactly once."""

    validated = validate_authorization_payload(payload, expected=payload)
    try:
        raw = (
            json.dumps(validated, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HeldoutContractError(
            "owner final-eval authorization cannot be encoded"
        ) from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise HeldoutContractError(
            f"owner final-eval authorization already exists: {path}"
        ) from exc
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        # Persist the directory entry where supported.
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except (OSError, TypeError, ValueError) as exc:
        raise HeldoutContractError(
            "could not publish immutable owner final-eval authorization"
        ) from exc
    return hashlib.sha256(raw).hexdigest()
