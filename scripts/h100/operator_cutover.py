"""Validate the operator-authored V100 diagnostic-isolation attestation.

The owner may keep the V100 campaign running as a non-reportable diagnostic.
This module never creates the attestation, never controls a process, and only
permits H100 launch after byte-bound namespace and suppression isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Mapping

from scripts.h100.contracts import (
    V100_DIAGNOSTIC_COMPLETE,
    validate_bound_cutover_forecast,
)

HEX64 = re.compile(r"^[0-9a-f]{64}$")
CUTOVER_READY_NAME = "CUTOVER_READY.json"
V100_DIAGNOSTIC_ISOLATION_NAME = "V100_DIAGNOSTIC_ISOLATION.json"
DIAGNOSTIC_ATTESTATION_KEYS = {
    "schema",
    "status",
    "created_utc",
    "attestation",
    "cutover_ready_sha256",
    "h100",
    "v100",
    "namespaces",
    "h100_suppression",
    "post_diagnostic",
}
DIAGNOSTIC_H100_KEYS = {"acceptance_uuid", "git_sha", "campaign_id"}
DIAGNOSTIC_V100_KEYS = {
    "git_sha",
    "campaign_id",
    "execution_status",
    "diagnostic_status",
}
DIAGNOSTIC_NAMESPACE_KEYS = {"v100_runs_root", "h100_runs_root", "disjoint"}
DIAGNOSTIC_SUPPRESSION_KEYS = {
    "v100_completions_suppress_h100",
    "v100_checkpoints_resume_h100",
    "mixed_hardware_curve_allowed",
}
DIAGNOSTIC_POST_KEYS = {"safe_stop_archive", "required_before_h100_campaign"}


def sha256_file(path: Path, chunk_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, description: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object: {path}")
    return value


def _require_regular_nonsymlink(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise RuntimeError(f"{label} path must be absolute")
    if path.is_symlink():
        raise RuntimeError(f"{label} path cannot be a symlink")
    if not path.is_file():
        raise RuntimeError(f"{label} is not a regular file")


def persist_immutable_copy(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> Path:
    """Install byte-identical evidence once; never replace an existing file."""

    _require_regular_nonsymlink(source, "operator evidence source")
    if not destination.is_absolute():
        raise RuntimeError("operator evidence destination must be absolute")
    if not HEX64.fullmatch(expected_sha256):
        raise RuntimeError("operator evidence expected SHA-256 is not 64-hex")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink():
        raise RuntimeError("operator evidence directory cannot be a symlink")
    if destination.exists() or destination.is_symlink():
        _require_regular_nonsymlink(destination, "persisted operator evidence")
        if sha256_file(destination) != expected_sha256:
            raise RuntimeError(
                f"immutable operator evidence already differs: {destination}"
            )
        destination.chmod(0o444)
        return destination

    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_fd = os.open(source, source_flags)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".installing",
        dir=destination.parent,
    )
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(source_fd).st_mode):
            raise RuntimeError("operator evidence source is not a regular file")
        with os.fdopen(source_fd, "rb") as source_handle, os.fdopen(
            temporary_fd, "wb"
        ) as destination_handle:
            source_fd = -1
            temporary_fd = -1
            while block := source_handle.read(8 * 2**20):
                digest.update(block)
                destination_handle.write(block)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError("operator evidence changed while it was copied")
        os.chmod(temporary_name, 0o444)
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            _require_regular_nonsymlink(destination, "persisted operator evidence")
            if sha256_file(destination) != expected_sha256:
                raise RuntimeError(
                    f"immutable operator evidence race differs: {destination}"
                )
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    _require_regular_nonsymlink(destination, "persisted operator evidence")
    if sha256_file(destination) != expected_sha256:
        raise RuntimeError("persisted operator evidence SHA-256 mismatch")
    return destination


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{label} is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} must include a timezone")
    return parsed



def _normalized_external_root(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise RuntimeError(f"{label} must be an absolute POSIX path")
    normalized = os.path.normpath(value)
    if normalized != value or normalized == "/" or ".." in Path(value).parts:
        raise RuntimeError(f"{label} must be normalized and cannot be a broad root")
    return normalized


def _validate_reference_bindings(
    cutover: Mapping[str, object],
    *,
    expected_reference_git_sha: str,
    expected_reference_campaign_id: str,
    expected_v100_core_git_sha: str,
    expected_v100_core_campaign_id: str,
) -> None:
    campaign_record = cutover.get("reference_campaign")
    campaign = (
        campaign_record.get("manifest")
        if isinstance(campaign_record, Mapping)
        else None
    )
    manifest_keys = {
        "schema",
        "campaign_role",
        "campaign_id",
        "core_campaign_id",
        "core_git_sha",
        "git_sha",
        "environment_sha256",
        "environment_lock_sha256",
        "runtime_launcher_sha256",
    }
    expected_identity = {
        "schema": 1,
        "campaign_role": "corrected-v100-references",
        "campaign_id": expected_reference_campaign_id,
        "core_campaign_id": expected_v100_core_campaign_id,
        "core_git_sha": expected_v100_core_git_sha,
        "git_sha": expected_reference_git_sha,
    }
    if (
        not isinstance(campaign_record, Mapping)
        or set(campaign_record) != {"manifest", "manifest_sha256"}
        or not HEX64.fullmatch(str(campaign_record.get("manifest_sha256", "")))
        or not isinstance(campaign, Mapping)
        or set(campaign) != manifest_keys
        or hashlib.sha256(
            (json.dumps(dict(campaign), indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        ).hexdigest()
        != campaign_record.get("manifest_sha256")
        or any(campaign.get(key) != value for key, value in expected_identity.items())
        or any(
            not HEX64.fullmatch(str(campaign.get(key, "")))
            for key in (
                "environment_sha256",
                "environment_lock_sha256",
                "runtime_launcher_sha256",
            )
        )
    ):
        raise RuntimeError("CUTOVER_READY reference campaign binding is invalid")
    expected_hashes = {
        "environment_sha256": campaign["environment_sha256"],
        "environment_lock_sha256": campaign["environment_lock_sha256"],
        "campaign_manifest_sha256": campaign_record["manifest_sha256"],
        "runtime_launcher_sha256": campaign["runtime_launcher_sha256"],
    }
    references = cutover.get("references")
    if not isinstance(references, Mapping) or set(references) != {"r2", "r3"}:
        raise RuntimeError("CUTOVER_READY reference bindings are absent")
    for name, record in references.items():
        if not isinstance(record, Mapping):
            raise RuntimeError(f"CUTOVER_READY {name} binding is invalid")
        provenance = record.get("provenance")
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("git_sha") != expected_reference_git_sha
            or provenance.get("campaign_id") != expected_reference_campaign_id
            or any(
                provenance.get(key) != value
                for key, value in expected_hashes.items()
            )
        ):
            raise RuntimeError(f"CUTOVER_READY {name} V100 identity mismatch")
    control = cutover.get("references_control")
    control_keys = {
        "package_id", "kind", "direction", "identity_sha256",
        "producer_git_sha",
        "manifest_sha256", "ready_sha256", "sha256sums_sha256",
    }
    if (
        not isinstance(control, Mapping)
        or set(control) != control_keys
        or control.get("kind") != "references"
        or control.get("direction") != "v100-to-judy"
        or not re.fullmatch(r"[0-9a-f]{40}", str(control.get("producer_git_sha", "")))
        or any(
            not HEX64.fullmatch(str(control.get(key, "")))
            for key in (
                "identity_sha256", "manifest_sha256", "ready_sha256",
                "sha256sums_sha256",
            )
        )
    ):
        raise RuntimeError("CUTOVER_READY references control-package binding is invalid")


def _validate_package_sha256sums_bindings(
    cutover: Mapping[str, object],
    *,
    expected_base_payload_sha256sums_sha256: str,
    expected_runtime_amendment_sha256sums_sha256: str,
) -> None:
    """Bind operator evidence to the exact accepted base/runtime packages."""

    acceptance = cutover.get("acceptance")
    base_payload = (
        acceptance.get("base_payload")
        if isinstance(acceptance, Mapping)
        else None
    )
    runtime_amendment = (
        acceptance.get("runtime_amendment")
        if isinstance(acceptance, Mapping)
        else None
    )
    for value, label in (
        (
            expected_base_payload_sha256sums_sha256,
            "base-payload SHA256SUMS",
        ),
        (
            expected_runtime_amendment_sha256sums_sha256,
            "runtime-amendment SHA256SUMS",
        ),
    ):
        if HEX64.fullmatch(value) is None:
            raise RuntimeError(f"expected {label} digest is not 64-hex")
    if (
        not isinstance(base_payload, Mapping)
        or base_payload.get("sha256sums_sha256")
        != expected_base_payload_sha256sums_sha256
    ):
        raise RuntimeError("CUTOVER_READY base-payload SHA256SUMS binding mismatch")
    if (
        not isinstance(runtime_amendment, Mapping)
        or runtime_amendment.get("sha256sums_sha256")
        != expected_runtime_amendment_sha256sums_sha256
    ):
        raise RuntimeError(
            "CUTOVER_READY runtime-amendment SHA256SUMS binding mismatch"
        )


def validate_diagnostic_isolation(
    *,
    cutover_ready: Path,
    cutover_ready_sha256: str,
    attestation: Path,
    attestation_sha256: str,
    expected_h100_git_sha: str,
    expected_h100_campaign_id: str,
    expected_h100_runs_root: str,
    expected_reference_git_sha: str,
    expected_reference_campaign_id: str,
    expected_v100_core_git_sha: str,
    expected_v100_core_campaign_id: str,
) -> dict[str, object]:
    """Validate the human decision to keep V100 running only diagnostically."""

    for path, label in (
        (cutover_ready, "CUTOVER_READY"),
        (attestation, "V100_DIAGNOSTIC_ISOLATION"),
    ):
        _require_regular_nonsymlink(path, label)
    for digest, label in (
        (cutover_ready_sha256, "CUTOVER_READY"),
        (attestation_sha256, "V100_DIAGNOSTIC_ISOLATION"),
    ):
        if HEX64.fullmatch(digest) is None:
            raise RuntimeError(f"{label} expected SHA-256 is not 64-hex")
    if sha256_file(cutover_ready) != cutover_ready_sha256:
        raise RuntimeError("CUTOVER_READY SHA-256 mismatch")
    if sha256_file(attestation) != attestation_sha256:
        raise RuntimeError("V100_DIAGNOSTIC_ISOLATION SHA-256 mismatch")

    cutover = _json_object(cutover_ready, "CUTOVER_READY receipt")
    isolated = _json_object(attestation, "V100_DIAGNOSTIC_ISOLATION attestation")
    if cutover.get("schema") != 2 or cutover.get("status") != "cutover-ready":
        raise RuntimeError("CUTOVER_READY schema/status is invalid")
    if cutover.get("h100_campaign_id") != expected_h100_campaign_id:
        raise RuntimeError("CUTOVER_READY H100 campaign identity mismatch")
    if cutover.get("v100_action") != (
        "none; this guard never stops or signals V100 processes"
    ):
        raise RuntimeError("CUTOVER_READY crosses the no-process-action boundary")
    cutover_forecast = validate_bound_cutover_forecast(cutover)
    acceptance = cutover.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise RuntimeError("CUTOVER_READY acceptance binding is absent")
    source = acceptance.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("git_sha") != expected_h100_git_sha
        or not str(acceptance.get("uuid", "")).strip()
    ):
        raise RuntimeError("CUTOVER_READY H100 source/acceptance identity mismatch")
    if "evaluation_ground_truth" not in acceptance:
        raise RuntimeError(
            "CUTOVER_READY does not preserve the evaluation-ground-truth gate"
        )
    _validate_reference_bindings(
        cutover,
        expected_reference_git_sha=expected_reference_git_sha,
        expected_reference_campaign_id=expected_reference_campaign_id,
        expected_v100_core_git_sha=expected_v100_core_git_sha,
        expected_v100_core_campaign_id=expected_v100_core_campaign_id,
    )

    if set(isolated) != DIAGNOSTIC_ATTESTATION_KEYS:
        raise RuntimeError("diagnostic-isolation top-level keys do not match schema")
    if (
        isolated.get("schema") != 1
        or isolated.get("status") != "v100-diagnostic-isolated"
        or isolated.get("attestation") != "external-human-operator"
        or isolated.get("cutover_ready_sha256") != cutover_ready_sha256
    ):
        raise RuntimeError("diagnostic-isolation identity/status is invalid")

    h100 = isolated.get("h100")
    expected_h100 = {
        "acceptance_uuid": acceptance["uuid"],
        "git_sha": expected_h100_git_sha,
        "campaign_id": expected_h100_campaign_id,
    }
    if not isinstance(h100, Mapping) or set(h100) != DIAGNOSTIC_H100_KEYS:
        raise RuntimeError("diagnostic-isolation H100 keys do not match schema")
    if dict(h100) != expected_h100:
        raise RuntimeError("diagnostic-isolation H100 identity mismatch")

    v100 = isolated.get("v100")
    expected_execution_status = str(cutover_forecast["v100_diagnostic_status"])
    expected_v100 = {
        "git_sha": expected_v100_core_git_sha,
        "campaign_id": expected_v100_core_campaign_id,
        "execution_status": expected_execution_status,
        "diagnostic_status": "non-reportable-diagnostic",
    }
    if not isinstance(v100, Mapping) or set(v100) != DIAGNOSTIC_V100_KEYS:
        raise RuntimeError("diagnostic-isolation V100 keys do not match schema")
    if dict(v100) != expected_v100:
        raise RuntimeError("diagnostic-isolation V100 identity/disposition mismatch")
    if (
        expected_execution_status == V100_DIAGNOSTIC_COMPLETE
        and cutover_forecast["current_remaining_v100_wall_hours"] != 0.0
    ):
        raise RuntimeError("completed V100 diagnostic has nonzero remaining hours")

    namespaces = isolated.get("namespaces")
    if (
        not isinstance(namespaces, Mapping)
        or set(namespaces) != DIAGNOSTIC_NAMESPACE_KEYS
        or namespaces.get("disjoint") is not True
    ):
        raise RuntimeError("diagnostic-isolation namespace evidence is invalid")
    v100_root = _normalized_external_root(
        namespaces.get("v100_runs_root"), "V100 runs root"
    )
    h100_root = _normalized_external_root(
        namespaces.get("h100_runs_root"), "H100 runs root"
    )
    actual_h100_root = _normalized_external_root(
        expected_h100_runs_root, "expected H100 runs root"
    )
    if h100_root != actual_h100_root:
        raise RuntimeError(
            "diagnostic-isolation H100 runs root differs from the canonical "
            "Judy H100_RUNS_ROOT"
        )
    common = os.path.commonpath((v100_root, h100_root))
    if v100_root == h100_root or common in {v100_root, h100_root}:
        raise RuntimeError("V100 and H100 run namespaces overlap")

    suppression = isolated.get("h100_suppression")
    expected_suppression = {
        "v100_completions_suppress_h100": False,
        "v100_checkpoints_resume_h100": False,
        "mixed_hardware_curve_allowed": False,
    }
    if (
        not isinstance(suppression, Mapping)
        or set(suppression) != DIAGNOSTIC_SUPPRESSION_KEYS
        or dict(suppression) != expected_suppression
    ):
        raise RuntimeError("V100 state could suppress, resume, or mix with H100")

    post = isolated.get("post_diagnostic")
    expected_post = {
        "safe_stop_archive": "optional-after-diagnostic",
        "required_before_h100_campaign": False,
    }
    if (
        not isinstance(post, Mapping)
        or set(post) != DIAGNOSTIC_POST_KEYS
        or dict(post) != expected_post
    ):
        raise RuntimeError("post-diagnostic stop/archive disposition is invalid")
    cutover_time = _timestamp(cutover.get("created_utc"), "CUTOVER_READY.created_utc")
    attested_time = _timestamp(
        isolated.get("created_utc"), "V100_DIAGNOSTIC_ISOLATION.created_utc"
    )
    if attested_time < cutover_time:
        raise RuntimeError("diagnostic-isolation attestation predates CUTOVER_READY")
    return {
        "status": "operator-diagnostic-isolation-validated",
        "cutover_ready_sha256": cutover_ready_sha256,
        "v100_diagnostic_isolation_sha256": attestation_sha256,
        "attestation": isolated,
    }


def persist_diagnostic_isolation_evidence(
    *,
    meta_root: Path,
    cutover_ready: Path,
    cutover_ready_sha256: str,
    attestation: Path,
    attestation_sha256: str,
) -> dict[str, dict[str, str]]:
    """Install byte-identical, read-only evidence at canonical H100 paths."""

    root = meta_root.absolute()
    if root.is_symlink():
        raise RuntimeError("H100 metadata root cannot be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    canonical_cutover = persist_immutable_copy(
        cutover_ready,
        root / CUTOVER_READY_NAME,
        expected_sha256=cutover_ready_sha256,
    )
    canonical_attestation = persist_immutable_copy(
        attestation,
        root / V100_DIAGNOSTIC_ISOLATION_NAME,
        expected_sha256=attestation_sha256,
    )
    return {
        "cutover_ready": {
            "path": str(canonical_cutover),
            "sha256": cutover_ready_sha256,
        },
        "v100_diagnostic_isolation": {
            "path": str(canonical_attestation),
            "sha256": attestation_sha256,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutover-ready", type=Path, required=True)
    parser.add_argument("--cutover-ready-sha256", required=True)
    parser.add_argument(
        "--diagnostic-isolation-package-root", type=Path, required=True
    )
    parser.add_argument(
        "--expected-diagnostic-isolation-package-id", required=True
    )
    parser.add_argument(
        "--expected-diagnostic-isolation-producer-git-sha", required=True
    )
    parser.add_argument(
        "--expected-diagnostic-isolation-identity-sha256", required=True
    )
    parser.add_argument(
        "--expected-diagnostic-isolation-manifest-sha256", required=True
    )
    parser.add_argument(
        "--expected-diagnostic-isolation-ready-sha256", required=True
    )
    parser.add_argument(
        "--expected-diagnostic-isolation-sha256sums-sha256", required=True
    )
    parser.add_argument("--persist-meta-root", type=Path)
    parser.add_argument("--expected-h100-git-sha", required=True)
    parser.add_argument("--expected-h100-campaign-id", required=True)
    parser.add_argument("--expected-h100-runs-root", required=True)
    parser.add_argument(
        "--expected-base-payload-sha256sums-sha256", required=True
    )
    parser.add_argument(
        "--expected-runtime-amendment-sha256sums-sha256", required=True
    )
    parser.add_argument("--expected-reference-git-sha", required=True)
    parser.add_argument("--expected-reference-campaign-id", required=True)
    parser.add_argument("--expected-v100-core-git-sha", required=True)
    parser.add_argument("--expected-v100-core-campaign-id", required=True)
    args = parser.parse_args()

    from scripts.handoff.control import control_package_identity

    package_root = args.diagnostic_isolation_package_root.absolute()
    control_bindings = {
        "cutover_ready_sha256": args.cutover_ready_sha256,
        "h100_campaign_id": args.expected_h100_campaign_id,
        "v100_core_campaign_id": args.expected_v100_core_campaign_id,
        "v100_core_git_sha": args.expected_v100_core_git_sha,
    }
    expected_control_identity = {
        "package_id": args.expected_diagnostic_isolation_package_id,
        "kind": "diagnostic-isolation",
        "direction": "v100-to-judy",
        "producer_git_sha": (
            args.expected_diagnostic_isolation_producer_git_sha
        ),
        "identity_sha256": args.expected_diagnostic_isolation_identity_sha256,
        "manifest_sha256": args.expected_diagnostic_isolation_manifest_sha256,
        "ready_sha256": args.expected_diagnostic_isolation_ready_sha256,
        "sha256sums_sha256": (
            args.expected_diagnostic_isolation_sha256sums_sha256
        ),
    }
    control_identity = control_package_identity(
        package_root,
        expected_kind="diagnostic-isolation",
        expected_bindings=control_bindings,
        expected_identity=expected_control_identity,
    )
    if (
        args.expected_diagnostic_isolation_producer_git_sha
        != args.expected_h100_git_sha
    ):
        raise RuntimeError(
            "diagnostic-isolation producer Git SHA differs from the H100 source SHA"
        )
    attestation = package_root / V100_DIAGNOSTIC_ISOLATION_NAME
    attestation_sha256 = sha256_file(attestation)
    cutover_ready = args.cutover_ready.absolute()
    payload = validate_diagnostic_isolation(
        cutover_ready=cutover_ready,
        cutover_ready_sha256=args.cutover_ready_sha256,
        attestation=attestation,
        attestation_sha256=attestation_sha256,
        expected_h100_git_sha=args.expected_h100_git_sha,
        expected_h100_campaign_id=args.expected_h100_campaign_id,
        expected_h100_runs_root=args.expected_h100_runs_root,
        expected_reference_git_sha=args.expected_reference_git_sha,
        expected_reference_campaign_id=args.expected_reference_campaign_id,
        expected_v100_core_git_sha=args.expected_v100_core_git_sha,
        expected_v100_core_campaign_id=args.expected_v100_core_campaign_id,
    )
    _validate_package_sha256sums_bindings(
        _json_object(cutover_ready, "CUTOVER_READY receipt"),
        expected_base_payload_sha256sums_sha256=(
            args.expected_base_payload_sha256sums_sha256
        ),
        expected_runtime_amendment_sha256sums_sha256=(
            args.expected_runtime_amendment_sha256sums_sha256
        ),
    )
    payload["control_package"] = control_identity
    if args.persist_meta_root is not None:
        evidence = persist_diagnostic_isolation_evidence(
            meta_root=args.persist_meta_root,
            cutover_ready=cutover_ready,
            cutover_ready_sha256=args.cutover_ready_sha256,
            attestation=attestation,
            attestation_sha256=attestation_sha256,
        )
        payload = validate_diagnostic_isolation(
            cutover_ready=Path(evidence["cutover_ready"]["path"]),
            cutover_ready_sha256=args.cutover_ready_sha256,
            attestation=Path(evidence["v100_diagnostic_isolation"]["path"]),
            attestation_sha256=attestation_sha256,
            expected_h100_git_sha=args.expected_h100_git_sha,
            expected_h100_campaign_id=args.expected_h100_campaign_id,
            expected_h100_runs_root=args.expected_h100_runs_root,
            expected_reference_git_sha=args.expected_reference_git_sha,
            expected_reference_campaign_id=args.expected_reference_campaign_id,
            expected_v100_core_git_sha=args.expected_v100_core_git_sha,
            expected_v100_core_campaign_id=args.expected_v100_core_campaign_id,
        )
        payload["control_package"] = control_identity
        payload["evidence"] = evidence
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
