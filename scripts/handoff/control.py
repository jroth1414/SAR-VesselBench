"""Small content-addressed control-plane packages for isolated hosts.

The bulk Sprint 7d payload and the Sprint 7f schema-2 runtime amendment are
immutable.  The amendment contains one Git bundle, the source-audited
13,911-row TRAIN+fixed-DEV8 CSV, and package controls, with no TEST/eval-final
content.  Later evidence crosses the V100/Judy filesystem boundary only in one
of the closed, direction-specific package kinds defined here.  The Box
transport remains responsible for SHA-1/size comparison, ``.partial``
downloads, atomic publication, and uploading ``READY.json`` last.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping

from .package import (
    PackageError,
    _absolute_path,
    _artifact_part_paths,
    _canonical_json,
    _git_source,
    _hash_file,
    _inside_repository_worktrees,
    _load_json,
    _package_regular_files,
    _parse_sha256sums,
    _physical_record,
    _require_no_symlink_components,
    _scan_secret_content,
    _validate_control_record,
    _validate_safe_path,
    _write_bytes,
)

FORMAT_VERSION = 1
PACKAGE_TYPE = "h100-dynamic-control"
CONTROL_BRANCH = "sprint-7f-eval-contract"
MAX_CONTROL_FILE_BYTES = 50_000_000
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


@dataclass(frozen=True)
class ControlPolicy:
    direction: str
    payload_paths: tuple[str, ...]
    binding_keys: tuple[str, ...]


CONTROL_POLICIES: Mapping[str, ControlPolicy] = {
    "references": ControlPolicy(
        direction="v100-to-judy",
        payload_paths=(
            "references/REFERENCE_CAMPAIGN.json",
            "references/locateanything-zs/final_metrics.json",
            "references/locateanything-zs/runtime_provenance.json",
            "references/yolo26-f100/final_metrics.json",
            "references/yolo26-f100/runtime_provenance.json",
        ),
        binding_keys=(
            "evaluation_contract",
            "reference_git_sha",
            "reference_campaign_id",
            "v100_core_git_sha",
            "v100_core_campaign_id",
        ),
    ),
    "cutover-ready": ControlPolicy(
        direction="judy-to-v100",
        payload_paths=("CUTOVER_READY.json",),
        binding_keys=(
            "h100_campaign_id",
            "h100_git_sha",
            "h100_ready_sha256",
            "references_manifest_sha256",
            "references_package_id",
        ),
    ),
    "diagnostic-isolation": ControlPolicy(
        direction="v100-to-judy",
        payload_paths=("V100_DIAGNOSTIC_ISOLATION.json",),
        binding_keys=(
            "cutover_ready_sha256",
            "h100_campaign_id",
            "v100_core_campaign_id",
            "v100_core_git_sha",
        ),
    ),
    "results": ControlPolicy(
        direction="judy-to-v100",
        payload_paths=("H100_RESULTS_READY.json",),
        binding_keys=(
            "cutover_ready_sha256",
            "diagnostic_isolation_sha256",
            "h100_campaign_id",
            "h100_git_sha",
        ),
    ),
}

ControlVerifier = Callable[[Path], dict[str, object]]


def _policy(kind: str) -> ControlPolicy:
    try:
        return CONTROL_POLICIES[kind]
    except KeyError as exc:
        raise PackageError(
            f"unsupported control-package kind {kind!r}; expected one of "
            + ", ".join(sorted(CONTROL_POLICIES))
        ) from exc


def _validate_bindings(kind: str, bindings: Mapping[str, str]) -> dict[str, str]:
    policy = _policy(kind)
    expected = set(policy.binding_keys)
    if set(bindings) != expected:
        raise PackageError(
            f"{kind} binding keys differ from the closed contract: "
            f"missing={sorted(expected - set(bindings))}, "
            f"unexpected={sorted(set(bindings) - expected)}"
        )
    normalized: dict[str, str] = {}
    for key in policy.binding_keys:
        value = bindings[key]
        if not isinstance(value, str) or not value or value.strip() != value:
            raise PackageError(f"control binding {key} must be a non-empty string")
        if key.endswith("_sha256") and HEX64.fullmatch(value) is None:
            raise PackageError(f"control binding {key} must be a SHA-256")
        if key.endswith("_git_sha") and HEX40.fullmatch(value) is None:
            raise PackageError(f"control binding {key} must be a full Git SHA")
        if key.endswith("_campaign_id") and SAFE_ID.fullmatch(value) is None:
            raise PackageError(f"control binding {key} is not a safe campaign ID")
        if key.endswith("_package_id") and SAFE_ID.fullmatch(value) is None:
            raise PackageError(f"control binding {key} is not a safe package ID")
        normalized[key] = value
    if kind == "references" and normalized["evaluation_contract"] != (
        "vessel-hm-positive-low-ignore-v2"
    ):
        raise PackageError("references package must bind evaluation contract v2")
    return normalized


def _validate_sources(
    kind: str, source_files: Mapping[str, Path]
) -> dict[str, Path]:
    policy = _policy(kind)
    expected = set(policy.payload_paths)
    if set(source_files) != expected:
        raise PackageError(
            f"{kind} payload differs from the exact allowlist: "
            f"missing={sorted(expected - set(source_files))}, "
            f"unexpected={sorted(set(source_files) - expected)}"
        )
    validated: dict[str, Path] = {}
    for relative_name in policy.payload_paths:
        relative = PurePosixPath(relative_name)
        _validate_safe_path(relative)
        source = _require_no_symlink_components(
            _absolute_path(source_files[relative_name]), leaf="file"
        )
        size = source.stat().st_size
        if size <= 0 or size > MAX_CONTROL_FILE_BYTES:
            raise PackageError(
                f"control payload must be 1..{MAX_CONTROL_FILE_BYTES} bytes: {source}"
            )
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PackageError(f"control payload is not valid JSON: {source}") from exc
        if not isinstance(value, dict):
            raise PackageError(f"control payload must be a JSON object: {source}")
        _scan_secret_content(source)
        validated[relative_name] = source
    return validated


def _artifact(relative: str, destination: Path, staging: Path) -> dict[str, object]:
    physical = _physical_record(destination, staging)
    return {
        "kind": "control_file",
        "name": relative,
        "format": "json",
        "file_count": 1,
        "unpacked_bytes": destination.stat().st_size,
        "archive_bytes": destination.stat().st_size,
        "archive_sha256": physical["sha256"],
        "archive_sha1": physical["sha1"],
        "parts": [physical],
    }


def _identity(
    *,
    kind: str,
    direction: str,
    producer_git_sha: str,
    bindings: Mapping[str, str],
    files: list[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "format_version": FORMAT_VERSION,
        "package_type": PACKAGE_TYPE,
        "kind": kind,
        "direction": direction,
        "producer_git_sha": producer_git_sha,
        "bindings": dict(bindings),
        "files": [
            {
                "path": record["path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "sha1": record["sha1"],
            }
            for record in files
        ],
    }


def _build_control_package(
    *,
    kind: str,
    repo_root: Path,
    source_files: Mapping[str, Path],
    bindings: Mapping[str, str],
    output_dir: Path,
    branch: str,
    production: bool,
) -> Path:
    policy = _policy(kind)
    repo = _require_no_symlink_components(repo_root, leaf="directory")
    output_parent = _require_no_symlink_components(output_dir, leaf="directory")
    if _inside_repository_worktrees(output_parent, repo):
        raise PackageError("control-package output must be outside repository worktrees")
    normalized_bindings = _validate_bindings(kind, bindings)
    sources = _validate_sources(kind, source_files)
    commit, commit_epoch = _git_source(repo, branch, production=production)

    staging = Path(
        tempfile.mkdtemp(prefix=f".xview3-control-{kind}-building-", dir=output_parent)
    )
    try:
        artifacts: list[dict[str, object]] = []
        for relative in policy.payload_paths:
            source = sources[relative]
            destination = staging.joinpath(*PurePosixPath(relative).parts)
            # Read once, create exclusively, and verify the source did not
            # change between validation and publication.
            content = source.read_bytes()
            _write_bytes(destination, content)
            if _hash_file(source) != _hash_file(destination):
                raise PackageError(f"control source changed while copied: {source}")
            artifacts.append(_artifact(relative, destination, staging))

        file_records = [artifact["parts"][0] for artifact in artifacts]
        identity = _identity(
            kind=kind,
            direction=policy.direction,
            producer_git_sha=commit,
            bindings=normalized_bindings,
            files=file_records,
        )
        identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
        package_id = f"xview3-control-{kind}-{identity_sha256}"
        manifest = {
            "format_version": FORMAT_VERSION,
            "package_type": PACKAGE_TYPE,
            "package_id": package_id,
            "kind": kind,
            "direction": policy.direction,
            "created_utc": datetime.fromtimestamp(
                commit_epoch, timezone.utc
            ).isoformat(),
            "producer_git_sha": commit,
            "bindings": normalized_bindings,
            "identity": identity,
            "identity_sha256": identity_sha256,
            "counts": {"control_files": len(artifacts)},
            "artifacts": artifacts,
        }
        _write_bytes(staging / "manifest.json", _canonical_json(manifest))
        sums = "".join(
            f"{record['sha256']}  {record['path']}\n"
            for record in file_records
        )
        _write_bytes(staging / "SHA256SUMS", sums.encode("utf-8"))
        ready = {
            "format_version": FORMAT_VERSION,
            "status": "READY",
            "package_type": PACKAGE_TYPE,
            "package_id": package_id,
            "kind": kind,
            "direction": policy.direction,
            "identity_sha256": identity_sha256,
            "manifest": _physical_record(staging / "manifest.json", staging),
            "checksums": _physical_record(staging / "SHA256SUMS", staging),
        }
        # READY is deliberately the final package write.
        _write_bytes(staging / "READY.json", _canonical_json(ready))
        _verify_control_package(
            staging,
            expected_kind=kind,
            expected_bindings=normalized_bindings,
            expected_branch=branch if production else None,
        )
        destination = output_parent / package_id
        if os.path.lexists(destination):
            raise PackageError(f"control-package destination exists: {destination}")
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def build_control_package(
    *,
    kind: str,
    repo_root: Path,
    source_files: Mapping[str, Path],
    bindings: Mapping[str, str],
    output_dir: Path,
) -> Path:
    """Build a production Sprint 7f control package from its exact allowlist."""

    return _build_control_package(
        kind=kind,
        repo_root=repo_root,
        source_files=source_files,
        bindings=bindings,
        output_dir=output_dir,
        branch=CONTROL_BRANCH,
        production=True,
    )


def _verify_artifact(
    root: Path, artifact: Mapping[str, object], expected_path: str
) -> dict[str, object]:
    expected_keys = {
        "kind",
        "name",
        "format",
        "file_count",
        "unpacked_bytes",
        "archive_bytes",
        "archive_sha256",
        "archive_sha1",
        "parts",
    }
    if set(artifact) != expected_keys:
        raise PackageError(f"control artifact schema is invalid: {expected_path}")
    if (
        artifact.get("kind") != "control_file"
        or artifact.get("name") != expected_path
        or artifact.get("format") != "json"
        or artifact.get("file_count") != 1
    ):
        raise PackageError(f"control artifact identity is invalid: {expected_path}")
    parts = _artifact_part_paths(root, artifact)
    if len(parts) != 1 or parts[0].relative_to(root).as_posix() != expected_path:
        raise PackageError(f"control artifact must have one exact part: {expected_path}")
    physical = _physical_record(parts[0], root)
    size = parts[0].stat().st_size
    if (
        artifact.get("unpacked_bytes") != size
        or artifact.get("archive_bytes") != size
        or artifact.get("archive_sha256") != physical["sha256"]
        or artifact.get("archive_sha1") != physical["sha1"]
    ):
        raise PackageError(f"control artifact digest/size mismatch: {expected_path}")
    try:
        payload = json.loads(parts[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageError(f"control payload is not valid JSON: {expected_path}") from exc
    if not isinstance(payload, dict):
        raise PackageError(f"control payload is not a JSON object: {expected_path}")
    _scan_secret_content(parts[0])
    return physical


def _verify_control_package(
    package_root: Path,
    *,
    expected_kind: str | None,
    expected_bindings: Mapping[str, str] | None,
    expected_branch: str | None,
) -> dict[str, object]:
    root = _require_no_symlink_components(package_root, leaf="directory")
    ready = _load_json(_require_no_symlink_components(root / "READY.json", leaf="file"))
    manifest = _load_json(
        _require_no_symlink_components(root / "manifest.json", leaf="file")
    )
    sums_path = _require_no_symlink_components(root / "SHA256SUMS", leaf="file")
    _validate_control_record(root, ready.get("manifest", {}), "manifest.json")  # type: ignore[arg-type]
    _validate_control_record(root, ready.get("checksums", {}), "SHA256SUMS")  # type: ignore[arg-type]

    manifest_keys = {
        "format_version",
        "package_type",
        "package_id",
        "kind",
        "direction",
        "created_utc",
        "producer_git_sha",
        "bindings",
        "identity",
        "identity_sha256",
        "counts",
        "artifacts",
    }
    if set(manifest) != manifest_keys:
        raise PackageError("control manifest schema is invalid")
    kind = manifest.get("kind")
    if not isinstance(kind, str):
        raise PackageError("control manifest kind is invalid")
    policy = _policy(kind)
    if expected_kind is not None and kind != expected_kind:
        raise PackageError(f"control kind mismatch: expected {expected_kind}, found {kind}")
    bindings_raw = manifest.get("bindings")
    if not isinstance(bindings_raw, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in bindings_raw.items()
    ):
        raise PackageError("control manifest bindings are invalid")
    bindings = _validate_bindings(kind, dict(bindings_raw))
    if expected_bindings is not None and bindings != _validate_bindings(
        kind, expected_bindings
    ):
        raise PackageError("control manifest binding values mismatch")
    commit = manifest.get("producer_git_sha")
    if not isinstance(commit, str) or HEX40.fullmatch(commit) is None:
        raise PackageError("control producer Git SHA is invalid")
    if expected_branch is not None:
        # The bundle/runtime package separately proves source ancestry; this
        # gate only keeps production control packages on the approved branch.
        if expected_branch != CONTROL_BRANCH:
            raise PackageError("control verifier received an unsupported branch")

    artifacts_raw = manifest.get("artifacts")
    if not isinstance(artifacts_raw, list) or len(artifacts_raw) != len(
        policy.payload_paths
    ):
        raise PackageError("control artifact count is invalid")
    physical: list[dict[str, object]] = []
    for expected_path, artifact in zip(
        policy.payload_paths, artifacts_raw, strict=True
    ):
        if not isinstance(artifact, Mapping):
            raise PackageError("control artifact is not an object")
        physical.append(_verify_artifact(root, artifact, expected_path))
    if manifest.get("counts") != {"control_files": len(policy.payload_paths)}:
        raise PackageError("control file count is invalid")
    identity = _identity(
        kind=kind,
        direction=policy.direction,
        producer_git_sha=commit,
        bindings=bindings,
        files=physical,
    )
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    package_id = f"xview3-control-{kind}-{identity_sha256}"
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("package_type") != PACKAGE_TYPE
        or manifest.get("package_id") != package_id
        or manifest.get("direction") != policy.direction
        or manifest.get("identity") != identity
        or manifest.get("identity_sha256") != identity_sha256
    ):
        raise PackageError("control manifest identity is invalid")
    try:
        created = datetime.fromisoformat(str(manifest.get("created_utc", "")))
    except ValueError as exc:
        raise PackageError("control manifest timestamp is invalid") from exc
    if created.tzinfo is None:
        raise PackageError("control manifest timestamp must include a timezone")

    ready_keys = {
        "format_version",
        "status",
        "package_type",
        "package_id",
        "kind",
        "direction",
        "identity_sha256",
        "manifest",
        "checksums",
    }
    if set(ready) != ready_keys or any(
        (
            ready.get("format_version") != FORMAT_VERSION,
            ready.get("status") != "READY",
            ready.get("package_type") != PACKAGE_TYPE,
            ready.get("package_id") != package_id,
            ready.get("kind") != kind,
            ready.get("direction") != policy.direction,
            ready.get("identity_sha256") != identity_sha256,
        )
    ):
        raise PackageError("control READY marker is invalid")

    sums = _parse_sha256sums(sums_path)
    expected_sums = {
        str(record["path"]): str(record["sha256"]) for record in physical
    }
    if list(sums) != list(policy.payload_paths) or sums != expected_sums:
        raise PackageError("control SHA256SUMS does not match the exact allowlist")
    allowed = set(policy.payload_paths) | {
        "manifest.json",
        "SHA256SUMS",
        "READY.json",
    }
    actual = _package_regular_files(root)
    if actual != allowed:
        raise PackageError(
            "control package contains unexpected files: "
            f"missing={sorted(allowed - actual)}, unexpected={sorted(actual - allowed)}"
        )
    return manifest


def verify_control_package(
    package_root: Path,
    *,
    expected_kind: str | None = None,
    expected_bindings: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Verify a received package and its closed direction/kind contract."""

    return _verify_control_package(
        package_root,
        expected_kind=expected_kind,
        expected_bindings=expected_bindings,
        expected_branch=None,
    )


def prepare_control_verifier(
    expected_kind: str,
    expected_bindings: Mapping[str, str] | None = None,
) -> ControlVerifier:
    """Return the verifier callback consumed by the generic Box transport."""

    _policy(expected_kind)

    def verifier(package_root: Path) -> dict[str, object]:
        return verify_control_package(
            package_root,
            expected_kind=expected_kind,
            expected_bindings=expected_bindings,
        )

    return verifier


def control_package_identity(
    package_root: Path,
    *,
    expected_kind: str | None = None,
    expected_bindings: Mapping[str, str] | None = None,
    expected_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return and optionally out-of-band bind the immutable identity."""

    root = _absolute_path(package_root)
    manifest = verify_control_package(
        root,
        expected_kind=expected_kind,
        expected_bindings=expected_bindings,
    )
    identity = {
        "package_id": manifest["package_id"],
        "kind": manifest["kind"],
        "direction": manifest["direction"],
        "producer_git_sha": manifest["producer_git_sha"],
        "identity_sha256": manifest["identity_sha256"],
        "manifest_sha256": _hash_file(root / "manifest.json"),
        "ready_sha256": _hash_file(root / "READY.json"),
        "sha256sums_sha256": _hash_file(root / "SHA256SUMS"),
    }
    if expected_identity is not None and identity != dict(expected_identity):
        raise PackageError(
            "control package differs from its out-of-band identity"
        )
    return identity
