"""Read-only validation of the operator-authored V100 archive attestation.

This module never creates the attestation and contains no process-control
operations.  It is run once on the submission host and again in the Slurm
allocation before the H100 controller may launch.
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

from scripts.h100.contracts import validate_bound_cutover_forecast

HEX64 = re.compile(r"^[0-9a-f]{64}$")
CUTOVER_READY_NAME = "CUTOVER_READY.json"
V100_RECEIPT_NAME = "V100_CORE_ARCHIVED.json"
V100_ARCHIVE_MANIFEST_NAME = "V100_CORE_ARCHIVE_MANIFEST.json"
PACKAGE_KEYS = {
    "manifest_sha256",
    "ready_sha256",
    "sha256sums_sha256",
    "repo_bundle_sha256",
}
RECEIPT_KEYS = {
    "schema",
    "status",
    "created_utc",
    "attestation",
    "cutover_ready_sha256",
    "h100",
    "v100",
    "archive",
}
H100_KEYS = {"acceptance_uuid", "git_sha", "sif_sha256", "package"}
V100_KEYS = {
    "git_sha",
    "campaign_id",
    "stopped_utc",
    "stop_mode",
    "running_core_processes",
    "diagnostic_status",
}
ARCHIVE_KEYS = {"manifest_path", "manifest_sha256"}
ARCHIVE_MANIFEST_KEYS = {
    "schema",
    "status",
    "scope",
    "diagnostic_status",
    "git_sha",
    "campaign_id",
    "stopped_utc",
    "archived_utc",
    "file_count",
    "total_bytes",
}


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


def expected_package(
    *,
    manifest_sha256: str,
    ready_sha256: str,
    sha256sums_sha256: str,
    repo_bundle_sha256: str,
) -> dict[str, str]:
    package = {
        "manifest_sha256": manifest_sha256,
        "ready_sha256": ready_sha256,
        "sha256sums_sha256": sha256sums_sha256,
        "repo_bundle_sha256": repo_bundle_sha256,
    }
    if any(not HEX64.fullmatch(value) for value in package.values()):
        raise RuntimeError("operator cutover package bindings require 64-hex SHA-256")
    return package


def validate_operator_archive(
    *,
    cutover_ready: Path,
    cutover_ready_sha256: str,
    receipt: Path,
    receipt_sha256: str,
    archive_manifest: Path,
    archive_manifest_sha256: str,
    bound_archive_manifest: Path | None = None,
    expected_h100_git_sha: str,
    expected_sif_sha256: str,
    expected_package_hashes: Mapping[str, str],
    expected_reference_git_sha: str,
    expected_reference_campaign_id: str,
) -> dict:
    """Validate chronology, hashes, and all H100/V100 identity bindings."""

    for path, label in (
        (cutover_ready, "CUTOVER_READY"),
        (receipt, "V100_CORE_ARCHIVED"),
        (archive_manifest, "V100 archive manifest"),
    ):
        _require_regular_nonsymlink(path, label)
    bound_archive_manifest = bound_archive_manifest or archive_manifest
    if not bound_archive_manifest.is_absolute():
        raise RuntimeError("bound V100 archive manifest path must be absolute")
    for digest, label in (
        (cutover_ready_sha256, "CUTOVER_READY"),
        (receipt_sha256, "V100_CORE_ARCHIVED"),
        (archive_manifest_sha256, "V100 archive manifest"),
    ):
        if not HEX64.fullmatch(digest):
            raise RuntimeError(f"{label} expected SHA-256 is not 64-hex")
    if sha256_file(cutover_ready) != cutover_ready_sha256:
        raise RuntimeError("CUTOVER_READY SHA-256 mismatch")
    if sha256_file(receipt) != receipt_sha256:
        raise RuntimeError("V100_CORE_ARCHIVED SHA-256 mismatch")
    if sha256_file(archive_manifest) != archive_manifest_sha256:
        raise RuntimeError("V100 archive manifest SHA-256 mismatch")

    cutover = _json_object(cutover_ready, "CUTOVER_READY receipt")
    attestation = _json_object(receipt, "V100_CORE_ARCHIVED receipt")
    archive_payload = _json_object(archive_manifest, "V100 archive manifest")
    if cutover.get("status") != "cutover-ready":
        raise RuntimeError("CUTOVER_READY status is invalid")
    if cutover.get("v100_action") != (
        "none; this guard never stops or signals V100 processes"
    ):
        raise RuntimeError("CUTOVER_READY does not preserve the no-V100-action boundary")
    validate_bound_cutover_forecast(cutover)
    acceptance = cutover.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise RuntimeError("CUTOVER_READY lacks bound H100 acceptance")
    source = acceptance.get("source")
    sif = acceptance.get("sif")
    package = acceptance.get("package")
    if not isinstance(source, Mapping) or not isinstance(sif, Mapping):
        raise RuntimeError("CUTOVER_READY source/SIF bindings are malformed")
    if (
        not isinstance(package, Mapping)
        or set(package) != PACKAGE_KEYS
        or dict(package) != dict(expected_package_hashes)
    ):
        raise RuntimeError("CUTOVER_READY package bindings mismatch")
    h100_expected = {
        "acceptance_uuid": acceptance.get("uuid"),
        "git_sha": source.get("git_sha"),
        "sif_sha256": sif.get("sha256"),
        "package": dict(expected_package_hashes),
    }
    if h100_expected["git_sha"] != expected_h100_git_sha:
        raise RuntimeError("CUTOVER_READY H100 git binding mismatch")
    if h100_expected["sif_sha256"] != expected_sif_sha256:
        raise RuntimeError("CUTOVER_READY SIF binding mismatch")
    if not str(h100_expected["acceptance_uuid"] or ""):
        raise RuntimeError("CUTOVER_READY acceptance UUID is absent")

    references = cutover.get("references")
    if not isinstance(references, Mapping):
        raise RuntimeError("CUTOVER_READY reference provenance is absent")
    for name in ("r2", "r3"):
        record = references.get(name)
        if not isinstance(record, Mapping) or set(record) != {
            "metrics",
            "metrics_sha256",
            "provenance",
            "provenance_sha256",
        }:
            raise RuntimeError(f"CUTOVER_READY {name} result binding is absent")
        if not HEX64.fullmatch(str(record.get("metrics_sha256", ""))) or not HEX64.fullmatch(
            str(record.get("provenance_sha256", ""))
        ):
            raise RuntimeError(f"CUTOVER_READY {name} result hashes are invalid")
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            raise RuntimeError(f"CUTOVER_READY {name} provenance is absent")
        if provenance.get("git_sha") != expected_reference_git_sha:
            raise RuntimeError(f"CUTOVER_READY {name} reference git mismatch")
        if provenance.get("campaign_id") != expected_reference_campaign_id:
            raise RuntimeError(f"CUTOVER_READY {name} campaign mismatch")

    if set(attestation) != RECEIPT_KEYS:
        raise RuntimeError("V100_CORE_ARCHIVED top-level keys do not match the schema")
    if attestation.get("schema") != 1:
        raise RuntimeError("V100_CORE_ARCHIVED schema is unsupported")
    if attestation.get("status") != "v100-core-archived":
        raise RuntimeError("V100_CORE_ARCHIVED status is invalid")
    if attestation.get("attestation") != "external-human-operator":
        raise RuntimeError("V100 archive receipt is not an operator attestation")
    if attestation.get("cutover_ready_sha256") != cutover_ready_sha256:
        raise RuntimeError("V100 archive receipt does not bind CUTOVER_READY")
    h100 = attestation.get("h100")
    if not isinstance(h100, Mapping) or set(h100) != H100_KEYS:
        raise RuntimeError("V100 archive receipt H100 keys do not match the schema")
    if dict(h100) != h100_expected:
        raise RuntimeError("V100 archive receipt H100 bindings mismatch")

    v100 = attestation.get("v100")
    if not isinstance(v100, Mapping):
        raise RuntimeError("V100 archive receipt lacks V100 state")
    if set(v100) != V100_KEYS:
        raise RuntimeError("V100 archive receipt V100 keys do not match the schema")
    expected_v100 = {
        "git_sha": expected_reference_git_sha,
        "campaign_id": expected_reference_campaign_id,
        "stop_mode": "graceful",
        "running_core_processes": 0,
        "diagnostic_status": "non-reportable-diagnostic",
    }
    mismatches = {
        key: (value, v100.get(key))
        for key, value in expected_v100.items()
        if v100.get(key) != value
    }
    if type(v100.get("running_core_processes")) is not int:  # bool is not valid
        mismatches["running_core_processes.type"] = (
            "integer",
            type(v100.get("running_core_processes")).__name__,
        )
    if mismatches:
        raise RuntimeError(f"V100 archive state mismatch: {mismatches}")

    cutover_time = _timestamp(cutover.get("created_utc"), "CUTOVER_READY.created_utc")
    stopped_time = _timestamp(v100.get("stopped_utc"), "v100.stopped_utc")
    attested_time = _timestamp(
        attestation.get("created_utc"), "V100_CORE_ARCHIVED.created_utc"
    )
    if stopped_time <= cutover_time:
        raise RuntimeError("V100 core stop must occur after CUTOVER_READY")
    if attested_time < stopped_time:
        raise RuntimeError("operator attestation predates the V100 core stop")

    archive = attestation.get("archive")
    if not isinstance(archive, Mapping):
        raise RuntimeError("V100 archive receipt lacks archive evidence")
    if set(archive) != ARCHIVE_KEYS:
        raise RuntimeError("V100 archive receipt archive keys do not match the schema")
    if archive.get("manifest_path") != str(bound_archive_manifest):
        raise RuntimeError("V100 archive manifest path binding mismatch")
    if archive.get("manifest_sha256") != archive_manifest_sha256:
        raise RuntimeError("V100 archive manifest receipt SHA-256 mismatch")
    allowed_manifest_keys = (ARCHIVE_MANIFEST_KEYS, ARCHIVE_MANIFEST_KEYS | {"empty_reason"})
    if set(archive_payload) not in allowed_manifest_keys:
        raise RuntimeError("V100 archive manifest keys do not match the schema")
    manifest_expected = {
        "schema": 1,
        "status": "v100-core-diagnostics-archived",
        "scope": "v100-core-diagnostics",
        "diagnostic_status": "non-reportable-diagnostic",
        "git_sha": expected_reference_git_sha,
        "campaign_id": expected_reference_campaign_id,
    }
    manifest_mismatches = {
        key: (value, archive_payload.get(key))
        for key, value in manifest_expected.items()
        if archive_payload.get(key) != value
    }
    if manifest_mismatches:
        raise RuntimeError(
            f"V100 archive manifest identity/status mismatch: {manifest_mismatches}"
        )
    manifest_stopped = _timestamp(
        archive_payload.get("stopped_utc"), "archive_manifest.stopped_utc"
    )
    archived_time = _timestamp(
        archive_payload.get("archived_utc"), "archive_manifest.archived_utc"
    )
    if manifest_stopped != stopped_time:
        raise RuntimeError("archive manifest stopped_utc differs from the attestation")
    if archived_time < stopped_time or attested_time < archived_time:
        raise RuntimeError("V100 archive/attestation timestamps are out of order")
    file_count = archive_payload.get("file_count")
    total_bytes = archive_payload.get("total_bytes")
    if type(file_count) is not int or type(total_bytes) is not int:
        raise RuntimeError("V100 archive counts must be integers")
    if file_count > 0 and total_bytes > 0:
        if "empty_reason" in archive_payload:
            raise RuntimeError("non-empty V100 archive cannot declare empty_reason")
    elif file_count == 0 and total_bytes == 0:
        if not str(archive_payload.get("empty_reason", "")).strip():
            raise RuntimeError("empty V100 archive requires an explicit reason")
    else:
        raise RuntimeError("V100 archive file_count/total_bytes must both be positive or zero")
    return {
        "status": "operator-cutover-validated",
        "cutover_ready_sha256": cutover_ready_sha256,
        "v100_core_archived_sha256": receipt_sha256,
        "archive_manifest_sha256": archive_manifest_sha256,
        "attestation": attestation,
        "archive_manifest": archive_payload,
    }


def persist_operator_evidence(
    *,
    meta_root: Path,
    cutover_ready: Path,
    cutover_ready_sha256: str,
    receipt: Path,
    receipt_sha256: str,
    archive_manifest: Path,
    archive_manifest_sha256: str,
) -> dict[str, dict[str, str]]:
    """Persist validated operator evidence at stable campaign paths."""

    meta_root = meta_root.absolute()
    if meta_root.is_symlink():
        raise RuntimeError("H100 metadata root cannot be a symlink")
    expected_cutover = meta_root / CUTOVER_READY_NAME
    if cutover_ready != expected_cutover:
        raise RuntimeError(f"CUTOVER_READY must be the canonical {expected_cutover}")
    _require_regular_nonsymlink(cutover_ready, "CUTOVER_READY")
    if sha256_file(cutover_ready) != cutover_ready_sha256:
        raise RuntimeError("CUTOVER_READY changed before evidence persistence")
    cutover_ready.chmod(0o444)

    canonical_manifest = persist_immutable_copy(
        archive_manifest,
        meta_root / V100_ARCHIVE_MANIFEST_NAME,
        expected_sha256=archive_manifest_sha256,
    )
    canonical_receipt = persist_immutable_copy(
        receipt,
        meta_root / V100_RECEIPT_NAME,
        expected_sha256=receipt_sha256,
    )
    return {
        "cutover_ready": {
            "path": str(expected_cutover),
            "sha256": cutover_ready_sha256,
        },
        "v100_core_archived": {
            "path": str(canonical_receipt),
            "sha256": receipt_sha256,
        },
        "archive_manifest": {
            "path": str(canonical_manifest),
            "sha256": archive_manifest_sha256,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutover-ready", type=Path, required=True)
    parser.add_argument("--cutover-ready-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--archive-manifest", type=Path, required=True)
    parser.add_argument("--archive-manifest-sha256", required=True)
    parser.add_argument("--bound-archive-manifest", type=Path)
    parser.add_argument("--persist-meta-root", type=Path)
    parser.add_argument("--expected-h100-git-sha", required=True)
    parser.add_argument("--expected-sif-sha256", required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--expected-package-ready-sha256", required=True)
    parser.add_argument("--expected-package-sha256sums-sha256", required=True)
    parser.add_argument("--expected-package-repo-bundle-sha256", required=True)
    parser.add_argument("--expected-reference-git-sha", required=True)
    parser.add_argument("--expected-reference-campaign-id", required=True)
    args = parser.parse_args()
    package = expected_package(
        manifest_sha256=args.expected_package_manifest_sha256,
        ready_sha256=args.expected_package_ready_sha256,
        sha256sums_sha256=args.expected_package_sha256sums_sha256,
        repo_bundle_sha256=args.expected_package_repo_bundle_sha256,
    )
    cutover_ready = args.cutover_ready.absolute()
    receipt = args.receipt.absolute()
    archive_manifest = args.archive_manifest.absolute()
    bound_archive_manifest = (
        args.bound_archive_manifest.absolute()
        if args.bound_archive_manifest is not None
        else archive_manifest
    )
    payload = validate_operator_archive(
        cutover_ready=cutover_ready,
        cutover_ready_sha256=args.cutover_ready_sha256,
        receipt=receipt,
        receipt_sha256=args.receipt_sha256,
        archive_manifest=archive_manifest,
        archive_manifest_sha256=args.archive_manifest_sha256,
        bound_archive_manifest=bound_archive_manifest,
        expected_h100_git_sha=args.expected_h100_git_sha,
        expected_sif_sha256=args.expected_sif_sha256,
        expected_package_hashes=package,
        expected_reference_git_sha=args.expected_reference_git_sha,
        expected_reference_campaign_id=args.expected_reference_campaign_id,
    )
    if args.persist_meta_root is not None:
        evidence = persist_operator_evidence(
            meta_root=args.persist_meta_root,
            cutover_ready=cutover_ready,
            cutover_ready_sha256=args.cutover_ready_sha256,
            receipt=receipt,
            receipt_sha256=args.receipt_sha256,
            archive_manifest=archive_manifest,
            archive_manifest_sha256=args.archive_manifest_sha256,
        )
        if Path(evidence["archive_manifest"]["path"]) != bound_archive_manifest:
            raise RuntimeError(
                "operator receipt must bind the canonical persisted archive manifest"
            )
        payload = validate_operator_archive(
            cutover_ready=Path(evidence["cutover_ready"]["path"]),
            cutover_ready_sha256=args.cutover_ready_sha256,
            receipt=Path(evidence["v100_core_archived"]["path"]),
            receipt_sha256=args.receipt_sha256,
            archive_manifest=Path(evidence["archive_manifest"]["path"]),
            archive_manifest_sha256=args.archive_manifest_sha256,
            expected_h100_git_sha=args.expected_h100_git_sha,
            expected_sif_sha256=args.expected_sif_sha256,
            expected_package_hashes=package,
            expected_reference_git_sha=args.expected_reference_git_sha,
            expected_reference_campaign_id=args.expected_reference_campaign_id,
        )
        payload["evidence"] = evidence
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
