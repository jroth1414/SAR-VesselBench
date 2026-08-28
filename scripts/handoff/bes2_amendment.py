"""Code-only schema-2 Box amendment for the BES2 Judy diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable

from src.analysis.bes2_contract import MAX_APPROVED_GPU_HOURS

from .package import (
    PackageError,
    _canonical_json,
    _create_git_bundle,
    _git_source,
    _hash_file,
    _hash_paths,
    _inside_repository_worktrees,
    _load_json,
    _package_regular_files,
    _parse_sha256sums,
    _physical_record,
    _plain_artifact,
    _require_no_symlink_components,
    _require_source_ancestor,
    _validate_control_record,
    _validate_safe_path,
    _write_bytes,
)
from .runtime_amendment import _verify_git_bundle_round_trip

FORMAT_VERSION = 2
BRANCH = "sprint-10d-bes2-readiness-shape-fix"
REQUIRED_ANCESTOR = "eafb9c882fdbf491fbc6aad8c5f1c621fa5c5620"
BUNDLE_PATH = "code/xview3-bes2-diagnostic.bundle"
CONTROL_PATH = "controls/BES2_AMENDMENT.json"
S2_CHECKPOINT_RELATIVE = "bigearthnet_s2/model.safetensors"
S2_CHECKPOINT_SHA256 = (
    "b09d0e41cc683878243a9128a6f4724d6a71d562318beeae716f0dce9cbbf454"
)
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")

PackageVerifier = Callable[[Path], dict[str, object]]


def _h100_identity(path: Path) -> dict[str, object]:
    ready_path = _require_no_symlink_components(path, leaf="file")
    payload = _load_json(ready_path)
    strict = payload.get("strict_fp32")
    venv = payload.get("venv")
    expected_strict = {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }
    if (
        payload.get("schema") != 2
        or payload.get("status") != "ready"
        or not isinstance(strict, Mapping)
        or dict(strict) != expected_strict
        or not isinstance(venv, Mapping)
        or not isinstance(payload.get("base_payload"), Mapping)
        or not isinstance(payload.get("runtime_amendment"), Mapping)
    ):
        raise PackageError("accepted H100_READY identity is invalid")
    return {
        "schema": 1,
        "ready_sha256": _hash_file(ready_path),
        "acceptance_uuid": payload.get("acceptance_uuid"),
        "source": payload.get("source"),
        "strict_fp32": dict(strict),
        "venv": {
            key: venv.get(key)
            for key in (
                "sha256",
                "venv_build_sha256",
                "base_python",
                "wheelhouse",
            )
        },
        "base_payload": dict(payload["base_payload"]),
        "runtime_amendment": dict(payload["runtime_amendment"]),
    }


def _contract(maximum_physical_file_bytes: int) -> dict[str, object]:
    if (
        isinstance(maximum_physical_file_bytes, bool)
        or not isinstance(maximum_physical_file_bytes, int)
        or maximum_physical_file_bytes <= 0
    ):
        raise PackageError("BES2 maximum physical file size must be positive")
    return {
        "schema": 2,
        "production": True,
        "payload": "one-git-bundle-plus-control-json",
        "data_artifacts": 0,
        "checkpoint_artifacts": 0,
        "test_or_final_artifacts": 0,
        "credentials": 0,
        "base_payload_reused_immutable": True,
        "canonical_h100_ready_mutated": False,
        "runtime": "accepted-sealed-native-venv",
        "precision": "32-true",
        "tf32": False,
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
        "processes_per_gpu": 1,
        "ddp": False,
        "diagnostic_scope": "train111-fixed-dev8-only",
        "training_fractions": {
            "probe": [0.5],
            "full": [0.1, 0.5],
        },
        "fraction_analysis": "per-fraction-no-pooling",
        "diagnostic_run_count": {"probe": 2, "full": 4},
        "variants": ["current_replay", "first_conv_reset"],
        "maximum_approved_gpu_hours": MAX_APPROVED_GPU_HOURS,
        "maximum_physical_file_bytes": maximum_physical_file_bytes,
    }


def _artifact_paths(
    root: Path,
    artifact: Mapping[str, object],
    *,
    maximum: int,
    expected_kind: str,
    expected_name: str,
    expected_root: str,
) -> list[Path]:
    expected_fields = {
        "kind",
        "name",
        "format",
        "extraction_root",
        "file_count",
        "unpacked_bytes",
        "archive_bytes",
        "archive_sha256",
        "archive_sha1",
        "parts",
    }
    if set(artifact) != expected_fields:
        raise PackageError("BES2 amendment artifact schema is invalid")
    if (
        artifact.get("kind") != expected_kind
        or artifact.get("name") != expected_name
        or artifact.get("format") != "file"
        or artifact.get("extraction_root") != expected_root
        or artifact.get("file_count") != 1
    ):
        raise PackageError("BES2 amendment artifact identity is invalid")
    parts = artifact.get("parts")
    if not isinstance(parts, list) or not parts:
        raise PackageError("BES2 amendment artifact parts are absent")
    paths = []
    for part in parts:
        if not isinstance(part, Mapping) or set(part) != {
            "path",
            "bytes",
            "sha256",
            "sha1",
        }:
            raise PackageError("BES2 amendment part schema is invalid")
        relative = PurePosixPath(str(part.get("path", "")))
        _validate_safe_path(relative)
        size = part.get("bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > maximum
        ):
            raise PackageError("BES2 amendment part violates Box size limit")
        path = _require_no_symlink_components(
            root.joinpath(*relative.parts), leaf="file"
        )
        if (
            path.stat().st_size != size
            or _hash_file(path) != part.get("sha256")
            or _hash_file(path, "sha1") != part.get("sha1")
        ):
            raise PackageError("BES2 amendment part hash/size mismatch")
        paths.append(path)
    if sum(path.stat().st_size for path in paths) != artifact.get("archive_bytes"):
        raise PackageError("BES2 amendment logical artifact size mismatch")
    if artifact.get("unpacked_bytes") != artifact.get("archive_bytes"):
        raise PackageError("BES2 amendment file artifact unpacked size mismatch")
    if _hash_paths(paths, "sha256") != artifact.get("archive_sha256"):
        raise PackageError("BES2 amendment logical SHA-256 mismatch")
    if _hash_paths(paths, "sha1") != artifact.get("archive_sha1"):
        raise PackageError("BES2 amendment logical SHA-1 mismatch")
    return paths


def _verify(
    package_root: Path,
    *,
    expected_h100_ready_sha256: str | None,
) -> dict[str, object]:
    root = _require_no_symlink_components(package_root, leaf="directory")
    ready = _load_json(
        _require_no_symlink_components(root / "READY.json", leaf="file")
    )
    manifest = _load_json(
        _require_no_symlink_components(root / "manifest.json", leaf="file")
    )
    sums_path = _require_no_symlink_components(root / "SHA256SUMS", leaf="file")
    _validate_control_record(root, ready.get("manifest", {}), "manifest.json")
    _validate_control_record(root, ready.get("checksums", {}), "SHA256SUMS")

    expected_manifest_fields = {
        "format_version",
        "package_type",
        "package_id",
        "created_utc",
        "contract",
        "accepted_h100",
        "source",
        "checkpoint",
        "prospective_data_view",
        "amendment_identity",
        "amendment_identity_sha256",
        "counts",
        "artifacts",
    }
    expected_ready_fields = {
        "format_version",
        "status",
        "package_id",
        "git_commit",
        "amendment_identity_sha256",
        "manifest",
        "checksums",
    }
    contract = manifest.get("contract")
    source = manifest.get("source")
    accepted_h100 = manifest.get("accepted_h100")
    checkpoint = manifest.get("checkpoint")
    identity = manifest.get("amendment_identity")
    artifacts = manifest.get("artifacts")
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("package_type") != "bes2-diagnostic-amendment"
        or not isinstance(contract, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(accepted_h100, Mapping)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(identity, Mapping)
        or not isinstance(artifacts, list)
        or len(artifacts) != 2
    ):
        raise PackageError("BES2 amendment manifest schema is invalid")
    maximum = contract.get("maximum_physical_file_bytes")
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or dict(contract) != _contract(maximum)
    ):
        raise PackageError("BES2 amendment runtime contract is invalid")
    if (
        expected_h100_ready_sha256 is not None
        and accepted_h100.get("ready_sha256") != expected_h100_ready_sha256
    ):
        raise PackageError("BES2 amendment accepted H100 binding mismatch")
    if (
        checkpoint
        != {
            "relative_path": S2_CHECKPOINT_RELATIVE,
            "sha256": S2_CHECKPOINT_SHA256,
        }
    ):
        raise PackageError("BES2 amendment checkpoint identity is invalid")

    branch = source.get("branch")
    commit = source.get("git_commit")
    bundle_sha256 = source.get("git_bundle_sha256")
    if (
        branch != BRANCH
        or not isinstance(commit, str)
        or not _HEX40.fullmatch(commit)
        or source.get("required_ancestor") != REQUIRED_ANCESTOR
        or not isinstance(bundle_sha256, str)
        or not _HEX64.fullmatch(bundle_sha256)
    ):
        raise PackageError("BES2 amendment source identity is invalid")
    expected_identity = {
        "schema": 1,
        "accepted_h100": dict(accepted_h100),
        "source": dict(source),
        "checkpoint": dict(checkpoint),
        "contract": dict(contract),
        "prospective_data_view": manifest.get("prospective_data_view"),
    }
    if (
        dict(identity) != expected_identity
        or hashlib.sha256(_canonical_json(identity)).hexdigest()
        != manifest.get("amendment_identity_sha256")
    ):
        raise PackageError("BES2 amendment content identity is invalid")
    expected_package_id = (
        f"xview3-bes2-diagnostic-{commit}-"
        f"{manifest['amendment_identity_sha256']}"
    )
    if (
        set(ready) != expected_ready_fields
        or manifest.get("package_id") != expected_package_id
        or ready.get("format_version") != FORMAT_VERSION
        or ready.get("status") != "READY"
        or ready.get("package_id") != expected_package_id
        or ready.get("git_commit") != commit
        or ready.get("amendment_identity_sha256")
        != manifest.get("amendment_identity_sha256")
        or manifest.get("counts")
        != {"git_bundles": 1, "control_json": 1}
    ):
        raise PackageError("BES2 amendment READY/package identity is invalid")

    by_kind = {
        item.get("kind"): item
        for item in artifacts
        if isinstance(item, Mapping)
    }
    if set(by_kind) != {"git_bundle", "diagnostic_control"}:
        raise PackageError("BES2 amendment artifact allowlist is invalid")
    bundle_parts = _artifact_paths(
        root,
        by_kind["git_bundle"],
        maximum=maximum,
        expected_kind="git_bundle",
        expected_name=BRANCH,
        expected_root=BUNDLE_PATH,
    )
    control_parts = _artifact_paths(
        root,
        by_kind["diagnostic_control"],
        maximum=maximum,
        expected_kind="diagnostic_control",
        expected_name="BES2_AMENDMENT",
        expected_root=CONTROL_PATH,
    )
    if _hash_paths(bundle_parts, "sha256") != bundle_sha256:
        raise PackageError("BES2 amendment source bundle binding mismatch")
    if len(control_parts) != 1:
        raise PackageError("BES2 control JSON must remain one small physical file")
    control_payload = _load_json(control_parts[0])
    if control_payload != identity:
        raise PackageError("BES2 control JSON differs from content identity")

    expected_sums = {
        part["path"]: part["sha256"]
        for artifact in artifacts
        for part in artifact["parts"]
    }
    if _parse_sha256sums(sums_path) != expected_sums:
        raise PackageError("BES2 amendment SHA256SUMS inventory mismatch")
    expected_files = set(expected_sums) | {
        "manifest.json",
        "SHA256SUMS",
        "READY.json",
    }
    if _package_regular_files(root) != expected_files:
        raise PackageError("BES2 amendment package tree is not exact")

    # Reconstruct and verify the exact branch/commit/ancestor and scan the
    # received bundle's reachable history before it can be cloned on Judy.
    if len(bundle_parts) != 1:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="bes2-bundle-", suffix=".bundle"
        )
        os.close(descriptor)
        reconstructed = Path(temporary_name)
        try:
            with reconstructed.open("wb") as output:
                for part in bundle_parts:
                    with part.open("rb") as source_file:
                        shutil.copyfileobj(source_file, output)
            _verify_git_bundle_round_trip(
                reconstructed,
                branch=BRANCH,
                commit=commit,
                required_ancestor=REQUIRED_ANCESTOR,
                production=True,
            )
        finally:
            reconstructed.unlink(missing_ok=True)
    else:
        _verify_git_bundle_round_trip(
            bundle_parts[0],
            branch=BRANCH,
            commit=commit,
            required_ancestor=REQUIRED_ANCESTOR,
            production=True,
        )
    return manifest


def prepare_bes2_verifier(
    expected_h100_ready_sha256: str | None = None,
) -> PackageVerifier:
    if (
        expected_h100_ready_sha256 is not None
        and not _HEX64.fullmatch(expected_h100_ready_sha256)
    ):
        raise PackageError("expected H100_READY SHA-256 is malformed")

    def verifier(package_root: Path) -> dict[str, object]:
        return _verify(
            package_root,
            expected_h100_ready_sha256=expected_h100_ready_sha256,
        )

    return verifier


def verify_bes2_amendment(
    package_root: Path,
    *,
    expected_h100_ready_sha256: str | None = None,
) -> dict[str, object]:
    return prepare_bes2_verifier(expected_h100_ready_sha256)(package_root)


def build_bes2_amendment(
    *,
    repo_root: Path,
    h100_ready_json: Path,
    weights_root: Path,
    output_dir: Path,
    maximum_physical_file_bytes: int,
) -> Path:
    repo = _require_no_symlink_components(repo_root, leaf="directory")
    output = _require_no_symlink_components(output_dir, leaf="directory")
    if _inside_repository_worktrees(output, repo):
        raise PackageError("BES2 amendment output must be outside repository worktrees")
    accepted_h100 = _h100_identity(h100_ready_json)
    checkpoint_path = _require_no_symlink_components(
        weights_root / S2_CHECKPOINT_RELATIVE,
        leaf="file",
    )
    if _hash_file(checkpoint_path) != S2_CHECKPOINT_SHA256:
        raise PackageError("BigEarthNet-S2 checkpoint differs from the pinned bytes")
    contract = _contract(maximum_physical_file_bytes)
    commit, commit_epoch = _git_source(repo, BRANCH, production=True)
    _require_source_ancestor(repo, commit, REQUIRED_ANCESTOR)

    staging = Path(
        tempfile.mkdtemp(prefix=".xview3-bes2-amendment-", dir=output)
    )
    try:
        bundle_path = staging / BUNDLE_PATH
        _create_git_bundle(
            repo,
            bundle_path,
            BRANCH,
            commit,
            REQUIRED_ANCESTOR,
            production=False,
        )
        bundle_artifact = _plain_artifact(
            bundle_path,
            staging,
            kind="git_bundle",
            name=BRANCH,
            extraction_root=PurePosixPath(BUNDLE_PATH),
            max_part_bytes=maximum_physical_file_bytes,
        )
        source = {
            "branch": BRANCH,
            "git_commit": commit,
            "required_ancestor": REQUIRED_ANCESTOR,
            "git_bundle_sha256": bundle_artifact["archive_sha256"],
        }
        checkpoint = {
            "relative_path": S2_CHECKPOINT_RELATIVE,
            "sha256": S2_CHECKPOINT_SHA256,
        }
        prospective_data_view = {
            "contract": "train111-fixed-dev8-no-test-v1",
            "bound_on_judy_by": "BES2_DIAGNOSTIC_READY.json",
            "test": "forbidden",
            "eval_final": "forbidden",
        }
        identity = {
            "schema": 1,
            "accepted_h100": accepted_h100,
            "source": source,
            "checkpoint": checkpoint,
            "contract": contract,
            "prospective_data_view": prospective_data_view,
        }
        control_path = staging / CONTROL_PATH
        _write_bytes(control_path, _canonical_json(identity))
        control_artifact = _plain_artifact(
            control_path,
            staging,
            kind="diagnostic_control",
            name="BES2_AMENDMENT",
            extraction_root=PurePosixPath(CONTROL_PATH),
            max_part_bytes=maximum_physical_file_bytes,
        )
        if len(control_artifact["parts"]) != 1:
            raise PackageError("BES2 control JSON unexpectedly exceeded Box limit")

        identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
        package_id = f"xview3-bes2-diagnostic-{commit}-{identity_sha256}"
        artifacts = [bundle_artifact, control_artifact]
        manifest = {
            "format_version": FORMAT_VERSION,
            "package_type": "bes2-diagnostic-amendment",
            "package_id": package_id,
            "created_utc": datetime.fromtimestamp(
                commit_epoch, timezone.utc
            ).isoformat(),
            "contract": contract,
            "accepted_h100": accepted_h100,
            "source": source,
            "checkpoint": checkpoint,
            "prospective_data_view": prospective_data_view,
            "amendment_identity": identity,
            "amendment_identity_sha256": identity_sha256,
            "counts": {"git_bundles": 1, "control_json": 1},
            "artifacts": artifacts,
        }
        _write_bytes(staging / "manifest.json", _canonical_json(manifest))
        sums = "".join(
            f"{part['sha256']}  {part['path']}\n"
            for artifact in artifacts
            for part in artifact["parts"]
        )
        _write_bytes(staging / "SHA256SUMS", sums.encode("utf-8"))
        ready = {
            "format_version": FORMAT_VERSION,
            "status": "READY",
            "package_id": package_id,
            "git_commit": commit,
            "amendment_identity_sha256": identity_sha256,
            "manifest": _physical_record(staging / "manifest.json", staging),
            "checksums": _physical_record(staging / "SHA256SUMS", staging),
        }
        # READY is deliberately the final package write.
        _write_bytes(staging / "READY.json", _canonical_json(ready))
        verify_bes2_amendment(
            staging,
            expected_h100_ready_sha256=accepted_h100["ready_sha256"],
        )
        destination = output / package_id
        if os.path.lexists(destination):
            raise PackageError(f"BES2 amendment destination exists: {destination}")
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def generate_bes2_bootstrap(
    *,
    repo_root: Path,
    package_root: Path,
    output: Path,
    expected_h100_ready_sha256: str,
) -> dict[str, object]:
    from .runtime_bootstrap import _generate_runtime_bootstrap

    return _generate_runtime_bootstrap(
        repo_root=repo_root,
        runtime_package_root=package_root,
        output=output,
        verifier=prepare_bes2_verifier(expected_h100_ready_sha256),
        production=True,
    )
