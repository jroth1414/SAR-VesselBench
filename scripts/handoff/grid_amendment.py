"""Separate immutable code/control package for approved CPU-only grid finalization."""
from __future__ import annotations

import copy
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .bes2_amendment import _artifact_paths
from .package import (
    PackageError, _canonical_json, _create_git_bundle, _git_source, _git_value,
    _inside_repository_worktrees, _load_json, _package_regular_files,
    _parse_sha256sums, _physical_record, _plain_artifact,
    _require_no_symlink_components, _require_source_ancestor,
    _validate_control_record, _write_bytes,
)
from .grid_contract import (
    POLICY, PROTECTED_OBJECTS, GRID_ANCESTOR, runtime_contract,
    validate_grid_control,
)
from .replacement_contract import base_payload_identity
from .runtime_amendment import _verify_git_bundle_round_trip

PACKAGE_TYPE = "h100-grid-finalization-amendment"
BUNDLE_PATH = "code/xview3-grid-finalization.bundle"
CONTROL_PATH = "controls/GRID_FINALIZATION.json"


def verify_grid_source(repo: Path, *, runtime_runs_root=None) -> None:
    """Require the reviewed collector and every unchanged scientific object."""
    repo = _require_no_symlink_components(repo, leaf="directory")
    if runtime_runs_root is not None:
        from scripts.h100.resume_recovery import validate_runtime_checkout
        validate_runtime_checkout(repo, runtime_runs_root)
    elif _git_value(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise PackageError("grid finalization requires a clean execution checkout")
    _require_source_ancestor(repo, _git_value(repo, "rev-parse", "HEAD"), GRID_ANCESTOR)
    for relative, expected in PROTECTED_OBJECTS.items():
        if _git_value(repo, "rev-parse", f"HEAD:{relative}") != expected:
            raise PackageError(f"grid finalization altered reviewed model/recipe/guard source: {relative}")


def verify_grid_amendment(package_root: Path) -> dict:
    root = _require_no_symlink_components(package_root, leaf="directory")

    def control_file(name):
        return _require_no_symlink_components(root / name, leaf="file")

    ready = _load_json(control_file("READY.json"))
    manifest = _load_json(control_file("manifest.json"))
    _validate_control_record(root, ready.get("manifest", {}), "manifest.json")
    _validate_control_record(root, ready.get("checksums", {}), "SHA256SUMS")
    if set(manifest) != {
        "format_version", "package_type", "package_id", "created_utc", "source",
        "base_payload", "amendment_identity", "amendment_identity_sha256",
        "counts", "artifacts",
    } or manifest.get("format_version") != 2 or manifest.get("package_type") != PACKAGE_TYPE:
        raise PackageError("grid finalization manifest schema is invalid")
    control = validate_grid_control(manifest["amendment_identity"])
    source = control["source"]
    digest = hashlib.sha256(_canonical_json(control)).hexdigest()
    package_id = f"xview3-h100-runtime-{source['git_commit']}-{digest}"
    if (
        manifest["source"] != source or manifest["amendment_identity_sha256"] != digest
        or manifest["package_id"] != package_id
        or manifest["base_payload"] != base_payload_identity()
        or manifest["counts"] != {"git_bundles": 1, "control_json": 1}
    ):
        raise PackageError("grid finalization manifest content identity mismatch")
    if set(ready) != {
        "format_version", "status", "package_id", "git_commit",
        "amendment_identity_sha256", "manifest", "checksums",
    } or any(ready.get(key) != value for key, value in {
        "format_version": 2, "status": "READY", "package_id": package_id,
        "git_commit": source["git_commit"], "amendment_identity_sha256": digest,
    }.items()):
        raise PackageError("grid finalization READY identity mismatch")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 2 or any(not isinstance(a, Mapping) for a in artifacts):
        raise PackageError("grid finalization requires exactly two artifacts")
    by_kind = {a.get("kind"): a for a in artifacts}
    if set(by_kind) != {"git_bundle", "grid_finalization_control"}:
        raise PackageError("grid finalization artifact inventory is not allowlisted")
    maximum = control["contract"]["maximum_physical_file_bytes"]
    paths = {}
    for kind, name, relative in (
        ("git_bundle", source["branch"], BUNDLE_PATH),
        ("grid_finalization_control", "GRID_FINALIZATION", CONTROL_PATH),
    ):
        paths[kind] = _artifact_paths(
            root, by_kind[kind], maximum=maximum,
            expected_kind=kind, expected_name=name, expected_root=relative,
        )
        expected_paths = (
            [relative] if len(paths[kind]) == 1 else
            [f"{relative}.part-{index:05d}" for index in range(len(paths[kind]))]
        )
        if [part["path"] for part in by_kind[kind]["parts"]] != expected_paths:
            raise PackageError("grid finalization artifact physical paths are not canonical")
    if (
        len(paths["grid_finalization_control"]) != 1
        or paths["grid_finalization_control"][0].read_bytes() != _canonical_json(control)
    ):
        raise PackageError("grid finalization control artifact differs from canonical bound identity")
    if by_kind["git_bundle"]["archive_sha256"] != source["git_bundle_sha256"]:
        raise PackageError("grid finalization source bundle hash mismatch")
    expected_sums = {part["path"]: part["sha256"] for a in artifacts for part in a["parts"]}
    if len(expected_sums) != sum(len(a["parts"]) for a in artifacts):
        raise PackageError("grid finalization physical paths must be unique")
    if _parse_sha256sums(control_file("SHA256SUMS")) != expected_sums:
        raise PackageError("grid finalization checksums inventory mismatch")
    inventory = _package_regular_files(root)
    if inventory != set(expected_sums) | {"manifest.json", "SHA256SUMS", "READY.json"}:
        raise PackageError("grid finalization package file inventory is not exact")
    if any((root / relative).stat().st_size > maximum for relative in inventory):
        raise PackageError("grid finalization physical file exceeds the package limit")
    with tempfile.TemporaryDirectory(prefix="xview3-grid-finalization-verify-") as temp:
        bundle = Path(temp) / "source.bundle"
        with bundle.open("wb") as output:
            for part in paths["git_bundle"]:
                with part.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
        metadata = _verify_git_bundle_round_trip(
            bundle, branch=source["branch"], commit=source["git_commit"],
            required_ancestor=GRID_ANCESTOR, production=True,
            checkout_validator=verify_grid_source,
        )
        if manifest["created_utc"] != metadata["created_utc"]:
            raise PackageError("grid finalization creation time differs from committed source")
    return manifest


def build_grid_amendment(
    *, repo_root: Path, output_dir: Path, maximum_physical_file_bytes: int,
) -> Path:
    contract = runtime_contract(maximum_physical_file_bytes)
    repo = _require_no_symlink_components(repo_root, leaf="directory")
    output = _require_no_symlink_components(output_dir, leaf="directory")
    if _inside_repository_worktrees(output, repo):
        raise PackageError("grid finalization output must be outside repository worktrees")
    branch = _git_value(repo, "branch", "--show-current")
    commit, epoch = _git_source(repo, branch, production=True)
    verify_grid_source(repo)
    staging = Path(tempfile.mkdtemp(prefix=".xview3-grid-finalization-", dir=output))
    try:
        bundle = staging / BUNDLE_PATH
        _create_git_bundle(repo, bundle, branch, commit, GRID_ANCESTOR, production=False)
        bundle_artifact = _plain_artifact(
            bundle, staging, kind="git_bundle", name=branch,
            extraction_root=PurePosixPath(BUNDLE_PATH), max_part_bytes=maximum_physical_file_bytes,
        )
        source = {
            "branch": branch, "git_commit": commit, "required_ancestor": GRID_ANCESTOR,
            "git_bundle_sha256": bundle_artifact["archive_sha256"],
        }
        control = {
            "schema": 1, "policy": copy.deepcopy(POLICY), "source": source,
            "contract": contract,
            "protected_source": {"required_ancestor": GRID_ANCESTOR, "git_objects": dict(PROTECTED_OBJECTS)},
        }
        validate_grid_control(control)
        control_path = staging / CONTROL_PATH
        _write_bytes(control_path, _canonical_json(control))
        control_artifact = _plain_artifact(
            control_path, staging, kind="grid_finalization_control", name="GRID_FINALIZATION",
            extraction_root=PurePosixPath(CONTROL_PATH), max_part_bytes=maximum_physical_file_bytes,
        )
        if len(control_artifact["parts"]) != 1:
            raise PackageError("grid finalization control must fit in one physical file")
        identity_sha = hashlib.sha256(_canonical_json(control)).hexdigest()
        package_id = f"xview3-h100-runtime-{commit}-{identity_sha}"
        artifacts = [bundle_artifact, control_artifact]
        manifest = {
            "format_version": 2, "package_type": PACKAGE_TYPE, "package_id": package_id,
            "created_utc": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
            "source": source, "base_payload": base_payload_identity(),
            "amendment_identity": control, "amendment_identity_sha256": identity_sha,
            "counts": {"git_bundles": 1, "control_json": 1}, "artifacts": artifacts,
        }
        _write_bytes(staging / "manifest.json", _canonical_json(manifest))
        sums = "".join(f"{part['sha256']}  {part['path']}\n" for a in artifacts for part in a["parts"])
        _write_bytes(staging / "SHA256SUMS", sums.encode())
        ready = {
            "format_version": 2, "status": "READY", "package_id": package_id,
            "git_commit": commit, "amendment_identity_sha256": identity_sha,
            "manifest": _physical_record(staging / "manifest.json", staging),
            "checksums": _physical_record(staging / "SHA256SUMS", staging),
        }
        _write_bytes(staging / "READY.json", _canonical_json(ready))
        verify_grid_amendment(staging)
        destination = output / package_id
        if os.path.lexists(destination):
            raise PackageError("grid finalization destination already exists")
        os.rename(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def generate_grid_bootstrap(*, repo_root: Path, package_root: Path, output: Path) -> dict:
    from .runtime_bootstrap import _generate_runtime_bootstrap
    return _generate_runtime_bootstrap(
        repo_root=repo_root, runtime_package_root=package_root, output=output,
        verifier=verify_grid_amendment, production=True,
    )
