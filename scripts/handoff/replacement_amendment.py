"""Code/control-only replacement package; accepted data stays in its old package."""
from __future__ import annotations

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
    _hash_file, _inside_repository_worktrees, _load_json, _package_regular_files,
    _parse_sha256sums, _physical_record, _plain_artifact,
    _require_no_symlink_components, _require_source_ancestor,
    _validate_control_record, _write_bytes,
)
from .replacement_contract import (
    ACCEPTED_READY_SHA256, APPROVED_SOURCE, PROTECTED_OBJECTS, accepted_bindings,
    base_payload_identity, runtime_contract, validate_replacement_control, OWNER_LAUNCH_POLICY,
)
from .runtime_amendment import _verify_git_bundle_round_trip

PACKAGE_TYPE = "h100-replacement-amendment"
BUNDLE_PATH = "code/xview3-replacement.bundle"
CONTROL_PATH = "controls/REPLACEMENT_AMENDMENT.json"


def verify_protected_source(repo: Path) -> None:
    for relative, expected in PROTECTED_OBJECTS.items():
        if _git_value(repo, "rev-parse", f"HEAD:{relative}") != expected:
            raise PackageError(f"replacement altered reviewed model/recipe/guard source: {relative}")


def verify_replacement_amendment(package_root: Path) -> dict:
    root = _require_no_symlink_components(package_root, leaf="directory")
    def control_file(name):
        return _require_no_symlink_components(root / name, leaf="file")
    ready = _load_json(control_file("READY.json"))
    manifest = _load_json(control_file("manifest.json"))
    _validate_control_record(root, ready.get("manifest", {}), "manifest.json")
    _validate_control_record(root, ready.get("checksums", {}), "SHA256SUMS")
    if set(manifest) != {
        "format_version", "package_type", "package_id", "created_utc", "source",
        "base_payload", "evaluation_ground_truth", "amendment_identity",
        "amendment_identity_sha256", "counts", "artifacts",
    } or manifest.get("format_version") != 2 or manifest.get("package_type") != PACKAGE_TYPE:
        raise PackageError("replacement manifest schema is invalid")
    control = validate_replacement_control(manifest["amendment_identity"])
    source = control["source"]
    digest = hashlib.sha256(_canonical_json(control)).hexdigest()
    package_id = f"xview3-h100-runtime-{source['git_commit']}-{digest}"
    if (
        manifest["source"] != source or manifest["amendment_identity_sha256"] != digest
        or manifest["package_id"] != package_id
        or manifest["base_payload"] != base_payload_identity()
        or manifest["evaluation_ground_truth"] != control["evaluation_ground_truth"]["receipt"]
        or manifest["counts"] != {"git_bundles": 1, "control_json": 1}
    ):
        raise PackageError("replacement manifest content identity mismatch")
    if set(ready) != {
        "format_version", "status", "package_id", "git_commit",
        "amendment_identity_sha256", "manifest", "checksums",
    } or any(ready.get(k) != v for k, v in {
        "format_version": 2, "status": "READY", "package_id": package_id,
        "git_commit": source["git_commit"], "amendment_identity_sha256": digest,
    }.items()):
        raise PackageError("replacement READY identity mismatch")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 2 or any(not isinstance(a, Mapping) for a in artifacts):
        raise PackageError("replacement requires exactly two artifacts")
    by_kind = {a.get("kind"): a for a in artifacts}
    if set(by_kind) != {"git_bundle", "replacement_control"}:
        raise PackageError("replacement artifact inventory is not allowlisted")
    paths = {}
    for kind, name, relative in (
        ("git_bundle", source["branch"], BUNDLE_PATH),
        ("replacement_control", "REPLACEMENT_AMENDMENT", CONTROL_PATH),
    ):
        paths[kind] = _artifact_paths(
            root, by_kind[kind], maximum=control["contract"]["maximum_physical_file_bytes"],
            expected_kind=kind, expected_name=name, expected_root=relative,
        )
    if len(paths["replacement_control"]) != 1 or _load_json(paths["replacement_control"][0]) != control:
        raise PackageError("replacement control artifact differs from bound identity")
    if by_kind["git_bundle"]["archive_sha256"] != source["git_bundle_sha256"]:
        raise PackageError("replacement source bundle hash mismatch")
    expected_sums = {part["path"]: part["sha256"] for a in artifacts for part in a["parts"]}
    if len(expected_sums) != sum(len(a["parts"]) for a in artifacts):
        raise PackageError("replacement physical paths must be unique")
    if _parse_sha256sums(control_file("SHA256SUMS")) != expected_sums:
        raise PackageError("replacement checksums inventory mismatch")
    if _package_regular_files(root) != set(expected_sums) | {"manifest.json", "SHA256SUMS", "READY.json"}:
        raise PackageError("replacement package file inventory is not exact")
    with tempfile.TemporaryDirectory(prefix="xview3-replacement-verify-") as temp:
        bundle = Path(temp) / "source.bundle"
        with bundle.open("wb") as output:
            for part in paths["git_bundle"]:
                with part.open("rb") as input_file:
                    shutil.copyfileobj(input_file, output)
        metadata = _verify_git_bundle_round_trip(
            bundle, branch=source["branch"], commit=source["git_commit"],
            required_ancestor=APPROVED_SOURCE, production=True,
            checkout_validator=verify_protected_source,
        )
        if manifest["created_utc"] != metadata["created_utc"]:
            raise PackageError("replacement creation time differs from committed source")
    return manifest


def build_replacement_amendment(
    *, repo_root: Path, h100_ready_json: Path, output_dir: Path,
    maximum_physical_file_bytes: int,
    dgx18_owner_launch: bool = False,
) -> Path:
    repo = _require_no_symlink_components(repo_root, leaf="directory")
    output = _require_no_symlink_components(output_dir, leaf="directory")
    if _inside_repository_worktrees(output, repo):
        raise PackageError("replacement output must be outside repository worktrees")
    ready_path = _require_no_symlink_components(h100_ready_json, leaf="file")
    bindings = accepted_bindings(ready_path)
    branch = _git_value(repo, "branch", "--show-current")
    commit, epoch = _git_source(repo, branch, production=True)
    _require_source_ancestor(repo, commit, APPROVED_SOURCE)
    verify_protected_source(repo)
    staging = Path(tempfile.mkdtemp(prefix=".xview3-replacement-", dir=output))
    try:
        bundle = staging / BUNDLE_PATH
        _create_git_bundle(repo, bundle, branch, commit, APPROVED_SOURCE, production=False)
        bundle_artifact = _plain_artifact(
            bundle, staging, kind="git_bundle", name=branch,
            extraction_root=PurePosixPath(BUNDLE_PATH), max_part_bytes=maximum_physical_file_bytes,
        )
        source = {
            "branch": branch, "git_commit": commit, "required_ancestor": APPROVED_SOURCE,
            "git_bundle_sha256": bundle_artifact["archive_sha256"],
        }
        control = {
            "schema": 2, **bindings, "source": source,
            "contract": runtime_contract(maximum_physical_file_bytes),
            "protected_source": {"git_commit": APPROVED_SOURCE, "git_objects": PROTECTED_OBJECTS},
        }
        if dgx18_owner_launch:
            control["owner_launch_policy"] = dict(OWNER_LAUNCH_POLICY)
        validate_replacement_control(control)
        control_path = staging / CONTROL_PATH
        _write_bytes(control_path, _canonical_json(control))
        control_artifact = _plain_artifact(
            control_path, staging, kind="replacement_control", name="REPLACEMENT_AMENDMENT",
            extraction_root=PurePosixPath(CONTROL_PATH), max_part_bytes=maximum_physical_file_bytes,
        )
        if len(control_artifact["parts"]) != 1:
            raise PackageError("replacement control must fit in one physical file")
        identity_sha = hashlib.sha256(_canonical_json(control)).hexdigest()
        package_id = f"xview3-h100-runtime-{commit}-{identity_sha}"
        artifacts = [bundle_artifact, control_artifact]
        manifest = {
            "format_version": 2, "package_type": PACKAGE_TYPE, "package_id": package_id,
            "created_utc": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
            "source": source, "base_payload": base_payload_identity(),
            "evaluation_ground_truth": control["evaluation_ground_truth"]["receipt"],
            "amendment_identity": control, "amendment_identity_sha256": identity_sha,
            "counts": {"git_bundles": 1, "control_json": 1}, "artifacts": artifacts,
        }
        _write_bytes(staging / "manifest.json", _canonical_json(manifest))
        sums = "".join(f"{p['sha256']}  {p['path']}\n" for a in artifacts for p in a["parts"])
        _write_bytes(staging / "SHA256SUMS", sums.encode())
        ready = {
            "format_version": 2, "status": "READY", "package_id": package_id,
            "git_commit": commit, "amendment_identity_sha256": identity_sha,
            "manifest": _physical_record(staging / "manifest.json", staging),
            "checksums": _physical_record(staging / "SHA256SUMS", staging),
        }
        # No package writes follow READY; publish the verified staging tree atomically.
        _write_bytes(staging / "READY.json", _canonical_json(ready))
        verify_replacement_amendment(staging)
        destination = output / package_id
        if os.path.lexists(destination):
            raise PackageError("replacement destination already exists")
        os.rename(staging, destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def generate_replacement_bootstrap(*, repo_root: Path, package_root: Path, output: Path) -> dict:
    from .runtime_bootstrap import _generate_runtime_bootstrap
    return _generate_runtime_bootstrap(
        repo_root=repo_root, runtime_package_root=package_root, output=output,
        verifier=verify_replacement_amendment, production=True,
    )
