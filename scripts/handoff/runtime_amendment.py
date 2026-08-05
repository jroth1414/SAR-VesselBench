"""Content-addressed native-venv runtime amendment for the H100 handoff.

The 294 GB Sprint 7d payload is immutable. This module produces a separate,
mostly code/metadata package whose identity binds both the exact Sprint 7d
controls and the exact Sprint 7f Git bundle.  Its sole derived-data exception
is a small TRAIN+fixed-DEV8 label CSV built and audited on the source host; that
artifact is necessary to keep TEST rows out of Judy's pre-cohort data view.
The package contains no raster/chip data, weights, wheelhouse, run output,
virtual environment, or credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping

from scripts.h100.runtime_versions import EXPECTED_NATIVE_PYTHON_VERSION

from .package import (
    EXPECTED_TORCH,
    PackageError,
    _absolute_path,
    _artifact_part_paths,
    _canonical_json,
    _git_source,
    _git_value,
    _extract_tar_zst,
    _hash_file,
    _hash_paths,
    _inside_repository_worktrees,
    _load_json,
    _lock_requirements,
    _package_regular_files,
    _parse_sha256sums,
    _physical_record,
    _plain_artifact,
    _require_no_symlink_components,
    _require_source_ancestor,
    _run,
    _scan_reachable_history,
    _validate_control_record,
    _validate_safe_path,
    _write_bytes,
    verify_package,
)
from src.eval.ground_truth_audit import (
    audit_ground_truth_dataset,
    validate_ground_truth_audit_receipt,
)

FORMAT_VERSION = 2
RUNTIME_BRANCH = "sprint-7f-eval-contract"
BASE_SOURCE_GIT_SHA = "2726199efcebbebc89156e708b89df2a3415468a"
RUNTIME_REQUIRED_ANCESTOR = "26bece168cd3b9b262ffec5939b836df21b352cd"
BASE_PACKAGE_ID = (
    f"xview3-h100-fp32-{BASE_SOURCE_GIT_SHA}"
)
BASE_READY_SHA256 = (
    "b0d6ee18f9ddbd0d604cbea06610dcdbae6a9eb6d1f5ff3ea3431bd9e2d55f81"
)
BASE_MANIFEST_SHA256 = (
    "fccb0b505c89836a148afec709bb799f7af4908d955ea1b142e153154d830896"
)
BASE_SHA256SUMS_SHA256 = (
    "21c83b2e3b1b9d67bf00b8abca3ce267a5efd9362c1206b8d29ab21ca3e2d396"
)
BASE_REPO_BUNDLE_SHA256 = (
    "df21d64cd7ba2d884fcb4a454c46106b39a6c604459861681e2823ac460f2f4e"
)
BASE_ENVIRONMENT_LOCK_SHA256 = (
    "7c651f762b84801fcdf50a48accca05911b2b7f6bc1c536cb1acce0d7fa22154"
)
ENVIRONMENT_LOCK_PATH = "locks/env-v100node.txt"
RUNTIME_BUNDLE_PATH = "code/xview3-runtime.bundle"
RUNTIME_EXTRACTED_RECEIPT = "RUNTIME_AMENDMENT_EXTRACTED.json"
TRAINING_LABELS_ARTIFACT_PATH = "data/training-view/labels/train.csv"
BASE_LABEL_SHA256 = "42871b3ddf12d2a732d11d07897c21efc6c688c5d1a6c59a90839a5539e15415"
BASE_LABEL_BYTES = 11_134_981

_REQUIRED_RUNTIME_FILES = (
    ENVIRONMENT_LOCK_PATH,
    "docs/CORRECTED_REFERENCES_RUNBOOK.md",
    "scripts/h100/acceptance.py",
    "scripts/h100/build_venv.py",
    "scripts/h100/campaign.py",
    "scripts/h100/cell.py",
    "scripts/h100/contracts.py",
    "scripts/h100/cutover.py",
    "scripts/h100/data_staging.py",
    "scripts/h100/runtime_versions.py",
    "scripts/h100/lightning_contract.py",
    "scripts/h100/operator_cutover.py",
    "scripts/h100/reverse_results.py",
    "scripts/h100/wheelhouse.py",
    "scripts/handoff/control.py",
    "scripts/handoff/results.py",
    "scripts/handoff/runtime_bootstrap.py",
    "scripts/export_results.py",
    "scripts/run_corrected_references.py",
    "scripts/score_test_cohort.py",
    "scripts/score_test_split.py",
    "src/analysis/curves.py",
    "src/eval/final_eval.py",
    "src/eval/ground_truth.py",
    "src/eval/ground_truth_audit.py",
    "src/eval/heldout_contract.py",
    "src/eval/result_contract.py",
    "src/references/locateanything_zs.py",
    "src/references/runtime_provenance.py",
    "src/references/yolo26_ref.py",
    "slurm/h100/V100_DIAGNOSTIC_ISOLATION.schema.json",
    "slurm/h100/campaign.sbatch",
    "slurm/h100/smoke.sbatch",
    "slurm/h100/submit.sh",
    "slurm/h100/shims/scontrol",
)
_LAUNCH_FILES = _REQUIRED_RUNTIME_FILES[1:]
_FORBIDDEN_CONTAINER_TOKENS = re.compile(
    r"\b(apptainer|singularity|enroot|pyxis)\b|\.sif\b",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")

RuntimeVerifier = Callable[[Path], dict[str, object]]


def _expected_base_identity() -> dict[str, object]:
    return {
        "schema": 1,
        "package_id": BASE_PACKAGE_ID,
        "package_type": "h100-source-handoff",
        "source_git_commit": BASE_SOURCE_GIT_SHA,
        "ready_sha256": BASE_READY_SHA256,
        "manifest_sha256": BASE_MANIFEST_SHA256,
        "sha256sums_sha256": BASE_SHA256SUMS_SHA256,
        "repo_bundle_sha256": BASE_REPO_BUNDLE_SHA256,
        "environment_lock_sha256": BASE_ENVIRONMENT_LOCK_SHA256,
    }


def _runtime_contract(
    *,
    maximum_physical_file_bytes: int,
    production: bool,
) -> dict[str, object]:
    if (
        not isinstance(maximum_physical_file_bytes, int)
        or isinstance(maximum_physical_file_bytes, bool)
        or maximum_physical_file_bytes <= 0
    ):
        raise PackageError("maximum physical file size must be a positive integer")
    return {
        "production": production,
        "payload": "code-plus-source-audited-training-label-view",
        "base_payload_reused": True,
        "runtime": "native-venv",
        "python": EXPECTED_NATIVE_PYTHON_VERSION,
        "torch": EXPECTED_TORCH,
        "precision": "32-true",
        "strict_fp32": True,
        "tf32": False,
        "nvidia_tf32_override": "0",
        "micro_batch": 16,
        "effective_batch": 16,
        "processes_per_gpu": 1,
        "ddp": False,
        "evaluation_contract": "vessel-hm-positive-low-ignore-v2",
        "training_result_schema": 2,
        "heldout_barrier": "all-32-training-cohort-before-test",
        "pre_cohort_data_view": "train111-fixed-dev8-no-test-v1",
        "test_result_artifact": "separate-immutable-test_metrics.json",
        "v100_disposition": "continues-running-non-reportable-diagnostic",
        "control_plane": "box-transfer-v1",
        "maximum_physical_file_bytes": maximum_physical_file_bytes,
    }


def verify_base_payload(package_root: Path) -> dict[str, object]:
    """Run the Sprint 7d verifier, then require its exact immutable identity."""

    root = _require_no_symlink_components(package_root, leaf="directory")
    manifest = verify_package(root)
    artifacts = manifest.get("artifacts")
    source = manifest.get("source")
    if not isinstance(artifacts, list) or not isinstance(source, Mapping):
        raise PackageError("verified base payload has no source/artifact records")
    bundles = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("kind") == "git_bundle"
    ]
    if len(bundles) != 1:
        raise PackageError("verified base payload must contain one Git bundle")
    identity = {
        "schema": 1,
        "package_id": manifest.get("package_id"),
        "package_type": manifest.get("package_type"),
        "source_git_commit": source.get("git_commit"),
        "ready_sha256": _hash_file(root / "READY.json"),
        "manifest_sha256": _hash_file(root / "manifest.json"),
        "sha256sums_sha256": _hash_file(root / "SHA256SUMS"),
        "repo_bundle_sha256": bundles[0].get("archive_sha256"),
        "environment_lock_sha256": source.get("environment_lock_sha256"),
    }
    expected = _expected_base_identity()
    if identity != expected:
        mismatches = {
            key: {"expected": expected[key], "observed": identity.get(key)}
            for key in expected
            if identity.get(key) != expected[key]
        }
        raise PackageError(
            "base payload is not the immutable Sprint 7d payload: "
            f"{mismatches}"
        )
    return identity


def verify_base_payload_control(package_root: Path) -> dict[str, object]:
    """Verify pinned base controls and code without reading any data archive.

    The complete Sprint 7d package is verified when Box transfer publishes it.
    Compute allocations must not repeat that full 294 GB verifier before the
    held-out barrier, because doing so opens TEST raster and combined-label
    archives.  This narrower verifier hashes only the three package controls
    and the Git bundle; phase staging separately verifies every archive it is
    authorized to extract.
    """

    root = _require_no_symlink_components(package_root, leaf="directory")
    ready_path = _require_no_symlink_components(root / "READY.json", leaf="file")
    manifest_path = _require_no_symlink_components(
        root / "manifest.json", leaf="file"
    )
    sums_path = _require_no_symlink_components(root / "SHA256SUMS", leaf="file")
    ready = _load_json(ready_path)
    manifest = _load_json(manifest_path)
    _validate_control_record(
        root, ready.get("manifest", {}), "manifest.json"  # type: ignore[arg-type]
    )
    _validate_control_record(
        root, ready.get("checksums", {}), "SHA256SUMS"  # type: ignore[arg-type]
    )
    source = manifest.get("source")
    artifacts = manifest.get("artifacts")
    if not isinstance(source, Mapping) or not isinstance(artifacts, list):
        raise PackageError("base control manifest lacks source/artifact records")
    bundles = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("kind") == "git_bundle"
    ]
    if len(bundles) != 1:
        raise PackageError("base control manifest must contain one Git bundle")
    bundle = bundles[0]
    if (
        bundle.get("format") != "file"
        or bundle.get("extraction_root") != "code/xview3.bundle"
        or bundle.get("archive_sha256") != BASE_REPO_BUNDLE_SHA256
    ):
        raise PackageError("base control Git-bundle record is invalid")
    bundle_parts = _artifact_part_paths(root, bundle)
    if _hash_paths(bundle_parts, "sha256") != bundle.get("archive_sha256"):
        raise PackageError("base control Git-bundle SHA-256 mismatch")
    identity = {
        "schema": 1,
        "package_id": manifest.get("package_id"),
        "package_type": manifest.get("package_type"),
        "source_git_commit": source.get("git_commit"),
        "ready_sha256": _hash_file(ready_path),
        "manifest_sha256": _hash_file(manifest_path),
        "sha256sums_sha256": _hash_file(sums_path),
        "repo_bundle_sha256": bundle.get("archive_sha256"),
        "environment_lock_sha256": source.get("environment_lock_sha256"),
    }
    expected = _expected_base_identity()
    if identity != expected:
        mismatches = {
            key: {"expected": expected[key], "observed": identity.get(key)}
            for key in expected
            if identity.get(key) != expected[key]
        }
        raise PackageError(
            "base controls are not the immutable Sprint 7d identity: "
            f"{mismatches}"
        )
    if (
        ready.get("status") != "READY"
        or ready.get("package_id") != identity["package_id"]
        or ready.get("git_commit") != identity["source_git_commit"]
    ):
        raise PackageError("base READY/control identity mismatch")
    return identity


def _verify_runtime_checkout(
    checkout: Path,
    *,
    branch: str,
    commit: str,
    required_ancestor: str,
    production: bool,
) -> dict[str, object]:
    actual = _git_value(checkout, "rev-parse", "HEAD")
    dirty = _git_value(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    if actual != commit or dirty:
        raise PackageError("runtime bundle did not reproduce the exact clean commit")
    _require_source_ancestor(checkout, actual, required_ancestor)
    if production:
        _scan_reachable_history(checkout, branch)

    for relative in _REQUIRED_RUNTIME_FILES:
        _require_no_symlink_components(checkout / relative, leaf="file")
    lock_path = checkout / ENVIRONMENT_LOCK_PATH
    requirements = _lock_requirements(lock_path)
    if requirements.get("torch") != EXPECTED_TORCH:
        raise PackageError("runtime bundle does not carry the exact cu126 torch lock")

    if production:
        for relative in _LAUNCH_FILES:
            content = (checkout / relative).read_text(encoding="utf-8")
            match = _FORBIDDEN_CONTAINER_TOKENS.search(content)
            if match:
                raise PackageError(
                    "native runtime file retains container token "
                    f"{match.group(0)!r}: {relative}"
                )
    commit_epoch = int(_git_value(checkout, "show", "-s", "--format=%ct", commit))
    splits_path = _require_no_symlink_components(
        checkout / "data/splits.json", leaf="file"
    )
    return {
        "environment_lock_sha256": _hash_file(lock_path),
        "created_utc": datetime.fromtimestamp(
            commit_epoch, timezone.utc
        ).isoformat(),
        "splits_json": splits_path.read_text(encoding="utf-8"),
    }


def _verify_git_bundle_round_trip(
    bundle: Path,
    *,
    branch: str,
    commit: str,
    required_ancestor: str,
    production: bool,
) -> dict[str, object]:
    heads = _run(["git", "bundle", "list-heads", str(bundle)]).splitlines()
    expected = f"{commit} refs/heads/{branch}"
    if heads != [expected]:
        raise PackageError(f"runtime Git bundle branch/commit mismatch: {heads!r}")
    with tempfile.TemporaryDirectory(
        prefix="xview3-runtime-bundle-verify-"
    ) as temporary:
        temporary_root = Path(temporary)
        verifier = temporary_root / "verifier.git"
        _run(["git", "init", "--quiet", "--bare", str(verifier)])
        _run(
            [
                "git",
                "-c",
                f"safe.directory={verifier}",
                "-C",
                str(verifier),
                "bundle",
                "verify",
                str(bundle),
            ]
        )
        checkout = temporary_root / "checkout"
        _run(
            [
                "git",
                "clone",
                "--quiet",
                "--branch",
                branch,
                "--single-branch",
                str(bundle),
                str(checkout),
            ]
        )
        return _verify_runtime_checkout(
            checkout,
            branch=branch,
            commit=commit,
            required_ancestor=required_ancestor,
            production=production,
        )


def _create_git_bundle(
    repo_root: Path,
    output: Path,
    *,
    branch: str,
    commit: str,
    required_ancestor: str,
    production: bool,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "bundle",
            "create",
            str(output),
            f"refs/heads/{branch}",
        ],
        cwd=repo_root,
    )
    return _verify_git_bundle_round_trip(
        output,
        branch=branch,
        commit=commit,
        required_ancestor=required_ancestor,
        production=production,
    )


def _runtime_identity(
    *,
    base_identity: Mapping[str, object],
    source: Mapping[str, object],
    contract: Mapping[str, object],
    training_view: Mapping[str, object],
    evaluation_ground_truth: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": 1,
        "base_payload": dict(base_identity),
        "source": dict(source),
        "runtime_contract": dict(contract),
        "training_view": dict(training_view),
        "evaluation_ground_truth": dict(evaluation_ground_truth),
    }


def _physical_paths(artifact: Mapping[str, object]) -> list[str]:
    parts = artifact.get("parts")
    if not isinstance(parts, list):
        raise PackageError("runtime bundle artifact has no parts")
    paths: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping):
            raise PackageError("runtime bundle part is not an object")
        relative = PurePosixPath(str(part.get("path", "")))
        _validate_safe_path(relative)
        paths.append(relative.as_posix())
    return paths


def _source_training_view(
    *,
    base_package_root: Path,
    repo_root: Path,
    temporary_root: Path,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    from scripts.h100.data_staging import filter_training_labels

    """Derive TRAIN+DEV8 labels and the full GT audit on the source host."""

    base_manifest = _load_json(base_package_root / "manifest.json")
    artifacts = base_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PackageError("base package lacks artifacts for training-view derivation")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == "labels"
        and item.get("name") == "train.csv"
    ]
    if len(matches) != 1:
        raise PackageError("base package must contain one train.csv label artifact")
    label_artifact = matches[0]
    member_hashes = label_artifact.get("member_sha256")
    if (
        label_artifact.get("format") != "tar.zst"
        or label_artifact.get("file_count") != 1
        or label_artifact.get("unpacked_bytes") != BASE_LABEL_BYTES
        or not isinstance(member_hashes, Mapping)
        or member_hashes.get("data/raw/xview3/labels/train.csv")
        != BASE_LABEL_SHA256
    ):
        raise PackageError("immutable base label artifact identity is invalid")
    parts = _artifact_part_paths(base_package_root, label_artifact)
    _extract_tar_zst(
        parts,
        temporary_root,
        extraction_root=PurePosixPath("data/raw/xview3/labels/train.csv"),
        expected_file_count=1,
        expected_unpacked_bytes=BASE_LABEL_BYTES,
    )
    full_labels = temporary_root / "data/raw/xview3/labels/train.csv"
    if _hash_file(full_labels) != BASE_LABEL_SHA256:
        raise PackageError("extracted immutable base label SHA-256 mismatch")
    splits_path = repo_root / "data/splits.json"
    audit = audit_ground_truth_dataset(
        train_csv=full_labels,
        splits_json=splits_path,
    )
    validate_ground_truth_audit_receipt(
        audit,
        splits_json=splits_path,
        expected_train_csv_sha256=BASE_LABEL_SHA256,
        expected_train_csv_bytes=BASE_LABEL_BYTES,
    )
    filtered = temporary_root / TRAINING_LABELS_ARTIFACT_PATH
    summary = filter_training_labels(
        full_labels,
        filtered,
        splits_path=splits_path,
        production=True,
    )
    if summary.get("row_count") != 13_911 or summary.get("scene_count") != 119:
        raise PackageError("source training-label view counts are not 13,911/119")
    training_view = {
        "schema": 1,
        "contract": "train111-fixed-dev8-no-test-v1",
        "source_labels": {
            "sha256": BASE_LABEL_SHA256,
            "bytes": BASE_LABEL_BYTES,
            "archive_sha256": label_artifact.get("archive_sha256"),
        },
        "splits_sha256": _hash_file(splits_path),
        "labels": summary,
    }
    return filtered, training_view, audit


def _build_runtime_amendment(
    *,
    repo_root: Path,
    output_dir: Path,
    maximum_physical_file_bytes: int,
    base_identity: Mapping[str, object],
    branch: str,
    required_ancestor: str,
    production: bool,
    training_labels_path: Path,
    training_view: Mapping[str, object],
    evaluation_ground_truth: Mapping[str, object],
) -> Path:
    repo = _require_no_symlink_components(repo_root, leaf="directory")
    output_parent = _require_no_symlink_components(output_dir, leaf="directory")
    if _inside_repository_worktrees(output_parent, repo):
        raise PackageError(
            "runtime package output must be outside repository worktrees"
        )
    contract = _runtime_contract(
        maximum_physical_file_bytes=maximum_physical_file_bytes,
        production=production,
    )
    commit, commit_epoch = _git_source(repo, branch, production=production)
    _require_source_ancestor(repo, commit, required_ancestor)
    staging = Path(
        tempfile.mkdtemp(
            prefix=".xview3-h100-runtime-building-",
            dir=output_parent,
        )
    )
    try:
        bundle_path = staging / RUNTIME_BUNDLE_PATH
        bundle_source = _create_git_bundle(
            repo,
            bundle_path,
            branch=branch,
            commit=commit,
            required_ancestor=required_ancestor,
            production=production,
        )
        if (
            production
            and bundle_source["environment_lock_sha256"]
            != base_identity["environment_lock_sha256"]
        ):
            raise PackageError(
                "Sprint 7f environment lock differs from the immutable base payload"
            )
        bundle_artifact = _plain_artifact(
            bundle_path,
            staging,
            kind="git_bundle",
            name=branch,
            extraction_root=PurePosixPath(RUNTIME_BUNDLE_PATH),
            max_part_bytes=maximum_physical_file_bytes,
        )
        packaged_labels = staging / TRAINING_LABELS_ARTIFACT_PATH
        packaged_labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(training_labels_path, packaged_labels)
        labels_artifact = _plain_artifact(
            packaged_labels,
            staging,
            kind="training_labels",
            name="train-dev8.csv",
            extraction_root=PurePosixPath(TRAINING_LABELS_ARTIFACT_PATH),
            max_part_bytes=maximum_physical_file_bytes,
        )
        if labels_artifact["archive_sha256"] != training_view.get("labels", {}).get(
            "sha256"
        ):
            raise PackageError("training-label artifact differs from its source summary")
        source = {
            "branch": branch,
            "git_bundle_ref": f"refs/heads/{branch}",
            "git_commit": commit,
            "required_base_commit": required_ancestor,
            "environment_lock": ENVIRONMENT_LOCK_PATH,
            "environment_lock_sha256": bundle_source["environment_lock_sha256"],
            "git_bundle_sha256": bundle_artifact["archive_sha256"],
        }
        identity = _runtime_identity(
            base_identity=base_identity,
            source=source,
            contract=contract,
            training_view=training_view,
            evaluation_ground_truth=evaluation_ground_truth,
        )
        identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
        package_id = f"xview3-h100-runtime-{commit}-{identity_sha256}"
        manifest = {
            "format_version": FORMAT_VERSION,
            "package_type": "h100-runtime-amendment",
            "package_id": package_id,
            "created_utc": datetime.fromtimestamp(
                commit_epoch, timezone.utc
            ).isoformat(),
            "contract": contract,
            "base_payload": dict(base_identity),
            "source": source,
            "training_view": dict(training_view),
            "evaluation_ground_truth": dict(evaluation_ground_truth),
            "runtime_identity": identity,
            "runtime_identity_sha256": identity_sha256,
            "counts": {"git_bundles": 1, "training_label_artifacts": 1},
            "artifacts": [bundle_artifact, labels_artifact],
        }
        _write_bytes(staging / "manifest.json", _canonical_json(manifest))
        checksums = "".join(
            f"{_hash_file(staging / relative)}  {relative}\n"
            for artifact in (bundle_artifact, labels_artifact)
            for relative in _physical_paths(artifact)
        )
        _write_bytes(staging / "SHA256SUMS", checksums.encode("utf-8"))
        ready = {
            "format_version": FORMAT_VERSION,
            "status": "READY",
            "package_id": package_id,
            "git_commit": commit,
            "runtime_identity_sha256": identity_sha256,
            "manifest": _physical_record(staging / "manifest.json", staging),
            "checksums": _physical_record(staging / "SHA256SUMS", staging),
        }
        # READY is deliberately the final package write.
        _write_bytes(staging / "READY.json", _canonical_json(ready))
        _verify_runtime_amendment(
            staging, base_identity=base_identity, require_production=production
        )
        destination = output_parent / package_id
        if os.path.lexists(destination):
            raise PackageError(f"runtime package destination exists: {destination}")
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def build_runtime_amendment(
    *,
    repo_root: Path,
    base_package_root: Path,
    output_dir: Path,
    maximum_physical_file_bytes: int,
) -> Path:
    """Build a production amendment after fully verifying the Sprint 7d base."""

    base_identity = verify_base_payload(base_package_root)
    with tempfile.TemporaryDirectory(prefix="xview3-training-view-source-") as temporary:
        labels, training_view, audit = _source_training_view(
            base_package_root=base_package_root,
            repo_root=repo_root,
            temporary_root=Path(temporary),
        )
        return _build_runtime_amendment(
            repo_root=repo_root,
            output_dir=output_dir,
            maximum_physical_file_bytes=maximum_physical_file_bytes,
            base_identity=base_identity,
            branch=RUNTIME_BRANCH,
            required_ancestor=RUNTIME_REQUIRED_ANCESTOR,
            production=True,
            training_labels_path=labels,
            training_view=training_view,
            evaluation_ground_truth=audit,
        )

def _validate_artifact(
    package_root: Path,
    artifact: Mapping[str, object],
    *,
    maximum_physical_file_bytes: int,
    expected_kind: str,
    expected_name: str,
    expected_root: str,
) -> list[Path]:
    expected_fields = {
        "kind", "name", "format", "extraction_root", "file_count",
        "unpacked_bytes", "archive_bytes", "archive_sha256", "archive_sha1",
        "parts",
    }
    if set(artifact) != expected_fields:
        raise PackageError("runtime file artifact schema is invalid")
    if (
        artifact.get("kind") != expected_kind
        or artifact.get("name") != expected_name
        or artifact.get("format") != "file"
        or artifact.get("extraction_root") != expected_root
        or artifact.get("file_count") != 1
    ):
        raise PackageError(f"runtime {expected_kind} artifact identity is invalid")
    parts = artifact.get("parts")
    if not isinstance(parts, list) or not parts:
        raise PackageError("runtime bundle artifact has no parts")
    observed_names: list[str] = []
    for raw in parts:
        if not isinstance(raw, Mapping) or set(raw) != {
            "path", "bytes", "sha256", "sha1"
        }:
            raise PackageError("runtime bundle part record is invalid")
        name = str(raw.get("path", ""))
        size = raw.get("bytes")
        _validate_safe_path(PurePosixPath(name))
        if (
            not isinstance(size, int) or isinstance(size, bool)
            or size <= 0 or size > maximum_physical_file_bytes
        ):
            raise PackageError("runtime bundle part violates the physical limit")
        observed_names.append(name)
    expected_names = (
        [expected_root]
        if len(parts) == 1
        else [
            f"{expected_root}.part-{index:05d}"
            for index in range(len(parts))
        ]
    )
    if observed_names != expected_names:
        raise PackageError("runtime bundle parts are not deterministically named")
    if len(parts) > 1 and any(
        int(raw["bytes"]) != maximum_physical_file_bytes
        for raw in parts[:-1] if isinstance(raw, Mapping)
    ):
        raise PackageError("non-final runtime bundle parts are not full-sized")
    paths = _artifact_part_paths(package_root, artifact)
    if sum(path.stat().st_size for path in paths) != artifact.get("archive_bytes"):
        raise PackageError("runtime bundle logical size mismatch")
    if artifact.get("archive_bytes") != artifact.get("unpacked_bytes"):
        raise PackageError("runtime bundle unpacked size mismatch")
    if _hash_paths(paths, "sha256") != artifact.get("archive_sha256"):
        raise PackageError("runtime bundle logical SHA-256 mismatch")
    if _hash_paths(paths, "sha1") != artifact.get("archive_sha1"):
        raise PackageError("runtime bundle logical SHA-1 mismatch")
    return paths


def _verify_runtime_amendment(
    package_root: Path,
    *, base_identity: Mapping[str, object], require_production: bool,
) -> dict[str, object]:
    from scripts.h100.data_staging import training_labels_summary

    root = _require_no_symlink_components(package_root, leaf="directory")
    ready = _load_json(
        _require_no_symlink_components(root / "READY.json", leaf="file")
    )
    manifest = _load_json(
        _require_no_symlink_components(root / "manifest.json", leaf="file")
    )
    sums_path = _require_no_symlink_components(root / "SHA256SUMS", leaf="file")
    _validate_control_record(
        root, ready.get("manifest", {}), "manifest.json"  # type: ignore[arg-type]
    )
    _validate_control_record(
        root, ready.get("checksums", {}), "SHA256SUMS"  # type: ignore[arg-type]
    )
    expected_manifest_fields = {
        "format_version", "package_type", "package_id", "created_utc",
        "contract", "base_payload", "source", "training_view",
        "evaluation_ground_truth", "runtime_identity",
        "runtime_identity_sha256", "counts", "artifacts",
    }
    if set(manifest) != expected_manifest_fields:
        raise PackageError("runtime amendment manifest schema is invalid")
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("package_type") != "h100-runtime-amendment"
        or manifest.get("counts")
        != {"git_bundles": 1, "training_label_artifacts": 1}
    ):
        raise PackageError("runtime amendment manifest identity is invalid")
    contract = manifest.get("contract")
    source = manifest.get("source")
    training_view = manifest.get("training_view")
    evaluation_ground_truth = manifest.get("evaluation_ground_truth")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(contract, Mapping) or not isinstance(source, Mapping)
        or not isinstance(training_view, Mapping)
        or not isinstance(evaluation_ground_truth, Mapping)
        or not isinstance(artifacts, list) or len(artifacts) != 2
        or any(not isinstance(item, Mapping) for item in artifacts)
    ):
        raise PackageError("runtime amendment records are invalid")
    production = contract.get("production")
    if not isinstance(production, bool):
        raise PackageError("runtime amendment production flag is invalid")
    if require_production and not production:
        raise PackageError("target verification requires a production amendment")
    maximum = contract.get("maximum_physical_file_bytes")
    if not isinstance(maximum, int) or isinstance(maximum, bool):
        raise PackageError("runtime amendment maximum file size is invalid")
    if dict(contract) != _runtime_contract(
        maximum_physical_file_bytes=maximum, production=production
    ):
        raise PackageError("runtime amendment native-venv contract mismatch")
    if manifest.get("base_payload") != dict(base_identity):
        raise PackageError("runtime amendment base-payload identity mismatch")

    expected_source_fields = {
        "branch", "git_bundle_ref", "git_commit", "required_base_commit",
        "environment_lock", "environment_lock_sha256", "git_bundle_sha256",
    }
    if set(source) != expected_source_fields:
        raise PackageError("runtime amendment source schema is invalid")
    branch = str(source.get("branch", ""))
    commit = str(source.get("git_commit", ""))
    required_ancestor = str(source.get("required_base_commit", ""))
    if (
        not re.fullmatch(r"[0-9a-f]{40}", commit)
        or not re.fullmatch(r"[0-9a-f]{40}", required_ancestor)
        or source.get("git_bundle_ref") != f"refs/heads/{branch}"
        or source.get("environment_lock") != ENVIRONMENT_LOCK_PATH
        or not _SHA256.fullmatch(str(source.get("environment_lock_sha256", "")))
        or not _SHA256.fullmatch(str(source.get("git_bundle_sha256", "")))
    ):
        raise PackageError("runtime amendment source values are invalid")
    if require_production and (
        branch != RUNTIME_BRANCH
        or required_ancestor != RUNTIME_REQUIRED_ANCESTOR
        or source.get("environment_lock_sha256") != BASE_ENVIRONMENT_LOCK_SHA256
    ):
        raise PackageError("production runtime source recipe mismatch")

    bundle_artifact = artifacts[0]
    labels_artifact = artifacts[1]
    bundle_parts = _validate_artifact(
        root,
        bundle_artifact,
        maximum_physical_file_bytes=maximum,
        expected_kind="git_bundle",
        expected_name=branch,
        expected_root=RUNTIME_BUNDLE_PATH,
    )
    label_parts = _validate_artifact(
        root,
        labels_artifact,
        maximum_physical_file_bytes=maximum,
        expected_kind="training_labels",
        expected_name="train-dev8.csv",
        expected_root=TRAINING_LABELS_ARTIFACT_PATH,
    )
    if source.get("git_bundle_sha256") != bundle_artifact.get("archive_sha256"):
        raise PackageError("runtime source/bundle SHA-256 mismatch")
    sums = _parse_sha256sums(sums_path)
    physical = [
        *(_physical_paths(bundle_artifact)),
        *(_physical_paths(labels_artifact)),
    ]
    if list(sums) != physical or set(sums) != set(physical):
        raise PackageError("runtime SHA256SUMS physical-file set mismatch")
    for relative, digest in sums.items():
        if _hash_file(root / relative) != digest:
            raise PackageError(f"runtime SHA256SUMS mismatch: {relative}")

    identity = _runtime_identity(
        base_identity=base_identity,
        source=source,
        contract=contract,
        training_view=training_view,
        evaluation_ground_truth=evaluation_ground_truth,
    )
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    if (
        manifest.get("runtime_identity") != identity
        or manifest.get("runtime_identity_sha256") != identity_sha256
    ):
        raise PackageError("runtime identity digest mismatch")
    expected_id = f"xview3-h100-runtime-{commit}-{identity_sha256}"
    if manifest.get("package_id") != expected_id:
        raise PackageError("runtime package ID is not fully content-addressed")
    if set(ready) != {
        "format_version", "status", "package_id", "git_commit",
        "runtime_identity_sha256", "manifest", "checksums",
    } or (
        ready.get("format_version") != FORMAT_VERSION
        or ready.get("status") != "READY"
        or ready.get("package_id") != expected_id
        or ready.get("git_commit") != commit
        or ready.get("runtime_identity_sha256") != identity_sha256
    ):
        raise PackageError("runtime READY marker is invalid")

    with tempfile.TemporaryDirectory(
        prefix="xview3-runtime-bundle-parts-"
    ) as temporary:
        bundle = Path(temporary) / "xview3-runtime.bundle"
        with bundle.open("xb") as output:
            for part in bundle_parts:
                with part.open("rb") as source_part:
                    shutil.copyfileobj(source_part, output)
        bundle_source = _verify_git_bundle_round_trip(
            bundle, branch=branch, commit=commit,
            required_ancestor=required_ancestor, production=production,
        )
        splits_path = Path(temporary) / "splits.json"
        splits_path.write_text(str(bundle_source["splits_json"]), encoding="utf-8")
        labels_path = Path(temporary) / "train-dev8.csv"
        with labels_path.open("xb") as output:
            for part in label_parts:
                with part.open("rb") as source_part:
                    shutil.copyfileobj(source_part, output)
        summary = training_labels_summary(
            labels_path,
            splits_path=splits_path,
            production=production,
        )
        if training_view.get("labels") != summary:
            raise PackageError("runtime training-label summary mismatch")
        source_labels = training_view.get("source_labels")
        if (
            training_view.get("schema") != 1
            or training_view.get("contract") != "train111-fixed-dev8-no-test-v1"
            or training_view.get("splits_sha256") != _hash_file(splits_path)
            or not isinstance(source_labels, Mapping)
        ):
            raise PackageError("runtime training-view contract is invalid")
        if production:
            if (
                source_labels.get("sha256") != BASE_LABEL_SHA256
                or source_labels.get("bytes") != BASE_LABEL_BYTES
                or summary.get("row_count") != 13_911
                or summary.get("scene_count") != 119
            ):
                raise PackageError("production training-view source/count binding is invalid")
            validate_ground_truth_audit_receipt(
                evaluation_ground_truth,
                splits_json=splits_path,
                expected_train_csv_sha256=BASE_LABEL_SHA256,
                expected_train_csv_bytes=BASE_LABEL_BYTES,
            )
    if (
        bundle_source["environment_lock_sha256"]
        != source["environment_lock_sha256"]
        or bundle_source["created_utc"] != manifest["created_utc"]
    ):
        raise PackageError("runtime bundle source binding mismatch")
    all_files = _package_regular_files(root)
    allowed = set(physical) | {"manifest.json", "SHA256SUMS", "READY.json"}
    if all_files != allowed:
        raise PackageError(
            f"unexpected runtime package files: {sorted(all_files - allowed)}"
        )
    return manifest


def prepare_runtime_verifier(base_package_root: Path) -> RuntimeVerifier:
    """Verify the large base once and return a reusable fail-closed callback."""

    base_identity = verify_base_payload(base_package_root)

    def verifier(package_root: Path) -> dict[str, object]:
        return _verify_runtime_amendment(
            package_root,
            base_identity=base_identity,
            require_production=True,
        )

    return verifier


def prepare_runtime_control_verifier(base_package_root: Path) -> RuntimeVerifier:
    """Verify safe base controls once, then fully verify the small amendment."""

    base_identity = verify_base_payload_control(base_package_root)

    def verifier(package_root: Path) -> dict[str, object]:
        return _verify_runtime_amendment(
            package_root,
            base_identity=base_identity,
            require_production=True,
        )

    return verifier


def verify_runtime_amendment(
    package_root: Path,
    base_package_root: Path,
) -> dict[str, object]:
    """Verify a production amendment and its actual immutable base payload."""

    return prepare_runtime_verifier(base_package_root)(package_root)


def _extract_runtime_amendment(
    package_root: Path,
    destination: Path,
    *, verifier: RuntimeVerifier,
) -> Path:
    manifest = verifier(package_root)
    root = _absolute_path(package_root)
    output_root = _absolute_path(destination)
    if os.path.lexists(output_root):
        raise PackageError(
            f"runtime extraction destination must not exist: {output_root}"
        )
    parent = _require_no_symlink_components(output_root.parent, leaf="directory")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.extracting-",
            dir=parent,
        )
    )
    try:
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, list) and len(artifacts) == 2
        extracted: dict[str, dict[str, object]] = {}
        for artifact in artifacts:
            assert isinstance(artifact, Mapping)
            parts = _artifact_part_paths(root, artifact)
            relative = str(artifact["extraction_root"])
            output_path = staging / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("xb") as output:
                for part in parts:
                    with part.open("rb") as source:
                        shutil.copyfileobj(source, output)
            if _hash_file(output_path) != artifact["archive_sha256"]:
                raise PackageError(f"extracted runtime artifact SHA-256 mismatch: {relative}")
            extracted[str(artifact["kind"])] = {
                "path": relative,
                "sha256": artifact["archive_sha256"],
                "bytes": artifact["archive_bytes"],
            }
        receipt = {
            "format_version": FORMAT_VERSION,
            "package_id": manifest["package_id"],
            "manifest_sha256": _hash_file(root / "manifest.json"),
            "base_payload": manifest["base_payload"],
            "runtime_bundle": extracted["git_bundle"],
            "training_labels": extracted["training_labels"],
            "training_view": manifest["training_view"],
        }
        _write_bytes(
            staging / RUNTIME_EXTRACTED_RECEIPT, _canonical_json(receipt)
        )
        os.rename(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_root


def extract_runtime_amendment(
    package_root: Path,
    base_package_root: Path,
    destination: Path,
) -> Path:
    """Verify the base/amendment pair and atomically extract the runtime bundle."""

    verifier = prepare_runtime_verifier(base_package_root)
    return _extract_runtime_amendment(
        package_root, destination, verifier=verifier
    )
