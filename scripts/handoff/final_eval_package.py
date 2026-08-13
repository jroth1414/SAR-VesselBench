"""Narrow, content-addressed package for the once-only final evaluation.

The package intentionally carries only the committed final-evaluation code,
the already-audited TRAIN+fixed-DEV8 label view, the human-verified label CSV
as *opaque bytes*, and one deterministic ``tar.gz`` containing VH, VV, and
bathymetry from each frozen scene's original xView3 archive.  Bathymetry is
required by the frozen inference/scorer contract for prediction-side shore
distance.  The package carries no model weights, checkpoints, run outputs,
TEST assets, environment, cache, or credentials.

Neither building nor verifying this package decodes or parses
``validation.csv``.  The final evaluator is the sole semantic consumer, and it
does so only after publishing its immutable once-only lock.  Staging performs
only safe archive extraction of the three required raster files and records
their byte hashes; it does not open them with a geospatial library.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from scripts.h100.data_staging import training_labels_summary

from .package import (
    PackageError,
    _absolute_path,
    _artifact_part_paths,
    _canonical_json,
    _git_source,
    _git_value,
    _hash_file,
    _hash_paths,
    _inside,
    _inside_repository_worktrees,
    _load_json,
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
)

FORMAT_VERSION = 1
PACKAGE_TYPE = "h100-final-eval-inputs"
FINAL_BRANCH = "sprint-8-final-eval-amendment"
CAMPAIGN_GIT_SHA = "1a82d508fbeb9fdf6868a9637611e9018952fb43"
EXPECTED_FINAL_SCENES = 50
EXPECTED_TRAINING_ROWS = 13_911
EXPECTED_TRAINING_SCENES = 119

BUNDLE_PATH = "code/xview3-final-eval.bundle"
TRAINING_LABELS_PATH = "data/final-inputs/training-view/train.csv"
VALIDATION_LABELS_PATH = "data/final-inputs/labels/validation.csv"
SCENE_ARCHIVE_ROOT = "data/final-inputs/rasters"
SCENE_RASTER_NAMES = ("VH_dB.tif", "VV_dB.tif", "bathymetry.tif")
STAGED_BUNDLE_PATH = "code/xview3-final-eval.bundle"
STAGED_REPO_PATH = "repo"

_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA64 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_PRODUCTION_FILES = (
    "data/splits.json",
    "scripts/authorize_final_eval.py",
    "scripts/handoff/final_eval_package.py",
    "src/eval/final_authorization.py",
    "src/eval/final_eval.py",
    "src/eval/final_worker.py",
    "slurm/h100/final_eval.sbatch",
    "slurm/h100/submit_final_eval.sh",
)


def _contract(*, maximum_physical_file_bytes: int, production: bool) -> dict[str, object]:
    if (
        not isinstance(maximum_physical_file_bytes, int)
        or isinstance(maximum_physical_file_bytes, bool)
        or maximum_physical_file_bytes <= 0
    ):
        raise PackageError("maximum physical file size must be a positive integer")
    return {
        "production": production,
        "purpose": "owner-amended-once-only-all-32-final-evaluation",
        "payload": (
            "code-train-dev8-opaque-final-labels-"
            "final-vh-vv-bathymetry-archives"
        ),
        "final_scene_count": EXPECTED_FINAL_SCENES if production else None,
        "training_view": "train111-fixed-dev8-no-test-v1",
        "training_rows": EXPECTED_TRAINING_ROWS if production else None,
        "validation_access": "opaque-byte-transfer-no-semantic-read-before-lock",
        "validation_consumer": "src.eval.final_eval-after-immutable-lock",
        "raster_staging": "safe-extraction-without-raster-decoding",
        "downloaded_weights": False,
        "checkpoints": False,
        "runs": False,
        "test_assets": False,
        "maximum_physical_file_bytes": maximum_physical_file_bytes,
    }


def _split_inventory(repo: Path, *, production: bool) -> dict[str, object]:
    path = _require_no_symlink_components(repo / "data/splits.json", leaf="file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_splits = payload["splits"]
        splits = {
            name: tuple(map(str, raw_splits[name]))
            for name in ("train", "dev", "test", "eval_final")
        }
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise PackageError("committed split manifest is invalid") from exc
    if any(not scene for values in splits.values() for scene in values):
        raise PackageError("committed split manifest contains an empty scene ID")
    if any(len(values) != len(set(values)) for values in splits.values()):
        raise PackageError("committed split manifest contains duplicate scene IDs")
    owners: dict[str, str] = {}
    for name, values in splits.items():
        for scene in values:
            previous = owners.setdefault(scene, name)
            if previous != name:
                raise PackageError(
                    f"scene {scene!r} occurs in both {previous} and {name}"
                )
    final_scenes = tuple(sorted(splits["eval_final"]))
    if production and (
        len(final_scenes) != EXPECTED_FINAL_SCENES
        or any(not scene.endswith("v") for scene in final_scenes)
    ):
        raise PackageError("production final split is not the frozen 50-scene set")
    return {
        "splits_sha256": _hash_file(path),
        "final_scenes": final_scenes,
    }


def _verify_checkout(
    checkout: Path,
    *,
    branch: str,
    commit: str,
    required_ancestor: str,
    production: bool,
) -> dict[str, object]:
    actual = _git_value(checkout, "rev-parse", "HEAD")
    dirty = _git_value(
        checkout, "status", "--porcelain=v1", "--untracked-files=all"
    )
    if actual != commit or dirty:
        raise PackageError("final-eval bundle did not reproduce its exact clean commit")
    _require_source_ancestor(checkout, actual, required_ancestor)
    if production:
        _scan_reachable_history(checkout, branch)
        for relative in _REQUIRED_PRODUCTION_FILES:
            _require_no_symlink_components(checkout / relative, leaf="file")
    commit_epoch = int(_git_value(checkout, "show", "-s", "--format=%ct", commit))
    inventory = _split_inventory(checkout, production=production)
    return {
        **inventory,
        "created_utc": datetime.fromtimestamp(
            commit_epoch, timezone.utc
        ).isoformat(),
    }


def _verify_bundle_round_trip(
    bundle: Path,
    *,
    branch: str,
    commit: str,
    required_ancestor: str,
    production: bool,
) -> dict[str, object]:
    heads = _run(
        ["git", "bundle", "list-heads", str(bundle), f"refs/heads/{branch}"]
    ).splitlines()
    expected = f"{commit} refs/heads/{branch}"
    if heads != [expected]:
        raise PackageError(f"final-eval Git bundle branch/commit mismatch: {heads!r}")
    with tempfile.TemporaryDirectory(prefix="xview3-final-bundle-verify-") as temporary:
        root = Path(temporary)
        verifier = root / "verifier.git"
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
        checkout = root / "checkout"
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
        return _verify_checkout(
            checkout,
            branch=branch,
            commit=commit,
            required_ancestor=required_ancestor,
            production=production,
        )


def _create_bundle(
    repo: Path,
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
            f"safe.directory={repo}",
            "bundle",
            "create",
            str(output),
            f"refs/heads/{branch}",
        ],
        cwd=repo,
    )
    return _verify_bundle_round_trip(
        output,
        branch=branch,
        commit=commit,
        required_ancestor=required_ancestor,
        production=production,
    )


def _source_file(path: Path, description: str) -> Path:
    source = _require_no_symlink_components(path, leaf="file")
    if source.stat().st_size <= 0:
        raise PackageError(f"{description} is empty: {source}")
    return source


def _copy_source(source: Path, destination: Path, description: str) -> None:
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle, destination.open("xb") as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)
    after = source.stat()
    if (
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or _hash_file(source) != _hash_file(destination)
    ):
        raise PackageError(f"{description} changed while it was copied")


def _scene_sources(
    archive_dir: Path, final_scenes: Sequence[str]
) -> dict[str, Path]:
    root = _require_no_symlink_components(archive_dir, leaf="directory")
    expected = {f"{scene}.tar.gz" for scene in final_scenes}
    observed = {path.name for path in root.glob("*.tar.gz")}
    if observed != expected:
        raise PackageError(
            "final archive inventory mismatch; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
    return {
        scene: _source_file(root / f"{scene}.tar.gz", f"archive for {scene}")
        for scene in final_scenes
    }


def _physical_paths(artifacts: Sequence[Mapping[str, object]]) -> list[str]:
    paths: list[str] = []
    for artifact in artifacts:
        parts = artifact.get("parts")
        if not isinstance(parts, list):
            raise PackageError("final-eval artifact has no parts")
        for raw in parts:
            if not isinstance(raw, Mapping):
                raise PackageError("final-eval artifact part is invalid")
            relative = PurePosixPath(str(raw.get("path", "")))
            _validate_safe_path(relative)
            paths.append(relative.as_posix())
    return paths


def _file_artifact(
    *,
    staging: Path,
    source: Path,
    relative: str,
    kind: str,
    name: str,
    maximum: int,
) -> dict[str, object]:
    destination = staging.joinpath(*PurePosixPath(relative).parts)
    _copy_source(source, destination, f"{kind}/{name}")
    return _plain_artifact(
        destination,
        staging,
        kind=kind,
        name=name,
        extraction_root=PurePosixPath(relative),
        max_part_bytes=maximum,
    )


class _HashingReader:
    """Minimal file wrapper used while re-archiving one opaque raster."""

    def __init__(self, source: object) -> None:
        self.source = source
        self.digest = hashlib.sha256()
        self.written = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.source.read(size)  # type: ignore[attr-defined]
        self.digest.update(chunk)
        self.written += len(chunk)
        return chunk


def _original_scene_members(
    archive: tarfile.TarFile, scene_id: str
) -> dict[str, tarfile.TarInfo]:
    """Validate an original DIU archive and locate its required raster members."""

    expected = {f"{scene_id}/{name}": name for name in SCENE_RASTER_NAMES}
    selected: dict[str, tarfile.TarInfo] = {}
    seen: set[str] = set()
    for member in archive.getmembers():
        raw_name = member.name.rstrip("/")
        if "\\" in raw_name or "\x00" in raw_name:
            raise PackageError(f"unsafe original archive member in {scene_id}")
        relative = PurePosixPath(raw_name)
        _validate_safe_path(relative)
        if not relative.parts or relative.parts[0] != scene_id:
            raise PackageError(
                f"original archive member escapes scene root {scene_id}: {member.name}"
            )
        normalized = relative.as_posix()
        if normalized in seen:
            raise PackageError(f"duplicate original archive member: {normalized}")
        seen.add(normalized)
        if member.isdir():
            continue
        if not member.isfile():
            raise PackageError(
                f"links/devices are forbidden in original archive: {normalized}"
            )
        output_name = expected.get(normalized)
        if output_name is not None:
            if member.size <= 0:
                raise PackageError(f"empty final raster: {normalized}")
            selected[output_name] = member
    if set(selected) != set(SCENE_RASTER_NAMES):
        raise PackageError(
            f"original archive {scene_id} lacks exact inference rasters: "
            f"{sorted(selected)}"
        )
    return selected


def _create_scene_archive(
    source_path: Path, destination: Path, *, scene_id: str
) -> list[dict[str, object]]:
    """Create a deterministic archive containing only required inference rasters."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    try:
        with tarfile.open(source_path, mode="r:gz") as source_archive:
            selected = _original_scene_members(source_archive, scene_id)
            with destination.open("xb") as raw_output:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_output, mtime=0
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed,
                        mode="w|",
                        format=tarfile.GNU_FORMAT,
                    ) as output_archive:
                        directory = tarfile.TarInfo(scene_id)
                        directory.type = tarfile.DIRTYPE
                        directory.mode = 0o755
                        directory.uid = 0
                        directory.gid = 0
                        directory.uname = "root"
                        directory.gname = "root"
                        directory.mtime = 0
                        directory.pax_headers = {}
                        output_archive.addfile(directory)
                        for name in SCENE_RASTER_NAMES:
                            member = selected[name]
                            source = source_archive.extractfile(member)
                            if source is None:
                                raise PackageError(
                                    f"cannot read original raster {scene_id}/{name}"
                                )
                            reader = _HashingReader(source)
                            info = tarfile.TarInfo(f"{scene_id}/{name}")
                            info.size = member.size
                            info.mode = 0o644
                            info.uid = 0
                            info.gid = 0
                            info.uname = "root"
                            info.gname = "root"
                            info.mtime = 0
                            info.pax_headers = {}
                            output_archive.addfile(info, reader)  # type: ignore[arg-type]
                            if reader.written != member.size:
                                raise PackageError(
                                    f"original raster changed or truncated: {scene_id}/{name}"
                                )
                            records.append(
                                {
                                    "path": f"data/raw/xview3/GRD/{scene_id}/{name}",
                                    "bytes": reader.written,
                                    "sha256": reader.digest.hexdigest(),
                                }
                            )
    except (OSError, tarfile.TarError) as exc:
        destination.unlink(missing_ok=True)
        raise PackageError(f"cannot package final archive for {scene_id}: {exc}") from exc
    return records


def _scene_artifact(
    *,
    staging: Path,
    source: Path,
    scene_id: str,
    maximum: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    relative = f"{SCENE_ARCHIVE_ROOT}/{scene_id}.tar.gz"
    destination = staging.joinpath(*PurePosixPath(relative).parts)
    before = source.stat()
    rasters = _create_scene_archive(source, destination, scene_id=scene_id)
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise PackageError(f"source archive changed while packaged: {scene_id}")
    artifact = _plain_artifact(
        destination,
        staging,
        kind="final_scene_archive",
        name=scene_id,
        extraction_root=PurePosixPath(relative),
        max_part_bytes=maximum,
    )
    return artifact, rasters


def _training_summary(
    path: Path, *, splits_path: Path, production: bool
) -> dict[str, object]:
    try:
        summary = training_labels_summary(
            path,
            splits_path=splits_path,
            production=production,
        )
    except RuntimeError as exc:
        raise PackageError(f"TRAIN+DEV8 label view is invalid: {exc}") from exc
    if production and (
        summary.get("row_count") != EXPECTED_TRAINING_ROWS
        or summary.get("scene_count") != EXPECTED_TRAINING_SCENES
    ):
        raise PackageError("production TRAIN+DEV8 labels are not 13,911 rows/119 scenes")
    return summary


def _identity(
    *,
    source: Mapping[str, object],
    contract: Mapping[str, object],
    training_view: Mapping[str, object],
    final_inputs: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": 1,
        "source": dict(source),
        "contract": dict(contract),
        "training_view": dict(training_view),
        "final_inputs": dict(final_inputs),
    }


def _build_final_eval_package(
    *,
    repo_root: Path,
    training_labels_path: Path,
    validation_labels_path: Path,
    archive_dir: Path,
    output_dir: Path,
    maximum_physical_file_bytes: int,
    branch: str,
    required_ancestor: str,
    production: bool,
) -> Path:
    """Internal builder; production callers use :func:`build_final_eval_package`."""

    repo = _require_no_symlink_components(repo_root, leaf="directory")
    output_parent = _require_no_symlink_components(output_dir, leaf="directory")
    if _inside_repository_worktrees(output_parent, repo):
        raise PackageError("final-eval package output must be outside worktrees")
    contract = _contract(
        maximum_physical_file_bytes=maximum_physical_file_bytes,
        production=production,
    )
    commit, commit_epoch = _git_source(repo, branch, production=production)
    _require_source_ancestor(repo, commit, required_ancestor)
    source_inventory = _split_inventory(repo, production=production)
    final_scenes = tuple(source_inventory["final_scenes"])
    train_source = _source_file(training_labels_path, "TRAIN+DEV8 labels")
    validation_source = _source_file(validation_labels_path, "validation labels")
    scene_sources = _scene_sources(archive_dir, final_scenes)
    training_view = {
        "schema": 1,
        "contract": "train111-fixed-dev8-no-test-v1",
        "splits_sha256": source_inventory["splits_sha256"],
        "labels": _training_summary(
            train_source,
            splits_path=repo / "data/splits.json",
            production=production,
        ),
    }

    staging = Path(
        tempfile.mkdtemp(prefix=".xview3-final-eval-building-", dir=output_parent)
    )
    try:
        bundle = staging / BUNDLE_PATH
        bundle_source = _create_bundle(
            repo,
            bundle,
            branch=branch,
            commit=commit,
            required_ancestor=required_ancestor,
            production=production,
        )
        if (
            bundle_source["splits_sha256"] != source_inventory["splits_sha256"]
            or tuple(bundle_source["final_scenes"]) != final_scenes
        ):
            raise PackageError("Git bundle split inventory differs from source checkout")
        bundle_artifact = _plain_artifact(
            bundle,
            staging,
            kind="git_bundle",
            name=branch,
            extraction_root=PurePosixPath(BUNDLE_PATH),
            max_part_bytes=maximum_physical_file_bytes,
        )
        training_artifact = _file_artifact(
            staging=staging,
            source=train_source,
            relative=TRAINING_LABELS_PATH,
            kind="training_labels",
            name="train-dev8.csv",
            maximum=maximum_physical_file_bytes,
        )
        validation_artifact = _file_artifact(
            staging=staging,
            source=validation_source,
            relative=VALIDATION_LABELS_PATH,
            kind="opaque_validation_labels",
            name="validation.csv",
            maximum=maximum_physical_file_bytes,
        )
        scene_artifacts: list[dict[str, object]] = []
        scene_rasters: dict[str, list[dict[str, object]]] = {}
        for scene in final_scenes:
            artifact, rasters = _scene_artifact(
                staging=staging,
                source=scene_sources[scene],
                scene_id=scene,
                maximum=maximum_physical_file_bytes,
            )
            scene_artifacts.append(artifact)
            scene_rasters[scene] = rasters
        artifacts = [
            bundle_artifact,
            training_artifact,
            validation_artifact,
            *scene_artifacts,
        ]
        source = {
            "branch": branch,
            "git_bundle_ref": f"refs/heads/{branch}",
            "git_commit": commit,
            "required_campaign_commit": required_ancestor,
            "git_bundle_sha256": bundle_artifact["archive_sha256"],
            "splits_sha256": source_inventory["splits_sha256"],
        }
        final_inputs = {
            "schema": 1,
            "validation_labels": {
                "path": VALIDATION_LABELS_PATH,
                "sha256": validation_artifact["archive_sha256"],
                "bytes": validation_artifact["archive_bytes"],
                "access": "opaque-bytes-only-before-final-lock",
            },
            "scenes": [
                {
                    "scene_id": artifact["name"],
                    "path": artifact["extraction_root"],
                    "sha256": artifact["archive_sha256"],
                    "bytes": artifact["archive_bytes"],
                    "rasters": scene_rasters[str(artifact["name"])],
                }
                for artifact in scene_artifacts
            ],
        }
        identity = _identity(
            source=source,
            contract=contract,
            training_view=training_view,
            final_inputs=final_inputs,
        )
        identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
        package_id = f"xview3-h100-final-eval-{commit}-{identity_sha256}"
        manifest = {
            "format_version": FORMAT_VERSION,
            "package_type": PACKAGE_TYPE,
            "package_id": package_id,
            "created_utc": datetime.fromtimestamp(
                commit_epoch, timezone.utc
            ).isoformat(),
            "source": source,
            "contract": contract,
            "training_view": training_view,
            "final_inputs": final_inputs,
            "identity": identity,
            "identity_sha256": identity_sha256,
            "counts": {
                "git_bundles": 1,
                "training_label_artifacts": 1,
                "opaque_validation_label_artifacts": 1,
                "final_scene_archives": len(final_scenes),
            },
            "artifacts": artifacts,
        }
        _write_bytes(staging / "manifest.json", _canonical_json(manifest))
        physical = _physical_paths(artifacts)
        _write_bytes(
            staging / "SHA256SUMS",
            "".join(
                f"{_hash_file(staging / relative)}  {relative}\n"
                for relative in physical
            ).encode("utf-8"),
        )
        ready = {
            "format_version": FORMAT_VERSION,
            "status": "READY",
            "package_type": PACKAGE_TYPE,
            "package_id": package_id,
            "git_commit": commit,
            "identity_sha256": identity_sha256,
            "manifest": _physical_record(staging / "manifest.json", staging),
            "checksums": _physical_record(staging / "SHA256SUMS", staging),
        }
        # READY is deliberately the final package write.
        _write_bytes(staging / "READY.json", _canonical_json(ready))
        _verify_final_eval_package(staging, require_production=production)
        destination = output_parent / package_id
        if os.path.lexists(destination):
            raise PackageError(f"final-eval package destination exists: {destination}")
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def build_final_eval_package(
    *,
    repo_root: Path,
    training_labels_path: Path,
    validation_labels_path: Path,
    archive_dir: Path,
    output_dir: Path,
    maximum_physical_file_bytes: int,
) -> Path:
    """Build the production package from one clean committed amendment."""

    return _build_final_eval_package(
        repo_root=repo_root,
        training_labels_path=training_labels_path,
        validation_labels_path=validation_labels_path,
        archive_dir=archive_dir,
        output_dir=output_dir,
        maximum_physical_file_bytes=maximum_physical_file_bytes,
        branch=FINAL_BRANCH,
        required_ancestor=CAMPAIGN_GIT_SHA,
        production=True,
    )


def _validate_artifact(
    root: Path,
    artifact: Mapping[str, object],
    *,
    maximum: int,
    kind: str,
    name: str,
    relative: str,
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
        raise PackageError(f"{kind}/{name} artifact schema is invalid")
    if (
        artifact.get("kind") != kind
        or artifact.get("name") != name
        or artifact.get("format") != "file"
        or artifact.get("extraction_root") != relative
        or artifact.get("file_count") != 1
    ):
        raise PackageError(f"{kind}/{name} artifact identity is invalid")
    parts = artifact.get("parts")
    if not isinstance(parts, list) or not parts:
        raise PackageError(f"{kind}/{name} has no physical parts")
    observed: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping) or set(part) != {
            "path",
            "bytes",
            "sha256",
            "sha1",
        }:
            raise PackageError(f"{kind}/{name} part schema is invalid")
        part_path = str(part.get("path", ""))
        size = part.get("bytes")
        _validate_safe_path(PurePosixPath(part_path))
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or size > maximum
            or _SHA64.fullmatch(str(part.get("sha256", ""))) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(part.get("sha1", ""))) is None
        ):
            raise PackageError(f"{kind}/{name} part record is invalid")
        observed.append(part_path)
    expected = (
        [relative]
        if len(parts) == 1
        else [f"{relative}.part-{index:05d}" for index in range(len(parts))]
    )
    if observed != expected:
        raise PackageError(f"{kind}/{name} parts are not deterministically named")
    if len(parts) > 1 and any(int(part["bytes"]) != maximum for part in parts[:-1]):
        raise PackageError(f"{kind}/{name} non-final split part is not full-sized")
    paths = _artifact_part_paths(root, artifact)
    logical_bytes = sum(path.stat().st_size for path in paths)
    if (
        logical_bytes != artifact.get("archive_bytes")
        or logical_bytes != artifact.get("unpacked_bytes")
        or _hash_paths(paths, "sha256") != artifact.get("archive_sha256")
        or _hash_paths(paths, "sha1") != artifact.get("archive_sha1")
    ):
        raise PackageError(f"{kind}/{name} logical byte identity mismatch")
    return paths


def _join_parts(parts: Sequence[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)


def _verify_final_eval_package(
    package_root: Path, *, require_production: bool
) -> dict[str, object]:
    root = _require_no_symlink_components(package_root, leaf="directory")
    ready = _load_json(_require_no_symlink_components(root / "READY.json", leaf="file"))
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
        "format_version",
        "package_type",
        "package_id",
        "created_utc",
        "source",
        "contract",
        "training_view",
        "final_inputs",
        "identity",
        "identity_sha256",
        "counts",
        "artifacts",
    }
    if set(manifest) != expected_manifest_fields:
        raise PackageError("final-eval manifest schema is invalid")
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("package_type") != PACKAGE_TYPE
    ):
        raise PackageError("final-eval package identity is invalid")
    contract = manifest.get("contract")
    source = manifest.get("source")
    training_view = manifest.get("training_view")
    final_inputs = manifest.get("final_inputs")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(contract, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(training_view, Mapping)
        or not isinstance(final_inputs, Mapping)
        or not isinstance(artifacts, list)
        or any(not isinstance(item, Mapping) for item in artifacts)
    ):
        raise PackageError("final-eval manifest records are invalid")
    production = contract.get("production")
    maximum = contract.get("maximum_physical_file_bytes")
    if (
        not isinstance(production, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or (require_production and not production)
        or dict(contract)
        != _contract(maximum_physical_file_bytes=maximum, production=production)
    ):
        raise PackageError("final-eval package contract is invalid")
    expected_source_fields = {
        "branch",
        "git_bundle_ref",
        "git_commit",
        "required_campaign_commit",
        "git_bundle_sha256",
        "splits_sha256",
    }
    if set(source) != expected_source_fields:
        raise PackageError("final-eval source schema is invalid")
    branch = str(source.get("branch", ""))
    commit = str(source.get("git_commit", ""))
    ancestor = str(source.get("required_campaign_commit", ""))
    if (
        _SHA40.fullmatch(commit) is None
        or _SHA40.fullmatch(ancestor) is None
        or source.get("git_bundle_ref") != f"refs/heads/{branch}"
        or _SHA64.fullmatch(str(source.get("git_bundle_sha256", ""))) is None
        or _SHA64.fullmatch(str(source.get("splits_sha256", ""))) is None
    ):
        raise PackageError("final-eval source values are invalid")
    if require_production and (branch != FINAL_BRANCH or ancestor != CAMPAIGN_GIT_SHA):
        raise PackageError("production final-eval source recipe is invalid")

    expected_counts = {
        "git_bundles": 1,
        "training_label_artifacts": 1,
        "opaque_validation_label_artifacts": 1,
        "final_scene_archives": EXPECTED_FINAL_SCENES if production else None,
    }
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or set(counts) != set(expected_counts):
        raise PackageError("final-eval counts schema is invalid")
    if any(counts[key] != value for key, value in expected_counts.items() if value is not None):
        raise PackageError("final-eval production counts are invalid")
    scene_count = counts.get("final_scene_archives")
    if not isinstance(scene_count, int) or isinstance(scene_count, bool) or scene_count <= 0:
        raise PackageError("final-eval scene archive count is invalid")
    if len(artifacts) != 3 + scene_count:
        raise PackageError("final-eval artifact count is invalid")

    bundle_artifact = artifacts[0]
    training_artifact = artifacts[1]
    validation_artifact = artifacts[2]
    bundle_parts = _validate_artifact(
        root,
        bundle_artifact,
        maximum=maximum,
        kind="git_bundle",
        name=branch,
        relative=BUNDLE_PATH,
    )
    training_parts = _validate_artifact(
        root,
        training_artifact,
        maximum=maximum,
        kind="training_labels",
        name="train-dev8.csv",
        relative=TRAINING_LABELS_PATH,
    )
    validation_parts = _validate_artifact(
        root,
        validation_artifact,
        maximum=maximum,
        kind="opaque_validation_labels",
        name="validation.csv",
        relative=VALIDATION_LABELS_PATH,
    )
    if source.get("git_bundle_sha256") != bundle_artifact.get("archive_sha256"):
        raise PackageError("final-eval source/bundle SHA-256 mismatch")

    with tempfile.TemporaryDirectory(prefix="xview3-final-package-verify-") as temporary:
        temporary_root = Path(temporary)
        bundle = temporary_root / "xview3-final-eval.bundle"
        _join_parts(bundle_parts, bundle)
        bundle_source = _verify_bundle_round_trip(
            bundle,
            branch=branch,
            commit=commit,
            required_ancestor=ancestor,
            production=production,
        )
        final_scenes = tuple(bundle_source["final_scenes"])
        if source.get("splits_sha256") != bundle_source["splits_sha256"]:
            raise PackageError("final-eval bundle/split SHA-256 mismatch")
        if len(final_scenes) != scene_count:
            raise PackageError("final-eval scene count differs from bundled split")
        labels = temporary_root / "train-dev8.csv"
        _join_parts(training_parts, labels)
        splits = temporary_root / "splits.json"
        checkout = temporary_root / "checkout-for-splits"
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
        shutil.copyfile(checkout / "data/splits.json", splits)
        observed_training = _training_summary(
            labels, splits_path=splits, production=production
        )

    if training_view != {
        "schema": 1,
        "contract": "train111-fixed-dev8-no-test-v1",
        "splits_sha256": source["splits_sha256"],
        "labels": observed_training,
    }:
        raise PackageError("final-eval training-view binding is invalid")
    if production and (
        observed_training.get("row_count") != EXPECTED_TRAINING_ROWS
        or observed_training.get("scene_count") != EXPECTED_TRAINING_SCENES
    ):
        raise PackageError("production TRAIN+DEV8 label count changed")

    expected_scene_records: list[dict[str, object]] = []
    for scene, artifact in zip(final_scenes, artifacts[3:], strict=True):
        relative = f"{SCENE_ARCHIVE_ROOT}/{scene}.tar.gz"
        scene_parts = _validate_artifact(
            root,
            artifact,
            maximum=maximum,
            kind="final_scene_archive",
            name=scene,
            relative=relative,
        )
        with tempfile.TemporaryDirectory(
            prefix="xview3-final-scene-verify-"
        ) as temporary:
            archive = Path(temporary) / f"{scene}.tar.gz"
            _join_parts(scene_parts, archive)
            extracted = _extract_scene_archive(
                archive,
                scene_id=scene,
                output_root=Path(temporary) / "GRD",
            )
        expected_scene_records.append(
            {
                "scene_id": scene,
                "path": relative,
                "sha256": artifact["archive_sha256"],
                "bytes": artifact["archive_bytes"],
                "rasters": extracted["rasters"],
            }
        )
    expected_final_inputs = {
        "schema": 1,
        "validation_labels": {
            "path": VALIDATION_LABELS_PATH,
            "sha256": validation_artifact["archive_sha256"],
            "bytes": validation_artifact["archive_bytes"],
            "access": "opaque-bytes-only-before-final-lock",
        },
        "scenes": expected_scene_records,
    }
    if dict(final_inputs) != expected_final_inputs:
        raise PackageError("final-eval input binding is invalid")
    # This is deliberately a logical byte hash only.  Do not decode, parse, or
    # inspect the human-verified label CSV here.
    if _hash_paths(validation_parts, "sha256") != expected_final_inputs[
        "validation_labels"
    ]["sha256"]:
        raise PackageError("opaque validation-label SHA-256 mismatch")

    identity = _identity(
        source=source,
        contract=contract,
        training_view=training_view,
        final_inputs=final_inputs,
    )
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    expected_id = f"xview3-h100-final-eval-{commit}-{identity_sha256}"
    if (
        manifest.get("identity") != identity
        or manifest.get("identity_sha256") != identity_sha256
        or manifest.get("package_id") != expected_id
        or manifest.get("created_utc") != bundle_source["created_utc"]
    ):
        raise PackageError("final-eval content identity is invalid")
    expected_ready_fields = {
        "format_version",
        "status",
        "package_type",
        "package_id",
        "git_commit",
        "identity_sha256",
        "manifest",
        "checksums",
    }
    if set(ready) != expected_ready_fields or (
        ready.get("format_version") != FORMAT_VERSION
        or ready.get("status") != "READY"
        or ready.get("package_type") != PACKAGE_TYPE
        or ready.get("package_id") != expected_id
        or ready.get("git_commit") != commit
        or ready.get("identity_sha256") != identity_sha256
    ):
        raise PackageError("final-eval READY marker is invalid")

    physical = _physical_paths(artifacts)
    sums = _parse_sha256sums(sums_path)
    if list(sums) != physical:
        raise PackageError("final-eval SHA256SUMS physical-file order/set mismatch")
    for relative, digest in sums.items():
        if _hash_file(root / relative) != digest:
            raise PackageError(f"final-eval SHA256SUMS mismatch: {relative}")
    all_files = _package_regular_files(root)
    allowed = set(physical) | {"manifest.json", "SHA256SUMS", "READY.json"}
    if all_files != allowed:
        raise PackageError(
            f"unexpected final-eval package files: {sorted(all_files - allowed)}"
        )
    if any(
        artifact.get("kind") in {"core_weight", "checkpoint", "run_output"}
        for artifact in artifacts
    ):
        raise PackageError("final-eval package contains a forbidden artifact kind")
    return manifest


def verify_final_eval_package(package_root: Path) -> dict[str, object]:
    """Fully verify one production final-evaluation package."""

    return _verify_final_eval_package(package_root, require_production=True)


def _extract_scene_archive(
    archive_path: Path, *, scene_id: str, output_root: Path
) -> dict[str, object]:
    """Safely extract required raster bytes without decoding their content."""

    expected = {f"{scene_id}/{name}": name for name in SCENE_RASTER_NAMES}
    observed: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    scene_root = output_root / scene_id
    scene_root.mkdir(parents=True, exist_ok=False)
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                raw_name = member.name.rstrip("/")
                if "\\" in raw_name or "\x00" in raw_name:
                    raise PackageError(f"unsafe archive member in {scene_id}")
                relative = PurePosixPath(raw_name)
                _validate_safe_path(relative)
                if not relative.parts or relative.parts[0] != scene_id:
                    raise PackageError(
                        f"archive member escapes scene root {scene_id}: {member.name}"
                    )
                normalized = relative.as_posix()
                if normalized in seen:
                    raise PackageError(f"duplicate archive member: {normalized}")
                seen.add(normalized)
                if member.isdir():
                    if normalized != scene_id:
                        raise PackageError(
                            f"unexpected directory in final archive: {normalized}"
                        )
                    continue
                if not member.isfile():
                    raise PackageError(
                        f"links/devices are forbidden in final archive: {normalized}"
                    )
                output_name = expected.get(normalized)
                if output_name is None:
                    raise PackageError(
                        f"unexpected file in narrowed final archive: {normalized}"
                    )
                if member.size <= 0:
                    raise PackageError(f"empty final raster: {normalized}")
                source = archive.extractfile(member)
                if source is None:
                    raise PackageError(f"cannot read final raster member: {normalized}")
                output = scene_root / output_name
                digest = hashlib.sha256()
                written = 0
                with output.open("xb") as target:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        written += len(chunk)
                        if written > member.size:
                            raise PackageError(f"raster exceeds tar size: {normalized}")
                        digest.update(chunk)
                        target.write(chunk)
                if written != member.size:
                    raise PackageError(f"truncated final raster: {normalized}")
                observed[output_name] = {
                    "path": f"data/raw/xview3/GRD/{scene_id}/{output_name}",
                    "bytes": written,
                    "sha256": digest.hexdigest(),
                }
    except (OSError, tarfile.TarError) as exc:
        raise PackageError(f"cannot safely stage archive for {scene_id}: {exc}") from exc
    if set(observed) != set(SCENE_RASTER_NAMES):
        raise PackageError(
            f"archive {scene_id} lacks exact inference rasters: {sorted(observed)}"
        )
    return {
        "scene_id": scene_id,
        "rasters": [observed[name] for name in SCENE_RASTER_NAMES],
    }


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    parent = _require_no_symlink_components(path.parent, leaf="directory")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        data = _canonical_json(payload)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short immutable-receipt write")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _seal_tree(root: Path) -> None:
    files: list[Path] = []
    directories: list[Path] = [root]
    for current, names, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in names:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise PackageError(f"staged view contains a non-directory: {path}")
            directories.append(path)
        for name in filenames:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise PackageError(f"staged view contains a non-regular file: {path}")
            files.append(path)
    for path in files:
        path.chmod(0o555 if path.stat().st_mode & stat.S_IXUSR else 0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)


def _check_expected_controls(
    root: Path,
    manifest: Mapping[str, object],
    *,
    expected_package_id: str | None,
    expected_ready_sha256: str | None,
    expected_manifest_sha256: str | None,
    expected_sha256sums_sha256: str | None,
) -> None:
    supplied = (
        expected_package_id,
        expected_ready_sha256,
        expected_manifest_sha256,
        expected_sha256sums_sha256,
    )
    if any(value is not None for value in supplied) and any(
        value is None for value in supplied
    ):
        raise PackageError("expected package ID and all three control hashes travel together")
    if expected_package_id is None:
        return
    observed = (
        manifest.get("package_id"),
        _hash_file(root / "READY.json"),
        _hash_file(root / "manifest.json"),
        _hash_file(root / "SHA256SUMS"),
    )
    if observed != supplied:
        raise PackageError("final-eval package differs from its pinned control identity")


def _unread_artifact_metadata(
    artifact: Mapping[str, object],
    *,
    maximum: int,
    kind: str,
    name: str,
    relative: str,
) -> None:
    """Validate one artifact record without opening its payload bytes."""

    if set(artifact) != {
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
    } or (
        artifact.get("kind") != kind
        or artifact.get("name") != name
        or artifact.get("format") != "file"
        or artifact.get("extraction_root") != relative
        or artifact.get("file_count") != 1
    ):
        raise PackageError(f"{kind}/{name} metadata identity is invalid")
    archive_bytes = artifact.get("archive_bytes")
    if (
        not isinstance(archive_bytes, int)
        or isinstance(archive_bytes, bool)
        or archive_bytes <= 0
        or artifact.get("unpacked_bytes") != archive_bytes
        or _SHA64.fullmatch(str(artifact.get("archive_sha256", ""))) is None
        or _SHA40.fullmatch(str(artifact.get("archive_sha1", ""))) is None
    ):
        raise PackageError(f"{kind}/{name} metadata byte identity is invalid")
    parts = artifact.get("parts")
    if not isinstance(parts, list) or not parts:
        raise PackageError(f"{kind}/{name} metadata has no parts")
    observed: list[str] = []
    total = 0
    for part in parts:
        if not isinstance(part, Mapping) or set(part) != {
            "path",
            "bytes",
            "sha256",
            "sha1",
        }:
            raise PackageError(f"{kind}/{name} part metadata is invalid")
        part_path = str(part.get("path", ""))
        part_bytes = part.get("bytes")
        _validate_safe_path(PurePosixPath(part_path))
        if (
            not isinstance(part_bytes, int)
            or isinstance(part_bytes, bool)
            or part_bytes <= 0
            or part_bytes > maximum
            or _SHA64.fullmatch(str(part.get("sha256", ""))) is None
            or _SHA40.fullmatch(str(part.get("sha1", ""))) is None
        ):
            raise PackageError(f"{kind}/{name} part metadata is invalid")
        observed.append(part_path)
        total += part_bytes
    expected = (
        [relative]
        if len(parts) == 1
        else [f"{relative}.part-{index:05d}" for index in range(len(parts))]
    )
    if observed != expected or total != archive_bytes:
        raise PackageError(f"{kind}/{name} split metadata is invalid")
    if len(parts) > 1 and any(int(part["bytes"]) != maximum for part in parts[:-1]):
        raise PackageError(f"{kind}/{name} non-final part metadata is not full-sized")


def _verify_training_view_only(
    package_root: Path,
    *,
    require_production: bool,
    expected_package_id: str,
    expected_ready_sha256: str,
    expected_manifest_sha256: str,
    expected_sha256sums_sha256: str,
) -> tuple[dict[str, object], list[Path]]:
    """Verify controls, code, and TRAIN+DEV8 without reading final payloads."""

    root = _require_no_symlink_components(package_root, leaf="directory")
    ready = _load_json(_require_no_symlink_components(root / "READY.json", leaf="file"))
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
    _check_expected_controls(
        root,
        manifest,
        expected_package_id=expected_package_id,
        expected_ready_sha256=expected_ready_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_sha256sums_sha256=expected_sha256sums_sha256,
    )
    if set(manifest) != {
        "format_version",
        "package_type",
        "package_id",
        "created_utc",
        "source",
        "contract",
        "training_view",
        "final_inputs",
        "identity",
        "identity_sha256",
        "counts",
        "artifacts",
    } or set(ready) != {
        "format_version",
        "status",
        "package_type",
        "package_id",
        "git_commit",
        "identity_sha256",
        "manifest",
        "checksums",
    }:
        raise PackageError("final-eval control schema is invalid")
    contract = manifest.get("contract")
    source = manifest.get("source")
    training_view = manifest.get("training_view")
    final_inputs = manifest.get("final_inputs")
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("format_version") != FORMAT_VERSION
        or manifest.get("package_type") != PACKAGE_TYPE
        or not isinstance(contract, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(training_view, Mapping)
        or not isinstance(final_inputs, Mapping)
        or not isinstance(artifacts, list)
        or any(not isinstance(artifact, Mapping) for artifact in artifacts)
    ):
        raise PackageError("final-eval control manifest is invalid")
    production = contract.get("production")
    maximum = contract.get("maximum_physical_file_bytes")
    if (
        not isinstance(production, bool)
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or (require_production and not production)
        or dict(contract)
        != _contract(maximum_physical_file_bytes=maximum, production=production)
    ):
        raise PackageError("final-eval control contract is invalid")
    branch = str(source.get("branch", ""))
    commit = str(source.get("git_commit", ""))
    ancestor = str(source.get("required_campaign_commit", ""))
    if (
        set(source)
        != {
            "branch",
            "git_bundle_ref",
            "git_commit",
            "required_campaign_commit",
            "git_bundle_sha256",
            "splits_sha256",
        }
        or _SHA40.fullmatch(commit) is None
        or _SHA40.fullmatch(ancestor) is None
        or source.get("git_bundle_ref") != f"refs/heads/{branch}"
        or _SHA64.fullmatch(str(source.get("git_bundle_sha256", ""))) is None
        or _SHA64.fullmatch(str(source.get("splits_sha256", ""))) is None
        or (require_production and (branch != FINAL_BRANCH or ancestor != CAMPAIGN_GIT_SHA))
    ):
        raise PackageError("final-eval control source is invalid")
    counts = manifest.get("counts")
    scene_count = counts.get("final_scene_archives") if isinstance(counts, Mapping) else None
    if (
        not isinstance(counts, Mapping)
        or set(counts)
        != {
            "git_bundles",
            "training_label_artifacts",
            "opaque_validation_label_artifacts",
            "final_scene_archives",
        }
        or counts.get("git_bundles") != 1
        or counts.get("training_label_artifacts") != 1
        or counts.get("opaque_validation_label_artifacts") != 1
        or not isinstance(scene_count, int)
        or isinstance(scene_count, bool)
        or scene_count <= 0
        or (production and scene_count != EXPECTED_FINAL_SCENES)
        or len(artifacts) != 3 + scene_count
    ):
        raise PackageError("final-eval control artifact count is invalid")

    bundle_artifact = artifacts[0]
    training_artifact = artifacts[1]
    validation_artifact = artifacts[2]
    assert isinstance(bundle_artifact, Mapping)
    assert isinstance(training_artifact, Mapping)
    assert isinstance(validation_artifact, Mapping)
    bundle_parts = _validate_artifact(
        root,
        bundle_artifact,
        maximum=maximum,
        kind="git_bundle",
        name=branch,
        relative=BUNDLE_PATH,
    )
    training_parts = _validate_artifact(
        root,
        training_artifact,
        maximum=maximum,
        kind="training_labels",
        name="train-dev8.csv",
        relative=TRAINING_LABELS_PATH,
    )
    _unread_artifact_metadata(
        validation_artifact,
        maximum=maximum,
        kind="opaque_validation_labels",
        name="validation.csv",
        relative=VALIDATION_LABELS_PATH,
    )

    with tempfile.TemporaryDirectory(prefix="xview3-final-training-verify-") as temporary:
        temporary_root = Path(temporary)
        bundle = temporary_root / "xview3-final-eval.bundle"
        _join_parts(bundle_parts, bundle)
        bundle_source = _verify_bundle_round_trip(
            bundle,
            branch=branch,
            commit=commit,
            required_ancestor=ancestor,
            production=production,
        )
        labels = temporary_root / "train-dev8.csv"
        _join_parts(training_parts, labels)
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
        observed_training = _training_summary(
            labels,
            splits_path=checkout / "data/splits.json",
            production=production,
        )
    if (
        source.get("splits_sha256") != bundle_source["splits_sha256"]
        or source.get("git_bundle_sha256") != bundle_artifact.get("archive_sha256")
        or training_view
        != {
            "schema": 1,
            "contract": "train111-fixed-dev8-no-test-v1",
            "splits_sha256": source["splits_sha256"],
            "labels": observed_training,
        }
    ):
        raise PackageError("final-eval TRAIN+DEV8 binding is invalid")

    scenes = final_inputs.get("scenes")
    validation_record = final_inputs.get("validation_labels")
    final_scene_ids = tuple(bundle_source["final_scenes"])
    if (
        final_inputs.get("schema") != 1
        or not isinstance(validation_record, Mapping)
        or validation_record
        != {
            "path": VALIDATION_LABELS_PATH,
            "sha256": validation_artifact["archive_sha256"],
            "bytes": validation_artifact["archive_bytes"],
            "access": "opaque-bytes-only-before-final-lock",
        }
        or not isinstance(scenes, list)
        or len(scenes) != scene_count
        or len(final_scene_ids) != scene_count
    ):
        raise PackageError("final-eval final-input metadata is invalid")
    for scene, record, artifact in zip(
        final_scene_ids, scenes, artifacts[3:], strict=True
    ):
        if not isinstance(record, Mapping) or not isinstance(artifact, Mapping):
            raise PackageError("final-eval scene metadata is invalid")
        relative = f"{SCENE_ARCHIVE_ROOT}/{scene}.tar.gz"
        _unread_artifact_metadata(
            artifact,
            maximum=maximum,
            kind="final_scene_archive",
            name=scene,
            relative=relative,
        )
        rasters = record.get("rasters")
        if (
            record.get("scene_id") != scene
            or record.get("path") != relative
            or record.get("sha256") != artifact.get("archive_sha256")
            or record.get("bytes") != artifact.get("archive_bytes")
            or not isinstance(rasters, list)
            or [raster.get("path") if isinstance(raster, Mapping) else None for raster in rasters]
            != [
                *(f"data/raw/xview3/GRD/{scene}/{name}" for name in SCENE_RASTER_NAMES),
            ]
            or any(
                not isinstance(raster, Mapping)
                or set(raster) != {"path", "bytes", "sha256"}
                or not isinstance(raster.get("bytes"), int)
                or isinstance(raster.get("bytes"), bool)
                or int(raster["bytes"]) <= 0
                or _SHA64.fullmatch(str(raster.get("sha256", ""))) is None
                for raster in rasters
            )
        ):
            raise PackageError(f"final-eval scene metadata is invalid: {scene}")

    identity = _identity(
        source=source,
        contract=contract,
        training_view=training_view,
        final_inputs=final_inputs,
    )
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    expected_id = f"xview3-h100-final-eval-{commit}-{identity_sha256}"
    if (
        manifest.get("identity") != identity
        or manifest.get("identity_sha256") != identity_sha256
        or manifest.get("package_id") != expected_id
        or expected_id != expected_package_id
        or manifest.get("created_utc") != bundle_source["created_utc"]
        or ready.get("status") != "READY"
        or ready.get("package_id") != expected_id
        or ready.get("git_commit") != commit
        or ready.get("identity_sha256") != identity_sha256
    ):
        raise PackageError("final-eval control content identity is invalid")
    sums = _parse_sha256sums(sums_path)
    physical = _physical_paths(artifacts)
    if list(sums) != physical:
        raise PackageError("final-eval control physical inventory is invalid")
    for artifact in artifacts:
        assert isinstance(artifact, Mapping)
        for part in artifact["parts"]:  # type: ignore[index]
            assert isinstance(part, Mapping)
            if sums.get(str(part["path"])) != part.get("sha256"):
                raise PackageError("final-eval control checksum metadata differs")
    all_files = _package_regular_files(root)
    if all_files != set(physical) | {"manifest.json", "SHA256SUMS", "READY.json"}:
        raise PackageError("final-eval package file inventory is invalid")
    # Only the bundle and training-label parts above were opened. The final
    # label and raster artifacts remain untouched byte-for-byte.
    return manifest, training_parts


def _stage_training_labels(
    package_root: Path,
    output: Path,
    *,
    require_production: bool,
    expected_package_id: str,
    expected_ready_sha256: str,
    expected_manifest_sha256: str,
    expected_sha256sums_sha256: str,
) -> dict[str, object]:
    manifest, parts = _verify_training_view_only(
        package_root,
        require_production=require_production,
        expected_package_id=expected_package_id,
        expected_ready_sha256=expected_ready_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_sha256sums_sha256=expected_sha256sums_sha256,
    )
    target = _absolute_path(output)
    parent = _require_no_symlink_components(target.parent, leaf="directory")
    if os.path.lexists(target):
        raise PackageError(f"training-label output already exists: {target}")
    training_artifact = manifest["artifacts"][1]  # type: ignore[index]
    assert isinstance(training_artifact, Mapping)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o444)
    published_info = os.fstat(descriptor)
    try:
        digest = hashlib.sha256()
        written = 0
        for part in parts:
            with part.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    view = memoryview(chunk)
                    while view:
                        count = os.write(descriptor, view)
                        if count <= 0:
                            raise OSError("short TRAIN+DEV8 staging write")
                        view = view[count:]
                    digest.update(chunk)
                    written += len(chunk)
        if (
            written != training_artifact.get("archive_bytes")
            or digest.hexdigest() != training_artifact.get("archive_sha256")
        ):
            raise PackageError("published TRAIN+DEV8 bytes differ from package")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        try:
            current = target.lstat()
            if (current.st_dev, current.st_ino) == (
                published_info.st_dev,
                published_info.st_ino,
            ):
                target.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {
        "status": "training-labels-staged",
        "package_id": manifest["package_id"],
        "evaluator_git_sha": manifest["source"]["git_commit"],  # type: ignore[index]
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": _hash_file(target),
        "row_count": manifest["training_view"]["labels"]["row_count"],  # type: ignore[index]
        "scene_count": manifest["training_view"]["labels"]["scene_count"],  # type: ignore[index]
    }


def stage_training_labels(
    package_root: Path,
    output: Path,
    *,
    expected_package_id: str,
    expected_ready_sha256: str,
    expected_manifest_sha256: str,
    expected_sha256sums_sha256: str,
) -> dict[str, object]:
    """Publish only TRAIN+DEV8; never open final label or raster bytes."""

    return _stage_training_labels(
        package_root,
        output,
        require_production=True,
        expected_package_id=expected_package_id,
        expected_ready_sha256=expected_ready_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_sha256sums_sha256=expected_sha256sums_sha256,
    )


def _stage_final_eval_package(
    package_root: Path,
    destination: Path,
    receipt: Path,
    *,
    require_production: bool,
    expected_package_id: str | None = None,
    expected_ready_sha256: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_sha256sums_sha256: str | None = None,
) -> Path:
    manifest = _verify_final_eval_package(
        package_root, require_production=require_production
    )
    package = _absolute_path(package_root)
    output = _absolute_path(destination)
    receipt_path = _absolute_path(receipt)
    _check_expected_controls(
        package,
        manifest,
        expected_package_id=expected_package_id,
        expected_ready_sha256=expected_ready_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_sha256sums_sha256=expected_sha256sums_sha256,
    )
    if os.path.lexists(output):
        raise PackageError(f"final data-view destination already exists: {output}")
    if os.path.lexists(receipt_path):
        raise PackageError(f"final data-view receipt already exists: {receipt_path}")
    parent = _require_no_symlink_components(output.parent, leaf="directory")
    receipt_parent = _require_no_symlink_components(
        receipt_path.parent, leaf="directory"
    )
    if _inside(receipt_path, output):
        raise PackageError("final data-view receipt must be outside the sealed view")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent)
    )
    published = False
    temporary_receipt: Path | None = None
    canonical_receipt_linked = False
    canonical_receipt_committed = False
    try:
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, list)
        bundle_artifact = artifacts[0]
        training_artifact = artifacts[1]
        validation_artifact = artifacts[2]
        assert isinstance(bundle_artifact, Mapping)
        assert isinstance(training_artifact, Mapping)
        assert isinstance(validation_artifact, Mapping)
        bundle = staging / STAGED_BUNDLE_PATH
        _join_parts(_artifact_part_paths(package, bundle_artifact), bundle)
        checkout = staging / STAGED_REPO_PATH
        source = manifest["source"]
        assert isinstance(source, Mapping)
        _run(
            [
                "git",
                "clone",
                "--quiet",
                "--branch",
                str(source["branch"]),
                "--single-branch",
                str(bundle),
                str(checkout),
            ]
        )
        _run(["git", "remote", "remove", "origin"], cwd=checkout)
        _verify_checkout(
            checkout,
            branch=str(source["branch"]),
            commit=str(source["git_commit"]),
            required_ancestor=str(source["required_campaign_commit"]),
            production=require_production,
        )

        labels_root = checkout / "data/raw/xview3/labels"
        labels_root.mkdir(parents=True, exist_ok=True)
        train_output = labels_root / "train.csv"
        validation_output = labels_root / "validation.csv"
        _join_parts(_artifact_part_paths(package, training_artifact), train_output)
        _join_parts(_artifact_part_paths(package, validation_artifact), validation_output)
        if (
            _hash_file(train_output) != training_artifact["archive_sha256"]
            or _hash_file(validation_output) != validation_artifact["archive_sha256"]
        ):
            raise PackageError("staged label bytes differ from the verified package")

        raster_root = checkout / "data/raw/xview3/GRD"
        raster_root.mkdir(parents=True, exist_ok=True)
        scene_receipts: list[dict[str, object]] = []
        for artifact in artifacts[3:]:
            assert isinstance(artifact, Mapping)
            scene_id = str(artifact["name"])
            archive = staging / "archive-work" / f"{scene_id}.tar.gz"
            _join_parts(_artifact_part_paths(package, artifact), archive)
            if _hash_file(archive) != artifact["archive_sha256"]:
                raise PackageError(f"staged archive hash mismatch: {scene_id}")
            scene_receipt = _extract_scene_archive(
                archive, scene_id=scene_id, output_root=raster_root
            )
            expected_scene = manifest["final_inputs"]["scenes"][  # type: ignore[index]
                len(scene_receipts)
            ]
            if (
                not isinstance(expected_scene, Mapping)
                or expected_scene.get("scene_id") != scene_id
                or expected_scene.get("rasters") != scene_receipt["rasters"]
            ):
                raise PackageError(
                    f"staged raster bytes differ from manifest binding: {scene_id}"
                )
            scene_receipt["source_archive"] = {
                "path": artifact["extraction_root"],
                "bytes": artifact["archive_bytes"],
                "sha256": artifact["archive_sha256"],
            }
            scene_receipts.append(scene_receipt)
            archive.unlink()
        (staging / "archive-work").rmdir()

        dirty = _git_value(
            checkout, "status", "--porcelain=v1", "--untracked-files=all"
        )
        if dirty:
            raise PackageError("staged final checkout is not Git-clean")
        receipt_payload = {
            "schema": 1,
            "status": "final-eval-data-view-staged",
            "package": {
                "package_id": manifest["package_id"],
                "identity_sha256": manifest["identity_sha256"],
                "ready_sha256": _hash_file(package / "READY.json"),
                "manifest_sha256": _hash_file(package / "manifest.json"),
                "sha256sums_sha256": _hash_file(package / "SHA256SUMS"),
            },
            "source": dict(source),
            "view": {
                "root": str(output),
                "repo": str(output / STAGED_REPO_PATH),
                "bundle": str(output / STAGED_BUNDLE_PATH),
                "training_labels": {
                    "path": "repo/data/raw/xview3/labels/train.csv",
                    "bytes": training_artifact["archive_bytes"],
                    "sha256": training_artifact["archive_sha256"],
                },
                "validation_labels": {
                    "path": "repo/data/raw/xview3/labels/validation.csv",
                    "bytes": validation_artifact["archive_bytes"],
                    "sha256": validation_artifact["archive_sha256"],
                    "access": "opaque-bytes-staged-not-semantically-read",
                },
                "scenes": scene_receipts,
            },
        }
        _seal_tree(staging)
        os.rename(staging, output)
        published = True
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        # Validate an immutable receipt inode before making its canonical name
        # visible.  A post-stage validation failure must never strand a
        # canonical FINAL_DATA_VIEW.json and thereby consume the pre-lock lane.
        descriptor, raw_temporary_receipt = tempfile.mkstemp(
            prefix=f".{receipt_path.name}.validated-",
            dir=receipt_parent,
        )
        temporary_receipt = Path(raw_temporary_receipt)
        try:
            data = _canonical_json(receipt_payload)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short temporary immutable-receipt write")
                view = view[written:]
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # Validation labels remain byte-opaque here.  The validator hashes the
        # exact staged bytes and clean checkout while the receipt is still
        # non-canonical.
        validate_staged_view(
            temporary_receipt,
            expected_repo=output / STAGED_REPO_PATH,
            expected_package_id=str(manifest["package_id"]),
            expected_evaluator_git_sha=str(source["git_commit"]),
            expected_campaign_git_sha=str(source["required_campaign_commit"]),
            expected_splits_sha256=str(source["splits_sha256"]),
            production=require_production,
        )
        try:
            os.link(temporary_receipt, receipt_path, follow_symlinks=False)
            canonical_receipt_linked = True
        except FileExistsError as exc:
            raise PackageError(
                f"final data-view receipt appeared during staging: {receipt_path}"
            ) from exc
        directory = os.open(receipt_parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        canonical_receipt_committed = True
        # Once the canonical name has been fsynced, hidden-temp cleanup must
        # not turn a successful publication into an ambiguous failure.  A
        # leftover dotfile is non-authoritative and can be reviewed/removed;
        # FINAL_DATA_VIEW.json remains the sole consumable receipt.
        try:
            temporary_receipt.unlink()
            temporary_receipt = None
            directory = os.open(receipt_parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    except BaseException:
        if canonical_receipt_linked and not canonical_receipt_committed:
            try:
                if (
                    temporary_receipt is not None
                    and receipt_path.stat(follow_symlinks=False).st_ino
                    == temporary_receipt.stat(follow_symlinks=False).st_ino
                    and receipt_path.stat(follow_symlinks=False).st_dev
                    == temporary_receipt.stat(follow_symlinks=False).st_dev
                ):
                    receipt_path.unlink()
                    directory = os.open(receipt_parent, os.O_RDONLY)
                    try:
                        os.fsync(directory)
                    finally:
                        os.close(directory)
            except (FileNotFoundError, OSError):
                # The Slurm EXIT guard independently recognizes and archives
                # this allocation-bound pre-lock receipt if local rollback
                # cannot be made durable.
                pass
        if temporary_receipt is not None:
            try:
                temporary_receipt.unlink()
            except FileNotFoundError:
                pass
        if not published:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def stage_final_eval_package(
    package_root: Path,
    destination: Path,
    receipt: Path,
    *,
    expected_package_id: str,
    expected_ready_sha256: str,
    expected_manifest_sha256: str,
    expected_sha256sums_sha256: str,
) -> Path:
    """Verify and atomically publish one sealed allocation-private data view."""

    return _stage_final_eval_package(
        package_root,
        destination,
        receipt,
        require_production=True,
        expected_package_id=expected_package_id,
        expected_ready_sha256=expected_ready_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_sha256sums_sha256=expected_sha256sums_sha256,
    )


def _receipt_json(path: Path) -> tuple[dict[str, object], str]:
    receipt = _require_no_symlink_components(path, leaf="file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(receipt, flags)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o444
        ):
            raise PackageError("final data-view receipt must have exact mode 0444")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise PackageError("final data-view receipt changed while read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise PackageError("final data-view receipt is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PackageError("final data-view receipt is not a JSON object")
    if raw != _canonical_json(value):
        raise PackageError("final data-view receipt is not canonical JSON")
    return value, hashlib.sha256(raw).hexdigest()


def _receipt_file(
    repo: Path,
    record: Mapping[str, object],
    expected_relative: str,
    *,
    expected_access: str | None = None,
) -> None:
    expected_fields = {"path", "bytes", "sha256"}
    if expected_access is not None:
        expected_fields.add("access")
    if set(record) != expected_fields or (
        expected_access is not None and record.get("access") != expected_access
    ):
        raise PackageError(f"unexpected staged-file receipt fields: {expected_relative}")
    if record.get("path") != expected_relative:
        raise PackageError(f"staged-file receipt path mismatch: {expected_relative}")
    relative = PurePosixPath(expected_relative)
    _validate_safe_path(relative)
    try:
        repo_relative = relative.relative_to("repo")
    except ValueError:
        # Raster records are rooted at the staged repository's data directory;
        # label records retain the explicit view-root `repo/` prefix.
        repo_relative = relative
    path = _require_no_symlink_components(
        repo.joinpath(*repo_relative.parts), leaf="file"
    )
    size = record.get("bytes")
    digest = record.get("sha256")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or _SHA64.fullmatch(str(digest)) is None
        or path.stat().st_size != size
        or _hash_file(path) != digest
        or stat.S_IMODE(path.stat().st_mode) != 0o444
    ):
        raise PackageError(f"staged-file byte identity mismatch: {expected_relative}")


def validate_staged_view(
    receipt_path: Path,
    *,
    expected_repo: Path,
    expected_package_id: str,
    expected_evaluator_git_sha: str,
    expected_campaign_git_sha: str,
    expected_splits_sha256: str,
    production: bool = True,
) -> tuple[dict[str, object], str]:
    """Validate a durable staged-view receipt without parsing final labels.

    Returns the normalized receipt and SHA-256 of its exact canonical bytes.
    Validation hashes the staged validation-label file as opaque bytes and
    hashes every staged inference raster, but it never decodes CSV or raster content.
    """

    payload, receipt_sha256 = _receipt_json(receipt_path)
    if set(payload) != {"schema", "status", "package", "source", "view"} or (
        payload.get("schema") != 1
        or payload.get("status") != "final-eval-data-view-staged"
    ):
        raise PackageError("final data-view receipt schema/status is invalid")
    package = payload.get("package")
    source = payload.get("source")
    view = payload.get("view")
    if (
        not isinstance(package, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(view, Mapping)
    ):
        raise PackageError("final data-view receipt records are invalid")
    if set(package) != {
        "package_id",
        "identity_sha256",
        "ready_sha256",
        "manifest_sha256",
        "sha256sums_sha256",
    } or (
        package.get("package_id") != expected_package_id
        or not expected_package_id.startswith(
            f"xview3-h100-final-eval-{expected_evaluator_git_sha}-"
        )
        or any(
            _SHA64.fullmatch(str(package.get(key, ""))) is None
            for key in (
                "identity_sha256",
                "ready_sha256",
                "manifest_sha256",
                "sha256sums_sha256",
            )
        )
    ):
        raise PackageError("final data-view package identity is invalid")
    if set(source) != {
        "branch",
        "git_bundle_ref",
        "git_commit",
        "required_campaign_commit",
        "git_bundle_sha256",
        "splits_sha256",
    } or (
        source.get("git_commit") != expected_evaluator_git_sha
        or source.get("required_campaign_commit") != expected_campaign_git_sha
        or source.get("splits_sha256") != expected_splits_sha256
        or source.get("git_bundle_ref")
        != f"refs/heads/{source.get('branch')}"
        or (production and source.get("branch") != FINAL_BRANCH)
        or _SHA64.fullmatch(str(source.get("git_bundle_sha256", ""))) is None
    ):
        raise PackageError("final data-view source identity is invalid")
    repo = _require_no_symlink_components(expected_repo, leaf="directory")
    if _git_value(repo, "rev-parse", "HEAD") != expected_evaluator_git_sha:
        raise PackageError("staged repo HEAD differs from evaluator binding")
    if _git_value(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise PackageError("staged final repo is not Git-clean")
    expected_view_fields = {
        "root",
        "repo",
        "bundle",
        "training_labels",
        "validation_labels",
        "scenes",
    }
    if set(view) != expected_view_fields or view.get("repo") != str(repo):
        raise PackageError("final data-view path binding is invalid")
    root = _require_no_symlink_components(Path(str(view.get("root"))), leaf="directory")
    if stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise PackageError("final data-view root must have exact mode 0555")
    if repo != root / STAGED_REPO_PATH:
        raise PackageError("final data-view repo is outside its exact staged root")
    bundle = _require_no_symlink_components(
        Path(str(view.get("bundle"))), leaf="file"
    )
    if bundle != root / STAGED_BUNDLE_PATH:
        raise PackageError("final data-view bundle path is invalid")
    training = view.get("training_labels")
    validation = view.get("validation_labels")
    scenes = view.get("scenes")
    if (
        not isinstance(training, Mapping)
        or not isinstance(validation, Mapping)
        or validation.get("access") != "opaque-bytes-staged-not-semantically-read"
        or not isinstance(scenes, list)
        or (production and len(scenes) != EXPECTED_FINAL_SCENES)
    ):
        raise PackageError("final data-view labels/scenes schema is invalid")
    _receipt_file(repo, training, "repo/data/raw/xview3/labels/train.csv")
    _receipt_file(
        repo,
        validation,
        "repo/data/raw/xview3/labels/validation.csv",
        expected_access="opaque-bytes-staged-not-semantically-read",
    )
    split_inventory = _split_inventory(repo, production=production)
    if split_inventory["splits_sha256"] != expected_splits_sha256:
        raise PackageError("staged split bytes differ from receipt binding")
    expected_scenes = tuple(split_inventory["final_scenes"])
    observed_scenes: list[str] = []
    for index, raw_scene in enumerate(scenes):
        if not isinstance(raw_scene, Mapping) or set(raw_scene) != {
            "scene_id",
            "rasters",
            "source_archive",
        }:
            raise PackageError("final data-view scene receipt schema is invalid")
        scene = str(raw_scene.get("scene_id", ""))
        if scene != expected_scenes[index]:
            raise PackageError("final data-view scene ordering/inventory is invalid")
        observed_scenes.append(scene)
        archive = raw_scene.get("source_archive")
        rasters = raw_scene.get("rasters")
        if (
            not isinstance(archive, Mapping)
            or set(archive) != {"path", "bytes", "sha256"}
            or archive.get("path") != f"{SCENE_ARCHIVE_ROOT}/{scene}.tar.gz"
            or not isinstance(archive.get("bytes"), int)
            or isinstance(archive.get("bytes"), bool)
            or int(archive["bytes"]) <= 0
            or _SHA64.fullmatch(str(archive.get("sha256", ""))) is None
            or not isinstance(rasters, list)
            or len(rasters) != len(SCENE_RASTER_NAMES)
        ):
            raise PackageError(f"final data-view scene binding is invalid: {scene}")
        for raster, name in zip(rasters, SCENE_RASTER_NAMES, strict=True):
            if not isinstance(raster, Mapping):
                raise PackageError(f"invalid raster receipt for {scene}")
            _receipt_file(
                repo,
                raster,
                f"data/raw/xview3/GRD/{scene}/{name}",
            )
    if tuple(observed_scenes) != expected_scenes:
        raise PackageError("final data-view scene inventory is incomplete")
    return json.loads(_canonical_json(payload)), receipt_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build the committed final-input package")
    build.add_argument("--repo", type=Path, required=True)
    build.add_argument("--training-labels", type=Path, required=True)
    build.add_argument("--validation-labels", type=Path, required=True)
    build.add_argument("--archive-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--max-part-bytes", type=int, required=True)

    verify = commands.add_parser("verify", help="fully verify a production package")
    verify.add_argument("--package-root", type=Path, required=True)

    stage = commands.add_parser("stage", help="publish a sealed final data view")
    stage.add_argument("--package-root", type=Path, required=True)
    stage.add_argument("--destination", type=Path, required=True)
    stage.add_argument("--receipt", type=Path, required=True)
    stage.add_argument("--expected-package-id", required=True)
    stage.add_argument("--expected-ready-sha256", required=True)
    stage.add_argument("--expected-manifest-sha256", required=True)
    stage.add_argument("--expected-sha256sums-sha256", required=True)

    stage_training = commands.add_parser(
        "stage-training-labels",
        help="publish only verified TRAIN+DEV8 labels for owner authorization",
    )
    stage_training.add_argument("--package-root", type=Path, required=True)
    stage_training.add_argument("--output", type=Path, required=True)
    stage_training.add_argument("--expected-package-id", required=True)
    stage_training.add_argument("--expected-ready-sha256", required=True)
    stage_training.add_argument("--expected-manifest-sha256", required=True)
    stage_training.add_argument("--expected-sha256sums-sha256", required=True)

    upload = commands.add_parser("upload", help="verified Box upload, READY last")
    upload.add_argument("--repo", type=Path, required=True)
    upload.add_argument("--package-root", type=Path, required=True)
    upload.add_argument("--receipt", type=Path, required=True)

    download = commands.add_parser("download", help="hash-pinned Box download")
    download.add_argument("--repo", type=Path, required=True)
    download.add_argument("--package-root", type=Path, required=True)
    download.add_argument("--expected-ready-sha256", required=True)
    download.add_argument("--expected-manifest-sha256", required=True)
    download.add_argument("--expected-sha256sums-sha256", required=True)
    download.add_argument("--expected-package-id", required=True)

    bootstrap = commands.add_parser(
        "build-bootstrap",
        help="generate a standalone hash-pinned Judy package puller",
    )
    bootstrap.add_argument("--repo", type=Path, required=True)
    bootstrap.add_argument("--package-root", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    return parser


def _print(value: Mapping[str, object]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            package = build_final_eval_package(
                repo_root=args.repo,
                training_labels_path=args.training_labels,
                validation_labels_path=args.validation_labels,
                archive_dir=args.archive_dir,
                output_dir=args.output_dir,
                maximum_physical_file_bytes=args.max_part_bytes,
            )
            manifest = _load_json(package / "manifest.json")
            _print(
                {
                    "status": "built",
                    "package_root": str(package),
                    "package_id": manifest["package_id"],
                    "git_commit": manifest["source"]["git_commit"],  # type: ignore[index]
                    "ready_sha256": _hash_file(package / "READY.json"),
                    "manifest_sha256": _hash_file(package / "manifest.json"),
                    "sha256sums_sha256": _hash_file(package / "SHA256SUMS"),
                }
            )
        elif args.command == "verify":
            manifest = verify_final_eval_package(args.package_root)
            _print(
                {
                    "status": "verified",
                    "package_id": manifest["package_id"],
                    "package_type": manifest["package_type"],
                    "identity_sha256": manifest["identity_sha256"],
                }
            )
        elif args.command == "stage":
            output = stage_final_eval_package(
                args.package_root,
                args.destination,
                args.receipt,
                expected_package_id=args.expected_package_id,
                expected_ready_sha256=args.expected_ready_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_sha256sums_sha256=args.expected_sha256sums_sha256,
            )
            _print(
                {
                    "status": "staged",
                    "destination": str(output),
                    "repo": str(output / STAGED_REPO_PATH),
                    "receipt": str(_absolute_path(args.receipt)),
                }
            )
        elif args.command == "stage-training-labels":
            _print(
                stage_training_labels(
                    args.package_root,
                    args.output,
                    expected_package_id=args.expected_package_id,
                    expected_ready_sha256=args.expected_ready_sha256,
                    expected_manifest_sha256=args.expected_manifest_sha256,
                    expected_sha256sums_sha256=args.expected_sha256sums_sha256,
                )
            )
        elif args.command == "upload":
            from .box import (
                client_from_environment,
                upload_package_with_verifier,
            )

            repo = args.repo.resolve()
            client, folder_id = client_from_environment(repo)
            result = upload_package_with_verifier(
                client,
                folder_id,
                args.package_root,
                repo_root=repo,
                receipt_path=args.receipt,
                verifier=verify_final_eval_package,
                minimum_free_bytes=0,
            )
            _print(result)
        elif args.command == "download":
            from .box import (
                client_from_environment,
                download_package_with_verifier,
            )

            repo = args.repo.resolve()
            client, folder_id = client_from_environment(repo)
            output = download_package_with_verifier(
                client,
                folder_id,
                args.package_root,
                repo_root=repo,
                expected_ready_sha256=args.expected_ready_sha256,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_sha256sums_sha256=args.expected_sha256sums_sha256,
                expected_package_id=args.expected_package_id,
                verifier=verify_final_eval_package,
            )
            _print({"status": "downloaded", "package_root": str(output)})
        elif args.command == "build-bootstrap":
            from .final_eval_bootstrap import generate_final_eval_bootstrap

            _print(
                generate_final_eval_bootstrap(
                    repo_root=args.repo,
                    package_root=args.package_root,
                    output=args.output,
                )
            )
        else:  # pragma: no cover - argparse makes this unreachable
            raise AssertionError(args.command)
    except (PackageError, RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
