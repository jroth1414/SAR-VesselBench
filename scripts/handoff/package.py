"""Deterministic, allowlist-only builder for the H100 core handoff.

The production builder is deliberately strict: the git worktree must be clean,
the frozen split must describe exactly 150 non-eval scenes, only dev/test raw
rasters are included, and all six core checkpoint directories must carry their
source and license notes.  Archives are created with normalized tar metadata
and single-threaded Zstandard level 3.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from scripts.h100.wheelhouse import wheelhouse_identity

FORMAT_VERSION = 1
# The immutable 294 GB payload remains a Sprint 7d package.  Sprint 7e is a
# separate code-only amendment and must never change this verifier's identity.
EXPECTED_BRANCH = "sprint-7d-h100-fp32"
SOURCE_BASE_COMMIT = "48e10534a8c7baf0662acd548f52928da69f23c8"
RUNTIME_BRANCH = "sprint-7e-judy-venv"
RUNTIME_REQUIRED_ANCESTOR = "2726199efcebbebc89156e708b89df2a3415468a"
ENVIRONMENT_LOCK_PATH = "locks/env-v100node.txt"
COMMITTED_APPTAINER_DEFINITION = "containers/h100-strict-fp32.def"
EXPECTED_PYTHON = "3.11.15"
EXPECTED_TORCH = "2.11.0+cu126"
EXPECTED_CHIP_ARCHIVES = 150
EXPECTED_RASTER_ARCHIVES = 39
ZSTD_LEVEL = 3
SPLIT_COPY_BLOCK_BYTES = 8 * 1024 * 1024
EXPECTED_CORE_RESULT_IDS = frozenset(
    f"{short}-f{fraction}-s0"
    for short in (
        "vitrand",
        "satdino",
        "sarmae",
        "vitin1k",
        "cnnrand",
        "beS2",
        "beS1",
        "cnnin1k",
    )
    for fraction in (10, 25, 50, 100)
)

REQUIRED_WEIGHT_DIRS: Mapping[str, Mapping[str, str]] = {
    "satdino": {
        "model": "strakajk/satdino-vit_base-16",
        "revision": "22b7a253cbd6f44d9d435c00be31bfdbb8cedb4f",
        "license": "Apache-2.0",
    },
    "sarmae": {
        "model": "Wenquandan777/SARMAE",
        "revision": "88a8d768665e6070464f239dabb9650795ba5f32",
        "license": "CC BY-NC 4.0",
    },
    "imagenet_vit_augreg_in1k": {
        "model": "timm/vit_base_patch16_224.augreg_in1k",
        "revision": "458542882691a06a8b667c6fb5fe5c9573093a81",
        "license": "Apache-2.0",
    },
    "bigearthnet_s2": {
        "model": "BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0",
        "revision": "1afe6c91a20184d207caf8a605ac39c7e7df03ab",
        "license": "MIT",
    },
    "bigearthnet_s1": {
        "model": "BIFOLD-BigEarthNetv2-0/convnextv2_base-s1-v0.2.0",
        "revision": "a0b43b44a090cb40b61ae77933b3a1a442e2bd04",
        "license": "MIT",
    },
    "imagenet_cnn_fcmae_ft_in1k": {
        "model": "timm/convnextv2_base.fcmae_ft_in1k",
        "revision": "7b29800e499fdc06de5b612970f3384dc8d29ca5",
        "license": "CC BY-NC 4.0",
    },
}

REQUIRED_CHECKPOINTS: Mapping[str, tuple[str, str]] = {
    "satdino": (
        "satdino-vit_base-16.pth",
        "74d1a3f88b4f554d5a74607d80b8ffbeeabff0bd009ac6b288fcdb40128ddd44",
    ),
    "sarmae": (
        "SARMAE_vitb_checkpoint-last",
        "ead8d93aaebeebbeeef43805495071738d9d63c05d7138015fdb1d9957308f6f",
    ),
    "bigearthnet_s1": (
        "model.safetensors",
        "39025759eacef4a6e668b1123ae4ae67fc982fed06f74abdc11de3c5179b9dad",
    ),
    "bigearthnet_s2": (
        "model.safetensors",
        "b09d0e41cc683878243a9128a6f4724d6a71d562318beeae716f0dce9cbbf454",
    ),
    "imagenet_vit_augreg_in1k": (
        "model.safetensors",
        "678a1ce471be7da9822fe2508497a5bcf6da4c6802053151b232ba88a42c21a2",
    ),
    "imagenet_cnn_fcmae_ft_in1k": (
        "model.safetensors",
        "ec152f1e375edc2b3dfac7a81155a449b4c5cbb7c5cf0b9494838f6c87518d73",
    ),
}

REQUIRED_WEIGHT_MEMBERS: Mapping[str, frozenset[str]] = {
    name: frozenset({"SOURCE.note", "LICENSE.note", checkpoint})
    for name, (checkpoint, _digest) in REQUIRED_CHECKPOINTS.items()
}

_DENIED_SEGMENTS = {
    ".cache",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    "__pycache__",
    "runs",
    "eval_final",
    "lsssdd",
    "raw_lsssdd",
    "locateanything",
    "yolo26",
    # The exact superseded directory.  The accepted replacement has a longer
    # exact segment and is therefore not rejected.
    "imagenet_cnn",
}
_SECRET_NAME = re.compile(
    r"(^|[._-])(jwt|credentials?|secrets?|private[_-]?key|access[_-]?token)"
    r"([._-]|$)",
    re.IGNORECASE,
)
_SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}
_SECRET_CONTENT_MARKERS = (
    b'"boxAppSettings"',
    b'"clientSecret"',
    b"-----BEGIN PRIVATE KEY-----",
)
_TEXT_SCAN_SUFFIXES = {
    ".cfg",
    ".conf",
    ".csv",
    ".env",
    ".ini",
    ".json",
    ".log",
    ".md",
    ".note",
    ".out",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_CONFIG_SCAN_SUFFIXES = {".cfg", ".conf", ".env", ".ini", ".json", ".toml", ".yaml", ".yml"}
_CREDENTIAL_CONTENT_SCAN_SUFFIXES = _CONFIG_SCAN_SUFFIXES | {".log", ".out", ".txt"}
_ENV_SECRET_SCAN_SUFFIXES = _CONFIG_SCAN_SUFFIXES | {".log", ".out", ".sh", ".txt"}
_ENV_SECRET_MARKERS = tuple(
    marker.encode("ascii")
    for marker in ("BOX_" + "JWT_CONFIG=", "BOX_" + "FOLDER_ID=")
)


class PackageError(RuntimeError):
    """The requested package violates a handoff invariant."""


@dataclass(frozen=True)
class BuildOptions:
    repo_root: Path
    data_root: Path
    package_root: Path
    wheelhouse: Path
    apptainer_definition: Path
    max_part_bytes: int
    branch: str = EXPECTED_BRANCH
    production: bool = True


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)


def _hash_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_paths(paths: Sequence[Path], algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    capture: bool = True,
) -> str:
    try:
        result = subprocess.run(
            list(argv),
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        raise PackageError(f"command failed: {argv[0]}: {detail or exc}") from exc
    return (result.stdout or "").strip()


def _repository_worktree_roots(repo_root: Path) -> tuple[Path, ...]:
    """Resolve every checkout sharing this repository's common Git directory."""

    repository = repo_root.resolve()
    output = _run(
        [
            "git",
            "-c",
            f"safe.directory={repository}",
            "worktree",
            "list",
            "--porcelain",
        ],
        cwd=repository,
    )
    roots = tuple(
        Path(line.removeprefix("worktree ")).resolve()
        for line in output.splitlines()
        if line.startswith("worktree ")
    )
    if not roots or repository not in roots:
        raise PackageError("cannot resolve all linked repository worktrees")
    return roots


def _inside_repository_worktrees(path: Path, repo_root: Path) -> bool:
    return any(_inside(path, root) for root in _repository_worktree_roots(repo_root))


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _absolute_path(path: Path) -> Path:
    """Return an absolute path without erasing a symlink component."""

    return Path(os.path.abspath(os.fspath(path)))


def _require_no_symlink_components(
    path: Path,
    *,
    leaf: str | None = None,
) -> Path:
    """Validate every existing component with lstat, including the leaf."""

    absolute = _absolute_path(path)
    chain = list(reversed(absolute.parents)) + [absolute]
    for index, component in enumerate(chain):
        try:
            info = component.lstat()
        except OSError as exc:
            raise PackageError(f"required path is absent: {component}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise PackageError(f"symlink path component is forbidden: {component}")
        is_leaf = index == len(chain) - 1
        if not is_leaf and not stat.S_ISDIR(info.st_mode):
            raise PackageError(f"path component is not a directory: {component}")
        if is_leaf and leaf == "directory" and not stat.S_ISDIR(info.st_mode):
            raise PackageError(f"required path is not a directory: {component}")
        if is_leaf and leaf == "file" and not stat.S_ISREG(info.st_mode):
            raise PackageError(f"required path is not a regular file: {component}")
    return absolute


def _validate_safe_path(relative: PurePosixPath) -> None:
    if relative.is_absolute() or not relative.parts:
        raise PackageError(f"unsafe package path: {relative}")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise PackageError(f"unsafe package path: {relative}")
    lowered = [part.lower() for part in relative.parts]
    if any(part in _DENIED_SEGMENTS for part in lowered):
        raise PackageError(f"denied payload path: {relative}")
    if any(part == "venv" or part.startswith(".venv") for part in lowered):
        raise PackageError(f"denied virtual-environment path: {relative}")
    basename = relative.name
    if _SECRET_NAME.search(basename) or Path(basename).suffix.lower() in _SECRET_SUFFIXES:
        raise PackageError(f"credential-like payload path: {relative}")


class _PrivateKeyStreamScanner:
    """Bounded-memory detector for complete PEM private-key blocks."""

    _BEGIN = (
        (b"-----BEGIN PRIVATE KEY-----", b"-----END PRIVATE KEY-----"),
        (
            b"-----BEGIN ENCRYPTED PRIVATE KEY-----",
            b"-----END ENCRYPTED PRIVATE KEY-----",
        ),
        (b"-----BEGIN RSA PRIVATE KEY-----", b"-----END RSA PRIVATE KEY-----"),
        (b"-----BEGIN EC PRIVATE KEY-----", b"-----END EC PRIVATE KEY-----"),
        (b"-----BEGIN DSA PRIVATE KEY-----", b"-----END DSA PRIVATE KEY-----"),
        (b"-----BEGIN OPENSSH PRIVATE KEY-----", b"-----END OPENSSH PRIVATE KEY-----"),
    )
    _BODY_BYTES = frozenset(
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    )
    _WHITESPACE = frozenset(b" \t\r\n")

    def __init__(self) -> None:
        self.search_tail = b""
        self.end_token: bytes | None = None
        self.end_tail = b""
        self.body_valid = True
        self.body_chars = 0

    def _consume_body(self, body: bytes) -> None:
        for value in body:
            if value in self._BODY_BYTES:
                self.body_chars += 1
            elif value not in self._WHITESPACE:
                self.body_valid = False

    def feed(self, chunk: bytes) -> bool:
        data = chunk
        while data:
            if self.end_token is None:
                candidate = self.search_tail + data
                matches = [
                    (candidate.find(begin), begin, end)
                    for begin, end in self._BEGIN
                    if candidate.find(begin) >= 0
                ]
                if not matches:
                    retain = max(len(begin) for begin, _end in self._BEGIN) - 1
                    self.search_tail = candidate[-retain:]
                    return False
                index, begin, end = min(matches, key=lambda item: item[0])
                self.search_tail = b""
                self.end_token = end
                self.end_tail = b""
                self.body_valid = True
                self.body_chars = 0
                data = candidate[index + len(begin):]
                continue

            candidate = self.end_tail + data
            end_index = candidate.find(self.end_token)
            if end_index >= 0:
                self._consume_body(candidate[:end_index])
                matched = self.body_valid and self.body_chars >= 64
                data = candidate[end_index + len(self.end_token):]
                self.end_token = None
                self.end_tail = b""
                self.body_valid = True
                self.body_chars = 0
                if matched:
                    return True
                continue
            retain = len(self.end_token) - 1
            body = candidate[:-retain] if retain else candidate
            self._consume_body(body)
            self.end_tail = candidate[-retain:] if retain else b""
            return False
        return False


def _scan_secret_stream(
    handle: object,
    *,
    suffixes: set[str],
    description: str,
) -> None:
    if not suffixes & _TEXT_SCAN_SUFFIXES:
        return
    credential_content_scan = bool(suffixes & _CREDENTIAL_CONTENT_SCAN_SUFFIXES)
    environment_scan = bool(suffixes & _ENV_SECRET_SCAN_SUFFIXES)
    box_settings = False
    client_secret = False
    overlap = max(
        len(marker)
        for marker in (*_SECRET_CONTENT_MARKERS, *_ENV_SECRET_MARKERS)
    ) - 1
    tail = b""
    pem = _PrivateKeyStreamScanner()
    while True:
        chunk = handle.read(1024 * 1024)  # type: ignore[attr-defined]
        if not chunk:
            break
        window = tail + chunk
        lowered = window.lower()
        if credential_content_scan:
            box_settings = box_settings or _SECRET_CONTENT_MARKERS[0].lower() in lowered
            client_secret = client_secret or _SECRET_CONTENT_MARKERS[1].lower() in lowered
        environment_secret = environment_scan and any(
            marker.lower() in lowered for marker in _ENV_SECRET_MARKERS
        )
        if environment_secret or pem.feed(chunk):
            raise PackageError(f"credential material detected in {description}")
        tail = window[-overlap:]
    if box_settings and client_secret:
        raise PackageError(f"credential material detected in {description}")


def _scan_secret_content(path: Path) -> None:
    name = path.name.lower()
    suffix = ".env" if name == ".env" or name.startswith(".env.") else path.suffix.lower()
    if suffix not in _TEXT_SCAN_SUFFIXES:
        return
    with path.open("rb") as handle:
        _scan_secret_stream(
            handle,
            suffixes={suffix},
            description=str(path),
        )


def _source_entries(
    source: Path,
    archive_root: PurePosixPath,
    *,
    excluded_segments: frozenset[str] = frozenset(),
) -> list[tuple[Path, PurePosixPath]]:
    """Return deterministic source/archive pairs, rejecting special files."""

    original = source
    try:
        original_info = original.lstat()
    except OSError as exc:
        raise PackageError(f"required source is absent: {original}") from exc
    if stat.S_ISLNK(original_info.st_mode):
        raise PackageError(f"symlinks are not allowed in payloads: {original}")
    if stat.S_ISREG(original_info.st_mode):
        leaf = "file"
    elif stat.S_ISDIR(original_info.st_mode):
        leaf = "directory"
    else:
        raise PackageError(f"special file is not allowed in payloads: {original}")
    source = _require_no_symlink_components(original, leaf=leaf)

    result: list[tuple[Path, PurePosixPath]] = []

    def add(path: Path, arcname: PurePosixPath) -> None:
        _validate_safe_path(arcname)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise PackageError(f"symlinks are not allowed in payloads: {path}")
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise PackageError(f"special file is not allowed in payloads: {path}")
        if stat.S_ISREG(info.st_mode):
            _scan_secret_content(path)
        result.append((path, arcname))

    if source.is_file():
        add(source, archive_root)
        return result

    add(source, archive_root)
    for path in sorted(source.rglob("*"), key=lambda item: item.relative_to(source).as_posix()):
        relative = path.relative_to(source)
        if any(part.lower() in excluded_segments for part in relative.parts):
            continue
        add(path, archive_root / PurePosixPath(relative.as_posix()))
    return result


def _normalized_tarinfo(path: Path, arcname: PurePosixPath) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname.as_posix())
    source = path.stat()
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    info.pax_headers = {}
    if path.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
        info.size = 0
    else:
        info.type = tarfile.REGTYPE
        info.mode = 0o755 if source.st_mode & stat.S_IXUSR else 0o644
        info.size = source.st_size
    return info


def _create_tar_zst(
    output: Path,
    entries: Sequence[tuple[Path, PurePosixPath]],
) -> tuple[int, int]:
    if not entries:
        raise PackageError(f"refusing to create an empty archive: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as compressed:
        process = subprocess.Popen(
            [
                "zstd",
                f"-{ZSTD_LEVEL}",
                "--threads=1",
                "--no-progress",
                "--quiet",
                "--stdout",
            ],
            stdin=subprocess.PIPE,
            stdout=compressed,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        try:
            with tarfile.open(
                fileobj=process.stdin,
                mode="w|",
                format=tarfile.GNU_FORMAT,
            ) as archive:
                for source, arcname in entries:
                    info = _normalized_tarinfo(source, arcname)
                    if source.is_file():
                        with source.open("rb") as content:
                            archive.addfile(info, content)
                    else:
                        archive.addfile(info)
            process.stdin.close()
            stderr = process.stderr.read() if process.stderr is not None else b""
            return_code = process.wait()
        except BaseException:
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()
            process.kill()
            process.wait()
            with contextlib.suppress(FileNotFoundError):
                output.unlink()
            raise
    if return_code:
        with contextlib.suppress(FileNotFoundError):
            output.unlink()
        raise PackageError(f"zstd failed for {output}: {stderr.decode(errors='replace')}")
    file_entries = [path for path, _ in entries if path.is_file()]
    return len(file_entries), sum(path.stat().st_size for path in file_entries)


def _split_archive(path: Path, max_part_bytes: int) -> tuple[list[Path], int, str, str]:
    if max_part_bytes <= 0:
        raise PackageError("Box maximum file size must be a positive integer")
    size = path.stat().st_size
    sha256 = _hash_file(path, "sha256")
    sha1 = _hash_file(path, "sha1")
    if size <= max_part_bytes:
        return [path], size, sha256, sha1

    parts: list[Path] = []
    try:
        with path.open("rb") as source:
            remaining_total = size
            index = 0
            while remaining_total:
                part = path.with_name(f"{path.name}.part-{index:05d}")
                remaining_part = min(max_part_bytes, remaining_total)
                with part.open("xb") as destination:
                    while remaining_part:
                        chunk = source.read(
                            min(SPLIT_COPY_BLOCK_BYTES, remaining_part)
                        )
                        if not chunk:
                            raise PackageError(
                                f"archive changed or truncated while splitting: {path}"
                            )
                        destination.write(chunk)
                        remaining_part -= len(chunk)
                        remaining_total -= len(chunk)
                parts.append(part)
                index += 1
            if source.read(1):
                raise PackageError(f"archive grew while splitting: {path}")
    except BaseException:
        for part in parts:
            part.unlink(missing_ok=True)
        # The current part may have been opened but not appended yet.
        path.parent.joinpath(f"{path.name}.part-{len(parts):05d}").unlink(
            missing_ok=True
        )
        raise
    path.unlink()
    if not parts or any(part.stat().st_size > max_part_bytes for part in parts):
        raise PackageError(f"deterministic split failed for {path}")
    return parts, size, sha256, sha1


def _physical_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _hash_file(path, "sha256"),
        "sha1": _hash_file(path, "sha1"),
    }


def _archive_artifact(
    *,
    staging: Path,
    output_relative: PurePosixPath,
    entries: Sequence[tuple[Path, PurePosixPath]],
    kind: str,
    name: str,
    extraction_root: PurePosixPath,
    max_part_bytes: int,
) -> dict[str, object]:
    output = staging / output_relative
    file_count, unpacked_bytes = _create_tar_zst(output, entries)
    parts, archive_bytes, archive_sha256, archive_sha1 = _split_archive(
        output, max_part_bytes
    )
    return {
        "kind": kind,
        "name": name,
        "format": "tar.zst",
        "compression": {"algorithm": "zstd", "level": ZSTD_LEVEL, "threads": 1},
        "extraction_root": extraction_root.as_posix(),
        "file_count": file_count,
        "unpacked_bytes": unpacked_bytes,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "archive_sha1": archive_sha1,
        "parts": [_physical_record(part, staging) for part in parts],
    }


def _plain_artifact(
    path: Path,
    staging: Path,
    *,
    kind: str,
    name: str,
    extraction_root: PurePosixPath,
    max_part_bytes: int,
) -> dict[str, object]:
    parts, archive_bytes, archive_sha256, archive_sha1 = _split_archive(
        path, max_part_bytes
    )
    return {
        "kind": kind,
        "name": name,
        "format": "file",
        "extraction_root": extraction_root.as_posix(),
        "file_count": 1,
        "unpacked_bytes": archive_bytes,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha256,
        "archive_sha1": archive_sha1,
        "parts": [_physical_record(part, staging) for part in parts],
    }


def _git_value(repo_root: Path, *arguments: str) -> str:
    return _run(["git", "-c", f"safe.directory={repo_root}", *arguments], cwd=repo_root)


def _history_path(relative: PurePosixPath) -> None:
    # These two small, committed provenance artifacts are deliberately not
    # the bulk payloads denoted by their path tokens.
    exceptions = {"runs/decisions.md", "data/lsssdd_split.json"}
    if relative.as_posix() in exceptions:
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise PackageError(f"unsafe historical path: {relative}")
        return
    _validate_safe_path(relative)


def _reachable_history_blobs(
    repo_root: Path,
    branch: str,
) -> dict[str, set[str]]:
    blobs: dict[str, set[str]] = {}
    commits = _git_value(repo_root, "rev-list", branch).splitlines()
    for commit in commits:
        try:
            result = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={repo_root}",
                    "ls-tree",
                    "-r",
                    "-z",
                    "--full-tree",
                    commit,
                ],
                cwd=repo_root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PackageError("cannot enumerate reachable Git history") from exc
        for record in result.stdout.split(b"\0"):
            if not record:
                continue
            try:
                metadata, encoded_path = record.split(b"\t", 1)
                _mode, object_type, encoded_sha = metadata.split(b" ", 2)
                path = encoded_path.decode("utf-8", errors="strict")
                sha = encoded_sha.decode("ascii", errors="strict")
            except (ValueError, UnicodeError) as exc:
                raise PackageError("Git history contains an undecodable tree entry") from exc
            if object_type != b"blob":
                continue
            relative = PurePosixPath(path)
            _history_path(relative)
            blobs.setdefault(sha, set()).add(path)
    return blobs


def _scan_reachable_history(repo_root: Path, branch: str) -> None:
    """Scan every unique reachable blob and every historical pathname."""

    for sha, paths in sorted(_reachable_history_blobs(repo_root, branch).items()):
        for path in paths:
            _validate_safe_path(PurePosixPath(path))
        suffixes = {
            ".env"
            if Path(path).name.lower() == ".env"
            or Path(path).name.lower().startswith(".env.")
            else Path(path).suffix.lower()
            for path in paths
        }
        if not suffixes & _TEXT_SCAN_SUFFIXES:
            continue
        process = subprocess.Popen(
            [
                "git",
                "-c",
                f"safe.directory={repo_root}",
                "cat-file",
                "blob",
                sha,
            ],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        try:
            _scan_secret_stream(
                process.stdout,
                suffixes=suffixes,
                description=f"reachable Git blob {sha} ({sorted(paths)})",
            )
        except BaseException:
            process.kill()
            process.wait()
            raise
        stderr = process.stderr.read() if process.stderr is not None else b""
        if process.wait():
            raise PackageError(
                "cannot read reachable Git blob "
                f"{sha}: {stderr.decode(errors='replace').strip()}"
            )


def _git_source(
    repo_root: Path,
    expected_branch: str,
    *,
    production: bool,
) -> tuple[str, int]:
    if not (repo_root / ".git").exists():
        # Git worktrees have a .git text file, so exists() is intentional.
        raise PackageError(f"not a git worktree: {repo_root}")
    branch = _git_value(repo_root, "branch", "--show-current")
    if branch != expected_branch:
        raise PackageError(f"expected branch {expected_branch!r}, found {branch!r}")
    status = _git_value(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise PackageError("git worktree must be clean before creating the exact bundle")
    commit = _git_value(repo_root, "rev-parse", "HEAD")
    commit_epoch = int(_git_value(repo_root, "show", "-s", "--format=%ct", commit))

    for tracked in _git_value(repo_root, "ls-files").splitlines():
        relative = PurePosixPath(tracked)
        _validate_safe_path(relative)
        source = repo_root / tracked
        if source.is_file():
            _scan_secret_content(source)
    if production:
        _scan_reachable_history(repo_root, expected_branch)
    return commit, commit_epoch


def _require_source_ancestor(
    repo_root: Path,
    head: str,
    required_base: str = SOURCE_BASE_COMMIT,
) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", required_base):
        raise PackageError("required source base is not a full commit SHA")
    try:
        _git_value(repo_root, "cat-file", "-e", f"{required_base}^{{commit}}")
        _git_value(
            repo_root,
            "merge-base",
            "--is-ancestor",
            required_base,
            head,
        )
    except PackageError as exc:
        raise PackageError(
            f"source commit {head} is not descended from required base {required_base}"
        ) from exc


def _create_git_bundle(
    repo_root: Path,
    output: Path,
    branch: str,
    expected_commit: str,
    required_ancestor: str | None,
    *,
    production: bool,
) -> None:
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
    _verify_git_bundle(
        output,
        branch,
        expected_commit,
        required_ancestor=required_ancestor,
        production=production,
    )


def _verify_git_bundle(
    bundle: Path,
    branch: str,
    expected_commit: str,
    *,
    required_ancestor: str | None = None,
    production: bool = False,
) -> dict[str, object]:
    heads = _run(
        ["git", "bundle", "list-heads", str(bundle), f"refs/heads/{branch}"]
    ).splitlines()
    expected = f"{expected_commit} refs/heads/{branch}"
    if heads != [expected]:
        raise PackageError(f"git bundle branch/commit mismatch: {heads!r}")
    with tempfile.TemporaryDirectory(prefix="xview3-bundle-verify-") as temporary:
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
        actual = _git_value(checkout, "rev-parse", "HEAD")
        dirty = _git_value(checkout, "status", "--porcelain=v1")
        if actual != expected_commit or dirty:
            raise PackageError("git bundle did not reproduce the exact clean commit")
        if required_ancestor is not None:
            _require_source_ancestor(checkout, actual, required_ancestor)
        chips, rasters, eval_final = _load_split(checkout, production=production)
        lock_path = _require_no_symlink_components(
            checkout / ENVIRONMENT_LOCK_PATH,
            leaf="file",
        )
        definition = _require_no_symlink_components(
            checkout / COMMITTED_APPTAINER_DEFINITION,
            leaf="file",
        )
        _validate_apptainer_definition(definition)
        return {
            "chips": chips,
            "rasters": rasters,
            "eval_final": eval_final,
            "environment_lock_sha256": _hash_file(lock_path),
            "apptainer_definition_sha256": _hash_file(definition),
        }


def _load_split(
    repo_root: Path, *, production: bool
) -> tuple[list[str], list[str], list[str]]:
    split_path = repo_root / "data" / "splits.json"
    try:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        splits = payload["splits"]
        chip_ids = list(splits["train"]) + list(splits["dev"]) + list(splits["test"])
        raster_ids = list(splits["dev"]) + list(splits["test"])
        eval_ids = set(splits.get("eval_final", []))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PackageError(f"invalid split file: {split_path}") from exc
    if len(chip_ids) != len(set(chip_ids)):
        raise PackageError("duplicate scene ID in train/dev/test split")
    if set(chip_ids) & eval_ids:
        raise PackageError("eval_final scene leaked into the core split")
    if production and len(chip_ids) != EXPECTED_CHIP_ARCHIVES:
        raise PackageError(
            f"expected {EXPECTED_CHIP_ARCHIVES} chip scenes, found {len(chip_ids)}"
        )
    if production and len(raster_ids) != EXPECTED_RASTER_ARCHIVES:
        raise PackageError(
            f"expected {EXPECTED_RASTER_ARCHIVES} raster scenes, found {len(raster_ids)}"
        )
    return sorted(chip_ids), sorted(raster_ids), sorted(eval_ids)


def _validate_train_labels(
    path: Path,
    *,
    required_scene_ids: Sequence[str],
    forbidden_scene_ids: Sequence[str],
) -> dict[str, object]:
    """Require label coverage for core scenes and zero eval-final leakage."""

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "scene_id" not in reader.fieldnames:
                raise PackageError("train.csv must contain a scene_id column")
            observed: set[str] = set()
            row_count = 0
            for row in reader:
                scene_id = str(row.get("scene_id", "")).strip()
                if not scene_id:
                    raise PackageError("train.csv contains an empty scene_id")
                observed.add(scene_id)
                row_count += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PackageError(f"cannot validate train labels: {path}") from exc
    if row_count <= 0:
        raise PackageError("train.csv contains no label rows")
    required = set(required_scene_ids)
    forbidden = set(forbidden_scene_ids)
    missing = sorted(required - observed)
    leaked = sorted(forbidden & observed)
    if missing:
        raise PackageError(f"train.csv is missing core scenes: {missing[:8]}")
    if leaked:
        raise PackageError(f"eval_final scene leaked into train.csv: {leaked[:8]}")
    return {
        "path": "data/raw/xview3/labels/train.csv",
        "row_count": row_count,
        "scene_count": len(observed),
        "required_core_scene_count": len(required),
        "eval_final_intersection": [],
    }


def _validate_apptainer_definition(path: Path) -> None:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PackageError(f"cannot read Apptainer definition: {path}") from exc
    bootstrap = re.search(r"(?im)^\s*Bootstrap:\s*docker\s*$", content)
    pinned_from = re.search(r"(?im)^\s*From:\s*\S+@sha256:[0-9a-f]{64}\s*$", content)
    if not bootstrap or not pinned_from:
        raise PackageError(
            "Apptainer definition must bootstrap docker from an OCI sha256 digest"
        )


def _lock_requirements(lock_path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise PackageError(f"environment lock contains an unpinned entry: {line}")
        name, version = line.split("==", 1)
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        if canonical in requirements:
            raise PackageError(f"duplicate package in environment lock: {name}")
        requirements[canonical] = version
    if requirements.get("torch") != EXPECTED_TORCH:
        raise PackageError(
            f"environment lock must pin torch=={EXPECTED_TORCH}, "
            f"found {requirements.get('torch')!r}"
        )
    return requirements


def validate_wheelhouse(wheelhouse: Path, lock_path: Path) -> dict[str, object]:
    """Require the wheel name/version set to equal the exact lock."""

    if not wheelhouse.is_dir():
        raise PackageError(f"wheelhouse is absent: {wheelhouse}")
    requirements = _lock_requirements(lock_path)
    entries = sorted(wheelhouse.iterdir(), key=lambda path: path.name)
    invalid = [
        path.name
        for path in entries
        if path.is_symlink() or not path.is_file() or path.suffix != ".whl"
    ]
    if invalid:
        raise PackageError(
            "wheelhouse contains non-wheel, non-file, or symlink entries: "
            + ", ".join(invalid)
        )
    wheels = entries
    if not wheels:
        raise PackageError(f"wheelhouse contains no wheels: {wheelhouse}")
    present: set[tuple[str, str]] = set()
    for wheel in wheels:
        if not wheel.is_file() or wheel.is_symlink():
            raise PackageError(f"invalid wheelhouse entry: {wheel}")
        fields = wheel.name[:-4].split("-")
        if len(fields) < 5:
            raise PackageError(f"invalid wheel filename: {wheel.name}")
        name = re.sub(r"[-_.]+", "-", fields[0]).lower()
        identity = (name, fields[1])
        if identity in present:
            raise PackageError(
                f"duplicate wheel distribution/version in wheelhouse: "
                f"{name}=={fields[1]}"
            )
        python_tag = fields[-3].lower()
        abi_tag = fields[-2].lower()
        platform_tag = fields[-1].lower()
        python_tags = python_tag.split(".")
        native_311 = any(
            tag == "py3" or tag.startswith("cp311") for tag in python_tags
        )
        compatible_abi3 = abi_tag == "abi3" and any(
            re.fullmatch(r"cp3(?:[6-9]|10|11)", tag) for tag in python_tags
        )
        if not (native_311 or compatible_abi3):
            raise PackageError(f"wheel is not compatible with CPython 3.11: {wheel.name}")
        if platform_tag != "any" and (
            "linux" not in platform_tag or "x86_64" not in platform_tag
        ):
            raise PackageError(f"wheel is not Linux x86_64 compatible: {wheel.name}")
        present.add(identity)
    required = set(requirements.items())
    unexpected = sorted(present - required)
    if unexpected:
        rendered = ", ".join(f"{name}=={version}" for name, version in unexpected)
        raise PackageError(
            "wheelhouse contains name/version pairs outside the exact lock: "
            + rendered
        )
    missing = [
        f"{name}=={version}"
        for name, version in sorted(requirements.items())
        if (name, version) not in present
    ]
    if missing:
        raise PackageError(
            "wheelhouse does not cover the exact environment lock: "
            + ", ".join(missing)
        )
    return {
        "wheel_count": len(wheels),
        "lock_requirement_count": len(requirements),
    }


def build_wheelhouse(
    *,
    repo_root: Path,
    output: Path,
    python: Path = Path(sys.executable),
) -> None:
    """Download the exact cu126 lock into an atomic, binary-only wheelhouse."""

    version = _run([str(python), "--version"])
    if version != f"Python {EXPECTED_PYTHON}":
        raise PackageError(
            f"wheelhouse must be resolved by Python {EXPECTED_PYTHON}, found {version}"
        )
    if output.exists():
        raise PackageError(f"wheelhouse destination already exists: {output}")
    if _inside_repository_worktrees(output, repo_root):
        raise PackageError("wheelhouse must be outside the repository")
    lock_path = repo_root / "locks" / "env-v100node.txt"
    _lock_requirements(lock_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        _run(
            [
                str(python),
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--no-deps",
                "--requirement",
                str(lock_path),
                "--dest",
                str(staging),
                "--extra-index-url",
                "https://download.pytorch.org/whl/cu126",
            ],
            capture=False,
        )
        validate_wheelhouse(staging, lock_path)
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _environment_entries(
    *,
    temporary: Path,
    wheelhouse: Path,
    lock_path: Path,
    apptainer_definition: Path,
) -> list[tuple[Path, PurePosixPath]]:
    environment = temporary / "environment"
    (environment / "wheelhouse").mkdir(parents=True)
    shutil.copy2(lock_path, environment / "env-v100node.txt")
    shutil.copy2(apptainer_definition, environment / "Apptainer.def")
    _write_bytes(environment / "PYTHON_VERSION", f"{EXPECTED_PYTHON}\n".encode())
    install = (
        "python -m pip install --no-index --find-links=/opt/xview3/wheelhouse "
        "-r /opt/xview3/env-v100node.txt\n"
        "python -m pip install --no-index --no-deps /opt/xview3/repo\n"
    )
    _write_bytes(environment / "INSTALL.txt", install.encode())
    checksums: list[str] = []
    for wheel in sorted(wheelhouse.iterdir(), key=lambda item: item.name):
        destination = environment / "wheelhouse" / wheel.name
        try:
            os.link(wheel, destination)
        except OSError:
            shutil.copy2(wheel, destination)
        checksums.append(f"{_hash_file(destination)}  {wheel.name}\n")
    _write_bytes(
        environment / "WHEELHOUSE_SHA256SUMS",
        "".join(checksums).encode("utf-8"),
    )
    return _source_entries(environment, PurePosixPath("environment"))


def _weight_provenance(
    weight_root: Path,
    name: str,
    *,
    production: bool,
) -> dict[str, object]:
    metadata = dict(REQUIRED_WEIGHT_DIRS[name])
    source_note = weight_root / "SOURCE.note"
    license_note = weight_root / "LICENSE.note"
    if (
        not source_note.is_file()
        or source_note.is_symlink()
        or not license_note.is_file()
        or license_note.is_symlink()
    ):
        raise PackageError(f"{name} must include SOURCE.note and LICENSE.note")
    source_text = source_note.read_text(encoding="utf-8", errors="strict")
    license_text = license_note.read_text(encoding="utf-8", errors="strict")
    model = metadata["model"]
    revision = metadata["revision"]
    if f"{model}@{revision}" not in source_text:
        raise PackageError(
            f"{name}/SOURCE.note does not contain exact model@revision"
        )
    license_token = re.sub(r"[^a-z0-9]", "", metadata["license"].lower())
    normalized_license = re.sub(r"[^a-z0-9]", "", license_text.lower())
    if license_token not in normalized_license:
        raise PackageError(f"{name}/LICENSE.note does not match {metadata['license']}")
    checkpoint_name, expected_checkpoint_sha256 = REQUIRED_CHECKPOINTS[name]
    checkpoint = weight_root / checkpoint_name
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise PackageError(f"{name} required checkpoint is absent: {checkpoint_name}")
    checkpoint_sha256 = _hash_file(checkpoint)
    if production and checkpoint_sha256 != expected_checkpoint_sha256:
        raise PackageError(
            f"{name}/{checkpoint_name} SHA-256 does not match the approved checkpoint"
        )
    metadata.update(
        {
            "directory": name,
            "source_note_sha256": _hash_file(source_note),
            "license_note_sha256": _hash_file(license_note),
            "checkpoint": checkpoint_name,
            "checkpoint_sha256": checkpoint_sha256,
            "approved_checkpoint_sha256": expected_checkpoint_sha256,
        }
    )
    return metadata


def _weight_archive_entries(
    weight_root: Path,
    name: str,
) -> list[tuple[Path, PurePosixPath]]:
    """Archive only the three files consumed by the approved loader."""

    root = _require_no_symlink_components(weight_root, leaf="directory")
    archive_root = PurePosixPath(f"data/weights/{name}")
    entries: list[tuple[Path, PurePosixPath]] = [(root, archive_root)]
    for member in sorted(REQUIRED_WEIGHT_MEMBERS[name]):
        entries.extend(
            _source_entries(root / member, archive_root / member)
        )
    return entries


def _add_archive(
    artifacts: list[dict[str, object]],
    **kwargs: object,
) -> dict[str, object]:
    artifact = _archive_artifact(**kwargs)  # type: ignore[arg-type]
    artifacts.append(artifact)
    return artifact


def _require_exact_scene_directories(
    root: Path, expected: Sequence[str], description: str
) -> None:
    if not root.is_dir():
        raise PackageError(f"required {description} root is absent: {root}")
    entries = list(root.iterdir())
    non_directories = sorted(item.name for item in entries if not item.is_dir())
    observed = {item.name for item in entries if item.is_dir()}
    expected_set = set(expected)
    if non_directories or observed != expected_set:
        missing = sorted(expected_set - observed)
        extra = sorted(observed - expected_set)
        raise PackageError(
            f"{description} top-level inventory mismatch; "
            f"missing={missing}, extra={extra}, non_directories={non_directories}"
        )


def _physical_paths(artifacts: Sequence[Mapping[str, object]]) -> list[str]:
    paths: list[str] = []
    for artifact in artifacts:
        for part in artifact["parts"]:  # type: ignore[index]
            paths.append(part["path"])  # type: ignore[index]
    return sorted(paths)


def build_package(options: BuildOptions) -> Path:
    """Build into a private staging directory, publishing READY last."""

    repo_root = options.repo_root.resolve()
    data_root = options.data_root.resolve()
    package_root = options.package_root.resolve()
    wheelhouse = options.wheelhouse.resolve()
    requested_definition = _absolute_path(options.apptainer_definition)
    committed_definition = repo_root / COMMITTED_APPTAINER_DEFINITION
    if options.production and requested_definition != committed_definition:
        raise PackageError(
            "production --apptainer-definition must be exactly "
            f"{committed_definition}"
        )
    definition = _require_no_symlink_components(
        requested_definition,
        leaf="file",
    )
    if package_root.exists():
        raise PackageError(f"package destination already exists: {package_root}")
    if _inside_repository_worktrees(package_root, repo_root):
        raise PackageError("package output must be outside the repository")
    if options.max_part_bytes <= 0:
        raise PackageError("max_part_bytes must be positive")

    commit, commit_epoch = _git_source(
        repo_root,
        options.branch,
        production=options.production,
    )
    required_ancestor = SOURCE_BASE_COMMIT if options.production else None
    if required_ancestor is not None:
        _require_source_ancestor(repo_root, commit, required_ancestor)
    package_id = f"xview3-h100-fp32-{commit}"
    if options.production and package_root.name != package_id:
        raise PackageError(
            f"production package directory must be full-SHA-addressed as {package_id}"
        )
    chip_ids, raster_ids, eval_final_ids = _load_split(
        repo_root, production=options.production
    )
    chip_root = data_root / "chips"
    raster_root = data_root / "raw" / "xview3" / "GRD"
    _require_exact_scene_directories(chip_root, chip_ids, "chip scene")
    _require_exact_scene_directories(raster_root, raster_ids, "dev/test raster scene")
    lock_path = _require_no_symlink_components(
        repo_root / ENVIRONMENT_LOCK_PATH,
        leaf="file",
    )
    wheelhouse_summary = validate_wheelhouse(wheelhouse, lock_path)
    _validate_apptainer_definition(definition)

    package_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{package_root.name}.building-", dir=package_root.parent)
    )
    artifacts: list[dict[str, object]] = []
    weights: list[dict[str, object]] = []
    try:
        bundle = staging / "code" / "xview3.bundle"
        _create_git_bundle(
            repo_root,
            bundle,
            options.branch,
            commit,
            required_ancestor,
            production=options.production,
        )
        artifacts.append(
            _plain_artifact(
                bundle,
                staging,
                kind="git_bundle",
                name=options.branch,
                extraction_root=PurePosixPath("code/xview3.bundle"),
                max_part_bytes=options.max_part_bytes,
            )
        )

        for scene_id in chip_ids:
            source = chip_root / scene_id
            _add_archive(
                artifacts,
                staging=staging,
                output_relative=PurePosixPath(f"data/chips/{scene_id}.tar.zst"),
                entries=_source_entries(
                    source, PurePosixPath(f"data/chips/{scene_id}")
                ),
                kind="chip_scene",
                name=scene_id,
                extraction_root=PurePosixPath(f"data/chips/{scene_id}"),
                max_part_bytes=options.max_part_bytes,
            )

        for scene_id in raster_ids:
            source = raster_root / scene_id
            _add_archive(
                artifacts,
                staging=staging,
                output_relative=PurePosixPath(f"data/rasters/{scene_id}.tar.zst"),
                entries=_source_entries(
                    source, PurePosixPath(f"data/raw/xview3/GRD/{scene_id}")
                ),
                kind="raster_scene",
                name=scene_id,
                extraction_root=PurePosixPath(f"data/raw/xview3/GRD/{scene_id}"),
                max_part_bytes=options.max_part_bytes,
            )
        label = data_root / "raw" / "xview3" / "labels" / "train.csv"
        label_summary = _validate_train_labels(
            label,
            required_scene_ids=chip_ids,
            forbidden_scene_ids=eval_final_ids,
        )
        label_artifact = _add_archive(
            artifacts,
            staging=staging,
            output_relative=PurePosixPath("data/labels/train.csv.tar.zst"),
            entries=_source_entries(
                label, PurePosixPath("data/raw/xview3/labels/train.csv")
            ),
            kind="labels",
            name="train.csv",
            extraction_root=PurePosixPath("data/raw/xview3/labels/train.csv"),
            max_part_bytes=options.max_part_bytes,
        )
        label_artifact["member_sha256"] = {
            "data/raw/xview3/labels/train.csv": _hash_file(label)
        }

        for name in REQUIRED_WEIGHT_DIRS:
            source = data_root / "weights" / name
            provenance = _weight_provenance(
                source,
                name,
                production=options.production,
            )
            weights.append(provenance)
            artifact = _add_archive(
                artifacts,
                staging=staging,
                output_relative=PurePosixPath(f"data/weights/{name}.tar.zst"),
                entries=_weight_archive_entries(source, name),
                kind="core_weight",
                name=name,
                extraction_root=PurePosixPath(f"data/weights/{name}"),
                max_part_bytes=options.max_part_bytes,
            )
            artifact["member_sha256"] = {
                f"data/weights/{name}/SOURCE.note": provenance[
                    "source_note_sha256"
                ],
                f"data/weights/{name}/LICENSE.note": provenance[
                    "license_note_sha256"
                ],
                f"data/weights/{name}/{provenance['checkpoint']}": provenance[
                    "checkpoint_sha256"
                ],
            }

        with tempfile.TemporaryDirectory(prefix="xview3-environment-") as temporary:
            environment_entries = _environment_entries(
                temporary=Path(temporary),
                wheelhouse=wheelhouse,
                lock_path=lock_path,
                apptainer_definition=definition,
            )
            environment_artifact = _add_archive(
                artifacts,
                staging=staging,
                output_relative=PurePosixPath("environment/offline.tar.zst"),
                entries=environment_entries,
                kind="offline_environment",
                name="python311-cu126",
                extraction_root=PurePosixPath("environment"),
                max_part_bytes=options.max_part_bytes,
            )
            environment_artifact["member_sha256"] = {
                "environment/Apptainer.def": _hash_file(definition),
                "environment/env-v100node.txt": _hash_file(lock_path),
            }

        counts = {
            "chip_archives": sum(a["kind"] == "chip_scene" for a in artifacts),
            "raster_archives": sum(a["kind"] == "raster_scene" for a in artifacts),
            "label_archives": sum(a["kind"] == "labels" for a in artifacts),
            "core_weight_archives": sum(a["kind"] == "core_weight" for a in artifacts),
        }
        manifest = {
            "format_version": FORMAT_VERSION,
            "package_type": "h100-source-handoff",
            "package_id": package_id,
            "created_at": datetime.fromtimestamp(
                commit_epoch, tz=timezone.utc
            ).isoformat(),
            "source": {
                "branch": options.branch,
                "git_commit": commit,
                "git_bundle_ref": f"refs/heads/{options.branch}",
                "required_base_commit": required_ancestor,
                "environment_lock": ENVIRONMENT_LOCK_PATH,
                "environment_lock_sha256": _hash_file(lock_path),
                "python": EXPECTED_PYTHON,
                "torch": EXPECTED_TORCH,
                "apptainer_definition": COMMITTED_APPTAINER_DEFINITION,
                "apptainer_definition_sha256": _hash_file(definition),
                "zstd": _run(["zstd", "--version"]),
            },
            "contract": {
                "production": options.production,
                "core_only": True,
                "strict_fp32": True,
                "tf32": False,
                "maximum_physical_file_bytes": options.max_part_bytes,
                "excluded": [
                    "runs",
                    ".venv*",
                    "caches",
                    "YOLO/LocateAnything references",
                    "LS-SSDD",
                    "superseded imagenet_cnn",
                    "JWT credentials",
                    "eval_final rasters and validation.csv",
                ],
            },
            "counts": counts,
            "scenes": {"chips": chip_ids, "rasters": raster_ids},
            "labels": label_summary,
            "weights": weights,
            "wheelhouse": wheelhouse_summary,
            "artifacts": artifacts,
        }
        _write_bytes(staging / "manifest.json", _canonical_json(manifest))

        checksums = []
        for relative in _physical_paths(artifacts):
            path = staging / relative
            checksums.append(f"{_hash_file(path)}  {relative}\n")
        _write_bytes(staging / "SHA256SUMS", "".join(checksums).encode("utf-8"))

        # READY is deliberately the final write.  It authenticates the two
        # control files without introducing a checksum cycle.
        ready = {
            "format_version": FORMAT_VERSION,
            "status": "READY",
            "package_id": manifest["package_id"],
            "git_commit": commit,
            "manifest": _physical_record(staging / "manifest.json", staging),
            "checksums": _physical_record(staging / "SHA256SUMS", staging),
        }
        _write_bytes(staging / "READY.json", _canonical_json(ready))
        _verify_package(staging, require_production=options.production)
        os.replace(staging, package_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return package_root


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"invalid JSON control file: {path}") from exc
    if not isinstance(value, dict):
        raise PackageError(f"JSON control file is not an object: {path}")
    return value


def _validate_control_record(
    package_root: Path,
    record: Mapping[str, object],
    expected_path: str,
) -> None:
    if record.get("path") != expected_path:
        raise PackageError(f"control path mismatch for {expected_path}")
    path = _require_no_symlink_components(
        package_root / expected_path,
        leaf="file",
    )
    if path.stat().st_size != record.get("bytes"):
        raise PackageError(f"control size mismatch: {expected_path}")
    if _hash_file(path, "sha256") != record.get("sha256"):
        raise PackageError(f"control sha256 mismatch: {expected_path}")
    if _hash_file(path, "sha1") != record.get("sha1"):
        raise PackageError(f"control sha1 mismatch: {expected_path}")


def _parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if not match:
            raise PackageError(f"malformed SHA256SUMS line: {line!r}")
        relative = PurePosixPath(match.group(2))
        _validate_safe_path(relative)
        name = relative.as_posix()
        if name in result:
            raise PackageError(f"duplicate SHA256SUMS entry: {name}")
        result[name] = match.group(1)
    return result


def _artifact_part_paths(
    package_root: Path, artifact: Mapping[str, object]
) -> list[Path]:
    raw_parts = artifact.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise PackageError("artifact has no physical parts")
    result: list[Path] = []
    for raw in raw_parts:
        if not isinstance(raw, dict):
            raise PackageError("artifact part is not an object")
        relative = PurePosixPath(str(raw.get("path", "")))
        _validate_safe_path(relative)
        path = _require_no_symlink_components(
            package_root.joinpath(*relative.parts),
            leaf="file",
        )
        if path.stat().st_size != raw.get("bytes"):
            raise PackageError(f"artifact part size mismatch: {relative}")
        if _hash_file(path, "sha256") != raw.get("sha256"):
            raise PackageError(f"artifact part sha256 mismatch: {relative}")
        if _hash_file(path, "sha1") != raw.get("sha1"):
            raise PackageError(f"artifact part sha1 mismatch: {relative}")
        result.append(path)
    return result


def _verify_contract(
    manifest: Mapping[str, object],
    *,
    require_production: bool,
) -> int:
    if manifest.get("format_version") != FORMAT_VERSION:
        raise PackageError("unsupported package format")
    contract = manifest.get("contract")
    counts = manifest.get("counts")
    if not isinstance(contract, dict) or not isinstance(counts, dict):
        raise PackageError("package contract/counts missing")
    package_type = manifest.get("package_type")
    if package_type not in {"h100-source-handoff", "h100-core-results"}:
        raise PackageError(f"unsupported package type: {package_type!r}")
    if not contract.get("core_only") or not contract.get("strict_fp32"):
        raise PackageError("package is not the strict-FP32 core-only handoff")
    if contract.get("tf32") is not False:
        raise PackageError("package contract permits TF32")
    production = contract.get("production")
    if not isinstance(production, bool):
        raise PackageError("package production contract must be boolean")
    if require_production and production is not True:
        raise PackageError("target-facing verification requires a production package")
    maximum = contract.get("maximum_physical_file_bytes")
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or maximum <= 0
    ):
        raise PackageError("maximum_physical_file_bytes must be a positive integer")
    if production is True and package_type == "h100-source-handoff":
        expected = {
            "chip_archives": EXPECTED_CHIP_ARCHIVES,
            "raster_archives": EXPECTED_RASTER_ARCHIVES,
            "label_archives": 1,
            "core_weight_archives": len(REQUIRED_WEIGHT_DIRS),
        }
        if counts != expected:
            raise PackageError(f"production package counts mismatch: {counts!r}")
    if production is True and package_type == "h100-core-results":
        if counts != {"core_result_archives": 32, "provenance_archives": 1}:
            raise PackageError(f"production result-package counts mismatch: {counts!r}")
    return maximum


def _artifact_expectations(
    manifest: Mapping[str, object],
) -> dict[tuple[str, str], tuple[str, str, str]]:
    """Return the only accepted (root, format, physical base) identities."""

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise PackageError("manifest source is missing")
    expected: dict[tuple[str, str], tuple[str, str, str]] = {}
    if manifest.get("package_type") == "h100-source-handoff":
        scenes = manifest.get("scenes")
        if not isinstance(scenes, Mapping):
            raise PackageError("source handoff scenes are missing")
        chips = scenes.get("chips")
        rasters = scenes.get("rasters")
        if not isinstance(chips, list) or not isinstance(rasters, list):
            raise PackageError("source handoff scene lists are invalid")
        if len(chips) != len(set(chips)) or len(rasters) != len(set(rasters)):
            raise PackageError("source handoff contains duplicate scene IDs")
        for scene_id in chips:
            name = str(scene_id)
            _validate_safe_path(PurePosixPath(name))
            expected[("chip_scene", name)] = (
                f"data/chips/{name}",
                "tar.zst",
                f"data/chips/{name}.tar.zst",
            )
        for scene_id in rasters:
            name = str(scene_id)
            _validate_safe_path(PurePosixPath(name))
            expected[("raster_scene", name)] = (
                f"data/raw/xview3/GRD/{name}",
                "tar.zst",
                f"data/rasters/{name}.tar.zst",
            )
        expected[("labels", "train.csv")] = (
            "data/raw/xview3/labels/train.csv",
            "tar.zst",
            "data/labels/train.csv.tar.zst",
        )
        for name in REQUIRED_WEIGHT_DIRS:
            expected[("core_weight", name)] = (
                f"data/weights/{name}",
                "tar.zst",
                f"data/weights/{name}.tar.zst",
            )
        expected[("offline_environment", "python311-cu126")] = (
            "environment",
            "tar.zst",
            "environment/offline.tar.zst",
        )
        branch = str(source.get("branch", ""))
        expected[("git_bundle", branch)] = (
            "code/xview3.bundle",
            "file",
            "code/xview3.bundle",
        )
    else:
        cells = manifest.get("cells")
        if not isinstance(cells, list) or len(cells) != len(set(cells)):
            raise PackageError("result package cell list is invalid")
        contract = manifest.get("contract")
        if (
            isinstance(contract, Mapping)
            and contract.get("production")
            and set(cells) != EXPECTED_CORE_RESULT_IDS
        ):
            raise PackageError("result package is not the exact 32-cell core grid")
        for cell in cells:
            name = str(cell)
            _validate_safe_path(PurePosixPath(name))
            expected[("core_result", name)] = (
                f"results/core/{name}",
                "tar.zst",
                f"results/core/{name}.tar.zst",
            )
        campaign_id = str(source.get("campaign_id", ""))
        if not campaign_id:
            raise PackageError("result package campaign ID is absent")
        expected[("campaign_provenance", campaign_id)] = (
            "results/provenance",
            "tar.zst",
            "results/provenance/campaign.tar.zst",
        )
    return expected


def _validate_part_names(
    artifact: Mapping[str, object],
    physical_base: str,
    maximum_physical_file_bytes: int,
) -> None:
    parts = artifact.get("parts")
    if not isinstance(parts, list) or not parts:
        raise PackageError("artifact has no parts")
    observed: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping):
            raise PackageError("artifact part is not an object")
        relative = str(part.get("path", ""))
        _validate_safe_path(PurePosixPath(relative))
        size = part.get("bytes")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or size > maximum_physical_file_bytes
        ):
            raise PackageError(
                f"artifact part size violates physical limit: {relative}"
            )
        observed.append(relative)
    if len(set(observed)) != len(observed):
        raise PackageError("artifact contains duplicate part paths")
    expected = (
        [physical_base]
        if len(observed) == 1
        else [
            f"{physical_base}.part-{index:05d}"
            for index in range(len(observed))
        ]
    )
    if observed != expected:
        raise PackageError(
            f"artifact physical paths do not match expected base {physical_base}"
        )
    typed_parts = [part for part in parts if isinstance(part, Mapping)]
    if len(typed_parts) > 1 and any(
        int(part["bytes"]) != maximum_physical_file_bytes
        for part in typed_parts[:-1]
    ):
        raise PackageError(
            f"non-final parts are not deterministically sized for {physical_base}"
        )


def _validate_artifact_schema(
    manifest: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
    maximum_physical_file_bytes: int,
) -> None:
    expected = _artifact_expectations(manifest)
    observed: dict[tuple[str, str], Mapping[str, object]] = {}
    for artifact in artifacts:
        identity = (str(artifact.get("kind", "")), str(artifact.get("name", "")))
        if identity in observed:
            raise PackageError(f"duplicate artifact identity: {identity}")
        observed[identity] = artifact
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise PackageError(
            f"artifact identity schema mismatch; missing={missing}, extra={extra}"
        )
    for identity, artifact in observed.items():
        extraction_root, archive_format, physical_base = expected[identity]
        if artifact.get("extraction_root") != extraction_root:
            raise PackageError(f"artifact extraction root mismatch: {identity}")
        if artifact.get("format") != archive_format:
            raise PackageError(f"artifact format mismatch: {identity}")
        if (
            not isinstance(artifact.get("file_count"), int)
            or int(artifact["file_count"]) < 1
            or not isinstance(artifact.get("unpacked_bytes"), int)
            or int(artifact["unpacked_bytes"]) < 0
        ):
            raise PackageError(f"artifact count/size declaration is invalid: {identity}")
        _validate_part_names(
            artifact,
            physical_base,
            maximum_physical_file_bytes,
        )

    if manifest.get("package_type") == "h100-core-results":
        source = manifest.get("source")
        cells = manifest.get("cells")
        if not isinstance(source, Mapping) or not isinstance(cells, list):
            raise PackageError("result package source/cell schema is invalid")
        campaign_id = str(source.get("campaign_id", ""))
        provenance = observed[("campaign_provenance", campaign_id)]
        provenance_members = provenance.get("member_sha256")
        declared_provenance_members = source.get(
            "campaign_provenance_member_sha256"
        )
        if (
            not isinstance(provenance_members, Mapping)
            or not isinstance(declared_provenance_members, Mapping)
            or dict(declared_provenance_members) != dict(provenance_members)
        ):
            raise PackageError(
                "result campaign-provenance digest index is not archive-bound"
            )
        required_provenance = {
            "results/provenance/campaign_manifest.json",
            "results/provenance/H100_READY.json",
            "results/provenance/h100_runtime.json",
            "results/provenance/throughput_projection.json",
            "results/provenance/venv_build.json",
            "results/provenance/CUTOVER_READY.json",
            "results/provenance/SOURCE_VALIDATED.json",
            "results/provenance/HOST_HANDOFF_TESTS.json",
            "results/provenance/PYTEST_ACCEPTANCE.json",
            "results/provenance/V100_CORE_ARCHIVED.json",
            "results/provenance/V100_CORE_ARCHIVE_MANIFEST.json",
            "results/provenance/slurm-smoke/SLURM_SMOKE_READY.json",
            "results/provenance/slurm-smoke/SLURM_SMOKE_STATE.json",
            "results/provenance/summary/grid.csv",
            "results/provenance/acceptance-logs/pytest-handoff-host.log",
            "results/provenance/acceptance-logs/pytest-venv-remaining.log",
            "results/provenance/acceptance-logs/vit-fp32.log",
            "results/provenance/acceptance-logs/cnn-200step-fp32.log",
        }
        if not required_provenance <= set(provenance_members):
            raise PackageError(
                "result campaign provenance lacks mandatory acceptance/cutover evidence"
            )
        allocation_members = {
            name
            for name in provenance_members
            if str(name).startswith(
                "results/provenance/allocations/h100_runtime-"
            )
            and str(name).endswith(".json")
        }
        if not allocation_members:
            raise PackageError("result package lacks an allocation hardware record")
        direct_provenance = {
            "results/provenance/campaign_manifest.json": source.get(
                "campaign_manifest_sha256"
            ),
            "results/provenance/SOURCE_VALIDATED.json": source.get(
                "source_validation_sha256"
            ),
            "results/provenance/PYTEST_ACCEPTANCE.json": source.get(
                "acceptance_test_suite_sha256"
            ),
            "results/provenance/summary/grid.csv": source.get(
                "summary_grid_sha256"
            ),
        }
        operator_cutover = source.get("operator_cutover")
        if not isinstance(operator_cutover, Mapping):
            raise PackageError("result operator-cutover identity is absent")
        direct_provenance.update(
            {
                "results/provenance/CUTOVER_READY.json": operator_cutover.get(
                    "cutover_ready_sha256"
                ),
                "results/provenance/V100_CORE_ARCHIVED.json": (
                    operator_cutover.get("v100_core_archived_sha256")
                ),
                "results/provenance/V100_CORE_ARCHIVE_MANIFEST.json": (
                    operator_cutover.get("archive_manifest_sha256")
                ),
            }
        )
        if any(
            provenance_members.get(path) != digest
            or not re.fullmatch(r"[0-9a-f]{64}", str(digest or ""))
            for path, digest in direct_provenance.items()
        ):
            raise PackageError(
                "result campaign/grid digests are not campaign-archive-bound"
            )
        runtime_digests = source.get("runtime_provenance_sha256")
        if (
            not isinstance(runtime_digests, Mapping)
            or set(runtime_digests) != set(cells)
        ):
            raise PackageError(
                "result runtime-provenance digest index is not the exact cell grid"
            )
        for cell in cells:
            name = str(cell)
            members = observed[("core_result", name)].get("member_sha256")
            root = f"results/core/{name}"
            expected_members = {
                f"{root}/final_metrics.json",
                f"{root}/config.yaml",
                f"{root}/metrics/metrics.csv",
                f"{root}/runtime_provenance.json",
                f"{root}/checkpoints/best.ckpt",
                f"{root}/checkpoints/last.ckpt",
                f"{root}/log.txt",
            }
            if not isinstance(members, Mapping) or set(members) != expected_members:
                raise PackageError(
                    f"{name} core-result archive is not the exact result allowlist"
                )
            runtime_digest = runtime_digests.get(name)
            if (
                members.get(f"{root}/runtime_provenance.json")
                != runtime_digest
                or not re.fullmatch(
                    r"[0-9a-f]{64}", str(runtime_digest or "")
                )
            ):
                raise PackageError(
                    f"{name} runtime-provenance digest is not archive-bound"
                )
        return

    weights = manifest.get("weights")
    if not isinstance(weights, list):
        raise PackageError("weight provenance is missing")
    by_name: dict[str, Mapping[str, object]] = {}
    for item in weights:
        if not isinstance(item, Mapping):
            raise PackageError("weight provenance record is invalid")
        name = str(item.get("directory", ""))
        if name in by_name:
            raise PackageError(f"duplicate weight provenance: {name}")
        by_name[name] = item
    if set(by_name) != set(REQUIRED_WEIGHT_DIRS):
        raise PackageError("six exact core-weight provenance records are required")
    contract = manifest["contract"]
    assert isinstance(contract, Mapping)
    source = manifest["source"]
    assert isinstance(source, Mapping)
    environment_members = {
        "environment/Apptainer.def": source.get(
            "apptainer_definition_sha256"
        ),
        "environment/env-v100node.txt": source.get(
            "environment_lock_sha256"
        ),
    }
    if observed[("offline_environment", "python311-cu126")].get(
        "member_sha256"
    ) != environment_members:
        raise PackageError(
            "offline environment is not hash-bound to the committed recipe"
        )
    for name, canonical in REQUIRED_WEIGHT_DIRS.items():
        item = by_name[name]
        checkpoint, approved_sha256 = REQUIRED_CHECKPOINTS[name]
        for key in ("model", "revision", "license"):
            if item.get(key) != canonical[key]:
                raise PackageError(f"{name} {key} provenance mismatch")
        if item.get("checkpoint") != checkpoint:
            raise PackageError(f"{name} checkpoint filename mismatch")
        if item.get("approved_checkpoint_sha256") != approved_sha256:
            raise PackageError(f"{name} approved checkpoint hash mismatch")
        if contract.get("production") and item.get("checkpoint_sha256") != approved_sha256:
            raise PackageError(f"{name} production checkpoint hash mismatch")
        artifact = observed[("core_weight", name)]
        expected_members = {
            f"data/weights/{name}/SOURCE.note": item.get("source_note_sha256"),
            f"data/weights/{name}/LICENSE.note": item.get("license_note_sha256"),
            f"data/weights/{name}/{checkpoint}": item.get("checkpoint_sha256"),
        }
        if artifact.get("member_sha256") != expected_members:
            raise PackageError(f"{name} archive member hash bindings mismatch")
        if artifact.get("file_count") != len(REQUIRED_WEIGHT_MEMBERS[name]):
            raise PackageError(f"{name} archive is not the exact loader allowlist")


def _validate_production_source_identity(
    manifest: Mapping[str, object],
) -> None:
    if manifest.get("package_type") != "h100-source-handoff":
        return
    contract = manifest.get("contract")
    source = manifest.get("source")
    if not isinstance(contract, Mapping) or not isinstance(source, Mapping):
        raise PackageError("source handoff recipe is absent")
    if contract.get("production") is not True:
        return
    expected = {
        "branch": EXPECTED_BRANCH,
        "git_bundle_ref": f"refs/heads/{EXPECTED_BRANCH}",
        "required_base_commit": SOURCE_BASE_COMMIT,
        "environment_lock": ENVIRONMENT_LOCK_PATH,
        "python": EXPECTED_PYTHON,
        "torch": EXPECTED_TORCH,
        "apptainer_definition": COMMITTED_APPTAINER_DEFINITION,
    }
    mismatches = {
        key: (value, source.get(key))
        for key, value in expected.items()
        if source.get(key) != value
    }
    if mismatches:
        raise PackageError(f"production source recipe mismatch: {mismatches}")
    for key in (
        "environment_lock_sha256",
        "apptainer_definition_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get(key, ""))):
            raise PackageError(f"production source {key} is not a SHA-256")


def _package_regular_files(package_root: Path) -> set[str]:
    files: set[str] = set()
    for current, directories, names in os.walk(package_root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            directory = current_path / name
            info = directory.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise PackageError(f"non-directory or symlink in package tree: {directory}")
        for name in names:
            path = current_path / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise PackageError(f"non-regular or symlink file in package tree: {path}")
            files.add(path.relative_to(package_root).as_posix())
    return files


def _verify_package(
    package_root: Path,
    *,
    require_production: bool,
) -> dict[str, object]:
    """Verify all control, physical, logical, count, and bundle invariants."""

    package_root = _require_no_symlink_components(package_root, leaf="directory")
    ready_path = _require_no_symlink_components(
        package_root / "READY.json", leaf="file"
    )
    manifest_path = _require_no_symlink_components(
        package_root / "manifest.json", leaf="file"
    )
    sums_path = _require_no_symlink_components(
        package_root / "SHA256SUMS", leaf="file"
    )
    ready = _load_json(ready_path)
    if ready.get("status") != "READY" or ready.get("format_version") != FORMAT_VERSION:
        raise PackageError("READY marker is invalid")
    _validate_control_record(
        package_root, ready.get("manifest", {}), "manifest.json"  # type: ignore[arg-type]
    )
    _validate_control_record(
        package_root, ready.get("checksums", {}), "SHA256SUMS"  # type: ignore[arg-type]
    )

    manifest = _load_json(manifest_path)
    maximum_physical_file_bytes = _verify_contract(
        manifest,
        require_production=require_production,
    )
    _validate_production_source_identity(manifest)
    if ready.get("package_id") != manifest.get("package_id"):
        raise PackageError("READY/manifest package ID mismatch")
    source = manifest.get("source")
    if not isinstance(source, dict) or ready.get("git_commit") != source.get("git_commit"):
        raise PackageError("READY/manifest git commit mismatch")
    commit = str(source.get("git_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PackageError("manifest git commit is not a full 40-character SHA")
    expected_id_prefix = (
        "xview3-h100-results-"
        if manifest.get("package_type") == "h100-core-results"
        else "xview3-h100-fp32-"
    )
    if manifest.get("package_type") == "h100-core-results":
        identity = source.get("result_identity")
        identity_sha256 = source.get("result_identity_sha256")
        if not isinstance(identity, Mapping):
            raise PackageError("result package identity is absent")
        computed_identity = hashlib.sha256(_canonical_json(dict(identity))).hexdigest()
        if identity_sha256 != computed_identity:
            raise PackageError("result package identity digest mismatch")
        expected_package_id = expected_id_prefix + commit + "-" + computed_identity
    else:
        expected_package_id = expected_id_prefix + commit
    if manifest.get("package_id") != expected_package_id:
        raise PackageError("package ID is not fully content-addressed")
    contract = manifest.get("contract")
    assert isinstance(contract, dict)
    required_ancestor = source.get("required_base_commit")

    sums = _parse_sha256sums(sums_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise PackageError("manifest artifacts missing")
    if not all(isinstance(artifact, Mapping) for artifact in artifacts):
        raise PackageError("manifest artifact is not an object")
    typed_artifacts = [artifact for artifact in artifacts if isinstance(artifact, Mapping)]
    _validate_artifact_schema(
        manifest,
        typed_artifacts,
        maximum_physical_file_bytes,
    )
    physical: set[str] = set()
    if manifest.get("package_type") == "h100-core-results":
        observed_counts = {"core_result_archives": 0, "provenance_archives": 0}
    else:
        observed_counts = {
            "chip_archives": 0,
            "raster_archives": 0,
            "label_archives": 0,
            "core_weight_archives": 0,
        }
    git_artifacts: list[tuple[Mapping[str, object], list[Path]]] = []
    extraction_roots: set[str] = set()
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise PackageError("manifest artifact is not an object")
        parts = _artifact_part_paths(package_root, raw)
        extraction_root = str(raw.get("extraction_root", ""))
        _validate_safe_path(PurePosixPath(extraction_root))
        if extraction_root in extraction_roots:
            raise PackageError(f"duplicate artifact extraction root: {extraction_root}")
        extraction_roots.add(extraction_root)
        for part_record in raw["parts"]:  # type: ignore[index]
            relative = part_record["path"]  # type: ignore[index]
            if relative in physical:
                raise PackageError(f"physical file belongs to two artifacts: {relative}")
            physical.add(relative)
        if sum(path.stat().st_size for path in parts) != raw.get("archive_bytes"):
            raise PackageError(f"logical size mismatch for artifact {raw.get('name')}")
        if _hash_paths(parts, "sha256") != raw.get("archive_sha256"):
            raise PackageError(f"logical sha256 mismatch for artifact {raw.get('name')}")
        if _hash_paths(parts, "sha1") != raw.get("archive_sha1"):
            raise PackageError(f"logical sha1 mismatch for artifact {raw.get('name')}")
        if raw.get("format") == "tar.zst":
            member_hashes = raw.get("member_sha256")
            if member_hashes is not None and not isinstance(member_hashes, Mapping):
                raise PackageError("artifact member SHA-256 map is invalid")
            text_tokens: dict[str, str] = {}
            if raw.get("kind") == "core_weight":
                weight_name = str(raw.get("name"))
                canonical = REQUIRED_WEIGHT_DIRS[weight_name]
                text_tokens = {
                    f"data/weights/{weight_name}/SOURCE.note": (
                        f"{canonical['model']}@{canonical['revision']}"
                    ),
                    f"data/weights/{weight_name}/LICENSE.note": canonical[
                        "license"
                    ],
                }
            _inspect_tar_zst(
                parts,
                extraction_root=PurePosixPath(str(raw["extraction_root"])),
                expected_file_count=int(raw["file_count"]),
                expected_unpacked_bytes=int(raw["unpacked_bytes"]),
                member_sha256=member_hashes,
                member_text_tokens=text_tokens,
                exact_file_members=raw.get("kind") in {
                    "labels",
                    "core_weight",
                    "core_result",
                    "campaign_provenance",
                },
            )
        kind = raw.get("kind")
        count_key = {
            "chip_scene": "chip_archives",
            "raster_scene": "raster_archives",
            "labels": "label_archives",
            "core_weight": "core_weight_archives",
            "core_result": "core_result_archives",
            "campaign_provenance": "provenance_archives",
        }.get(kind)
        if count_key:
            observed_counts[count_key] += 1
        if kind == "git_bundle":
            git_artifacts.append((raw, parts))

    if set(sums) != physical:
        raise PackageError("SHA256SUMS physical-file set does not match manifest")
    for relative, expected in sums.items():
        if _hash_file(package_root / relative) != expected:
            raise PackageError(f"SHA256SUMS mismatch: {relative}")
    if manifest.get("counts") != observed_counts:
        raise PackageError("declared archive counts do not match artifacts")

    if manifest.get("package_type") == "h100-source-handoff":
        weights = manifest.get("weights")
        if not isinstance(weights, list):
            raise PackageError("weight provenance is missing")
        if {item.get("directory") for item in weights if isinstance(item, dict)} != set(
            REQUIRED_WEIGHT_DIRS
        ):
            raise PackageError("six exact core-weight provenance records are required")

        if len(git_artifacts) != 1:
            raise PackageError("exactly one git bundle is required")
        git_artifact, git_parts = git_artifacts[0]
        bundle_source: dict[str, object]
        if len(git_parts) == 1:
            bundle_source = _verify_git_bundle(
                git_parts[0],
                str(source["branch"]),
                str(source["git_commit"]),
                required_ancestor=(
                    str(required_ancestor) if required_ancestor is not None else None
                ),
                production=contract.get("production") is True,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="xview3-bundle-parts-") as temporary:
                bundle = Path(temporary) / "xview3.bundle"
                with bundle.open("wb") as output:
                    for part in git_parts:
                        with part.open("rb") as source_part:
                            shutil.copyfileobj(source_part, output)
                bundle_source = _verify_git_bundle(
                    bundle,
                    str(source["branch"]),
                    str(source["git_commit"]),
                    required_ancestor=(
                        str(required_ancestor) if required_ancestor is not None else None
                    ),
                    production=contract.get("production") is True,
                )
        scenes = manifest.get("scenes")
        if not isinstance(scenes, Mapping):
            raise PackageError("source handoff scene lists are absent")
        if (
            scenes.get("chips") != bundle_source["chips"]
            or scenes.get("rasters") != bundle_source["rasters"]
        ):
            raise PackageError(
                "manifest scene lists do not match data/splits.json "
                "from the verified Git bundle"
            )
        label_artifact = next(
            artifact
            for artifact in typed_artifacts
            if artifact.get("kind") == "labels"
        )
        label_parts = _artifact_part_paths(package_root, label_artifact)
        with tempfile.TemporaryDirectory(prefix="xview3-label-verify-") as temporary:
            label_root = Path(temporary)
            _extract_tar_zst(
                label_parts,
                label_root,
                extraction_root=PurePosixPath(
                    str(label_artifact["extraction_root"])
                ),
                expected_file_count=int(label_artifact["file_count"]),
                expected_unpacked_bytes=int(label_artifact["unpacked_bytes"]),
            )
            label_summary = _validate_train_labels(
                label_root / "data/raw/xview3/labels/train.csv",
                required_scene_ids=bundle_source["chips"],  # type: ignore[arg-type]
                forbidden_scene_ids=bundle_source["eval_final"],  # type: ignore[arg-type]
            )
        if manifest.get("labels") != label_summary:
            raise PackageError("manifest label summary differs from verified train.csv")
        if (
            source.get("environment_lock_sha256")
            != bundle_source["environment_lock_sha256"]
            or source.get("apptainer_definition_sha256")
            != bundle_source["apptainer_definition_sha256"]
        ):
            raise PackageError(
                "transferred environment recipe does not match the verified Git bundle"
            )
    elif git_artifacts:
        raise PackageError("result package must not contain a git bundle")

    all_files = _package_regular_files(package_root)
    allowed = physical | {"manifest.json", "SHA256SUMS", "READY.json"}
    if all_files != allowed:
        raise PackageError(f"unexpected package files: {sorted(all_files - allowed)}")
    return manifest


def verify_package(package_root: Path) -> dict[str, object]:
    """Target-facing verifier: production packages only."""

    return _verify_package(package_root, require_production=True)


def _verify_fixture_package(package_root: Path) -> dict[str, object]:
    """Internal-only verifier used by deterministic fixture tests."""

    return _verify_package(package_root, require_production=False)


def _safe_output_path(destination: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    _validate_safe_path(relative)
    output = destination.joinpath(*relative.parts)
    if not _inside(output, destination):
        raise PackageError(f"archive path escapes destination: {name}")
    return output


def _validate_tar_member_root(name: str, extraction_root: PurePosixPath) -> PurePosixPath:
    relative = PurePosixPath(name)
    _validate_safe_path(relative)
    try:
        relative.relative_to(extraction_root)
    except ValueError as exc:
        raise PackageError(
            f"archive member {relative} escapes declared root {extraction_root}"
        ) from exc
    return relative


def _start_zstd_reader(
    parts: Sequence[Path],
) -> tuple[subprocess.Popen[bytes], threading.Thread, list[BaseException]]:
    process = subprocess.Popen(
        ["zstd", "--decompress", "--stdout", "--quiet"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    pump_error: list[BaseException] = []

    def pump() -> None:
        try:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, process.stdin)
            process.stdin.close()
        except BaseException as exc:  # propagated below
            pump_error.append(exc)
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()

    thread = threading.Thread(target=pump, name="xview3-zstd-input", daemon=True)
    thread.start()
    return process, thread, pump_error


def _finish_zstd_reader(
    process: subprocess.Popen[bytes],
    thread: threading.Thread,
    pump_error: Sequence[BaseException],
) -> None:
    thread.join()
    stderr = process.stderr.read() if process.stderr is not None else b""
    if process.wait() or pump_error:
        detail = (
            str(pump_error[0])
            if pump_error
            else stderr.decode(errors="replace").strip()
        )
        raise PackageError(f"zstd stream failed: {detail}")


def _inspect_tar_zst(
    parts: Sequence[Path],
    *,
    extraction_root: PurePosixPath,
    expected_file_count: int,
    expected_unpacked_bytes: int,
    member_sha256: Mapping[str, object] | None = None,
    member_text_tokens: Mapping[str, str] | None = None,
    exact_file_members: bool = False,
) -> None:
    process, thread, pump_error = _start_zstd_reader(parts)
    assert process.stdout is not None
    files = 0
    unpacked = 0
    seen: set[str] = set()
    seen_files: set[str] = set()
    required_hashes = {
        str(path): str(digest)
        for path, digest in (member_sha256 or {}).items()
    }
    observed_hashes: dict[str, str] = {}
    text_tokens = dict(member_text_tokens or {})
    observed_text: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                relative = _validate_tar_member_root(
                    member.name, extraction_root
                )
                name = relative.as_posix()
                if name in seen:
                    raise PackageError(f"duplicate archive member: {name}")
                seen.add(name)
                if member.isdir():
                    continue
                if not member.isfile():
                    raise PackageError(
                        f"links/devices are forbidden in archive: {name}"
                    )
                files += 1
                seen_files.add(name)
                unpacked += member.size
                if files > expected_file_count or unpacked > expected_unpacked_bytes:
                    raise PackageError(
                        f"archive exceeds declared count/size under {extraction_root}"
                    )
                if name in required_hashes:
                    source = archive.extractfile(member)
                    if source is None:
                        raise PackageError(f"cannot read archive member: {name}")
                    digest = hashlib.sha256()
                    captured = bytearray()
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        if name in text_tokens:
                            captured.extend(chunk)
                            if len(captured) > 1024 * 1024:
                                raise PackageError(
                                    f"bound text member is unexpectedly large: {name}"
                                )
                    observed_hashes[name] = digest.hexdigest()
                    if name in text_tokens:
                        normalized = re.sub(
                            r"[^a-z0-9]",
                            "",
                            captured.decode("utf-8", errors="strict").lower(),
                        )
                        observed_text[name] = normalized
    except BaseException:
        process.kill()
        process.wait()
        thread.join()
        raise
    _finish_zstd_reader(process, thread, pump_error)
    if files != expected_file_count or unpacked != expected_unpacked_bytes:
        raise PackageError(
            f"archive declared count/size mismatch under {extraction_root}"
        )
    if observed_hashes != required_hashes:
        raise PackageError(
            f"archive bound-member SHA-256 mismatch under {extraction_root}"
        )
    if exact_file_members and seen_files != set(required_hashes):
        raise PackageError(
            f"archive file members violate exact allowlist under {extraction_root}"
        )
    for name, token in text_tokens.items():
        normalized_token = re.sub(r"[^a-z0-9]", "", token.lower())
        if normalized_token not in observed_text.get(name, ""):
            raise PackageError(f"archive bound text content mismatch: {name}")


def _extract_tar_zst(
    parts: Sequence[Path],
    destination: Path,
    *,
    extraction_root: PurePosixPath,
    expected_file_count: int,
    expected_unpacked_bytes: int,
) -> None:
    process, thread, pump_error = _start_zstd_reader(parts)
    assert process.stdout is not None
    files = 0
    unpacked = 0
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                relative = _validate_tar_member_root(
                    member.name, extraction_root
                )
                name = relative.as_posix()
                if name in seen:
                    raise PackageError(f"duplicate archive member: {name}")
                seen.add(name)
                output = _safe_output_path(destination, name)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise PackageError(
                        f"links/devices are forbidden in archive: {name}"
                    )
                files += 1
                unpacked += member.size
                if files > expected_file_count or unpacked > expected_unpacked_bytes:
                    raise PackageError(
                        f"archive exceeds declared count/size under {extraction_root}"
                    )
                if output.exists():
                    raise PackageError(f"archive extraction would overwrite: {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise PackageError(f"cannot read archive member: {name}")
                with output.open("xb") as target:
                    shutil.copyfileobj(source, target)
                output.chmod(member.mode & 0o777)
    except BaseException:
        process.kill()
        process.wait()
        thread.join()
        raise
    _finish_zstd_reader(process, thread, pump_error)
    if files != expected_file_count or unpacked != expected_unpacked_bytes:
        raise PackageError(
            f"archive declared count/size mismatch under {extraction_root}"
        )


def _count_root(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def _extract_package(
    package_root: Path,
    destination: Path,
    *,
    require_production: bool,
) -> Path:
    """Verify into a private sibling stage, then atomically publish."""

    manifest = _verify_package(
        package_root,
        require_production=require_production,
    )
    package_root = _absolute_path(package_root)
    destination = Path(os.path.abspath(destination))
    if os.path.lexists(destination):
        raise PackageError(f"extraction destination must not exist: {destination}")
    parent = destination.parent
    if not parent.is_dir():
        raise PackageError(f"extraction parent is not a directory: {parent}")
    current = parent
    while True:
        if current.is_symlink():
            raise PackageError(f"extraction parent chain contains symlink: {current}")
        if current.parent == current:
            break
        current = current.parent
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.extracting-",
            dir=parent,
        )
    )
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    try:
        for artifact in artifacts:
            assert isinstance(artifact, dict)
            parts = _artifact_part_paths(package_root, artifact)
            extraction_root = _safe_output_path(
                staging, str(artifact["extraction_root"])
            )
            if artifact["format"] == "tar.zst":
                _extract_tar_zst(
                    parts,
                    staging,
                    extraction_root=PurePosixPath(
                        str(artifact["extraction_root"])
                    ),
                    expected_file_count=int(artifact["file_count"]),
                    expected_unpacked_bytes=int(artifact["unpacked_bytes"]),
                )
            elif artifact["format"] == "file":
                if extraction_root.exists():
                    raise PackageError(
                        f"artifact extraction would overwrite: {extraction_root}"
                    )
                extraction_root.parent.mkdir(parents=True, exist_ok=True)
                with extraction_root.open("xb") as output:
                    for part in parts:
                        with part.open("rb") as source:
                            shutil.copyfileobj(source, output)
            else:
                raise PackageError(f"unknown artifact format: {artifact['format']}")
            count, size = _count_root(extraction_root)
            if count != artifact["file_count"] or size != artifact["unpacked_bytes"]:
                raise PackageError(
                    f"extracted content mismatch for {artifact.get('name')}"
                )
        extraction_receipt: dict[str, object] = {
            "format_version": FORMAT_VERSION,
            "package_id": manifest["package_id"],
            "manifest_sha256": _hash_file(package_root / "manifest.json"),
        }
        extracted_wheelhouse = staging / "environment/wheelhouse"
        if manifest.get("package_type") == "h100-source-handoff":
            if not extracted_wheelhouse.is_dir():
                raise PackageError(
                    "source handoff extraction lacks environment/wheelhouse"
                )
            try:
                extraction_receipt["wheelhouse"] = wheelhouse_identity(
                    extracted_wheelhouse
                )
            except RuntimeError as exc:
                raise PackageError(
                    f"extracted wheelhouse identity is invalid: {exc}"
                ) from exc
        _write_bytes(
            staging / "HANDOFF_EXTRACTED.json",
            _canonical_json(extraction_receipt),
        )
        os.replace(staging, destination)
    except BaseException:
        # Only the builder-owned sibling stage is ever removed.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def extract_package(package_root: Path, destination: Path) -> Path:
    """Target-facing atomic extraction: production packages only."""

    return _extract_package(
        package_root,
        destination,
        require_production=True,
    )


def _extract_fixture_package(package_root: Path, destination: Path) -> Path:
    """Internal-only atomic extraction for deterministic fixture packages."""

    return _extract_package(
        package_root,
        destination,
        require_production=False,
    )
