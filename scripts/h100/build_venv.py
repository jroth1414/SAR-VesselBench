"""Build and verify the immutable native Python environment for Judy H100 jobs.

The environment is created at its final path because Python virtual
environments are not relocatable.  Package resolution and installation are
strictly offline from the verified Sprint 7d wheelhouse.  A content receipt is
written only after the environment has been made read-only and re-verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from packaging.utils import canonicalize_name

from scripts.h100.contracts import atomic_write_json, atomic_write_text, sha256_file
from scripts.h100.wheelhouse import (
    assert_wheelhouse_unchanged,
    validate_base_extraction_receipt,
    wheelhouse_identity,
    wheelhouse_manifest,
)

EXPECTED_PYTHON_VERSION = "3.11.15"
RECEIPT_SCHEMA = 1
RECEIPT_KIND = "xview3-h100-native-venv"
TREE_DIGEST_ALGORITHM = "xview3-venv-tree-v1"
BASE_RUNTIME_DIGEST_ALGORITHM = "xview3-base-python-runtime-v1"
BASE_RUNTIME_EXCLUDED_DIRECTORIES = frozenset(
    {"__pycache__", "site-packages", "dist-packages"}
)
BASE_RUNTIME_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})
BASE_RUNTIME_PROBE_MODULES = (
    "_bz2",
    "_ctypes",
    "_decimal",
    "_hashlib",
    "_lzma",
    "_sqlite3",
    "_ssl",
    "readline",
    "uuid",
    "zlib",
)
BOOTSTRAP_PACKAGES = frozenset({"pip", "setuptools", "wheel"})
IMMUTABILITY_CONTRACT = {
    "directory_mode": "0555",
    "executable_file_mode": "0555",
    "regular_file_mode": "0444",
    "bytecode_removed": True,
}


def _sidecar(venv_root: Path, suffix: str) -> Path:
    return Path(f"{venv_root}{suffix}")


def checksum_path(venv_root: str | Path) -> Path:
    return _sidecar(Path(venv_root), ".sha256")


def receipt_path(venv_root: str | Path) -> Path:
    return _sidecar(Path(venv_root), ".build.json")


def _offline_env() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PIP_") and key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    environment.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _run(
    command: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=capture_output,
        env=_offline_env(),
    )


def _require_regular_executable(path: Path, label: str) -> Path:
    requested = path.expanduser()
    try:
        resolved = requested.resolve(strict=True)
        mode = resolved.lstat().st_mode
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"{label} does not exist: {requested}") from exc
    if not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"{label} must resolve to a regular executable: {requested}")
    return resolved


def _runtime_file_record(path: Path, *, identity: str) -> dict[str, object]:
    requested = path.absolute()
    try:
        metadata = requested.lstat()
        resolved = requested.resolve(strict=True)
        resolved_metadata = resolved.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"base runtime component is absent: {requested}") from exc
    if not stat.S_ISREG(resolved_metadata.st_mode):
        raise RuntimeError(f"base runtime component is not a regular file: {requested}")
    record: dict[str, object] = {
        "identity": identity,
        "path": str(requested),
        "resolved_path": str(resolved),
        "bytes": resolved_metadata.st_size,
        "sha256": sha256_file(resolved),
    }
    if stat.S_ISLNK(metadata.st_mode):
        record["symlink_target"] = os.readlink(requested)
    elif not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError(f"base runtime component has unsupported type: {requested}")
    return record


def _runtime_tree_records(
    root: Path,
    *,
    identity: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    requested = root.absolute()
    try:
        resolved = requested.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"base Python runtime tree is absent: {requested}") from exc
    if not resolved.is_dir():
        raise RuntimeError(f"base Python runtime tree is not a directory: {requested}")
    records: list[dict[str, object]] = []
    counts = {"files": 0, "symlinks": 0, "directories": 1, "bytes": 0}

    def visit(
        directory: Path,
        relative_root: Path,
        ancestors: frozenset[tuple[int, int]],
    ) -> None:
        directory_metadata = directory.stat()
        inode = (directory_metadata.st_dev, directory_metadata.st_ino)
        if inode in ancestors:
            raise RuntimeError(
                f"base Python runtime tree contains a symlink cycle: {directory}"
            )
        current_ancestors = ancestors | {inode}
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                if entry.name in BASE_RUNTIME_EXCLUDED_DIRECTORIES:
                    continue
                source = Path(entry.path)
                relative = relative_root / entry.name
                if source.suffix.lower() in BASE_RUNTIME_EXCLUDED_SUFFIXES:
                    continue
                metadata = entry.stat(follow_symlinks=False)
                record_identity = f"{identity}/{relative.as_posix()}"
                if stat.S_ISLNK(metadata.st_mode):
                    target = os.readlink(source)
                    try:
                        resolved_target = source.resolve(strict=True)
                    except (FileNotFoundError, OSError) as exc:
                        raise RuntimeError(
                            "base Python runtime tree has a dangling symlink: "
                            f"{source}"
                        ) from exc
                    counts["symlinks"] += 1
                    record: dict[str, object] = {
                        "identity": record_identity,
                        "type": "symlink",
                        "target": target,
                        "resolved_path": str(resolved_target),
                    }
                    if resolved_target.is_dir():
                        records.append(record)
                        counts["directories"] += 1
                        visit(resolved_target, relative, current_ancestors)
                        continue
                    resolved_metadata = resolved_target.lstat()
                    if not stat.S_ISREG(resolved_metadata.st_mode):
                        raise RuntimeError(
                            "base Python runtime symlink has unsupported target: "
                            f"{source}"
                        )
                    record.update(
                        {
                            "bytes": resolved_metadata.st_size,
                            "sha256": sha256_file(resolved_target),
                        }
                    )
                    counts["bytes"] += resolved_metadata.st_size
                    records.append(record)
                    continue
                if stat.S_ISDIR(metadata.st_mode):
                    counts["directories"] += 1
                    records.append(
                        {"identity": record_identity, "type": "directory"}
                    )
                    visit(source, relative, current_ancestors)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise RuntimeError(
                        f"base Python runtime tree has unsupported entry: {source}"
                    )
                counts["files"] += 1
                counts["bytes"] += metadata.st_size
                records.append(
                    {
                        "identity": record_identity,
                        "type": "file",
                        "bytes": metadata.st_size,
                        "sha256": sha256_file(source),
                    }
                )

    visit(resolved, Path(), frozenset())
    root_binding = {
        "identity": identity,
        "requested_path": str(requested),
        "resolved_path": str(resolved),
    }
    return records, {**root_binding, **counts}


def base_python_runtime_manifest(metadata: Mapping[str, object]) -> dict[str, object]:
    resolved_executable = Path(str(metadata.get("resolved_path", "")))
    roots: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    seen_roots: set[Path] = set()
    for label, raw_root in (
        ("stdlib", metadata.get("stdlib")),
        ("platstdlib", metadata.get("platstdlib")),
    ):
        if not isinstance(raw_root, str) or not raw_root:
            raise RuntimeError(f"Python provenance probe lacks {label}")
        resolved_root = Path(raw_root).resolve(strict=True)
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)
        tree_records, root_summary = _runtime_tree_records(
            Path(raw_root), identity=label
        )
        roots.append(root_summary)
        records.extend(tree_records)

    standalone: list[dict[str, object]] = [
        _runtime_file_record(resolved_executable, identity="python-executable")
    ]
    config = metadata.get("config_vars")
    if not isinstance(config, Mapping):
        raise RuntimeError("Python provenance probe lacks runtime config variables")
    libdirs = [
        Path(str(value))
        for key in ("LIBDIR", "LIBPL")
        if (value := config.get(key))
    ]
    libpython_paths: list[str] = []
    seen_libpython_candidates: set[Path] = set()
    seen_components = {Path(str(standalone[0]["resolved_path"]))}
    for key in ("LDLIBRARY", "LIBRARY", "INSTSONAME"):
        value = config.get(key)
        if not isinstance(value, str) or not value:
            continue
        candidates = (
            [Path(value)]
            if Path(value).is_absolute()
            else [root / value for root in libdirs]
        )
        for candidate in candidates:
            requested_candidate = candidate.absolute()
            if (
                requested_candidate in seen_libpython_candidates
                or not candidate.exists()
            ):
                continue
            seen_libpython_candidates.add(requested_candidate)
            record = _runtime_file_record(candidate, identity=f"libpython:{key}")
            resolved = Path(str(record["resolved_path"]))
            seen_components.add(resolved)
            standalone.append(record)
            libpython_paths.append(str(requested_candidate))

    mapped = metadata.get("mapped_runtime_files")
    if not isinstance(mapped, list) or any(not isinstance(item, str) for item in mapped):
        raise RuntimeError("Python provenance probe lacks mapped runtime files")
    mapped_paths: list[str] = []
    for raw_path in sorted(set(mapped)):
        record = _runtime_file_record(
            Path(raw_path), identity="mapped-runtime-library"
        )
        resolved = Path(str(record["resolved_path"]))
        mapped_paths.append(str(resolved))
        if resolved in seen_components:
            continue
        seen_components.add(resolved)
        standalone.append(record)

    records.extend(
        sorted(
            standalone,
            key=lambda item: (str(item["identity"]), str(item["path"])),
        )
    )
    encoded = json.dumps(
        {
            "algorithm": BASE_RUNTIME_DIGEST_ALGORITHM,
            "excluded_directories": sorted(BASE_RUNTIME_EXCLUDED_DIRECTORIES),
            "excluded_suffixes": sorted(BASE_RUNTIME_EXCLUDED_SUFFIXES),
            "roots": roots,
            "records": records,
            "probed_modules": metadata.get("probed_modules"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "algorithm": BASE_RUNTIME_DIGEST_ALGORITHM,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": sum(int(root["files"]) for root in roots) + len(standalone),
        "symlinks": sum(int(root["symlinks"]) for root in roots),
        "directories": sum(int(root["directories"]) for root in roots),
        "bytes": sum(int(root["bytes"]) for root in roots)
        + sum(int(item["bytes"]) for item in standalone),
        "roots": roots,
        "libpython_paths": sorted(set(libpython_paths)),
        "mapped_runtime_files": sorted(set(mapped_paths)),
        "probed_modules": metadata.get("probed_modules"),
        "excluded_directories": sorted(BASE_RUNTIME_EXCLUDED_DIRECTORIES),
        "excluded_suffixes": sorted(BASE_RUNTIME_EXCLUDED_SUFFIXES),
    }


def inspect_python(path: str | Path) -> dict[str, object]:
    requested = Path(path).expanduser()
    resolved = _require_regular_executable(requested, "Python interpreter")
    probe = """
import importlib
import json
import platform
import sys
import sysconfig

probe_modules = %r
probed_modules = {}
for name in probe_modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        probed_modules[name] = type(exc).__name__
    else:
        probed_modules[name] = "loaded"
mapped_runtime_files = set()
with open("/proc/self/maps", encoding="utf-8") as maps:
    for line in maps:
        fields = line.rstrip().split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            mapped_runtime_files.add(fields[5])
config_keys = ("LIBDIR", "LIBPL", "LDLIBRARY", "LIBRARY", "INSTSONAME")
print(json.dumps({
    "version": platform.python_version(),
    "implementation": sys.implementation.name,
    "implementation_version": list(sys.implementation.version),
    "soabi": sysconfig.get_config_var("SOABI"),
    "platform": platform.platform(),
    "libc": list(platform.libc_ver()),
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "stdlib": sysconfig.get_path("stdlib"),
    "platstdlib": sysconfig.get_path("platstdlib"),
    "executable": sys.executable,
    "config_vars": {key: sysconfig.get_config_var(key) for key in config_keys},
    "mapped_runtime_files": sorted(mapped_runtime_files),
    "probed_modules": probed_modules,
}, sort_keys=True))
""" % (BASE_RUNTIME_PROBE_MODULES,)
    completed = _run([str(resolved), "-I", "-c", probe], capture_output=True)
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Python provenance probe returned invalid JSON: {resolved}"
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Python provenance probe returned a non-object: {resolved}")
    probed_modules = metadata.get("probed_modules")
    expected_modules = set(BASE_RUNTIME_PROBE_MODULES)
    if (
        not isinstance(probed_modules, dict)
        or set(probed_modules) != expected_modules
        or set(probed_modules.values()) != {"loaded"}
    ):
        raise RuntimeError(
            "base Python failed required runtime-library probes: "
            f"{probed_modules}"
        )
    metadata.update(
        {
            "requested_path": str(requested.absolute()),
            "resolved_path": str(resolved),
            "executable_sha256": sha256_file(resolved),
        }
    )
    metadata["runtime"] = base_python_runtime_manifest(metadata)
    return metadata


def _exact_pins(text: str, *, source: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise RuntimeError(f"{source} contains a non-exact pin: {line}")
        name, version = line.split("==", 1)
        canonical = canonicalize_name(name.strip())
        version = version.strip()
        if not canonical or not version:
            raise RuntimeError(f"{source} contains an invalid exact pin: {line}")
        if canonical in pins:
            raise RuntimeError(f"{source} contains a duplicate pin: {canonical}")
        pins[canonical] = version
    if not pins:
        raise RuntimeError(f"{source} contains no package pins")
    return pins


def assert_freeze_matches_lock(lock_text: str, freeze_text: str) -> dict[str, str]:
    locked = _exact_pins(lock_text, source="environment lock")
    installed = _exact_pins(freeze_text, source="venv freeze")
    mismatches = {
        name: (version, installed.get(name))
        for name, version in locked.items()
        if installed.get(name) != version
    }
    if mismatches:
        first = dict(list(mismatches.items())[:8])
        raise RuntimeError(
            f"native venv freeze does not match exact environment lock: {first}"
        )
    extras = set(installed) - set(locked) - BOOTSTRAP_PACKAGES
    if extras:
        raise RuntimeError(
            f"native venv freeze has unexpected packages: {sorted(extras)}"
        )
    return {name: installed[name] for name in sorted(locked)}


def _walk_tree(root: Path) -> list[tuple[str, Path, os.stat_result]]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"venv root must be a real directory: {root}")
    found: list[tuple[str, Path, os.stat_result]] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                metadata = entry.stat(follow_symlinks=False)
                found.append((relative, path, metadata))
                if stat.S_ISDIR(metadata.st_mode):
                    visit(path)

    visit(root)
    return sorted(found, key=lambda item: item[0])


def tree_manifest(root: str | Path) -> dict[str, object]:
    base = Path(root)
    root_mode = stat.S_IMODE(base.lstat().st_mode)
    records: list[dict[str, object]] = [
        {"path": ".", "type": "directory", "mode": root_mode}
    ]
    files = symlinks = byte_count = 0
    directories = 1
    resolved_root = base.resolve(strict=True)
    for relative, path, metadata in _walk_tree(base):
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISREG(metadata.st_mode):
            files += 1
            byte_count += metadata.st_size
            record: dict[str, object] = {
                "path": relative,
                "type": "file",
                "mode": mode,
                "bytes": metadata.st_size,
                "sha256": sha256_file(path),
            }
        elif stat.S_ISLNK(metadata.st_mode):
            symlinks += 1
            target = os.readlink(path)
            target_path = Path(target)
            if target_path.is_absolute() or ".." in target_path.parts:
                raise RuntimeError(
                    f"venv symlink target must be safe and relative: {path} -> {target}"
                )
            try:
                resolved_target = path.resolve(strict=True)
            except (FileNotFoundError, OSError) as exc:
                raise RuntimeError(f"venv contains a dangling symlink: {path}") from exc
            if not resolved_target.is_relative_to(resolved_root):
                raise RuntimeError(f"venv symlink escapes the sealed tree: {path}")
            record = {
                "path": relative,
                "type": "symlink",
                "mode": mode,
                "target": target,
            }
        elif stat.S_ISDIR(metadata.st_mode):
            directories += 1
            record = {"path": relative, "type": "directory", "mode": mode}
        else:
            raise RuntimeError(f"venv contains unsupported filesystem entry: {path}")
        records.append(record)
    encoded = json.dumps(
        {
            "algorithm": TREE_DIGEST_ALGORITHM,
            "entries": records,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "algorithm": TREE_DIGEST_ALGORITHM,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
        "symlinks": symlinks,
        "directories": directories,
        "bytes": byte_count,
    }


def _remove_bytecode(root: Path) -> None:
    for _relative, path, metadata in reversed(_walk_tree(root)):
        if stat.S_ISREG(metadata.st_mode) and path.suffix == ".pyc":
            path.unlink()
        elif stat.S_ISDIR(metadata.st_mode) and path.name == "__pycache__":
            try:
                path.rmdir()
            except OSError as exc:
                raise RuntimeError(f"non-bytecode content remained in {path}") from exc


def make_tree_readonly(root: str | Path) -> None:
    base = Path(root)
    entries = _walk_tree(base)
    for _relative, path, metadata in entries:
        if stat.S_ISREG(metadata.st_mode):
            executable = bool(stat.S_IMODE(metadata.st_mode) & 0o111)
            path.chmod(0o555 if executable else 0o444)
    for _relative, path, metadata in reversed(entries):
        if stat.S_ISDIR(metadata.st_mode):
            path.chmod(0o555)
    base.chmod(0o555)


def assert_tree_readonly(root: str | Path) -> None:
    base = Path(root)
    if stat.S_IMODE(base.lstat().st_mode) != 0o555:
        raise RuntimeError("native venv root must be mode 0555")
    for relative, path, metadata in _walk_tree(base):
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            if mode != 0o555:
                raise RuntimeError(f"native venv directory is not mode 0555: {relative}")
            if path.name == "__pycache__":
                raise RuntimeError(f"native venv retains bytecode directory: {relative}")
        elif stat.S_ISREG(metadata.st_mode):
            expected = 0o555 if mode & 0o111 else 0o444
            if mode != expected:
                raise RuntimeError(
                    f"native venv file mode mismatch: {relative} is {mode:04o}"
                )
            if path.suffix == ".pyc":
                raise RuntimeError(f"native venv retains bytecode file: {relative}")


def _remove_owned_tree(root: Path) -> None:
    if not root.exists() and not root.is_symlink():
        return
    if root.is_symlink() or not root.is_dir():
        root.unlink()
        return
    for _relative, path, metadata in _walk_tree(root):
        if stat.S_ISDIR(metadata.st_mode):
            path.chmod(0o700)
        elif stat.S_ISREG(metadata.st_mode):
            path.chmod(0o600)
    root.chmod(0o700)
    shutil.rmtree(root)


def _freeze(venv_python: Path) -> str:
    return _run(
        [str(venv_python), "-I", "-m", "pip", "freeze", "--all"],
        capture_output=True,
    ).stdout


def _pip_check(venv_python: Path) -> None:
    _run([str(venv_python), "-I", "-m", "pip", "check"], capture_output=True)


def _assert_python_version(metadata: Mapping[str, object], *, label: str) -> None:
    version = metadata.get("version")
    if version != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"{label} Python version mismatch: expected "
            f"{EXPECTED_PYTHON_VERSION}, got {version}"
        )


def build(
    *,
    repo: Path,
    wheelhouse: Path,
    output: Path,
    base_python: Path,
    base_extraction_receipt: Path,
    expected_base_payload_package_id: str,
    expected_base_payload_manifest_sha256: str,
) -> dict[str, object]:
    repo = repo.resolve()
    wheelhouse = wheelhouse.resolve()
    output = output.absolute()
    checksum = checksum_path(output)
    receipt = receipt_path(output)
    for path, label in (
        (output, "venv"),
        (checksum, "venv checksum"),
        (receipt, "venv build receipt"),
    ):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to overwrite existing {label}: {path}")

    base_python_metadata = inspect_python(base_python)
    _assert_python_version(base_python_metadata, label="base")
    if (
        base_python_metadata.get("prefix")
        != base_python_metadata.get("base_prefix")
    ):
        raise RuntimeError("base Python must not itself be a virtual environment")
    resolved_base = Path(str(base_python_metadata["resolved_path"]))

    env_lock = repo / "locks/env-v100node.txt"
    if env_lock.is_symlink() or not env_lock.is_file():
        raise RuntimeError(f"environment lock is absent or unsafe: {env_lock}")
    lock_text = env_lock.read_text()
    locked = _exact_pins(lock_text, source="environment lock")
    wheels = wheelhouse_manifest(wheelhouse)
    wheelhouse_tree = wheelhouse_identity(wheelhouse)
    base_extraction = validate_base_extraction_receipt(
        base_extraction_receipt,
        wheelhouse=wheelhouse,
        expected_package_id=expected_base_payload_package_id,
        expected_manifest_sha256=expected_base_payload_manifest_sha256,
        expected_wheelhouse_sha256=str(wheelhouse_tree["sha256"]),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    owns_output = True
    try:
        _run(
            [
                str(resolved_base),
                "-I",
                "-m",
                "venv",
                "--copies",
                str(output),
            ]
        )
        venv_python = output / "bin/python"
        resolved_venv_python = _require_regular_executable(
            venv_python, "venv Python interpreter"
        )
        if venv_python.is_symlink():
            raise RuntimeError("venv Python interpreter must be copied, not symlinked")
        with tempfile.NamedTemporaryFile(suffix=".json") as report:
            _run(
                [
                    str(resolved_venv_python),
                    "-I",
                    "-m",
                    "pip",
                    "install",
                    "--dry-run",
                    "--ignore-installed",
                    "--no-index",
                    f"--find-links={wheelhouse}",
                    "--only-binary=:all:",
                    "--requirement",
                    str(env_lock),
                    "--report",
                    report.name,
                ]
            )
        _run(
            [
                str(resolved_venv_python),
                "-I",
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-compile",
                "--no-index",
                f"--find-links={wheelhouse}",
                "--only-binary=:all:",
                "--requirement",
                str(env_lock),
            ]
        )
        _pip_check(resolved_venv_python)
        freeze = _freeze(resolved_venv_python)
        matched_freeze = assert_freeze_matches_lock(lock_text, freeze)
        runtime_python = inspect_python(resolved_venv_python)
        _assert_python_version(runtime_python, label="venv")
        if (
            runtime_python.get("base_prefix")
            != base_python_metadata.get("base_prefix")
        ):
            raise RuntimeError(
                "venv does not bind the inspected base Python installation"
            )
        assert_wheelhouse_unchanged(wheelhouse, wheels)
        if wheelhouse_identity(wheelhouse) != wheelhouse_tree:
            raise RuntimeError("wheelhouse identity changed while the native venv was building")
        if validate_base_extraction_receipt(
            base_extraction_receipt,
            wheelhouse=wheelhouse,
            expected_package_id=expected_base_payload_package_id,
            expected_manifest_sha256=expected_base_payload_manifest_sha256,
            expected_receipt_sha256=str(base_extraction["sha256"]),
            expected_wheelhouse_sha256=str(wheelhouse_tree["sha256"]),
        ) != base_extraction:
            raise RuntimeError("base extraction changed while the native venv was building")
        if inspect_python(base_python) != base_python_metadata:
            raise RuntimeError("base Python changed while the native venv was building")

        _remove_bytecode(output)
        make_tree_readonly(output)
        assert_tree_readonly(output)
        tree = tree_manifest(output)
        checksum_text = (
            f"{tree['sha256']}  {output.name}/  ({TREE_DIGEST_ALGORITHM})\n"
        )
        atomic_write_text(checksum, checksum_text)
        checksum.chmod(0o444)
        checksum_sha256 = sha256_file(checksum)
        payload: dict[str, object] = {
            "schema": RECEIPT_SCHEMA,
            "kind": RECEIPT_KIND,
            "status": "ready",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "base_python": base_python_metadata,
            "environment_lock": {
                "path": "locks/env-v100node.txt",
                "sha256": sha256_file(env_lock),
                "normalized": {name: locked[name] for name in sorted(locked)},
            },
            "wheelhouse": {
                "identity": wheelhouse_tree,
                "artifacts": wheels,
                "base_extraction": base_extraction,
                "reverified_after_build": True,
            },
            "venv": {
                "path": str(output),
                "tree": tree,
                "python": runtime_python,
                "normalized_environment": matched_freeze,
                "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
                "checksum": {
                    "path": str(checksum),
                    "sha256": checksum_sha256,
                    "mode": "0444",
                },
            },
            "immutability": dict(IMMUTABILITY_CONTRACT),
            "network_policy": "pip --no-index; packages installed only from wheelhouse",
            "build_mode": "python -m venv --copies at final path",
        }
        # READY-equivalent receipt: this must remain the final publication step.
        atomic_write_json(receipt, payload)
        receipt.chmod(0o444)
        owns_output = False
        return payload
    finally:
        if owns_output:
            receipt.unlink(missing_ok=True)
            checksum.unlink(missing_ok=True)
            _remove_owned_tree(output)


def verify(
    *,
    repo: Path | None = None,
    environment_lock: Path | None = None,
    build_receipt: Path | None = None,
    venv_root: Path,
    base_python: Path,
    wheelhouse: Path,
    base_extraction_receipt: Path,
    expected_venv_sha256: str,
    expected_receipt_sha256: str,
    expected_base_python_sha256: str,
    expected_base_python_runtime_sha256: str,
    expected_wheelhouse_sha256: str,
    expected_base_extraction_receipt_sha256: str,
    expected_base_payload_package_id: str,
    expected_base_payload_manifest_sha256: str,
) -> dict[str, object]:
    root = venv_root.absolute()
    receipt = (
        build_receipt.absolute()
        if build_receipt is not None
        else receipt_path(root)
    )
    checksum = checksum_path(root)
    if not receipt.is_file() or receipt.is_symlink():
        raise RuntimeError(f"native venv build receipt is absent or unsafe: {receipt}")
    if stat.S_IMODE(receipt.lstat().st_mode) != 0o444:
        raise RuntimeError("native venv build receipt must be mode 0444")
    if sha256_file(receipt) != expected_receipt_sha256:
        raise RuntimeError("native venv build receipt SHA-256 mismatch")
    try:
        payload = json.loads(receipt.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("native venv build receipt is unreadable") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != RECEIPT_SCHEMA
        or payload.get("kind") != RECEIPT_KIND
        or payload.get("status") != "ready"
    ):
        raise RuntimeError("native venv build receipt contract mismatch")

    if payload.get("immutability") != IMMUTABILITY_CONTRACT:
        raise RuntimeError("native venv immutability receipt mismatch")
    assert_tree_readonly(root)

    recorded_venv = payload.get("venv")
    if not isinstance(recorded_venv, dict) or recorded_venv.get("path") != str(root):
        raise RuntimeError("native venv receipt is bound to a different path")
    recorded_tree = recorded_venv.get("tree")
    actual_tree = tree_manifest(root)
    if (
        not isinstance(recorded_tree, dict)
        or recorded_tree != actual_tree
        or actual_tree.get("sha256") != expected_venv_sha256
    ):
        raise RuntimeError("native venv tree digest mismatch")

    if not checksum.is_file() or checksum.is_symlink():
        raise RuntimeError("native venv tree checksum is absent or unsafe")
    if stat.S_IMODE(checksum.lstat().st_mode) != 0o444:
        raise RuntimeError("native venv tree checksum must be mode 0444")
    checksum_binding = recorded_venv.get("checksum")
    expected_checksum_text = (
        f"{expected_venv_sha256}  {root.name}/  ({TREE_DIGEST_ALGORITHM})\n"
    )
    if (
        not isinstance(checksum_binding, dict)
        or checksum_binding.get("path") != str(checksum)
        or checksum_binding.get("sha256") != sha256_file(checksum)
        or checksum_binding.get("mode") != "0444"
        or checksum.read_text() != expected_checksum_text
    ):
        raise RuntimeError("native venv checksum binding mismatch")

    actual_base = inspect_python(base_python)
    _assert_python_version(actual_base, label="base")
    actual_base_runtime = actual_base.get("runtime")
    if (
        actual_base.get("executable_sha256") != expected_base_python_sha256
        or not isinstance(actual_base_runtime, Mapping)
        or actual_base_runtime.get("sha256")
        != expected_base_python_runtime_sha256
        or payload.get("base_python") != actual_base
    ):
        raise RuntimeError("native venv base Python provenance mismatch")

    if (repo is None) == (environment_lock is None):
        raise RuntimeError("verify requires exactly one of repo or environment lock")
    env_lock = (
        environment_lock.absolute()
        if environment_lock is not None
        else repo.resolve() / "locks/env-v100node.txt"
    )
    if env_lock.is_symlink() or not env_lock.is_file():
        raise RuntimeError(f"environment lock is absent or unsafe: {env_lock}")
    lock_binding = payload.get("environment_lock")
    lock_text = env_lock.read_text()
    locked = _exact_pins(lock_text, source="environment lock")
    expected_lock = {
        "path": "locks/env-v100node.txt",
        "sha256": sha256_file(env_lock),
        "normalized": {name: locked[name] for name in sorted(locked)},
    }
    if lock_binding != expected_lock:
        raise RuntimeError("native venv environment-lock binding mismatch")

    wheelhouse = wheelhouse.resolve()
    actual_wheels = wheelhouse_manifest(wheelhouse)
    actual_wheelhouse_tree = wheelhouse_identity(wheelhouse)
    base_extraction = validate_base_extraction_receipt(
        base_extraction_receipt,
        wheelhouse=wheelhouse,
        expected_package_id=expected_base_payload_package_id,
        expected_manifest_sha256=expected_base_payload_manifest_sha256,
        expected_receipt_sha256=expected_base_extraction_receipt_sha256,
        expected_wheelhouse_sha256=expected_wheelhouse_sha256,
    )
    expected_wheelhouse = {
        "identity": actual_wheelhouse_tree,
        "artifacts": actual_wheels,
        "base_extraction": base_extraction,
        "reverified_after_build": True,
    }
    if (
        actual_wheelhouse_tree.get("sha256") != expected_wheelhouse_sha256
        or payload.get("wheelhouse") != expected_wheelhouse
    ):
        raise RuntimeError("native venv wheelhouse provenance mismatch")

    venv_python = _require_regular_executable(
        root / "bin/python", "venv Python interpreter"
    )
    if (root / "bin/python").is_symlink():
        raise RuntimeError("venv Python interpreter must be copied, not symlinked")
    runtime_python = inspect_python(venv_python)
    _assert_python_version(runtime_python, label="venv")
    if recorded_venv.get("python") != runtime_python:
        raise RuntimeError("native venv Python provenance mismatch")
    _pip_check(venv_python)
    freeze = _freeze(venv_python)
    normalized = assert_freeze_matches_lock(lock_text, freeze)
    if (
        recorded_venv.get("normalized_environment") != normalized
        or recorded_venv.get("pip_freeze_sha256")
        != hashlib.sha256(freeze.encode()).hexdigest()
    ):
        raise RuntimeError("native venv installed-environment binding mismatch")
    if tree_manifest(root) != actual_tree:
        raise RuntimeError("native venv changed while it was being verified")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    builder = commands.add_parser("build", help="build and seal the native venv")
    builder.add_argument("--repo", type=Path, default=Path.cwd())
    builder.add_argument("--wheelhouse", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--base-python", type=Path, required=True)
    builder.add_argument("--base-extraction-receipt", type=Path, required=True)
    builder.add_argument("--expected-base-payload-package-id", required=True)
    builder.add_argument("--expected-base-payload-manifest-sha256", required=True)

    verifier = commands.add_parser(
        "verify", help="verify the complete native venv receipt before a job"
    )
    lock_source = verifier.add_mutually_exclusive_group(required=True)
    lock_source.add_argument("--repo", type=Path)
    lock_source.add_argument("--lock", type=Path, dest="environment_lock")
    verifier.add_argument("--venv-root", type=Path, required=True)
    verifier.add_argument("--receipt", type=Path, dest="build_receipt")
    verifier.add_argument("--base-python", type=Path, required=True)
    verifier.add_argument("--wheelhouse", type=Path, required=True)
    verifier.add_argument("--base-extraction-receipt", type=Path, required=True)
    verifier.add_argument("--expected-venv-sha256", required=True)
    verifier.add_argument("--expected-receipt-sha256", required=True)
    verifier.add_argument("--expected-base-python-sha256", required=True)
    verifier.add_argument("--expected-base-python-runtime-sha256", required=True)
    verifier.add_argument("--expected-wheelhouse-sha256", required=True)
    verifier.add_argument("--expected-base-extraction-receipt-sha256", required=True)
    verifier.add_argument("--expected-base-payload-package-id", required=True)
    verifier.add_argument("--expected-base-payload-manifest-sha256", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "build":
        payload = build(
            repo=args.repo,
            wheelhouse=args.wheelhouse,
            output=args.output,
            base_python=args.base_python,
            base_extraction_receipt=args.base_extraction_receipt,
            expected_base_payload_package_id=args.expected_base_payload_package_id,
            expected_base_payload_manifest_sha256=(
                args.expected_base_payload_manifest_sha256
            ),
        )
    else:
        payload = verify(
            repo=args.repo,
            environment_lock=args.environment_lock,
            build_receipt=args.build_receipt,
            venv_root=args.venv_root,
            base_python=args.base_python,
            wheelhouse=args.wheelhouse,
            base_extraction_receipt=args.base_extraction_receipt,
            expected_venv_sha256=args.expected_venv_sha256,
            expected_receipt_sha256=args.expected_receipt_sha256,
            expected_base_python_sha256=args.expected_base_python_sha256,
            expected_base_python_runtime_sha256=(
                args.expected_base_python_runtime_sha256
            ),
            expected_wheelhouse_sha256=args.expected_wheelhouse_sha256,
            expected_base_extraction_receipt_sha256=(
                args.expected_base_extraction_receipt_sha256
            ),
            expected_base_payload_package_id=(
                args.expected_base_payload_package_id
            ),
            expected_base_payload_manifest_sha256=(
                args.expected_base_payload_manifest_sha256
            ),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
