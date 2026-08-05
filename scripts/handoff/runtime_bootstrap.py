"""Generate a standalone, hash-pinned Judy runtime bootstrap.

The generated shell file has no repository imports.  It needs only Bash,
Git, and the isolated boxsdk-v4 transfer Python already installed on Judy.
It downloads one dedicated runtime-amendment Box folder through ``.partial``
files, verifies every byte, reconstructs the Git bundle, and atomically
publishes a new SHA-addressed checkout.  Existing package, bundle, or checkout
paths are never overwritten.
"""

from __future__ import annotations

import json
import os
import re
import stat
import textwrap
from pathlib import Path
from typing import Callable, Mapping

from .package import (
    PackageError,
    _absolute_path,
    _artifact_part_paths,
    _canonical_json,
    _hash_file,
    _inside_repository_worktrees,
    _load_json,
    _package_regular_files,
    _require_no_symlink_components,
)
from .runtime_amendment import prepare_runtime_verifier

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _bootstrap_spec(
    package_root: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    source = manifest.get("source")
    artifacts = manifest.get("artifacts")
    if (
        not isinstance(source, Mapping)
        or not isinstance(artifacts, list)
        or len(artifacts) != 2
        or any(not isinstance(item, Mapping) for item in artifacts)
    ):
        raise PackageError("runtime amendment lacks its two bound artifacts")
    bundles = [item for item in artifacts if item.get("kind") == "git_bundle"]
    if len(bundles) != 1:
        raise PackageError("runtime amendment lacks one bound Git bundle")
    bundle_artifact = bundles[0]
    branch = str(source.get("branch", ""))
    commit = str(source.get("git_commit", ""))
    bundle_sha256 = str(source.get("git_bundle_sha256", ""))
    package_id = str(manifest.get("package_id", ""))
    if (
        HEX40.fullmatch(commit) is None
        or HEX64.fullmatch(bundle_sha256) is None
        or not branch
        or not package_id
    ):
        raise PackageError("runtime source/package identity is invalid")
    part_paths = _artifact_part_paths(package_root, bundle_artifact)
    parts = []
    for path in part_paths:
        parts.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha1": _hash_file(path, "sha1"),
                "sha256": _hash_file(path),
            }
        )
    files = {
        record["path"]: {
            "bytes": record["bytes"],
            "sha1": record["sha1"],
            "sha256": record["sha256"],
        }
        for record in parts
    }
    for artifact in artifacts:
        if artifact is bundle_artifact:
            continue
        for path in _artifact_part_paths(package_root, artifact):
            relative = path.relative_to(package_root).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha1": _hash_file(path, "sha1"),
                "sha256": _hash_file(path),
            }
    for relative in ("manifest.json", "SHA256SUMS", "READY.json"):
        path = _require_no_symlink_components(package_root / relative, leaf="file")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha1": _hash_file(path, "sha1"),
            "sha256": _hash_file(path),
        }
    if set(files) != _package_regular_files(package_root):
        raise PackageError("runtime bootstrap file inventory differs from package")
    return {
        "schema": 1,
        "package_id": package_id,
        "branch": branch,
        "commit": commit,
        "ready_sha256": files["READY.json"]["sha256"],
        "manifest_sha256": files["manifest.json"]["sha256"],
        "sha256sums_sha256": files["SHA256SUMS"]["sha256"],
        "bundle_sha256": bundle_sha256,
        "bundle_parts": parts,
        "files": files,
    }


_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail

: "${TRANSFER_PYTHON:?set TRANSFER_PYTHON to the boxsdk-v4 venv Python}"
: "${XVIEW3_TARGET_ROOT:?set XVIEW3_TARGET_ROOT to the handoff directory}"
: "${BOX_JWT_CONFIG:?set BOX_JWT_CONFIG to the mode-0600 JWT file}"
: "${BOX_FOLDER_ID:?set BOX_FOLDER_ID to the dedicated runtime folder}"

if [[ -n "${H100_BASE_PYTHON_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${H100_BASE_PYTHON_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

"$TRANSFER_PYTHON" -I - "$XVIEW3_TARGET_ROOT" "$BOX_JWT_CONFIG" "$BOX_FOLDER_ID" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

SPEC = json.loads(__SPEC_JSON__)


def fail(message: str) -> None:
    raise SystemExit(message)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def attr(item: object, name: str) -> object | None:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def object_type(item: object) -> str:
    return str(attr(item, "object_type") or attr(item, "type") or "")


def list_items(folder: object) -> list[object]:
    result = []
    offset = 0
    while True:
        batch = list(folder.get_items(
            limit=1000, offset=offset,
            fields=["id", "name", "type", "sha1", "size"],
        ))
        result.extend(batch)
        if len(batch) < 1000:
            return result
        offset += 1000


def remote_files(client: object, folder_id: str) -> dict[str, object]:
    result = {}

    def walk(current_id: str, prefix: PurePosixPath | None) -> None:
        for item in list_items(client.folder(current_id)):
            name = str(attr(item, "name") or "")
            if not name or "/" in name or name in {".", ".."}:
                fail("unsafe item name in dedicated Box runtime folder")
            relative = PurePosixPath(name) if prefix is None else prefix / name
            kind = object_type(item)
            if kind == "folder":
                walk(str(attr(item, "id")), relative)
            elif kind == "file":
                key = relative.as_posix()
                if key in result:
                    fail("duplicate path in dedicated Box runtime folder")
                result[key] = item
            else:
                fail("unsupported item in dedicated Box runtime folder")

    walk(folder_id, None)
    return result


target = Path(os.path.abspath(sys.argv[1]))
jwt = Path(os.path.abspath(sys.argv[2]))
folder_id = sys.argv[3]
if not target.is_dir() or target.is_symlink():
    fail("XVIEW3_TARGET_ROOT must be an existing non-symlink directory")
if not jwt.is_file() or jwt.is_symlink() or stat.S_IMODE(jwt.stat().st_mode) != 0o600:
    fail("BOX_JWT_CONFIG must be a regular non-symlink mode-0600 file")

try:
    from boxsdk import Client, JWTAuth
    from boxsdk.session.session import AuthorizedSession, Session
except ImportError as exc:
    fail(f"boxsdk-v4 transfer environment is unavailable: {exc}")

request_kwargs = {"timeout": (30, 300)}
auth = JWTAuth.from_settings_file(
    str(jwt), session=Session(default_network_request_kwargs=request_kwargs)
)
auth.authenticate_instance()
client = Client(
    auth,
    session=AuthorizedSession(
        auth, default_network_request_kwargs=request_kwargs
    ),
)
remote = remote_files(client, folder_id)
expected = SPEC["files"]
if set(remote) != set(expected):
    fail("dedicated Box runtime folder tree differs from the pinned package")
for name, record in expected.items():
    item = remote[name]
    if (
        int(attr(item, "size") or -1) != record["bytes"]
        or str(attr(item, "sha1") or "").lower() != record["sha1"]
    ):
        fail(f"Box SHA-1/size mismatch before download: {name}")

package = target / SPEC["package_id"]
bundle = target / f"xview3-runtime-{SPEC['commit']}.bootstrap.bundle"
checkout = target / f"bootstrap-{SPEC['commit']}"
for path in (package, bundle, checkout):
    if os.path.lexists(path):
        fail(f"refusing to overwrite existing path: {path}")

staging = Path(tempfile.mkdtemp(
    prefix=f".{SPEC['package_id']}.downloading-", dir=target
))
try:
    for name in sorted(expected):
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            fail(f"unsafe pinned package path: {name}")
        destination = staging.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        with partial.open("xb") as handle:
            client.file(str(attr(remote[name], "id"))).download_to(handle)
        record = expected[name]
        if (
            partial.stat().st_size != record["bytes"]
            or digest(partial, "sha1") != record["sha1"]
            or digest(partial, "sha256") != record["sha256"]
        ):
            fail(f"downloaded runtime file failed hash validation: {name}")
        os.replace(partial, destination)
    ready = json.loads((staging / "READY.json").read_text(encoding="utf-8"))
    manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
    if (
        ready.get("status") != "READY"
        or ready.get("package_id") != SPEC["package_id"]
        or manifest.get("package_id") != SPEC["package_id"]
        or digest(staging / "READY.json", "sha256") != SPEC["ready_sha256"]
        or digest(staging / "manifest.json", "sha256") != SPEC["manifest_sha256"]
        or digest(staging / "SHA256SUMS", "sha256") != SPEC["sha256sums_sha256"]
    ):
        fail("runtime package control identity mismatch")
    os.rename(staging, package)
except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise

bundle_partial = bundle.with_name(bundle.name + ".partial")
try:
    with bundle_partial.open("xb") as output:
        for part in SPEC["bundle_parts"]:
            with (package / part["path"]).open("rb") as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    if digest(bundle_partial, "sha256") != SPEC["bundle_sha256"]:
        fail("reconstructed runtime bundle SHA-256 mismatch")
    os.replace(bundle_partial, bundle)
except BaseException:
    bundle_partial.unlink(missing_ok=True)
    raise

clone_stage = target / f".bootstrap-{SPEC['commit']}.cloning-{os.getpid()}"
if os.path.lexists(clone_stage):
    fail(f"temporary clone path already exists: {clone_stage}")
try:
    subprocess.run(
        ["git", "clone", "--single-branch", "--branch", SPEC["branch"],
         str(bundle), str(clone_stage)],
        check=True,
    )
    git = ["git", "-c", f"safe.directory={clone_stage}", "-C", str(clone_stage)]
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    branch = subprocess.run(
        [*git, "branch", "--show-current"], check=True, text=True,
        capture_output=True,
    ).stdout.strip()
    dirty = subprocess.run(
        [*git, "status", "--porcelain=v1", "--untracked-files=all"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    subprocess.run([*git, "bundle", "verify", str(bundle)], check=True)
    if head != SPEC["commit"] or branch != SPEC["branch"] or dirty:
        fail("runtime bundle did not reproduce the exact clean branch/commit")
    os.rename(clone_stage, checkout)
except BaseException:
    shutil.rmtree(clone_stage, ignore_errors=True)
    raise

print(json.dumps({
    "status": "runtime-bootstrap-complete",
    "package_root": str(package),
    "bundle": str(bundle),
    "checkout": str(checkout),
    "branch": SPEC["branch"],
    "commit": SPEC["commit"],
    "bundle_sha256": SPEC["bundle_sha256"],
}, indent=2, sort_keys=True))
PY
'''


def _generate_runtime_bootstrap(
    *,
    repo_root: Path,
    runtime_package_root: Path,
    output: Path,
    verifier: Callable[[Path], Mapping[str, object]],
    production: bool,
) -> dict[str, object]:
    repo = _require_no_symlink_components(repo_root, leaf="directory")
    package = _require_no_symlink_components(
        runtime_package_root, leaf="directory"
    )
    destination = _absolute_path(output)
    if os.path.lexists(destination):
        raise PackageError(f"runtime bootstrap output exists: {destination}")
    parent = _require_no_symlink_components(destination.parent, leaf="directory")
    if production and _inside_repository_worktrees(destination, repo):
        raise PackageError("runtime bootstrap output must be outside the repository")
    manifest = dict(verifier(package))
    spec = _bootstrap_spec(package, manifest)
    encoded = repr(json.dumps(spec, sort_keys=True, separators=(",", ":")))
    content = textwrap.dedent(_TEMPLATE).replace("__SPEC_JSON__", encoded)
    if "__SPEC_JSON__" in content:
        raise PackageError("runtime bootstrap template substitution failed")
    try:
        with destination.open("xb") as handle:
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        destination.chmod(0o700)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if stat.S_IMODE(destination.stat().st_mode) != 0o700:
        raise PackageError("runtime bootstrap mode is not 0700")
    return {
        "path": str(destination),
        "sha256": _hash_file(destination),
        "bytes": destination.stat().st_size,
        "package_id": spec["package_id"],
        "branch": spec["branch"],
        "commit": spec["commit"],
        "bundle_sha256": spec["bundle_sha256"],
        "ready_sha256": spec["ready_sha256"],
        "manifest_sha256": spec["manifest_sha256"],
        "sha256sums_sha256": spec["sha256sums_sha256"],
    }


def generate_runtime_bootstrap(
    *,
    repo_root: Path,
    base_package_root: Path,
    runtime_package_root: Path,
    output: Path,
) -> dict[str, object]:
    """Verify the production amendment, then create its standalone puller."""

    return _generate_runtime_bootstrap(
        repo_root=repo_root,
        runtime_package_root=runtime_package_root,
        output=output,
        verifier=prepare_runtime_verifier(base_package_root),
        production=True,
    )
