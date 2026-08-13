"""Generate a standalone, hash-pinned Judy final-evaluation bootstrap.

The generated shell artifact has no repository imports.  It requires only
Bash, Git, and Judy's isolated boxsdk-v4 transfer Python.  It downloads the
exact contents of one dedicated final-evaluation Box folder, validates every
physical file against embedded SHA-1/SHA-256/size pins, reconstructs a
possibly multipart Git bundle, and atomically publishes both the untouched
package and an exact clean source checkout.

The bootstrap never parses ``validation.csv`` and never decodes a raster.  All
held-out payloads are handled as opaque byte streams until the separately
authorized final evaluator consumes them after its immutable lock.
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
    _hash_file,
    _inside_repository_worktrees,
    _package_regular_files,
    _require_no_symlink_components,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SHA_ADDRESS = re.compile(r"^[A-Za-z0-9._-]+-[0-9a-f]{40}-[0-9a-f]{64}$")


def _bootstrap_spec(
    package_root: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Derive the complete standalone trust specification from a package."""

    source = manifest.get("source")
    artifacts = manifest.get("artifacts")
    package_id = str(manifest.get("package_id", ""))
    package_type = str(manifest.get("package_type", ""))
    identity_sha256 = str(manifest.get("identity_sha256", ""))
    if (
        not isinstance(source, Mapping)
        or not isinstance(artifacts, list)
        or not artifacts
        or any(not isinstance(item, Mapping) for item in artifacts)
        or not package_type
        or SHA_ADDRESS.fullmatch(package_id) is None
        or HEX64.fullmatch(identity_sha256) is None
    ):
        raise PackageError("final-eval package identity is invalid")

    bundles = [item for item in artifacts if item.get("kind") == "git_bundle"]
    if len(bundles) != 1 or artifacts[0] is not bundles[0]:
        raise PackageError("final-eval package lacks one leading Git bundle")
    bundle_artifact = bundles[0]
    branch = str(source.get("branch", ""))
    commit = str(source.get("git_commit", ""))
    required_ancestor = str(source.get("required_campaign_commit", ""))
    bundle_ref = str(source.get("git_bundle_ref", ""))
    bundle_sha256 = str(source.get("git_bundle_sha256", ""))
    bundle_sha1 = str(bundle_artifact.get("archive_sha1", ""))
    bundle_bytes = bundle_artifact.get("archive_bytes")
    if (
        not branch
        or bundle_ref != f"refs/heads/{branch}"
        or HEX40.fullmatch(commit) is None
        or HEX40.fullmatch(required_ancestor) is None
        or HEX64.fullmatch(bundle_sha256) is None
        or re.fullmatch(r"[0-9a-f]{40}", bundle_sha1) is None
        or not isinstance(bundle_bytes, int)
        or isinstance(bundle_bytes, bool)
        or bundle_bytes <= 0
        or bundle_artifact.get("archive_sha256") != bundle_sha256
        or package_id
        != f"xview3-h100-final-eval-{commit}-{identity_sha256}"
    ):
        raise PackageError("final-eval Git source identity is invalid")

    bundle_parts: list[dict[str, object]] = []
    for path in _artifact_part_paths(package_root, bundle_artifact):
        bundle_parts.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha1": _hash_file(path, "sha1"),
                "sha256": _hash_file(path),
            }
        )

    payload_files: list[dict[str, object]] = []
    seen: set[str] = set()
    for artifact in artifacts:
        assert isinstance(artifact, Mapping)
        for path in _artifact_part_paths(package_root, artifact):
            relative = path.relative_to(package_root).as_posix()
            if relative in seen:
                raise PackageError("final-eval package repeats a physical artifact")
            seen.add(relative)
            payload_files.append(
                {
                    "path": relative,
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
        for record in payload_files
    }
    for relative in ("manifest.json", "SHA256SUMS", "READY.json"):
        path = _require_no_symlink_components(package_root / relative, leaf="file")
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha1": _hash_file(path, "sha1"),
            "sha256": _hash_file(path),
        }
    if set(files) != _package_regular_files(package_root):
        raise PackageError("final-eval bootstrap inventory differs from package")

    return {
        "schema": 1,
        "purpose": "owner-amended-once-only-all-32-final-evaluation",
        "package_type": package_type,
        "package_id": package_id,
        "identity_sha256": identity_sha256,
        "branch": branch,
        "commit": commit,
        "required_ancestor": required_ancestor,
        "bundle_ref": bundle_ref,
        "bundle_bytes": bundle_bytes,
        "bundle_sha1": bundle_sha1,
        "bundle_sha256": bundle_sha256,
        "bundle_parts": bundle_parts,
        "payload_files": payload_files,
        "ready_sha256": files["READY.json"]["sha256"],
        "manifest_sha256": files["manifest.json"]["sha256"],
        "sha256sums_sha256": files["SHA256SUMS"]["sha256"],
        "files": dict(sorted(files.items())),
    }


_TEMPLATE = r'''#!/usr/bin/env bash
set -euo pipefail

: "${TRANSFER_PYTHON:?set TRANSFER_PYTHON to the isolated Judy boxsdk-v4 venv Python}"
: "${XVIEW3_TARGET_ROOT:?set XVIEW3_TARGET_ROOT to an existing handoff directory}"
: "${BOX_JWT_CONFIG:?set BOX_JWT_CONFIG to the mode-0600 JWT file}"
: "${BOX_FOLDER_ID:?set BOX_FOLDER_ID to the dedicated final-eval folder}"

if [[ -n "${H100_BASE_PYTHON_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${H100_BASE_PYTHON_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

"$TRANSFER_PYTHON" -I - "$XVIEW3_TARGET_ROOT" "$BOX_JWT_CONFIG" "$BOX_FOLDER_ID" <<'PY'
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

SPEC = json.loads(__SPEC_JSON__)
CONTROL_FILES = ("READY.json", "manifest.json", "SHA256SUMS")


def fail(message: str) -> None:
    raise SystemExit(message)


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def safe_relative(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        not value
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
    ):
        fail(f"unsafe pinned package path: {value!r}")
    return relative


def require_no_symlink(path: Path, kind: str) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            info = current.lstat()
        except OSError:
            fail(f"required {kind} path is absent")
        if stat.S_ISLNK(info.st_mode):
            fail(f"required {kind} path has a symlink component")
    if kind == "directory" and not absolute.is_dir():
        fail("XVIEW3_TARGET_ROOT must be an existing directory")
    if kind == "file" and not absolute.is_file():
        fail("BOX_JWT_CONFIG must be a regular file")
    return absolute


def attr(item: object, name: str) -> object | None:
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def object_type(item: object) -> str:
    return str(attr(item, "object_type") or attr(item, "type") or "")


def list_items(folder: object) -> list[object]:
    result = []
    offset = 0
    while True:
        batch = list(folder.get_items(
            limit=1000,
            offset=offset,
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
            if not name or "/" in name or "\\" in name or name in {".", ".."}:
                fail("unsafe item name in dedicated final-eval Box folder")
            relative = PurePosixPath(name) if prefix is None else prefix / name
            kind = object_type(item)
            identifier = str(attr(item, "id") or "")
            if not identifier:
                fail("Box item has no identifier")
            if kind == "folder":
                walk(identifier, relative)
            elif kind == "file":
                key = relative.as_posix()
                safe_relative(key)
                if key in result:
                    fail("duplicate path in dedicated final-eval Box folder")
                result[key] = item
            else:
                fail("unsupported item in dedicated final-eval Box folder")

    walk(folder_id, None)
    return result


def run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        fail("final-eval Git bundle/checkout validation failed")


target = require_no_symlink(Path(sys.argv[1]), "directory")
jwt = require_no_symlink(Path(sys.argv[2]), "file")
folder_id = sys.argv[3]
if stat.S_IMODE(jwt.stat().st_mode) != 0o600:
    fail("BOX_JWT_CONFIG must have exact mode 0600")
if not folder_id.isdigit():
    fail("BOX_FOLDER_ID must be a numeric Box folder ID")

logging.getLogger("boxsdk").handlers[:] = [logging.NullHandler()]
logging.getLogger("boxsdk").propagate = False
logging.getLogger("boxsdk").setLevel(logging.CRITICAL + 1)
try:
    from boxsdk import Client, JWTAuth
    from boxsdk.session.session import AuthorizedSession, Session
except ImportError:
    fail("boxsdk-v4 transfer environment is unavailable")

try:
    request_kwargs = {"timeout": (30, 300)}
    auth = JWTAuth.from_settings_file(
        str(jwt),
        session=Session(default_network_request_kwargs=request_kwargs),
    )
    auth.authenticate_instance()
    client = Client(
        auth,
        session=AuthorizedSession(
            auth,
            default_network_request_kwargs=request_kwargs,
        ),
    )
    remote = remote_files(client, folder_id)
except SystemExit:
    raise
except Exception:
    fail("Box JWT authentication or folder inspection failed")

expected = SPEC["files"]
if set(remote) != set(expected):
    fail("dedicated final-eval Box folder tree differs from the pinned package")
for name, record in expected.items():
    item = remote[name]
    if (
        int(attr(item, "size") or -1) != record["bytes"]
        or str(attr(item, "sha1") or "").lower() != record["sha1"]
    ):
        fail(f"Box SHA-1/size mismatch before download: {name}")

bootstrap = target / f"xview3-final-eval-bootstrap-{SPEC['commit']}"
if os.path.lexists(bootstrap):
    fail(f"refusing to overwrite existing bootstrap: {bootstrap}")

staging = Path(tempfile.mkdtemp(
    prefix=f".{bootstrap.name}.downloading-",
    dir=target,
))
package = staging / SPEC["package_id"]
package.mkdir(mode=0o700)
try:
    ordered = list(CONTROL_FILES) + sorted(set(expected) - set(CONTROL_FILES))
    for name in ordered:
        relative = safe_relative(name)
        destination = package.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        try:
            with partial.open("xb") as handle:
                client.file(str(attr(remote[name], "id"))).download_to(handle)
        except Exception:
            partial.unlink(missing_ok=True)
            fail(f"Box download failed for pinned package path: {name}")
        record = expected[name]
        if (
            partial.stat().st_size != record["bytes"]
            or digest(partial, "sha1") != record["sha1"]
            or digest(partial, "sha256") != record["sha256"]
        ):
            fail(f"downloaded final-eval file failed hash validation: {name}")
        os.replace(partial, destination)

    try:
        ready = json.loads((package / "READY.json").read_text(encoding="utf-8"))
        manifest = json.loads(
            (package / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError):
        fail("final-eval package controls are invalid JSON")
    if (
        digest(package / "READY.json", "sha256") != SPEC["ready_sha256"]
        or digest(package / "manifest.json", "sha256") != SPEC["manifest_sha256"]
        or digest(package / "SHA256SUMS", "sha256") != SPEC["sha256sums_sha256"]
        or ready.get("status") != "READY"
        or ready.get("package_id") != SPEC["package_id"]
        or ready.get("git_commit") != SPEC["commit"]
        or ready.get("identity_sha256") != SPEC["identity_sha256"]
        or manifest.get("package_type") != SPEC["package_type"]
        or manifest.get("package_id") != SPEC["package_id"]
        or manifest.get("identity_sha256") != SPEC["identity_sha256"]
        or manifest.get("source", {}).get("branch") != SPEC["branch"]
        or manifest.get("source", {}).get("git_commit") != SPEC["commit"]
        or manifest.get("source", {}).get("required_campaign_commit")
        != SPEC["required_ancestor"]
        or manifest.get("source", {}).get("git_bundle_sha256")
        != SPEC["bundle_sha256"]
    ):
        fail("final-eval package control identity mismatch")
    if (package / "READY.json").read_bytes() != canonical_json(ready):
        fail("final-eval READY marker is not canonical JSON")
    if (package / "manifest.json").read_bytes() != canonical_json(manifest):
        fail("final-eval manifest is not canonical JSON")

    try:
        checksum_lines = (package / "SHA256SUMS").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeError):
        fail("final-eval SHA256SUMS is not valid text")
    expected_lines = [
        f"{record['sha256']}  {record['path']}"
        for record in SPEC["payload_files"]
    ]
    if checksum_lines != expected_lines:
        fail("final-eval SHA256SUMS inventory differs from embedded pins")

    bundle = staging / f"xview3-final-eval-{SPEC['commit']}.bootstrap.bundle"
    with bundle.open("xb") as output:
        for part in SPEC["bundle_parts"]:
            with (package / part["path"]).open("rb") as source:
                shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    if (
        bundle.stat().st_size != SPEC["bundle_bytes"]
        or digest(bundle, "sha1") != SPEC["bundle_sha1"]
        or digest(bundle, "sha256") != SPEC["bundle_sha256"]
    ):
        fail("reconstructed final-eval bundle identity mismatch")

    expected_head = f"{SPEC['commit']} {SPEC['bundle_ref']}"
    heads = run(["git", "bundle", "list-heads", str(bundle)]).stdout.splitlines()
    if heads != [expected_head]:
        fail("final-eval bundle exposes an unexpected ref set")
    checkout = staging / f"source-{SPEC['commit']}"
    run([
        "git",
        "clone",
        "--quiet",
        "--single-branch",
        "--branch",
        SPEC["branch"],
        str(bundle),
        str(checkout),
    ])
    git = ["git", "-c", f"safe.directory={checkout}", "-C", str(checkout)]
    run([*git, "bundle", "verify", str(bundle)])
    run([*git, "fsck", "--full", "--strict"])
    head = run([*git, "rev-parse", "HEAD"]).stdout.strip()
    branch = run([*git, "branch", "--show-current"]).stdout.strip()
    dirty = run([
        *git,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ]).stdout.strip()
    run([
        *git,
        "merge-base",
        "--is-ancestor",
        SPEC["required_ancestor"],
        SPEC["commit"],
    ])
    if head != SPEC["commit"] or branch != SPEC["branch"] or dirty:
        fail("final-eval bundle did not reproduce the exact clean branch/commit")
    run([*git, "remote", "remove", "origin"])
    if run([
        *git,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ]).stdout.strip():
        fail("final-eval checkout became dirty during bootstrap")

    os.rename(staging, bootstrap)
    directory = os.open(target, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
except BaseException:
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    raise

print(json.dumps({
    "status": "final-eval-bootstrap-complete",
    "bootstrap_root": str(bootstrap),
    "package_root": str(bootstrap / SPEC["package_id"]),
    "bundle": str(
        bootstrap / f"xview3-final-eval-{SPEC['commit']}.bootstrap.bundle"
    ),
    "checkout": str(bootstrap / f"source-{SPEC['commit']}"),
    "branch": SPEC["branch"],
    "commit": SPEC["commit"],
    "required_ancestor": SPEC["required_ancestor"],
    "package_id": SPEC["package_id"],
    "bundle_sha256": SPEC["bundle_sha256"],
    "ready_sha256": SPEC["ready_sha256"],
    "manifest_sha256": SPEC["manifest_sha256"],
    "sha256sums_sha256": SPEC["sha256sums_sha256"],
}, indent=2, sort_keys=True))
PY
'''


def _generate_final_eval_bootstrap(
    *,
    repo_root: Path,
    package_root: Path,
    output: Path,
    verifier: Callable[[Path], Mapping[str, object]],
    production: bool,
) -> dict[str, object]:
    repository = _require_no_symlink_components(repo_root, leaf="directory")
    package = _require_no_symlink_components(package_root, leaf="directory")
    destination = _absolute_path(output)
    if os.path.lexists(destination):
        raise PackageError(f"final-eval bootstrap output exists: {destination}")
    _require_no_symlink_components(destination.parent, leaf="directory")
    if production and _inside_repository_worktrees(destination, repository):
        raise PackageError("final-eval bootstrap output must be outside the repository")

    manifest = dict(verifier(package))
    spec = _bootstrap_spec(package, manifest)
    encoded = repr(json.dumps(spec, sort_keys=True, separators=(",", ":")))
    content = textwrap.dedent(_TEMPLATE).replace("__SPEC_JSON__", encoded)
    if "__SPEC_JSON__" in content:
        raise PackageError("final-eval bootstrap template substitution failed")
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
        raise PackageError("final-eval bootstrap mode is not 0700")
    return {
        "status": "final-eval-bootstrap-generated",
        "path": str(destination),
        "sha256": _hash_file(destination),
        "bytes": destination.stat().st_size,
        "package_id": spec["package_id"],
        "branch": spec["branch"],
        "commit": spec["commit"],
        "required_ancestor": spec["required_ancestor"],
        "bundle_sha256": spec["bundle_sha256"],
        "ready_sha256": spec["ready_sha256"],
        "manifest_sha256": spec["manifest_sha256"],
        "sha256sums_sha256": spec["sha256sums_sha256"],
        "remote_file_count": len(spec["files"]),
    }


def generate_final_eval_bootstrap(
    *,
    repo_root: Path,
    package_root: Path,
    output: Path,
) -> dict[str, object]:
    """Verify a production final-input package, then build its puller."""

    from .final_eval_package import verify_final_eval_package

    return _generate_final_eval_bootstrap(
        repo_root=repo_root,
        package_root=package_root,
        output=output,
        verifier=verify_final_eval_package,
        production=True,
    )
