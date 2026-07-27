"""Verified Box JWT upload/download for complete handoff packages.

This adapts the supplied helpers around content hashes rather than mtimes.
Files above 50 MB use Box upload sessions, failed sessions are resumed, and
the READY marker is always transferred last.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping

from .package import (
    PackageError,
    _absolute_path,
    _inside_repository_worktrees,
    _require_no_symlink_components,
    _validate_safe_path,
    _verify_package,
)

MINIMUM_BOX_FREE_BYTES = 500_000_000_000
CHUNKED_UPLOAD_THRESHOLD = 50_000_000
MAX_CHUNK_RESUME_ATTEMPTS = 5


class BoxTransferError(PackageError):
    """A Box preflight or verified transfer failed."""


@dataclass(frozen=True)
class BoxPreflight:
    folder_id: str
    quota_bytes: int
    used_bytes: int
    free_bytes: int
    maximum_file_bytes: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "quota_bytes": self.quota_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "maximum_file_bytes": self.maximum_file_bytes,
        }


@dataclass(frozen=True)
class RemoteFile:
    path: str
    item_id: str
    size: int
    sha1: str
    item: object


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def credentials_from_environment(repo_root: Path) -> tuple[Path, str]:
    """Resolve mandatory environment variables without printing their values."""

    config_value = os.environ.get("BOX_JWT_CONFIG", "")
    folder_id = os.environ.get("BOX_FOLDER_ID", "")
    if not config_value or not folder_id:
        raise BoxTransferError("BOX_JWT_CONFIG and BOX_FOLDER_ID are required")
    try:
        config = _require_no_symlink_components(
            Path(config_value).expanduser(),
            leaf="file",
        )
    except PackageError as exc:
        raise BoxTransferError(
            "BOX_JWT_CONFIG must be a regular file with no symlink components"
        ) from exc
    if _inside_repository_worktrees(config, repo_root):
        raise BoxTransferError("BOX_JWT_CONFIG must be outside the repository")
    mode = stat.S_IMODE(config.stat().st_mode)
    if mode != 0o600:
        raise BoxTransferError("BOX_JWT_CONFIG must have mode 0600")
    if not folder_id.isdigit():
        raise BoxTransferError("BOX_FOLDER_ID must be a numeric Box folder ID")
    return config, folder_id


def client_from_environment(repo_root: Path) -> tuple[object, str]:
    """Authenticate using the isolated boxsdk[jwt] v4 transfer environment."""

    config, folder_id = credentials_from_environment(repo_root)
    try:
        from boxsdk import Client, JWTAuth
    except ImportError as exc:
        raise BoxTransferError(
            "Box SDK is absent; install requirements-transfer.txt in a separate venv"
        ) from exc
    try:
        auth = JWTAuth.from_settings_file(str(config))
        auth.authenticate_instance()
        return Client(auth), folder_id
    except Exception as exc:
        raise BoxTransferError("Box JWT authentication failed") from exc


def _attribute(item: object, name: str) -> object | None:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _object_type(item: object) -> str:
    value = _attribute(item, "object_type") or _attribute(item, "type")
    return str(value or "")


def preflight_box(
    client: object,
    folder_id: str,
    *,
    minimum_free_bytes: int = MINIMUM_BOX_FREE_BYTES,
) -> BoxPreflight:
    """Validate destination access, quota, and the account file-size ceiling."""

    try:
        folder = client.folder(folder_id).get(  # type: ignore[attr-defined]
            fields=["id", "name", "permissions", "owned_by"]
        )
        uploader = client.user(user_id="me").get(  # type: ignore[attr-defined]
            fields=["id", "max_upload_size"]
        )
        owner_id = str(_attribute(_attribute(folder, "owned_by") or {}, "id") or "")
        if not owner_id:
            raise BoxTransferError("Box folder did not report its quota owner")
        owner = client.user(user_id=owner_id).get(  # type: ignore[attr-defined]
            fields=["id", "space_amount", "space_used"]
        )
        if str(_attribute(owner, "id") or "") != owner_id:
            raise BoxTransferError("Box quota response does not match the folder owner")
        quota = int(_attribute(owner, "space_amount") or 0)
        used = int(_attribute(owner, "space_used") or 0)
        maximum = int(_attribute(uploader, "max_upload_size") or 0)
    except Exception as exc:
        raise BoxTransferError("Box quota/folder preflight failed") from exc
    free = quota - used
    if quota <= 0 or used < 0 or free < minimum_free_bytes:
        raise BoxTransferError(
            f"Box free space is {free} bytes; at least {minimum_free_bytes} is required"
        )
    if maximum <= 0:
        raise BoxTransferError("Box did not report a positive maximum upload size")
    permissions = _attribute(folder, "permissions")
    required_permissions = ("can_upload", "can_delete", "can_rename")
    missing_permissions = [
        name
        for name in required_permissions
        if _attribute(permissions or {}, name) is not True
    ]
    if missing_permissions:
        raise BoxTransferError(
            "Box folder lacks required upload/update/delete permissions: "
            + ", ".join(missing_permissions)
        )
    return BoxPreflight(
        folder_id=folder_id,
        quota_bytes=quota,
        used_bytes=used,
        free_bytes=free,
        maximum_file_bytes=maximum,
    )


def _list_items(folder: object) -> list[object]:
    items: list[object] = []
    offset = 0
    limit = 1000
    while True:
        batch = list(
            folder.get_items(  # type: ignore[attr-defined]
                limit=limit,
                offset=offset,
                fields=["id", "name", "type", "sha1", "size"],
            )
        )
        items.extend(batch)
        if len(batch) < limit:
            return items
        offset += limit


def list_remote_files(client: object, folder_id: str) -> dict[str, RemoteFile]:
    result: dict[str, RemoteFile] = {}

    def recurse(current_id: str, prefix: PurePosixPath | None) -> None:
        folder = client.folder(current_id)  # type: ignore[attr-defined]
        for item in _list_items(folder):
            name = str(_attribute(item, "name") or "")
            if not name or "/" in name or name in {".", ".."}:
                raise BoxTransferError("Box package contains an unsafe item name")
            relative = PurePosixPath(name) if prefix is None else prefix / name
            item_type = _object_type(item)
            if item_type == "folder":
                recurse(str(_attribute(item, "id")), relative)
            elif item_type == "file":
                key = relative.as_posix()
                if key in result:
                    raise BoxTransferError(f"duplicate remote path: {key}")
                result[key] = RemoteFile(
                    path=key,
                    item_id=str(_attribute(item, "id")),
                    size=int(_attribute(item, "size") or 0),
                    sha1=str(_attribute(item, "sha1") or "").lower(),
                    item=item,
                )
            else:
                raise BoxTransferError(f"unsupported Box item type: {item_type}")

    recurse(folder_id, None)
    return result


def _local_files(package_root: Path) -> dict[str, Path]:
    return {
        path.relative_to(package_root).as_posix(): path
        for path in package_root.rglob("*")
        if path.is_file()
    }


def _remote_matches(remote: RemoteFile, local: Path) -> bool:
    return (
        remote.size == local.stat().st_size
        and remote.sha1 == _hash_file(local, "sha1")
    )


def _get_or_create_subfolder(client: object, parent_id: str, name: str) -> str:
    parent = client.folder(parent_id)  # type: ignore[attr-defined]
    for item in _list_items(parent):
        if _object_type(item) == "folder" and _attribute(item, "name") == name:
            return str(_attribute(item, "id"))
    try:
        created = parent.create_subfolder(name)
    except Exception as exc:
        raise BoxTransferError(f"cannot create Box subfolder {name!r}") from exc
    return str(_attribute(created, "id"))


def _ensure_remote_parent(
    client: object,
    root_folder_id: str,
    relative: PurePosixPath,
    cache: dict[PurePosixPath, str],
) -> str:
    parent_path = relative.parent
    if parent_path == PurePosixPath("."):
        return root_folder_id
    current_id = root_folder_id
    current = PurePosixPath()
    for segment in parent_path.parts:
        current /= segment
        if current not in cache:
            cache[current] = _get_or_create_subfolder(client, current_id, segment)
        current_id = cache[current]
    return current_id


def _refresh_file(client: object, item_id: str) -> RemoteFile:
    try:
        item = client.file(item_id).get(  # type: ignore[attr-defined]
            fields=["id", "name", "sha1", "size"]
        )
    except Exception as exc:
        raise BoxTransferError("cannot verify uploaded Box file") from exc
    return RemoteFile(
        path=str(_attribute(item, "name") or ""),
        item_id=str(_attribute(item, "id")),
        size=int(_attribute(item, "size") or 0),
        sha1=str(_attribute(item, "sha1") or "").lower(),
        item=item,
    )


def _upload_one(
    client: object,
    parent_id: str,
    local: Path,
    filename: str,
    existing: RemoteFile | None,
    *,
    chunked_threshold: int,
) -> None:
    try:
        if local.stat().st_size > chunked_threshold:
            if existing is None:
                uploader = client.folder(parent_id).get_chunked_uploader(  # type: ignore[attr-defined]
                    str(local), file_name=filename
                )
            else:
                uploader = client.file(existing.item_id).get_chunked_uploader(  # type: ignore[attr-defined]
                    str(local)
                )
            try:
                uploaded = uploader.start()
            except Exception:
                # The v4 SDK retains the upload session and committed parts;
                # resume() queries the session and sends only missing chunks.
                uploaded = None
            for _attempt in range(MAX_CHUNK_RESUME_ATTEMPTS):
                if uploaded is not None:
                    break
                uploaded = uploader.resume()
            if uploaded is None:
                raise BoxTransferError(
                    f"Box chunk commit remained pending for {filename}"
                )
        elif existing is None:
            uploaded = client.folder(parent_id).upload(  # type: ignore[attr-defined]
                str(local), filename
            )
        else:
            uploaded = client.file(existing.item_id).update_contents(  # type: ignore[attr-defined]
                str(local)
            )
    except Exception as exc:
        raise BoxTransferError(f"Box upload failed for {filename}") from exc
    item_id = str(_attribute(uploaded, "id") or (existing.item_id if existing else ""))
    if not item_id:
        raise BoxTransferError(f"Box upload returned no file ID for {filename}")
    remote = _refresh_file(client, item_id)
    if not _remote_matches(remote, local):
        raise BoxTransferError(f"Box SHA-1/size verification failed for {filename}")


def _delete_remote_ready(client: object, folder_id: str) -> None:
    remote = list_remote_files(client, folder_id)
    ready = remote.get("READY.json")
    if ready is None:
        return
    try:
        client.file(ready.item_id).delete()  # type: ignore[attr-defined]
    except Exception as exc:
        raise BoxTransferError("cannot invalidate remote READY marker") from exc


def _receipt_destination(
    receipt_path: Path,
    repo_root: Path,
) -> Path:
    repository = _require_no_symlink_components(repo_root, leaf="directory")
    destination = _absolute_path(receipt_path)
    if _inside_repository_worktrees(destination, repository):
        raise BoxTransferError("upload receipt must be outside the repository")
    if os.path.lexists(destination):
        raise BoxTransferError(f"upload receipt already exists: {destination}")
    _require_no_symlink_components(destination.parent, leaf="directory")
    return destination


def _write_upload_receipt(
    *,
    destination: Path,
    manifest: Mapping[str, object],
    package_root: Path,
    remote: Mapping[str, RemoteFile],
) -> dict[str, object]:
    payload = {
        "schema": 1,
        "status": "uploaded",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_id": manifest["package_id"],
        "ready_sha256": _hash_file(package_root / "READY.json", "sha256"),
        "manifest_sha256": _hash_file(package_root / "manifest.json", "sha256"),
        "sha256sums_sha256": _hash_file(package_root / "SHA256SUMS", "sha256"),
        "remote_file_count": len(remote),
        "remote_bytes": sum(item.size for item in remote.values()),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return payload


def _upload_package(
    client: object,
    folder_id: str,
    package_root: Path,
    *,
    repo_root: Path,
    receipt_path: Path,
    require_production: bool,
    chunked_threshold: int = CHUNKED_UPLOAD_THRESHOLD,
    minimum_free_bytes: int = 0,
) -> dict[str, object]:
    """Upload exact package contents, invalidating READY before any mutation."""

    manifest = _verify_package(
        package_root,
        require_production=require_production,
    )
    package_root = _absolute_path(package_root)
    receipt = _receipt_destination(receipt_path, repo_root)
    preflight = preflight_box(
        client, folder_id, minimum_free_bytes=minimum_free_bytes
    )
    local = _local_files(package_root)
    remote = list_remote_files(client, folder_id)
    unexpected = set(remote) - set(local)
    initial_mismatched = {
        relative
        for relative, path in local.items()
        if relative not in remote or not _remote_matches(remote[relative], path)
    }
    mutation_needed = bool(unexpected or initial_mismatched)
    ready_cleanup_required = mutation_needed
    if mutation_needed and "READY.json" in remote:
        _delete_remote_ready(client, folder_id)
        remote.pop("READY.json", None)
    if unexpected:
        raise BoxTransferError(
            f"Box folder has unexpected package files: {sorted(unexpected)}"
        )
    if any(path.stat().st_size > preflight.maximum_file_bytes for path in local.values()):
        raise BoxTransferError(
            "package has a file above the current Box maximum; rebuild with smaller parts"
        )

    mismatched = {
        relative
        for relative, path in local.items()
        if relative not in remote or not _remote_matches(remote[relative], path)
    }
    required_upload_bytes = sum(
        local[relative].stat().st_size for relative in mismatched
    )
    if preflight.free_bytes < required_upload_bytes:
        raise BoxTransferError(
            "Box has insufficient free space for the remaining verified "
            f"upload: need {required_upload_bytes} bytes, "
            f"have {preflight.free_bytes}"
        )

    uploaded = 0
    skipped = 0
    try:
        ordered = sorted(path for path in local if path != "READY.json") + [
            "READY.json"
        ]
        folder_cache: dict[PurePosixPath, str] = {}
        for relative in ordered:
            path = local[relative]
            existing = remote.get(relative)
            if existing is not None and _remote_matches(existing, path):
                skipped += 1
                continue
            # The local package may have changed after the initial comparison.
            # Invalidate an existing READY before the first late-discovered
            # upload, and retain cleanup responsibility for all later failures.
            if not ready_cleanup_required:
                ready_cleanup_required = True
                _delete_remote_ready(client, folder_id)
                remote.pop("READY.json", None)
                existing = remote.get(relative)
            parent_id = _ensure_remote_parent(
                client, folder_id, PurePosixPath(relative), folder_cache
            )
            _upload_one(
                client,
                parent_id,
                path,
                PurePosixPath(relative).name,
                existing,
                chunked_threshold=chunked_threshold,
            )
            uploaded += 1
        final_remote = list_remote_files(client, folder_id)
        if set(final_remote) != set(local) or any(
            not _remote_matches(final_remote[name], path)
            for name, path in local.items()
        ):
            ready_cleanup_required = True
            raise BoxTransferError("final Box tree does not match the local package")
        try:
            final_manifest = _verify_package(
                package_root,
                require_production=require_production,
            )
        except BaseException:
            ready_cleanup_required = True
            raise
        if final_manifest != manifest:
            ready_cleanup_required = True
            raise BoxTransferError(
                "local package changed after upload verification"
            )

        receipt_payload = _write_upload_receipt(
            destination=receipt,
            manifest=manifest,
            package_root=package_root,
            remote=final_remote,
        )
    except BaseException:
        if ready_cleanup_required:
            try:
                _delete_remote_ready(client, folder_id)
            except BaseException as cleanup_error:
                raise BoxTransferError(
                    "upload failed and remote READY could not be removed"
                ) from cleanup_error
        raise
    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "receipt": str(receipt),
        "package_id": receipt_payload["package_id"],
    }


def upload_package(
    client: object,
    folder_id: str,
    package_root: Path,
    *,
    repo_root: Path,
    receipt_path: Path,
    chunked_threshold: int = CHUNKED_UPLOAD_THRESHOLD,
    minimum_free_bytes: int = 0,
) -> dict[str, object]:
    """Target-facing verified production upload."""

    return _upload_package(
        client,
        folder_id,
        package_root,
        repo_root=repo_root,
        receipt_path=receipt_path,
        require_production=True,
        chunked_threshold=chunked_threshold,
        minimum_free_bytes=minimum_free_bytes,
    )


def _upload_fixture_package(
    client: object,
    folder_id: str,
    package_root: Path,
    *,
    repo_root: Path,
    receipt_path: Path,
    chunked_threshold: int = CHUNKED_UPLOAD_THRESHOLD,
    minimum_free_bytes: int = 0,
) -> dict[str, object]:
    """Internal-only upload path for the deterministic Box fixture."""

    return _upload_package(
        client,
        folder_id,
        package_root,
        repo_root=repo_root,
        receipt_path=receipt_path,
        require_production=False,
        chunked_threshold=chunked_threshold,
        minimum_free_bytes=minimum_free_bytes,
    )


def _download_atomic(
    client: object,
    remote: RemoteFile,
    destination: Path,
    *,
    expected_sha256: str | None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        with partial.open("xb") as handle:
            client.file(remote.item_id).download_to(handle)  # type: ignore[attr-defined]
        if partial.stat().st_size != remote.size:
            raise BoxTransferError(f"Box size mismatch for {remote.path}")
        if _hash_file(partial, "sha1") != remote.sha1:
            raise BoxTransferError(f"Box SHA-1 mismatch for {remote.path}")
        computed_sha256 = _hash_file(partial, "sha256")
        if expected_sha256 is not None and computed_sha256 != expected_sha256:
            raise BoxTransferError(f"SHA-256 mismatch for {remote.path}")
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _control_record(ready: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = ready.get(name)
    if not isinstance(value, Mapping):
        raise BoxTransferError(f"READY marker has no {name} record")
    return value


def _validate_remote_control_record(
    record: Mapping[str, object],
    *,
    expected_path: str,
    expected_sha256: str,
    remote: RemoteFile,
) -> None:
    if record.get("path") != expected_path:
        raise BoxTransferError(f"READY control path mismatch: {expected_path}")
    if record.get("sha256") != expected_sha256:
        raise BoxTransferError(f"READY control SHA-256 mismatch: {expected_path}")
    if record.get("bytes") != remote.size:
        raise BoxTransferError(f"READY control size mismatch: {expected_path}")
    if record.get("sha1") != remote.sha1:
        raise BoxTransferError(f"READY control SHA-1 mismatch: {expected_path}")


def _manifest_physical(manifest: Mapping[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise BoxTransferError("manifest artifacts missing")
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise BoxTransferError("invalid manifest artifact")
        parts = artifact.get("parts")
        if not isinstance(parts, list):
            raise BoxTransferError("invalid artifact parts")
        for part in parts:
            if not isinstance(part, Mapping):
                raise BoxTransferError("invalid artifact part")
            path = str(part.get("path", ""))
            sha256 = str(part.get("sha256", ""))
            relative = PurePosixPath(path)
            _validate_safe_path(relative)
            if path in result or len(sha256) != 64:
                raise BoxTransferError("duplicate/invalid physical artifact record")
            result[path] = sha256
    return result


def _download_package(
    client: object,
    folder_id: str,
    package_root: Path,
    *,
    repo_root: Path,
    require_production: bool,
    expected_ready_sha256: str,
    expected_manifest_sha256: str,
    expected_sha256sums_sha256: str,
    expected_package_id: str,
) -> Path:
    """Verify in a private sibling tree, then atomically publish the whole root."""

    repository = _require_no_symlink_components(repo_root, leaf="directory")
    package_root = _absolute_path(package_root)
    if _inside_repository_worktrees(package_root, repository):
        raise BoxTransferError("download package root must be outside the repository")
    if os.path.lexists(package_root):
        raise BoxTransferError(f"download destination already exists: {package_root}")
    _require_no_symlink_components(package_root.parent, leaf="directory")
    for label, digest in {
        "READY": expected_ready_sha256,
        "manifest": expected_manifest_sha256,
        "SHA256SUMS": expected_sha256sums_sha256,
    }.items():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise BoxTransferError(f"expected {label} SHA-256 is invalid")
    if not expected_package_id:
        raise BoxTransferError("expected package ID is required")
    remote = list_remote_files(client, folder_id)
    for required in ("READY.json", "manifest.json", "SHA256SUMS"):
        if required not in remote:
            raise BoxTransferError(f"remote package is incomplete: {required} absent")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{package_root.name}.downloading-",
            dir=package_root.parent,
        )
    )
    try:
        _download_atomic(
            client,
            remote["READY.json"],
            staging / "READY.json",
            expected_sha256=expected_ready_sha256,
        )
        ready = json.loads((staging / "READY.json").read_text(encoding="utf-8"))
        if (
            ready.get("status") != "READY"
            or ready.get("package_id") != expected_package_id
        ):
            raise BoxTransferError("remote READY marker is invalid")
        manifest_record = _control_record(ready, "manifest")
        sums_record = _control_record(ready, "checksums")
        _validate_remote_control_record(
            manifest_record,
            expected_path="manifest.json",
            expected_sha256=expected_manifest_sha256,
            remote=remote["manifest.json"],
        )
        _validate_remote_control_record(
            sums_record,
            expected_path="SHA256SUMS",
            expected_sha256=expected_sha256sums_sha256,
            remote=remote["SHA256SUMS"],
        )
        _download_atomic(
            client,
            remote["manifest.json"],
            staging / "manifest.json",
            expected_sha256=expected_manifest_sha256,
        )
        _download_atomic(
            client,
            remote["SHA256SUMS"],
            staging / "SHA256SUMS",
            expected_sha256=expected_sha256sums_sha256,
        )
        manifest = json.loads(
            (staging / "manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("package_id") != expected_package_id:
            raise BoxTransferError("downloaded manifest package ID mismatch")
        expected = _manifest_physical(manifest)
        expected_tree = set(expected) | {"READY.json", "manifest.json", "SHA256SUMS"}
        if set(remote) != expected_tree:
            raise BoxTransferError("remote package tree differs from its manifest")
        for relative in sorted(expected):
            safe_relative = PurePosixPath(relative)
            _validate_safe_path(safe_relative)
            destination = staging.joinpath(*safe_relative.parts)
            if not _inside(destination, staging):
                raise BoxTransferError(f"manifest path escapes destination: {relative}")
            _download_atomic(
                client,
                remote[relative],
                destination,
                expected_sha256=expected[relative],
            )
        _verify_package(
            staging,
            require_production=require_production,
        )
        os.replace(staging, package_root)
    except BaseException:
        # Only the private, builder-owned sibling stage is ever removed.
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return package_root


def download_package(
    client: object,
    folder_id: str,
    package_root: Path,
    *,
    repo_root: Path,
    expected_ready_sha256: str,
    expected_manifest_sha256: str,
    expected_sha256sums_sha256: str,
    expected_package_id: str,
) -> Path:
    """Target-facing atomic download: production packages only."""

    return _download_package(
        client,
        folder_id,
        package_root,
        repo_root=repo_root,
        require_production=True,
        expected_ready_sha256=expected_ready_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_sha256sums_sha256=expected_sha256sums_sha256,
        expected_package_id=expected_package_id,
    )


def _download_fixture_package(
    client: object,
    folder_id: str,
    package_root: Path,
    *,
    repo_root: Path,
    expected_ready_sha256: str,
    expected_manifest_sha256: str,
    expected_sha256sums_sha256: str,
    expected_package_id: str,
) -> Path:
    """Internal-only atomic download for the deterministic Box fixture."""

    return _download_package(
        client,
        folder_id,
        package_root,
        repo_root=repo_root,
        require_production=False,
        expected_ready_sha256=expected_ready_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_sha256sums_sha256=expected_sha256sums_sha256,
        expected_package_id=expected_package_id,
    )
