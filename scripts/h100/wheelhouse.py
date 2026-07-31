"""Deterministic identity for the immutable Sprint-7d wheelhouse."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


WHEELHOUSE_DIGEST_ALGORITHM = "xview3-wheelhouse-tree-v1"
BASE_EXTRACTION_FORMAT_VERSION = 1


def wheelhouse_manifest(wheelhouse: str | Path) -> dict[str, dict[str, object]]:
    root = Path(wheelhouse)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"offline wheelhouse is absent or unsafe: {root}")
    entries = sorted(root.iterdir(), key=lambda item: item.name)
    invalid = [
        item.name
        for item in entries
        if item.suffix != ".whl"
        or item.is_symlink()
        or not stat.S_ISREG(item.lstat().st_mode)
    ]
    if invalid:
        raise RuntimeError(
            "offline wheelhouse may contain only regular .whl files: "
            + ", ".join(invalid)
        )
    artifacts = {
        item.name: {"sha256": sha256_file(item), "bytes": item.stat().st_size}
        for item in entries
    }
    if not artifacts:
        raise RuntimeError("offline wheelhouse contains no files")
    return artifacts


def wheelhouse_identity(wheelhouse: str | Path) -> dict[str, object]:
    artifacts = wheelhouse_manifest(wheelhouse)
    encoded = json.dumps(
        {
            "algorithm": WHEELHOUSE_DIGEST_ALGORITHM,
            "artifacts": artifacts,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "algorithm": WHEELHOUSE_DIGEST_ALGORITHM,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "files": len(artifacts),
        "bytes": sum(int(item["bytes"]) for item in artifacts.values()),
    }


def assert_wheelhouse_unchanged(
    wheelhouse: str | Path,
    expected: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    actual = wheelhouse_manifest(wheelhouse)
    if actual != dict(expected):
        raise RuntimeError("offline wheelhouse changed while the native venv was building")
    return actual


def validate_base_extraction_receipt(
    receipt_path: str | Path,
    *,
    wheelhouse: str | Path,
    expected_package_id: str,
    expected_manifest_sha256: str,
    expected_receipt_sha256: str | None = None,
    expected_wheelhouse_sha256: str | None = None,
) -> dict[str, object]:
    requested = Path(receipt_path).absolute()
    if requested.is_symlink() or not requested.is_file():
        raise RuntimeError("base extraction receipt is absent or unsafe")
    receipt = requested.resolve(strict=True)
    digest = sha256_file(receipt)
    if expected_receipt_sha256 is not None and digest != expected_receipt_sha256:
        raise RuntimeError("base extraction receipt SHA-256 mismatch")
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("base extraction receipt is invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "package_id",
        "manifest_sha256",
        "wheelhouse",
    }:
        raise RuntimeError("base extraction receipt contract mismatch")
    if payload.get("format_version") != BASE_EXTRACTION_FORMAT_VERSION:
        raise RuntimeError("base extraction receipt format version mismatch")
    if (
        payload.get("package_id") != expected_package_id
        or payload.get("manifest_sha256") != expected_manifest_sha256
    ):
        raise RuntimeError("base extraction receipt payload identity mismatch")
    expected_root = receipt.parent / "environment/wheelhouse"
    actual_root = Path(wheelhouse).resolve(strict=True)
    if actual_root != expected_root.resolve(strict=True):
        raise RuntimeError("wheelhouse is not rooted in the verified base extraction")
    actual_identity = wheelhouse_identity(actual_root)
    if payload.get("wheelhouse") != actual_identity:
        raise RuntimeError("base extraction wheelhouse identity mismatch")
    if (
        expected_wheelhouse_sha256 is not None
        and actual_identity.get("sha256") != expected_wheelhouse_sha256
    ):
        raise RuntimeError("expected wheelhouse tree SHA-256 mismatch")
    return {
        "path": str(receipt),
        "sha256": digest,
        "receipt": payload,
    }
