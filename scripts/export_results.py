"""Export only small H100 contract receipts for Git sharing.

The 32 reportable H100 core cells are returned exclusively through the
verified reverse Box package.  This helper therefore never copies any core
experiment, summary grid, or label-efficiency plot: a same-ID V100 diagnostic
must not become reportable merely because it exists under ``runs/``.

Corrected R2/R3 live in an explicit external campaign root and travel only
through the validated ``build-control --kind references`` workflow.  The
legacy ``runs/<exp_id>`` reference export path is retired.  Managed output is
limited to three small immutable H100 contract receipts.  An ownership receipt
allows later invocations to prune only unchanged files previously written by
this helper; unknown, modified, symlinked, and historical paths (including
P3.6) are preserved or rejected rather than deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Mapping


RUNS = Path("runs")
OUT = Path("results")
OWNERSHIP_RECEIPT = ".export-results-owned.json"
_RECEIPT_SCHEMA = 2
_RECEIPT_POLICY = "small-h100-contract-receipts-only-no-experiments"
_LEGACY_RECEIPT = (1, "references-and-small-provenance-only-no-core")
_HEX64 = re.compile(r"[0-9a-f]{64}")
# Retained only so an unchanged receipt created by the pre-retirement exporter
# can be pruned safely. These paths are never selected as desired output.
_LEGACY_REFERENCE_IDS = frozenset({"yolo26-f100", "locateanything-zs"})
_REFERENCE_FILES = ("final_metrics.json", "runtime_provenance.json")
_META_FILES = (
    "TRAINING_COHORT.json",
    "EVAL_GROUND_TRUTH_VALIDATED.json",
    "V100_DIAGNOSTIC_ISOLATION.json",
)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular_source(path: Path, description: str, *, immutable: bool = False) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{description} is not a regular non-symlink file: {path}")
    if immutable and path.stat().st_mode & 0o222:
        raise RuntimeError(f"{description} is not immutable: {path}")
    return path


def _json_source(path: Path, description: str, *, immutable: bool = False) -> Path:
    path = _regular_source(path, description, immutable=immutable)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"{description} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} is not a JSON object: {path}")
    return path


def _managed_relative(relative: str) -> PurePosixPath:
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"unsafe exporter-owned path: {relative!r}")
    if len(path.parts) == 2 and path.parts[0] == ".h100" and path.name in _META_FILES:
        return path
    # Accept the former ``results/<exp_id>/`` namespace only when reading an
    # ownership receipt, so unchanged pre-retirement reference copies can be
    # removed. No current source selection can create one of these paths.
    if (
        len(path.parts) == 2
        and path.parts[0] in _LEGACY_REFERENCE_IDS
        and path.name in _REFERENCE_FILES
    ):
        return path
    raise RuntimeError(f"path is outside the exporter-owned namespace: {relative}")


def _load_ownership(root: Path) -> dict[str, str]:
    receipt = root / OWNERSHIP_RECEIPT
    if not receipt.exists() and not receipt.is_symlink():
        return {}
    receipt = _json_source(receipt, "export ownership receipt")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    files = payload.get("files")
    receipt_identity = (payload.get("schema"), payload.get("policy"))
    if (
        set(payload) != {"schema", "policy", "files"}
        or receipt_identity not in {
            (_RECEIPT_SCHEMA, _RECEIPT_POLICY),
            _LEGACY_RECEIPT,
        }
        or not isinstance(files, Mapping)
    ):
        raise RuntimeError("export ownership receipt has an invalid schema")
    normalized: dict[str, str] = {}
    for raw_relative, raw_digest in files.items():
        relative = _managed_relative(str(raw_relative)).as_posix()
        digest = str(raw_digest)
        if relative in normalized or not _HEX64.fullmatch(digest):
            raise RuntimeError("export ownership receipt has an invalid file binding")
        normalized[relative] = digest
    return normalized


def _destination(root: Path, relative: str) -> Path:
    safe = _managed_relative(relative)
    current = root
    for part in safe.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"symlinked export destination parent is forbidden: {current}")
        if current.exists() and not current.is_dir():
            raise RuntimeError(f"export destination parent is not a directory: {current}")
    destination = root.joinpath(*safe.parts)
    if destination.is_symlink():
        raise RuntimeError(f"symlinked export destination is forbidden: {destination}")
    return destination


def _desired_sources() -> dict[str, Path]:
    desired: dict[str, Path] = {}
    for name in _META_FILES:
        source = RUNS / ".h100" / name
        if source.exists() or source.is_symlink():
            desired[f".h100/{name}"] = _json_source(
                source, f"H100 {name}", immutable=True
            )

    return desired


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_ownership(root: Path, files: Mapping[str, str]) -> None:
    payload = {
        "schema": _RECEIPT_SCHEMA,
        "policy": _RECEIPT_POLICY,
        "files": dict(sorted(files.items())),
    }
    destination = root / OWNERSHIP_RECEIPT
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=root
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_empty_parents(path: Path, root: Path) -> None:
    current = path.parent
    while current != root:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def main() -> int:
    if OUT.is_symlink():
        raise RuntimeError(f"export root may not be a symlink: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    if not OUT.is_dir():
        raise RuntimeError(f"export root is not a directory: {OUT}")

    previous = _load_ownership(OUT)
    desired_sources = _desired_sources()
    desired_hashes = {
        relative: _sha256(source)
        for relative, source in desired_sources.items()
    }

    # Validate every managed destination before making any change.  A user edit
    # or an unknown pre-existing file fails closed instead of being overwritten.
    for relative in sorted(set(previous) | set(desired_sources)):
        destination = _destination(OUT, relative)
        if not destination.exists():
            continue
        if not destination.is_file():
            raise RuntimeError(f"managed export destination is not a file: {destination}")
        if relative not in previous:
            raise RuntimeError(f"refusing unowned export destination: {destination}")
        if _sha256(destination) != previous[relative]:
            raise RuntimeError(f"owned export destination was modified: {destination}")

    for relative, source in sorted(desired_sources.items()):
        destination = _destination(OUT, relative)
        if not destination.exists() or _sha256(destination) != desired_hashes[relative]:
            _atomic_copy(source, destination)

    removed = 0
    for relative in sorted(set(previous) - set(desired_sources), reverse=True):
        destination = _destination(OUT, relative)
        if destination.exists():
            destination.unlink()
            _remove_empty_parents(destination, OUT)
            removed += 1

    _write_ownership(OUT, desired_hashes)
    print(
        f"exported {len(desired_sources)} managed files, pruned {removed} stale files "
        f"-> {OUT}/ (all experiment directories excluded)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
