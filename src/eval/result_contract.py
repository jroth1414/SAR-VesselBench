"""Checkpoint-bound completion markers for reportable core experiments.

Schema 2 binds the dev-selected operating point to the exact Lightning
checkpoint that was selected with it.  All reportable consumers use this
module instead of assuming a checkpoint filename or falling back to the most
recent dev threshold.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any


RESULT_SCHEMA = 2
BEST_DEV_FIELDS = frozenset(
    {
        "epoch",
        "f1",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "ignored_predictions",
        "threshold",
        "n_candidates",
    }
)
BEST_CHECKPOINT_FIELDS = frozenset({"relative_path", "sha256", "epoch"})
RECIPE_FIELDS = (
    "exp_id",
    "git_sha",
    "detector_sha256",
    "precision",
    "micro_batch",
    "gradient_accumulation",
    "effective_batch",
)
COMPLETION_FIELDS = frozenset(
    {
        "result_schema",
        *RECIPE_FIELDS,
        "epochs_run",
        "best_dev_f1",
        "best_dev",
        "best_checkpoint",
        "last_dev",
        "train_loss",
    }
)
H100_RUNTIME_CONTRACT_FIELD = "h100_runtime_contract"
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")


class ResultContractError(RuntimeError):
    """A completion marker or checkpoint binding is not reportable."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one regular, non-symlink file."""

    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ResultContractError(f"checkpoint is not a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResultContractError(f"could not hash checkpoint: {source}") from exc
    return digest.hexdigest()


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ResultContractError(f"{field} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ResultContractError(f"{field} must be a finite number")
    return normalized


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ResultContractError(f"{field} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ResultContractError(f"{field} must be a non-negative integer")
    return normalized


def validate_dev_result(
    result: Mapping[str, object] | None,
    *,
    candidate_floor: float,
    description: str = "best_dev",
) -> dict[str, object]:
    """Validate one complete dev operating point and return a plain copy."""

    floor = _finite_number(candidate_floor, field="candidate_floor")
    if not 0.0 <= floor <= 1.0:
        raise ResultContractError("candidate_floor must be within [0, 1]")
    if not isinstance(result, Mapping):
        raise ResultContractError(f"{description} is missing")
    observed_fields = set(result)
    if observed_fields != BEST_DEV_FIELDS:
        missing = BEST_DEV_FIELDS - observed_fields
        unexpected = observed_fields - BEST_DEV_FIELDS
        raise ResultContractError(
            f"{description} keys do not match the exact contract; "
            f"missing: {', '.join(sorted(map(str, missing))) or '<none>'}; "
            f"unexpected: {', '.join(sorted(map(str, unexpected))) or '<none>'}"
        )

    epoch = _nonnegative_int(result["epoch"], field=f"{description}.epoch")
    metrics = {
        name: _finite_number(result[name], field=f"{description}.{name}")
        for name in ("f1", "precision", "recall")
    }
    for name, value in metrics.items():
        if not 0.0 <= value <= 1.0:
            raise ResultContractError(f"{description}.{name} must be within [0, 1]")
    threshold = _finite_number(
        result["threshold"], field=f"{description}.threshold"
    )
    if not floor <= threshold <= 1.0:
        raise ResultContractError(
            f"{description}.threshold must be within [candidate_floor, 1]"
        )
    counts = {
        name: _nonnegative_int(result[name], field=f"{description}.{name}")
        for name in ("tp", "fp", "fn", "ignored_predictions", "n_candidates")
    }
    if (
        counts["tp"] + counts["fp"] + counts["ignored_predictions"]
        > counts["n_candidates"]
    ):
        raise ResultContractError(
            f"{description} scored predictions exceed its decoded candidate count"
        )
    expected_precision = (
        counts["tp"] / (counts["tp"] + counts["fp"])
        if counts["tp"] + counts["fp"]
        else 0.0
    )
    expected_recall = (
        counts["tp"] / (counts["tp"] + counts["fn"])
        if counts["tp"] + counts["fn"]
        else 0.0
    )
    expected_f1 = (
        2.0 * expected_precision * expected_recall
        / (expected_precision + expected_recall)
        if expected_precision + expected_recall
        else 0.0
    )
    for name, expected in (
        ("precision", expected_precision),
        ("recall", expected_recall),
        ("f1", expected_f1),
    ):
        if not math.isclose(metrics[name], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ResultContractError(
                f"{description}.{name} is inconsistent with TP/FP/FN"
            )

    normalized = dict(result)
    normalized.update(
        {
            "epoch": epoch,
            **metrics,
            **counts,
            "threshold": threshold,
        }
    )
    return normalized


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _safe_checkpoint_path(run_dir: str | Path, checkpoint: str | Path) -> tuple[Path, str]:
    """Resolve a checkpoint while rejecting escapes and symlinks below run_dir."""

    root = _absolute_lexical(Path(run_dir))
    supplied = Path(checkpoint)
    candidate = _absolute_lexical(supplied if supplied.is_absolute() else root / supplied)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ResultContractError("best checkpoint must be inside its run directory") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ResultContractError("best checkpoint relative path is unsafe")

    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ResultContractError(f"best checkpoint path contains a symlink: {current}")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ResultContractError(
            f"best checkpoint is absent or escapes its run directory: {candidate}"
        ) from exc
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise ResultContractError(f"best checkpoint is absent or empty: {candidate}")
    return candidate, relative.as_posix()


def checkpoint_path_from_marker(
    run_dir: str | Path, relative_path: object
) -> Path:
    """Resolve the marker's safe run-relative checkpoint path."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ResultContractError("best_checkpoint.relative_path must be non-empty")
    relative = Path(relative_path)
    if relative.is_absolute() or relative.as_posix() != relative_path:
        raise ResultContractError("best_checkpoint.relative_path must be normalized and relative")
    path, normalized = _safe_checkpoint_path(run_dir, relative)
    if normalized != relative_path:
        raise ResultContractError("best_checkpoint.relative_path is not canonical")
    return path


def read_lightning_checkpoint_epoch(path: str | Path) -> int:
    """Read Lightning's zero-based ``epoch`` from a trusted local checkpoint."""

    import torch

    source = Path(path)
    try:
        checkpoint = torch.load(
            source,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise ResultContractError(f"could not load Lightning checkpoint: {source}") from exc
    if not isinstance(checkpoint, Mapping):
        raise ResultContractError("Lightning checkpoint root must be a mapping")
    return _nonnegative_int(checkpoint.get("epoch"), field="checkpoint.epoch")


def create_best_checkpoint_binding(
    *,
    run_dir: str | Path,
    checkpoint_path: str | Path,
    best_dev: Mapping[str, object] | None,
    candidate_floor: float,
) -> dict[str, object]:
    """Bind a validated best-dev result to its exact Lightning checkpoint."""

    result = validate_dev_result(best_dev, candidate_floor=candidate_floor)
    supplied = Path(checkpoint_path)
    if not supplied.is_absolute():
        cwd_candidate = _absolute_lexical(supplied)
        if cwd_candidate.exists() or cwd_candidate.is_symlink():
            supplied = cwd_candidate
    checkpoint, relative_path = _safe_checkpoint_path(run_dir, supplied)
    epoch = read_lightning_checkpoint_epoch(checkpoint)
    if epoch != result["epoch"]:
        raise ResultContractError(
            "best-dev epoch does not match the selected Lightning checkpoint epoch"
        )
    return {
        "relative_path": relative_path,
        "sha256": sha256_file(checkpoint),
        "epoch": epoch,
    }


def _validate_recipe(
    payload: Mapping[str, object],
    *,
    run_dir: Path,
    expected_recipe: Mapping[str, object] | None,
) -> None:
    missing = [name for name in RECIPE_FIELDS if name not in payload]
    if missing:
        raise ResultContractError(
            "completion recipe is incomplete; missing: " + ", ".join(missing)
        )
    if payload["exp_id"] != run_dir.name:
        raise ResultContractError("completion exp_id does not match the run directory")
    if not isinstance(payload["git_sha"], str) or not _HEX_40.fullmatch(payload["git_sha"]):
        raise ResultContractError("completion git_sha must be full lowercase 40-hex")
    if not isinstance(payload["detector_sha256"], str) or not _HEX_64.fullmatch(
        payload["detector_sha256"]
    ):
        raise ResultContractError("completion detector_sha256 must be lowercase 64-hex")
    if not isinstance(payload["precision"], str) or not payload["precision"]:
        raise ResultContractError("completion precision must be non-empty")
    micro_batch = _nonnegative_int(payload["micro_batch"], field="micro_batch")
    accumulation = _nonnegative_int(
        payload["gradient_accumulation"], field="gradient_accumulation"
    )
    effective = _nonnegative_int(payload["effective_batch"], field="effective_batch")
    if micro_batch <= 0 or accumulation <= 0 or effective != micro_batch * accumulation:
        raise ResultContractError("completion batch recipe is inconsistent")

    if expected_recipe is not None:
        missing_expected = [name for name in RECIPE_FIELDS if name not in expected_recipe]
        if missing_expected:
            raise ResultContractError(
                "expected recipe is incomplete; missing: " + ", ".join(missing_expected)
            )
        mismatches = {
            name: (expected_recipe[name], payload.get(name))
            for name in RECIPE_FIELDS
            if payload.get(name) != expected_recipe[name]
        }
        if mismatches:
            raise ResultContractError(f"completion marker is not recipe-matched: {mismatches}")


def validate_completion_payload(
    payload: Mapping[str, object],
    *,
    run_dir: str | Path,
    candidate_floor: float,
    expected_recipe: Mapping[str, object] | None = None,
) -> Path:
    """Validate an exact schema-2 completion payload and return its checkpoint."""

    if not isinstance(payload, Mapping):
        raise ResultContractError("completion marker root must be a mapping")
    if payload.get("result_schema") != RESULT_SCHEMA:
        raise ResultContractError(f"completion marker requires result_schema == {RESULT_SCHEMA}")
    observed_fields = set(payload)
    fields_with_h100_runtime = COMPLETION_FIELDS | {H100_RUNTIME_CONTRACT_FIELD}
    if (
        observed_fields != COMPLETION_FIELDS
        and observed_fields != fields_with_h100_runtime
    ):
        missing = COMPLETION_FIELDS - observed_fields
        unexpected = observed_fields - fields_with_h100_runtime
        raise ResultContractError(
            "completion marker keys do not match the exact contract; "
            f"missing: {', '.join(sorted(map(str, missing))) or '<none>'}; "
            f"unexpected: {', '.join(sorted(map(str, unexpected))) or '<none>'}"
        )
    root = Path(run_dir)
    _validate_recipe(payload, run_dir=root, expected_recipe=expected_recipe)
    best_dev = validate_dev_result(
        payload.get("best_dev"), candidate_floor=candidate_floor
    )
    best_dev_f1 = _finite_number(payload.get("best_dev_f1"), field="best_dev_f1")
    if best_dev_f1 != best_dev["f1"]:
        raise ResultContractError("best_dev_f1 does not equal best_dev.f1")

    best_checkpoint = payload.get("best_checkpoint")
    if not isinstance(best_checkpoint, Mapping):
        raise ResultContractError("best_checkpoint is missing")
    observed_checkpoint_fields = set(best_checkpoint)
    if observed_checkpoint_fields != BEST_CHECKPOINT_FIELDS:
        missing = BEST_CHECKPOINT_FIELDS - observed_checkpoint_fields
        unexpected = observed_checkpoint_fields - BEST_CHECKPOINT_FIELDS
        raise ResultContractError(
            "best_checkpoint keys do not match the exact contract; "
            f"missing: {', '.join(sorted(map(str, missing))) or '<none>'}; "
            f"unexpected: {', '.join(sorted(map(str, unexpected))) or '<none>'}"
        )
    marker_epoch = _nonnegative_int(
        best_checkpoint.get("epoch"), field="best_checkpoint.epoch"
    )
    if marker_epoch != best_dev["epoch"]:
        raise ResultContractError("best checkpoint epoch does not equal best-dev epoch")
    expected_sha256 = best_checkpoint.get("sha256")
    if not isinstance(expected_sha256, str) or not _HEX_64.fullmatch(expected_sha256):
        raise ResultContractError("best_checkpoint.sha256 must be lowercase 64-hex")
    checkpoint = checkpoint_path_from_marker(
        root, best_checkpoint.get("relative_path")
    )
    if sha256_file(checkpoint) != expected_sha256:
        raise ResultContractError("best checkpoint SHA-256 differs from completion marker")
    if read_lightning_checkpoint_epoch(checkpoint) != marker_epoch:
        raise ResultContractError("Lightning checkpoint epoch differs from completion marker")

    _nonnegative_int(payload.get("epochs_run"), field="epochs_run")
    _finite_number(payload.get("train_loss"), field="train_loss")
    validate_dev_result(
        payload.get("last_dev"),
        candidate_floor=candidate_floor,
        description="last_dev",
    )
    return checkpoint


def load_completion_marker(
    marker_path: str | Path,
    *,
    candidate_floor: float,
    expected_recipe: Mapping[str, object] | None = None,
) -> tuple[dict[str, Any], Path]:
    """Load and validate one completion marker and its bound checkpoint."""

    marker = Path(marker_path)
    if marker.is_symlink() or not marker.is_file():
        raise ResultContractError(f"completion marker is absent or a symlink: {marker}")
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultContractError(f"invalid completion marker JSON: {marker}") from exc
    checkpoint = validate_completion_payload(
        payload,
        run_dir=marker.parent,
        candidate_floor=candidate_floor,
        expected_recipe=expected_recipe,
    )
    return dict(payload), checkpoint


def atomic_write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """Durably replace a JSON file from a same-directory temporary file."""

    destination = Path(path)
    if destination.is_symlink():
        raise ResultContractError(f"refusing to replace symlink: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=1, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        temp_path = None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except (OSError, TypeError, ValueError) as exc:
        raise ResultContractError(f"could not atomically write JSON: {destination}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
