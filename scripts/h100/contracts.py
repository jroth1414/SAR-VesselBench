"""Pure contracts shared by the H100 acceptance and campaign tools.

Keep this module free of torch/CUDA imports so its safety checks remain
fixture-testable on CPU-only CI.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml
from scripts.h100.runtime_versions import EXPECTED_NATIVE_PYTHON_VERSION


EXPECTED_PRECISION = "32-true"
MICRO_BATCH = 16
GRADIENT_ACCUMULATION = 1
EFFECTIVE_BATCH = 16
EXPECTED_GPU_COUNT = 8
EXPECTED_COMPUTE_CAPABILITY = (9, 0)
MIN_SCRATCH_BYTES = 500_000_000_000
SLURM_ALLOCATION_WALL_HOURS = 36.5
SLURM_SIGNAL_LEAD_HOURS = 0.25
V100_DIAGNOSTIC_RUNNING = "continues-running-non-reportable-diagnostic"
V100_DIAGNOSTIC_COMPLETE = "complete-non-reportable-diagnostic"
V100_DIAGNOSTIC_STATUSES = frozenset(
    {V100_DIAGNOSTIC_RUNNING, V100_DIAGNOSTIC_COMPLETE}
)

FROZEN_PATHS = (
    "configs/detector.yaml",
    "src/eval/scorer.py",
    "data/splits.json",
    "data/stats.json",
    "data/lsssdd_split.json",
)


def cutover_acceptance_bindings(ready: Mapping[str, object]) -> dict[str, object]:
    """Return the exact H100 acceptance subset embedded in CUTOVER_READY."""

    keys = (
        "schema",
        "source",
        "venv",
        "base_payload",
        "runtime_amendment",
        "gates",
        "source_validation",
        "evaluation_ground_truth",
        "test_suite",
        "projection",
        "slurm_smoke",
    )
    try:
        return {
            "uuid": ready["acceptance_uuid"],
            **{key: ready[key] for key in keys},
        }
    except KeyError as exc:
        raise RuntimeError(
            f"H100_READY lacks cutover acceptance binding {exc.args[0]!r}"
        ) from exc


def validate_bound_cutover_forecast(
    cutover: Mapping[str, object],
) -> dict[str, object]:
    """Validate the cutover-time V100 forecast against the bound H100 receipt."""

    forecast = cutover.get("cutover_forecast")
    acceptance = cutover.get("acceptance")
    projection = acceptance.get("projection") if isinstance(acceptance, Mapping) else None
    numeric_keys = {
        "conservative_h100_wall_hours",
        "acceptance_remaining_v100_wall_hours",
        "current_remaining_v100_wall_hours",
    }
    keys = {
        *numeric_keys,
        "v100_diagnostic_status",
        "h100_scientifically_mandatory",
    }
    if (
        not isinstance(forecast, Mapping)
        or set(forecast) != keys
        or not isinstance(projection, Mapping)
    ):
        raise RuntimeError("CUTOVER_READY forecast binding is absent or malformed")
    try:
        values = {key: float(forecast[key]) for key in numeric_keys}
        accepted_h100 = float(projection["conservative_h100_wall_hours"])
        accepted_v100 = float(projection["remaining_v100_wall_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("CUTOVER_READY forecast values are invalid") from exc
    if any(not math.isfinite(value) for value in values.values()):
        raise RuntimeError("CUTOVER_READY forecast values must be finite")
    if (
        values["conservative_h100_wall_hours"] <= 0
        or values["acceptance_remaining_v100_wall_hours"] <= 0
        or values["current_remaining_v100_wall_hours"] < 0
    ):
        raise RuntimeError(
            "CUTOVER_READY H100/acceptance forecasts must be positive and "
            "current V100 remaining hours must be nonnegative"
        )
    if (
        not math.isclose(values["conservative_h100_wall_hours"], accepted_h100)
        or not math.isclose(
            values["acceptance_remaining_v100_wall_hours"], accepted_v100
        )
    ):
        raise RuntimeError("CUTOVER_READY forecast differs from H100 acceptance")
    if accepted_h100 >= accepted_v100:
        raise RuntimeError("CUTOVER_READY does not preserve the acceptance comparison")

    status = forecast.get("v100_diagnostic_status")
    if status not in V100_DIAGNOSTIC_STATUSES:
        raise RuntimeError("CUTOVER_READY V100 diagnostic status is invalid")
    if forecast.get("h100_scientifically_mandatory") is not True:
        raise RuntimeError("CUTOVER_READY must keep the H100 rerun scientifically mandatory")
    current = values["current_remaining_v100_wall_hours"]
    if current > 0:
        if status != V100_DIAGNOSTIC_RUNNING or accepted_h100 >= current:
            raise RuntimeError("CUTOVER_READY no longer proves a current H100 advantage")
    elif status != V100_DIAGNOSTIC_COMPLETE:
        raise RuntimeError(
            "zero remaining V100 hours require an explicit complete "
            "non-reportable diagnostic status"
        )
    return {
        **values,
        "v100_diagnostic_status": status,
        "h100_scientifically_mandatory": True,
    }

# CNN cells are longer than ViT cells.  The scientific matrix is unchanged;
# this is only the launch order within each expensive-first fraction.
FAMILY_EXPENSE_ORDER = ("cnn", "vit")
FRACTION_EXPENSE_ORDER = (1.0, 0.5, 0.25, 0.1)


@dataclass(frozen=True)
class Cell:
    init: str
    short: str
    track: str
    fraction: float
    seed: int = 0

    @property
    def exp_id(self) -> str:
        return f"{self.short}-f{int(round(self.fraction * 100))}-s{self.seed}"


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: object) -> None:
    """Write JSON durably enough that a preemption cannot leave half a marker."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_cells(repo: str | Path) -> list[Cell]:
    """Derive the exact 32-cell matrix from the active arm manifest."""

    root = Path(repo)
    config = yaml.safe_load((root / "configs/arms.yaml").read_text())
    arms: Mapping[str, Mapping[str, str]] = config["arms"]
    if tuple(config["seeds"]["core"]) != (0,):
        raise RuntimeError("H100 lane requires the frozen seed-0-only core matrix")
    if set(float(value) for value in config["label_fracs"]) != {
        0.1,
        0.25,
        0.5,
        1.0,
    }:
        raise RuntimeError("H100 lane requires exactly f10/f25/f50/f100")

    cells = [
        Cell(
            init=init,
            short=str(meta["short"]),
            track=str(meta["track"]),
            fraction=fraction,
        )
        for fraction in FRACTION_EXPENSE_ORDER
        for track in FAMILY_EXPENSE_ORDER
        for init, meta in arms.items()
        if str(meta["track"]) == track
    ]
    if len(cells) != 32 or len({cell.exp_id for cell in cells}) != 32:
        raise RuntimeError("active arms.yaml did not resolve to 32 unique core cells")
    return cells


def frozen_hashes(repo: str | Path) -> dict[str, str]:
    root = Path(repo)
    return {relative: sha256_file(root / relative) for relative in FROZEN_PATHS}


def verify_expected_hashes(
    root: str | Path, expected: Mapping[str, str]
) -> dict[str, str]:
    """Hash all explicitly expected files and fail on the first mismatch."""

    base = Path(root)
    actual: dict[str, str] = {}
    for relative, wanted in expected.items():
        path = base / relative
        if not path.is_file():
            raise RuntimeError(f"required file is absent: {path}")
        got = sha256_file(path)
        if got != wanted:
            raise RuntimeError(
                f"SHA-256 mismatch for {relative}: expected {wanted}, got {got}"
            )
        actual[relative] = got
    return actual


def validate_gpu_inventory(devices: Iterable[Mapping[str, object]]) -> list[dict]:
    """Validate a mocked or real nvidia-smi/torch inventory."""

    normalized = [dict(device) for device in devices]
    if len(normalized) != EXPECTED_GPU_COUNT:
        raise RuntimeError(
            f"expected exactly {EXPECTED_GPU_COUNT} H100s, found {len(normalized)}"
        )
    seen: set[str] = set()
    names: set[str] = set()
    memory_sizes: set[int] = set()
    for index, device in enumerate(normalized):
        name = str(device.get("name", ""))
        capability = tuple(device.get("compute_capability", ()))
        identifier = str(device.get("uuid", "")).strip()
        try:
            total_memory = int(device.get("total_memory_bytes", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"device {index} has invalid total memory") from exc
        if "H100" not in name.upper():
            raise RuntimeError(f"device {index} is not an H100: {name!r}")
        if capability != EXPECTED_COMPUTE_CAPABILITY:
            raise RuntimeError(
                f"device {index} must be compute capability 9.0, got {capability}"
            )
        if not identifier:
            raise RuntimeError(f"device {index} has no hardware UUID")
        if identifier in seen:
            raise RuntimeError(f"duplicate GPU identity: {identifier}")
        if total_memory <= 0:
            raise RuntimeError(f"device {index} has no positive total memory")
        seen.add(identifier)
        names.add(name)
        memory_sizes.add(total_memory)
    if len(names) != 1 or len(memory_sizes) != 1:
        raise RuntimeError("H100 allocation must have one exact GPU name and memory size")
    return normalized


def assert_empty_core_namespaces(repo: str | Path, runs_root: str | Path) -> None:
    occupied = [
        cell.exp_id
        for cell in load_cells(repo)
        if (Path(runs_root) / cell.exp_id).exists()
    ]
    if occupied:
        raise RuntimeError(
            "H100 canonical namespaces are not empty: " + ", ".join(occupied)
        )


def validate_completion_marker(
    path: str | Path,
    *,
    cell: Cell,
    git_sha: str,
    detector_sha256: str,
) -> dict:
    marker = Path(path)
    try:
        payload = json.loads(marker.read_text())
        score = float(payload["best_dev_f1"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid completion marker: {marker}") from exc
    expected = {
        "exp_id": cell.exp_id,
        "git_sha": git_sha,
        "detector_sha256": detector_sha256,
        "precision": EXPECTED_PRECISION,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch": EFFECTIVE_BATCH,
    }
    mismatches = {
        key: (expected_value, payload.get(key))
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    }
    if mismatches or not math.isfinite(score):
        raise RuntimeError(
            f"completion marker is not recipe-matched: {marker}; {mismatches}"
        )
    return payload


def estimate_grid_projection(
    *,
    probe_steps: int,
    probe_seconds: float,
    steps_per_epoch: Mapping[str, int],
    expected_epochs: float = 42.2,
    ceiling_epochs: int = 50,
    gpu_count: int = EXPECTED_GPU_COUNT,
) -> dict[str, object]:
    """Conservative projection from one strict-FP32, 200-step probe.

    The four per-fraction step counts must be measured from the received chips
    and frozen nested split.  Both expected and 50-epoch ceiling estimates are
    retained; measured campaign telemetry replaces this estimate later.
    """

    required = {"f10", "f25", "f50", "f100"}
    if (
        probe_steps <= 0
        or probe_seconds <= 0
        or set(steps_per_epoch) != required
        or any(type(value) is not int or value <= 0 for value in steps_per_epoch.values())
    ):
        raise ValueError("projection inputs must be positive")
    steps_per_second = probe_steps / probe_seconds
    grid_steps_per_epoch = 8 * sum(steps_per_epoch.values())
    total_expected_steps = grid_steps_per_epoch * expected_epochs
    total_ceiling_steps = grid_steps_per_epoch * ceiling_epochs
    expected_gpu_hours = total_expected_steps / steps_per_second / 3600.0
    ceiling_gpu_hours = total_ceiling_steps / steps_per_second / 3600.0
    return {
        "steps_per_second": steps_per_second,
        "steps_per_epoch": dict(steps_per_epoch),
        "grid_steps_per_epoch": grid_steps_per_epoch,
        "expected_total_optimizer_steps": total_expected_steps,
        "ceiling_total_optimizer_steps": total_ceiling_steps,
        "expected_gpu_hours": expected_gpu_hours,
        "ceiling_gpu_hours": ceiling_gpu_hours,
        "expected_wall_hours_ideal": expected_gpu_hours / gpu_count,
        "ceiling_wall_hours_ideal": ceiling_gpu_hours / gpu_count,
    }


def staging_aware_wall_clock(
    *,
    training_wall_hours: float,
    staging_seconds: float,
    allocation_wall_hours: float = SLURM_ALLOCATION_WALL_HOURS,
    signal_lead_hours: float = SLURM_SIGNAL_LEAD_HOURS,
) -> dict[str, float | int]:
    """Add measured per-allocation staging to a conservative training forecast."""

    values = (training_wall_hours, staging_seconds, allocation_wall_hours, signal_lead_hours)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("wall-clock projection inputs must be finite")
    if training_wall_hours <= 0 or staging_seconds < 0 or allocation_wall_hours <= 0:
        raise ValueError("wall-clock projection inputs are out of range")
    staging_hours = staging_seconds / 3600.0
    usable_training_hours = allocation_wall_hours - signal_lead_hours - staging_hours
    if signal_lead_hours < 0 or usable_training_hours <= 0:
        raise ValueError("staging leaves no usable training time in an allocation")
    allocation_count = max(1, math.ceil(training_wall_hours / usable_training_hours))
    return {
        "staging_seconds": staging_seconds,
        "staging_hours_per_allocation": staging_hours,
        "allocation_wall_hours": allocation_wall_hours,
        "signal_lead_hours": signal_lead_hours,
        "usable_training_hours_per_allocation": usable_training_hours,
        "projected_allocation_count": allocation_count,
        "training_wall_hours_before_staging": training_wall_hours,
        "conservative_h100_wall_hours": (
            training_wall_hours + allocation_count * staging_hours
        ),
    }
