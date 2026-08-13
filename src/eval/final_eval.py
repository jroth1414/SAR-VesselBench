"""Once-only verified-scene evaluation after the complete TEST cohort gate.

The final labels are opened only after all 32 training bindings and all 32
immutable TEST results validate.  The once-only lock is then created
exclusively before ``validation.csv`` is read.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.h100.contracts import load_cells
from src.analysis.curves import (
    GRID_COLUMNS,
    GRID_COUNT_COLUMNS,
    MONOTONICITY_TOLERANCE,
    training_fraction_counts,
)
from src.eval.heldout_contract import (
    COHORT_FILENAME,
    TEST_RESULT_FILENAME,
    HeldoutContractError,
    cohort_record,
    validate_complete_test_cohort,
    validate_training_cohort,
)
from src.eval.final_authorization import (
    AUTHORIZATION_FILENAME,
    build_authorization,
    validate_authorization,
)
from src.eval.result_contract import (
    ResultContractError,
    read_lightning_checkpoint_epoch,
    sha256_file,
)

LOCKFILE = Path("runs/final_eval.lock")
FINAL_CELL_RESULT_FILENAME = "final_verified_metrics.json"
FINAL_CELL_RESULT_SCHEMA = 1
FINAL_DATA_VIEW_FILENAME = "FINAL_DATA_VIEW.json"
FINAL_CONSUMPTION_FILENAME = "FINAL_GROUND_TRUTH_CONSUMED.json"
FINAL_NORMALIZED_GT_FILENAME = "FINAL_NORMALIZED_GROUND_TRUTH.json"
FINAL_COMPLETE_FILENAME = "FINAL_EVAL_COMPLETE.json"
LEGACY_EVAL_FRACS = (0.1, 0.25, 1.0)
AMENDED_EVAL_FRACS = (0.1, 0.25, 0.5, 1.0)
AMENDED_FINAL_INTERPRETATION = (
    "descriptive-exploratory-post-test-owner-amendment;"
    "predeclared-test-monotonicity-failed"
)
LEGACY_FINAL_INTERPRETATION = "predeclared-green-grid-final-evaluation"
FINAL_STUDY_DESIGN = "single-seed-0-point-estimate-no-uncertainty-estimate"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_STRICT_FP32_BACKEND = {
    "cuda_matmul_fp32_precision": "ieee",
    "cudnn_conv_fp32_precision": "ieee",
    "cudnn_rnn_fp32_precision": "ieee",
}

# A clean amendment may change orchestration, tests, and protocol prose, but
# not the scientific runtime that produced/consumes model predictions.
_SCIENTIFIC_RUNTIME_PATHS = (
    "configs",
    "data/splits.json",
    "data/stats.json",
    "data/lsssdd_split.json",
    "src/data",
    "src/models",
    "src/train",
    "src/eval/decode.py",
    "src/eval/ground_truth.py",
    "src/eval/heldout_contract.py",
    "src/eval/infer_scene.py",
    "src/eval/result_contract.py",
    "src/eval/scorer.py",
    "src/eval/threshold.py",
    "src/analysis/curves.py",
    "scripts/h100/contracts.py",
)


def _repo_path(repo: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo / path


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HeldoutContractError("could not resolve the active source SHA") from exc


def _require_clean_evaluator(repo: Path, *, campaign_git_sha: str) -> str:
    """Bind a clean descendant whose scientific runtime matches the campaign."""

    evaluator_git_sha = _git_sha(repo)
    if not _HEX40.fullmatch(campaign_git_sha) or not _HEX40.fullmatch(
        evaluator_git_sha
    ):
        raise HeldoutContractError("campaign/evaluator Git SHA is malformed")
    commands = (
        ["git", "diff", "--quiet"],
        ["git", "diff", "--cached", "--quiet"],
        ["git", "merge-base", "--is-ancestor", campaign_git_sha, evaluator_git_sha],
        [
            "git",
            "diff",
            "--quiet",
            campaign_git_sha,
            evaluator_git_sha,
            "--",
            *_SCIENTIFIC_RUNTIME_PATHS,
        ],
    )
    try:
        for command in commands:
            subprocess.run(command, cwd=repo, check=True)
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HeldoutContractError(
            "final evaluator must be a clean campaign descendant with an "
            "unchanged scientific runtime"
        ) from exc
    unexpected = [line for line in status if line != "?? slurm/h100/site.env"]
    if unexpected:
        raise HeldoutContractError(
            "final evaluator checkout contains uncommitted or untracked files: "
            + ", ".join(unexpected[:5])
        )
    return evaluator_git_sha


def _write_once_lock(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise HeldoutContractError(
            f"{path} exists; the once-only final evaluation was already attempted"
        ) from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=1, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
        _fsync_directory(path.parent)
    except Exception:
        # The lock intentionally remains after any post-create failure: opening
        # verified-final resources is a once-only event requiring human review.
        raise


def _write_once_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise HeldoutContractError(f"refusing to replace immutable output: {path}") from exc
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finite_grid_float(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HeldoutContractError(f"grid.csv {field} must be finite") from exc
    if not math.isfinite(result):
        raise HeldoutContractError(f"grid.csv {field} must be finite")
    return result


def _cohort_grid_training_values(
    *,
    cohort: Mapping[str, object],
    runs_root: Path,
    exp_id: str,
) -> tuple[float, float, int]:
    """Rebind grid training fields to the frozen marker bytes for one cell."""

    record = cohort_record(cohort, exp_id)
    marker_binding = record.get("completion_marker")
    best_dev = record.get("best_dev")
    expected_relative = f"{exp_id}/final_metrics.json"
    if (
        not isinstance(marker_binding, Mapping)
        or set(marker_binding) != {"relative_path", "sha256"}
        or marker_binding.get("relative_path") != expected_relative
        or not isinstance(best_dev, Mapping)
    ):
        raise HeldoutContractError(
            f"{exp_id}: frozen cohort training binding is invalid"
        )

    run_dir = runs_root.absolute() / exp_id
    marker = run_dir / "final_metrics.json"
    if run_dir.is_symlink() or marker.is_symlink() or not marker.is_file():
        raise HeldoutContractError(
            f"{exp_id}: cohort-bound completion marker path is unsafe"
        )
    try:
        if marker.stat().st_mode & 0o222:
            raise HeldoutContractError(
                f"{exp_id}: cohort-bound completion marker is not immutable"
            )
        marker_bytes = marker.read_bytes()
    except OSError as exc:
        raise HeldoutContractError(
            f"{exp_id}: could not read cohort-bound completion marker"
        ) from exc
    if hashlib.sha256(marker_bytes).hexdigest() != marker_binding.get("sha256"):
        raise HeldoutContractError(
            f"{exp_id}: cohort-bound completion marker drifted"
        )
    try:
        marker_payload = json.loads(marker_bytes)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HeldoutContractError(
            f"{exp_id}: cohort-bound completion marker is invalid JSON"
        ) from exc
    if (
        not isinstance(marker_payload, Mapping)
        or marker_payload.get("exp_id") != exp_id
        or marker_payload.get("best_dev") != best_dev
    ):
        raise HeldoutContractError(
            f"{exp_id}: cohort and completion marker training bindings differ"
        )
    epochs_run = marker_payload.get("epochs_run")
    if type(epochs_run) is not int or epochs_run <= 0:
        raise HeldoutContractError(
            f"{exp_id}: cohort-bound epochs_run must be a positive integer"
        )
    return (
        _finite_grid_float(best_dev.get("f1"), field=f"{exp_id}.cohort_dev_f1"),
        _finite_grid_float(
            best_dev.get("threshold"), field=f"{exp_id}.cohort_dev_threshold"
        ),
        epochs_run,
    )


def _validated_final_checkpoint(
    *,
    record: Mapping[str, object],
    runs_root: Path,
    exp_id: str,
) -> Path:
    """Rehash and epoch-check one sealed cohort checkpoint immediately."""

    if record.get("exp_id") != exp_id:
        raise HeldoutContractError(f"{exp_id}: cohort checkpoint record is misbound")
    binding = record.get("best_checkpoint")
    if not isinstance(binding, Mapping) or set(binding) != {
        "relative_path",
        "sha256",
        "epoch",
    }:
        raise HeldoutContractError(f"{exp_id}: cohort checkpoint binding is invalid")
    relative_value = binding.get("relative_path")
    if not isinstance(relative_value, str) or not relative_value:
        raise HeldoutContractError(f"{exp_id}: cohort checkpoint path is invalid")
    relative = Path(relative_value)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_value
        or not relative.parts
        or relative.parts[0] != exp_id
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise HeldoutContractError(f"{exp_id}: cohort checkpoint path is unsafe")
    root = runs_root.absolute()
    run_dir = root / exp_id
    checkpoint = root / relative
    if root.is_symlink() or run_dir.is_symlink():
        raise HeldoutContractError(f"{exp_id}: cohort checkpoint path is unsafe")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise HeldoutContractError(
                f"{exp_id}: cohort checkpoint path contains a symlink"
            )
    try:
        resolved_run = run_dir.resolve(strict=True)
        resolved_checkpoint = checkpoint.resolve(strict=True)
        resolved_checkpoint.relative_to(resolved_run)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HeldoutContractError(
            f"{exp_id}: cohort checkpoint is absent or escapes its run directory"
        ) from exc
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise HeldoutContractError(f"{exp_id}: cohort checkpoint is absent or empty")
    if checkpoint.stat().st_mode & 0o222:
        raise HeldoutContractError(f"{exp_id}: cohort checkpoint is writable")
    expected_epoch = binding.get("epoch")
    if type(expected_epoch) is not int or expected_epoch < 0:
        raise HeldoutContractError(f"{exp_id}: cohort checkpoint epoch is invalid")
    try:
        if sha256_file(checkpoint) != binding.get("sha256"):
            raise HeldoutContractError(
                f"{exp_id}: cohort checkpoint SHA-256 drifted"
            )
        actual_epoch = read_lightning_checkpoint_epoch(checkpoint)
    except ResultContractError as exc:
        raise HeldoutContractError(
            f"{exp_id}: cohort checkpoint could not be revalidated"
        ) from exc
    if actual_epoch != expected_epoch:
        raise HeldoutContractError(
            f"{exp_id}: cohort checkpoint epoch differs from its binding"
        )
    return checkpoint


def validate_grid_contract(
    path: Path,
    *,
    cells: Sequence[object],
    test_results: dict[str, dict],
    cohort: Mapping[str, object],
    runs_root: Path,
    arms: dict[str, dict],
    fraction_counts: dict[float, dict[str, int]],
    git_sha: str,
    detector_sha256: str,
    precision: str,
) -> dict[str, object]:
    """Validate all grid bindings and return the truthful STOP diagnostic.

    This function never waives monotonicity; it only separates the diagnostic
    from every other hard contract so an exact owner receipt can bind it.
    """

    if path.is_symlink() or not path.is_file():
        raise HeldoutContractError(
            "verified final evaluation requires a regular non-symlink grid.csv"
        )
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != GRID_COLUMNS:
                raise HeldoutContractError(
                    "grid.csv columns do not match the exact reportable schema"
                )
            rows = list(reader)
    except OSError as exc:
        raise HeldoutContractError("could not read grid.csv") from exc

    expected = {str(cell.exp_id): cell for cell in cells}
    if len(rows) != 32 or len(expected) != 32:
        raise HeldoutContractError("grid.csv must contain the exact 32-cell cohort")
    observed_ids = [row["exp_id"] for row in rows]
    if len(set(observed_ids)) != 32 or set(observed_ids) != set(expected):
        raise HeldoutContractError("grid.csv IDs do not match the exact 32-cell cohort")

    points: dict[str, list[tuple[float, float]]] = {}
    flags: dict[str, list[str]] = {}
    for row in rows:
        exp_id = row["exp_id"]
        cell = expected[exp_id]
        try:
            arm = arms[str(cell.init)]
            expected_role = str(arm["role"])
            test_f1 = _finite_grid_float(row["test_f1"], field=f"{exp_id}.test_f1")
            fraction = _finite_grid_float(
                row["label_frac"], field=f"{exp_id}.label_frac"
            )
            seed = int(row["seed"])
            epochs_run = int(row["epochs_run"])
            counts = {name: int(row[name]) for name in GRID_COUNT_COLUMNS}
            dev_f1 = _finite_grid_float(row["dev_f1"], field=f"{exp_id}.dev_f1")
            dev_threshold = _finite_grid_float(
                row["dev_threshold"], field=f"{exp_id}.dev_threshold"
            )
            bound_test_f1 = _finite_grid_float(
                test_results[exp_id]["metrics"]["f1"],
                field=f"{exp_id}.bound_test_f1",
            )
            bound_dev_f1, bound_dev_threshold, bound_epochs_run = (
                _cohort_grid_training_values(
                    cohort=cohort, runs_root=runs_root, exp_id=exp_id
                )
            )
            monotonicity_flag = row["monotonicity_ok"].strip().casefold()
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise HeldoutContractError(
                f"grid.csv row for {exp_id} is malformed or unbound"
            ) from exc
        if (
            row["init"] != str(cell.init)
            or row["track"] != str(cell.track)
            or row["role"] != expected_role
            or not math.isclose(fraction, float(cell.fraction), abs_tol=0.0)
            or seed != int(cell.seed)
            or row["precision"] != precision
            or row["detector_sha256"] != detector_sha256
            or row["git_sha"] != git_sha
            or not 0.0 <= dev_f1 <= 1.0
            or not math.isclose(
                dev_f1, bound_dev_f1, rel_tol=1e-12, abs_tol=1e-12
            )
            or not 0.0 <= dev_threshold <= 1.0
            or not math.isclose(
                dev_threshold,
                bound_dev_threshold,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not 0.0 <= test_f1 <= 1.0
            or not math.isclose(test_f1, bound_test_f1, rel_tol=1e-12, abs_tol=1e-12)
            or epochs_run != bound_epochs_run
            or counts != fraction_counts.get(float(cell.fraction))
            or counts["train_scene_count"] <= 0
            or counts["train_vessel_count"] <= 0
            or counts["train_dark_vessel_count"] < 0
            or counts["train_near_shore_vessel_count"] < 0
            or counts["train_dark_vessel_count"] > counts["train_vessel_count"]
            or counts["train_near_shore_vessel_count"]
            > counts["train_vessel_count"]
            or monotonicity_flag not in {"true", "false"}
        ):
            raise HeldoutContractError(
                f"grid.csv row for {exp_id} disagrees with the frozen cohort"
            )
        points.setdefault(str(cell.init), []).append((fraction, test_f1))
        flags.setdefault(str(cell.init), []).append(monotonicity_flag)

    if set(points) != {str(cell.init) for cell in cells}:
        raise HeldoutContractError("grid.csv arm set is incomplete")
    violations: list[dict[str, object]] = []
    violating_inits: set[str] = set()
    for init_name, arm_points in sorted(points.items()):
        ordered = sorted(arm_points)
        if [point[0] for point in ordered] != [0.1, 0.25, 0.5, 1.0]:
            raise HeldoutContractError(
                f"grid.csv fractions are incomplete for {init_name}"
            )
        for (from_fraction, previous), (to_fraction, current) in zip(
            ordered, ordered[1:]
        ):
            if current < previous - MONOTONICITY_TOLERANCE:
                violating_inits.add(init_name)
                violations.append(
                    {
                        "init": init_name,
                        "from_fraction": from_fraction,
                        "to_fraction": to_fraction,
                        "from_test_f1": previous,
                        "to_test_f1": current,
                        "drop": previous - current,
                    }
                )

    for init_name, observed_flags in flags.items():
        expected_flag = "false" if init_name in violating_inits else "true"
        if len(observed_flags) != 4 or set(observed_flags) != {expected_flag}:
            raise HeldoutContractError(
                "grid.csv monotonicity STOP: flags disagree with the independently "
                f"recomputed diagnostic for {init_name}"
            )
    return {
        "sha256": sha256_file(path),
        "monotonicity_tolerance": MONOTONICITY_TOLERANCE,
        "monotonicity_ok": not violations,
        "violations": violations,
    }


def validate_grid_gate(path: Path, **kwargs: object) -> str:
    """Legacy green-only gate retained as the default fail-closed behavior."""

    audit = validate_grid_contract(path, **kwargs)
    violations = audit["violations"]
    if violations:
        first = violations[0]
        raise HeldoutContractError(
            "grid.csv monotonicity STOP condition is false for "
            f"{first['init']}"
        )
    return str(audit["sha256"])


def _regular_json(path: Path, description: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise HeldoutContractError(
            f"{description} must be a regular non-symlink: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeldoutContractError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise HeldoutContractError(f"{description} root must be a JSON object")
    return payload


def _ground_truth_points_from_normalized(
    value: object, *, expected_scene_ids: Sequence[str]
) -> dict[str, list[object]]:
    """Rebuild frozen-scorer GT from the post-lock canonical normalized form."""

    from src.eval.scorer import GroundTruthPoint

    if not isinstance(value, Mapping) or list(sorted(value)) != list(
        expected_scene_ids
    ):
        raise HeldoutContractError(
            "normalized final ground-truth scene inventory is invalid"
        )
    expected_keys = {
        "x_m",
        "y_m",
        "confidence",
        "source",
        "distance_from_shore_km",
    }
    normalized: dict[str, list[object]] = {}
    for scene_id in expected_scene_ids:
        rows = value.get(scene_id)
        if not isinstance(rows, list):
            raise HeldoutContractError(
                f"normalized final ground-truth rows are invalid for {scene_id}"
            )
        points: list[object] = []
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != expected_keys:
                raise HeldoutContractError(
                    f"normalized final ground-truth point is invalid for {scene_id}"
                )
            x_m = row.get("x_m")
            y_m = row.get("y_m")
            confidence = row.get("confidence")
            source = row.get("source")
            shore = row.get("distance_from_shore_km")
            if (
                isinstance(x_m, bool)
                or not isinstance(x_m, (int, float))
                or not math.isfinite(float(x_m))
                or isinstance(y_m, bool)
                or not isinstance(y_m, (int, float))
                or not math.isfinite(float(y_m))
                or confidence not in {"HIGH", "MEDIUM", "LOW"}
                or (source is not None and not isinstance(source, str))
                or (
                    shore is not None
                    and (
                        isinstance(shore, bool)
                        or not isinstance(shore, (int, float))
                        or not math.isfinite(float(shore))
                    )
                )
            ):
                raise HeldoutContractError(
                    f"normalized final ground-truth value is invalid for {scene_id}"
                )
            points.append(
                GroundTruthPoint(
                    x_m=float(x_m),
                    y_m=float(y_m),
                    confidence=str(confidence),
                    source=source,
                    distance_from_shore_km=(
                        float(shore) if shore is not None else None
                    ),
                )
            )
        normalized[scene_id] = points
    return normalized


def _campaign_git_sha_from_cohort(path: Path) -> str:
    payload = _regular_json(path, "training cohort")
    value = payload.get("git_sha")
    if not isinstance(value, str) or not _HEX40.fullmatch(value):
        raise HeldoutContractError("training cohort Git SHA is malformed")
    return value


def _campaign_binding(
    *,
    runs_root: Path,
    campaign_git_sha: str,
    detector_sha256: str,
    cohort_sha256: str,
    expected_ids: Sequence[str],
) -> dict[str, object]:
    manifest_path = runs_root / ".h100" / "campaign_manifest.json"
    manifest = _regular_json(manifest_path, "H100 campaign manifest")
    events = manifest.get("events")
    failures = manifest.get("fail_stop")
    cohort_binding = manifest.get("training_cohort")
    cell_order = manifest.get("cell_order")
    complete = manifest.get("complete")
    training_complete = manifest.get("training_complete")
    test_complete = manifest.get("test_complete")
    if (
        manifest.get("schema") != 2
        or manifest.get("status") != "failed"
        or manifest.get("phase") != "score-test"
        or manifest.get("git_sha") != campaign_git_sha
        or manifest.get("detector_sha256") != detector_sha256
        or cell_order != list(expected_ids)
        or not isinstance(complete, list)
        or len(complete) != 32
        or len(set(complete)) != 32
        or set(complete) != set(expected_ids)
        or not isinstance(training_complete, list)
        or len(training_complete) != 32
        or len(set(training_complete)) != 32
        or set(training_complete) != set(expected_ids)
        or not isinstance(test_complete, list)
        or len(test_complete) != 32
        or len(set(test_complete)) != 32
        or set(test_complete) != set(expected_ids)
        or manifest.get("running") != {}
        or not isinstance(failures, Mapping)
        or dict(failures)
        != {"engaged": True, "failed": [], "allowed_to_finish": []}
        or not isinstance(cohort_binding, Mapping)
        or cohort_binding.get("sha256") != cohort_sha256
        or not isinstance(events, list)
        or not events
    ):
        raise HeldoutContractError(
            "owner amendment requires the exact completed TEST cohort and "
            "scientific-stop campaign manifest"
        )
    terminal = events[-1]
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("event") != "grid_validation_failed"
        or not isinstance(terminal.get("error"), str)
        or "monotonicity STOP" not in str(terminal.get("error"))
        or any(
            isinstance(event, Mapping) and event.get("event") == "grid_validated"
            for event in events
        )
    ):
        raise HeldoutContractError(
            "campaign did not terminate at the disclosed monotonicity STOP"
        )
    campaign_id = manifest.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise HeldoutContractError("campaign manifest lacks its campaign ID")
    return {
        "campaign_id": campaign_id,
        "git_sha": campaign_git_sha,
        "runs_root": str(runs_root),
        "manifest": {
            "relative_path": ".h100/campaign_manifest.json",
            "sha256": sha256_file(manifest_path),
        },
        "terminal_event": {
            "event": "grid_validation_failed",
            "error": str(terminal["error"]),
            "utc": str(terminal["utc"]),
        },
    }


def _validated_h100_hardware_class(hardware: object) -> dict[str, object]:
    """Validate one exact eight-H100 strict-FP32 allocation receipt."""

    if not isinstance(hardware, Mapping):
        raise HeldoutContractError("H100 hardware receipt is not a JSON object")
    backend = hardware.get("backend")
    devices = hardware.get("devices")
    children = hardware.get("child_probes")
    if backend != _STRICT_FP32_BACKEND:
        raise HeldoutContractError("H100 hardware backend is not exact strict IEEE FP32")
    try:
        from scripts.h100.acceptance import validate_hardware_runtime_contracts
        from scripts.h100.campaign import hardware_class
        from scripts.h100.contracts import validate_gpu_inventory
        from scripts.h100.strict_fp32_probe import bind_child_probes

        inventory = validate_gpu_inventory(devices if isinstance(devices, list) else [])
        if not isinstance(children, list):
            raise RuntimeError("H100 hardware child-probe list is absent")
        rebound = bind_child_probes(
            inventory,
            children,
            expected_backend=_STRICT_FP32_BACKEND,
        )
        if rebound != children:
            raise RuntimeError("H100 child probes are not in canonical bound form")
        validate_hardware_runtime_contracts(hardware)
        return hardware_class(hardware)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise HeldoutContractError(f"invalid strict-FP32 H100 hardware: {exc}") from exc


def _phase5_control_bindings(
    *,
    runs_root: Path,
    campaign_git_sha: str,
    campaign_id: str,
) -> dict[str, object]:
    """Bind the already-persisted reporting controls without weakening them."""

    ready_path = runs_root / ".h100" / "H100_READY.json"
    cutover_path = runs_root / ".h100" / "CUTOVER_READY.json"
    isolation_path = runs_root / ".h100" / "V100_DIAGNOSTIC_ISOLATION.json"
    ready = _regular_json(ready_path, "H100_READY")
    cutover = _regular_json(cutover_path, "CUTOVER_READY")
    isolation = _regular_json(isolation_path, "V100 diagnostic isolation")
    for path, description in (
        (ready_path, "H100_READY"),
        (cutover_path, "CUTOVER_READY"),
        (isolation_path, "V100 diagnostic isolation"),
    ):
        try:
            if path.stat().st_mode & 0o222:
                raise HeldoutContractError(f"{description} is writable")
        except OSError as exc:
            raise HeldoutContractError(f"could not inspect {description}") from exc

    cutover_sha256 = sha256_file(cutover_path)
    ready_sha256 = sha256_file(ready_path)
    campaign_manifest = _regular_json(
        runs_root / ".h100" / "campaign_manifest.json", "H100 campaign manifest"
    )
    acceptance = cutover.get("acceptance")
    source = acceptance.get("source") if isinstance(acceptance, Mapping) else None
    references = cutover.get("references")
    ready_source = ready.get("source")
    ready_hardware = ready.get("hardware")
    if (
        ready.get("schema") != 2
        or ready.get("status") != "ready"
        or not isinstance(ready_source, Mapping)
        or ready_source.get("git_sha") != campaign_git_sha
        or ready.get("strict_fp32") != _STRICT_FP32_BACKEND
        or not isinstance(ready_hardware, Mapping)
        or ready_hardware.get("backend") != _STRICT_FP32_BACKEND
        or campaign_manifest.get("campaign_id") != campaign_id
        or campaign_manifest.get("git_sha") != campaign_git_sha
        or campaign_manifest.get("h100_ready_sha256") != ready_sha256
        or campaign_manifest.get("strict_fp32") != _STRICT_FP32_BACKEND
        or cutover.get("schema") != 2
        or cutover.get("status") != "cutover-ready"
        or cutover.get("h100_campaign_id") != campaign_id
        or not isinstance(source, Mapping)
        or source.get("git_sha") != campaign_git_sha
        or cutover.get("h100_ready") != ready
        or not isinstance(references, Mapping)
        or set(references) != {"r2", "r3"}
        or cutover.get("v100_action")
        != "none; this guard never stops or signals V100 processes"
    ):
        raise HeldoutContractError(
            "CUTOVER_READY does not bind the corrected references and H100 campaign"
        )
    try:
        from scripts.h100.contracts import (
            cutover_acceptance_bindings,
            validate_bound_cutover_forecast,
        )

        accepted_hardware_class = _validated_h100_hardware_class(ready_hardware)
        campaign_hardware_class = _validated_h100_hardware_class(
            campaign_manifest.get("hardware")
        )
        if (
            campaign_manifest.get("accepted_hardware_class")
            != accepted_hardware_class
            or campaign_manifest.get("allocation_hardware_class")
            != accepted_hardware_class
            or campaign_hardware_class != accepted_hardware_class
        ):
            raise RuntimeError(
                "campaign H100 hardware class differs from canonical H100_READY"
            )
        validate_bound_cutover_forecast(cutover)
        if acceptance != cutover_acceptance_bindings(ready):
            raise RuntimeError(
                "CUTOVER_READY acceptance differs from canonical H100_READY"
            )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise HeldoutContractError(
            f"H100 readiness/cutover validation failed: {exc}"
        ) from exc

    h100 = isolation.get("h100")
    namespaces = isolation.get("namespaces")
    suppression = isolation.get("h100_suppression")
    if (
        isolation.get("schema") != 1
        or isolation.get("status") != "v100-diagnostic-isolated"
        or isolation.get("attestation") != "external-human-operator"
        or isolation.get("cutover_ready_sha256") != cutover_sha256
        or not isinstance(h100, Mapping)
        or h100.get("git_sha") != campaign_git_sha
        or h100.get("campaign_id") != campaign_id
        or not isinstance(namespaces, Mapping)
        or namespaces.get("h100_runs_root") != str(runs_root)
        or namespaces.get("disjoint") is not True
        or not isinstance(suppression, Mapping)
        or suppression.get("v100_completions_suppress_h100") is not False
        or suppression.get("v100_checkpoints_resume_h100") is not False
        or suppression.get("mixed_hardware_curve_allowed") is not False
    ):
        raise HeldoutContractError(
            "V100 diagnostic-isolation evidence does not bind the H100 namespace"
        )
    try:
        from scripts.h100.operator_cutover import validate_diagnostic_isolation

        reference_record = cutover.get("reference_campaign")
        reference_manifest = (
            reference_record.get("manifest")
            if isinstance(reference_record, Mapping)
            else None
        )
        v100 = isolation.get("v100")
        if not isinstance(reference_manifest, Mapping) or not isinstance(v100, Mapping):
            raise RuntimeError("reference/V100 identity records are absent")
        validate_diagnostic_isolation(
            cutover_ready=cutover_path,
            cutover_ready_sha256=cutover_sha256,
            attestation=isolation_path,
            attestation_sha256=sha256_file(isolation_path),
            expected_h100_git_sha=campaign_git_sha,
            expected_h100_campaign_id=campaign_id,
            expected_h100_runs_root=str(runs_root),
            expected_reference_git_sha=str(reference_manifest["git_sha"]),
            expected_reference_campaign_id=str(reference_manifest["campaign_id"]),
            expected_v100_core_git_sha=str(v100["git_sha"]),
            expected_v100_core_campaign_id=str(v100["campaign_id"]),
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise HeldoutContractError(
            f"diagnostic-isolation validation failed: {exc}"
        ) from exc
    return {
        "status": "reporting-controls-persisted",
        "h100_ready": {
            "relative_path": ".h100/H100_READY.json",
            "sha256": ready_sha256,
        },
        "cutover_ready": {
            "relative_path": ".h100/CUTOVER_READY.json",
            "sha256": cutover_sha256,
        },
        "v100_diagnostic_isolation": {
            "relative_path": ".h100/V100_DIAGNOSTIC_ISOLATION.json",
            "sha256": sha256_file(isolation_path),
        },
    }


def _test_result_bindings(
    *, cells: Sequence[object], runs_root: Path
) -> list[dict[str, str]]:
    bindings = []
    for cell in cells:
        exp_id = str(cell.exp_id)
        relative = f"{exp_id}/{TEST_RESULT_FILENAME}"
        bindings.append(
            {
                "exp_id": exp_id,
                "relative_path": relative,
                "sha256": sha256_file(runs_root / relative),
            }
        )
    return bindings


def _metric_payload(metric: object) -> dict[str, int | float]:
    return {
        "f1": float(getattr(metric, "f1")),
        "precision": float(getattr(metric, "precision")),
        "recall": float(getattr(metric, "recall")),
        "tp": int(getattr(metric, "tp")),
        "fp": int(getattr(metric, "fp")),
        "fn": int(getattr(metric, "fn")),
        "ignored_predictions": int(getattr(metric, "ignored_predictions")),
    }


def _per_scene_payload(result: object) -> dict[str, object]:
    return {
        scene_id: {
            "aggregate": _metric_payload(scene.aggregate),
            "slices": {
                name: _metric_payload(metric)
                for name, metric in sorted(scene.slices.items())
            },
            "matches": [
                {
                    "prediction_index": int(match.prediction_index),
                    "ground_truth_index": (
                        int(match.ground_truth_index)
                        if match.ground_truth_index is not None
                        else None
                    ),
                    "distance_m": (
                        float(match.distance_m)
                        if match.distance_m is not None
                        else None
                    ),
                    "outcome": str(match.outcome),
                }
                for match in scene.matches
            ],
        }
        for scene_id, scene in sorted(result.scene_results.items())
    }


def _validated_metric_record(
    value: object, *, description: str
) -> dict[str, int | float]:
    keys = {"f1", "precision", "recall", "tp", "fp", "fn", "ignored_predictions"}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise HeldoutContractError(f"{description} metric schema is invalid")
    counts: dict[str, int] = {}
    for field in ("tp", "fp", "fn", "ignored_predictions"):
        observed = value.get(field)
        if type(observed) is not int or observed < 0:
            raise HeldoutContractError(
                f"{description}.{field} must be a nonnegative integer"
            )
        counts[field] = observed
    floating: dict[str, float] = {}
    for field in ("f1", "precision", "recall"):
        observed = value.get(field)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise HeldoutContractError(f"{description}.{field} is invalid")
        floating[field] = float(observed)
        if not math.isfinite(floating[field]) or not 0.0 <= floating[field] <= 1.0:
            raise HeldoutContractError(f"{description}.{field} is outside [0,1]")
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    for field, expected in (("precision", precision), ("recall", recall), ("f1", f1)):
        if not math.isclose(floating[field], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise HeldoutContractError(f"{description}.{field} disagrees with counts")
    return {**floating, **counts}


def _validate_final_cell_payload(
    payload: object,
    *,
    cell: object,
    record: Mapping[str, object],
    eval_scene_ids: Sequence[str],
    campaign_git_sha: str,
    evaluator_git_sha: str,
    detector_sha256: str,
    cohort_sha256: str,
    test_result_sha256: str,
    grid_audit: Mapping[str, object],
    authorization_sha256: str | None,
    final_access: Mapping[str, str] | None,
    ground_truth_by_scene: Mapping[str, Sequence[object]],
) -> dict[str, object]:
    """Independently validate one worker artifact before aggregation."""

    keys = {
        "final_result_schema",
        "created_utc",
        "exp_id",
        "init",
        "label_frac",
        "seed",
        "study_design",
        "interpretation",
        "threshold",
        "threshold_source",
        "dev_epoch",
        "checkpoint",
        "campaign_git_sha",
        "evaluator_git_sha",
        "detector_sha256",
        "cohort_sha256",
        "test_result_sha256",
        "owner_authorization_sha256",
        "final_access",
        "test_monotonicity",
        "inference_precision",
        "strict_fp32",
        "eval_scene_ids",
        "metrics",
        "per_scene",
        "thresholded_predictions",
    }
    if not isinstance(payload, Mapping) or set(payload) != keys:
        raise HeldoutContractError(f"{cell.exp_id}: final result schema is invalid")
    created = payload.get("created_utc")
    try:
        parsed = dt.datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HeldoutContractError(
            f"{cell.exp_id}: final result timestamp is invalid"
        ) from exc
    best_dev = record.get("best_dev")
    checkpoint = record.get("best_checkpoint")
    expected_monotonicity = {
        "ok": bool(grid_audit["monotonicity_ok"]),
        "tolerance": float(grid_audit["monotonicity_tolerance"]),
        "violations": list(grid_audit["violations"]),
    }
    expected_interpretation = (
        AMENDED_FINAL_INTERPRETATION
        if authorization_sha256 is not None
        else LEGACY_FINAL_INTERPRETATION
    )
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != dt.timedelta(0)
        or not isinstance(best_dev, Mapping)
        or not isinstance(checkpoint, Mapping)
        or payload.get("final_result_schema") != FINAL_CELL_RESULT_SCHEMA
        or payload.get("exp_id") != str(cell.exp_id)
        or payload.get("init") != str(cell.init)
        or payload.get("label_frac") != float(cell.fraction)
        or payload.get("seed") != int(cell.seed)
        or payload.get("seed") != 0
        or payload.get("study_design") != FINAL_STUDY_DESIGN
        or payload.get("interpretation") != expected_interpretation
        or payload.get("threshold") != float(best_dev.get("threshold"))
        or payload.get("threshold_source") != "best-dev-checkpoint-bound"
        or payload.get("dev_epoch") != int(best_dev.get("epoch"))
        or payload.get("checkpoint") != dict(checkpoint)
        or payload.get("campaign_git_sha") != campaign_git_sha
        or payload.get("evaluator_git_sha") != evaluator_git_sha
        or payload.get("detector_sha256") != detector_sha256
        or payload.get("cohort_sha256") != cohort_sha256
        or payload.get("test_result_sha256") != test_result_sha256
        or payload.get("owner_authorization_sha256") != authorization_sha256
        or payload.get("final_access")
        != (dict(final_access) if final_access is not None else None)
        or payload.get("test_monotonicity") != expected_monotonicity
        or payload.get("inference_precision") != "32-true"
        or payload.get("strict_fp32") != _STRICT_FP32_BACKEND
        or payload.get("eval_scene_ids") != list(eval_scene_ids)
        or list(sorted(ground_truth_by_scene)) != list(eval_scene_ids)
    ):
        raise HeldoutContractError(
            f"{cell.exp_id}: final result provenance differs from frozen evidence"
        )

    metrics = payload.get("metrics")
    aggregate_keys = {
        "f1", "precision", "recall", "tp", "fp", "fn", "ignored_predictions"
    }
    extra_keys = {"dark_recall", "dark_support", "near_shore_f1", "near_shore_support"}
    if not isinstance(metrics, Mapping) or set(metrics) != aggregate_keys | extra_keys:
        raise HeldoutContractError(f"{cell.exp_id}: final aggregate metrics are invalid")
    aggregate = _validated_metric_record(
        {key: metrics[key] for key in aggregate_keys},
        description=f"{cell.exp_id}.aggregate",
    )
    for field in ("dark_support", "near_shore_support"):
        if type(metrics.get(field)) is not int or int(metrics[field]) < 0:
            raise HeldoutContractError(f"{cell.exp_id}: {field} is invalid")
    for field in ("dark_recall", "near_shore_f1"):
        value = metrics.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise HeldoutContractError(f"{cell.exp_id}: {field} is invalid")

    per_scene = payload.get("per_scene")
    predictions = payload.get("thresholded_predictions")
    if (
        not isinstance(per_scene, Mapping)
        or list(sorted(per_scene)) != list(eval_scene_ids)
        or not isinstance(predictions, Mapping)
        or list(sorted(predictions)) != list(eval_scene_ids)
    ):
        raise HeldoutContractError(f"{cell.exp_id}: final scene inventory is invalid")
    aggregate_sums = {key: 0 for key in ("tp", "fp", "fn", "ignored_predictions")}
    slice_sums = {
        name: {key: 0 for key in ("tp", "fp", "fn", "ignored_predictions")}
        for name in ("dark", "near_shore")
    }
    reconstructed_predictions: dict[str, list[object]] = {}
    for scene_id in eval_scene_ids:
        scene = per_scene[scene_id]
        scene_predictions = predictions[scene_id]
        if (
            not isinstance(scene, Mapping)
            or set(scene) != {"aggregate", "slices", "matches"}
            or not isinstance(scene_predictions, list)
        ):
            raise HeldoutContractError(
                f"{cell.exp_id}/{scene_id}: final scene result is invalid"
            )
        scene_aggregate = _validated_metric_record(
            scene["aggregate"], description=f"{cell.exp_id}/{scene_id}.aggregate"
        )
        slices = scene.get("slices")
        matches = scene.get("matches")
        if not isinstance(slices, Mapping) or set(slices) != {"dark", "near_shore"}:
            raise HeldoutContractError(
                f"{cell.exp_id}/{scene_id}: final slice schema is invalid"
            )
        validated_slices = {
            name: _validated_metric_record(
                slices[name], description=f"{cell.exp_id}/{scene_id}.{name}"
            )
            for name in ("dark", "near_shore")
        }
        if not isinstance(matches, list) or len(matches) != len(scene_predictions):
            raise HeldoutContractError(
                f"{cell.exp_id}/{scene_id}: match/prediction counts differ"
            )
        outcomes = {"tp": 0, "fp": 0, "ignored": 0}
        indices: set[int] = set()
        matched_ground_indices: set[int] = set()
        for match in matches:
            if not isinstance(match, Mapping) or set(match) != {
                "prediction_index", "ground_truth_index", "distance_m", "outcome"
            }:
                raise HeldoutContractError(
                    f"{cell.exp_id}/{scene_id}: match schema is invalid"
                )
            pred_index = match.get("prediction_index")
            ground_index = match.get("ground_truth_index")
            distance = match.get("distance_m")
            outcome = match.get("outcome")
            if (
                type(pred_index) is not int
                or pred_index < 0
                or pred_index >= len(scene_predictions)
                or pred_index in indices
                or outcome not in outcomes
                or (ground_index is not None and (type(ground_index) is not int or ground_index < 0))
                or (outcome == "fp" and (ground_index is not None or distance is not None))
                or (
                    outcome != "fp"
                    and (
                        ground_index is None
                        or ground_index >= len(ground_truth_by_scene[scene_id])
                        or (
                            outcome == "tp"
                            and ground_index in matched_ground_indices
                        )
                        or distance is None
                    )
                )
                or (
                    distance is not None
                    and (
                        isinstance(distance, bool)
                        or not isinstance(distance, (int, float))
                        or not math.isfinite(float(distance))
                        or float(distance) < 0.0
                    )
                )
            ):
                raise HeldoutContractError(
                    f"{cell.exp_id}/{scene_id}: match evidence is invalid"
                )
            indices.add(pred_index)
            if outcome == "tp" and ground_index is not None:
                matched_ground_indices.add(ground_index)
            outcomes[str(outcome)] += 1
        if indices != set(range(len(scene_predictions))) or (
            outcomes["tp"] != scene_aggregate["tp"]
            or outcomes["fp"] != scene_aggregate["fp"]
            or outcomes["ignored"] != scene_aggregate["ignored_predictions"]
        ):
            raise HeldoutContractError(
                f"{cell.exp_id}/{scene_id}: match evidence disagrees with metrics"
            )
        threshold = float(payload["threshold"])
        from src.eval.scorer import PredictionPoint

        rebuilt_scene_predictions: list[object] = []
        for prediction in scene_predictions:
            if not isinstance(prediction, Mapping) or set(prediction) != {
                "x_m", "y_m", "score", "distance_from_shore_km"
            }:
                raise HeldoutContractError(
                    f"{cell.exp_id}/{scene_id}: prediction schema is invalid"
                )
            for coordinate in ("x_m", "y_m", "score"):
                value = prediction.get(coordinate)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise HeldoutContractError(
                        f"{cell.exp_id}/{scene_id}: prediction value is invalid"
                    )
            shore = prediction.get("distance_from_shore_km")
            if shore is not None and (
                isinstance(shore, bool)
                or not isinstance(shore, (int, float))
                or not math.isfinite(float(shore))
            ):
                raise HeldoutContractError(
                    f"{cell.exp_id}/{scene_id}: prediction shore distance is invalid"
                )
            if not threshold <= float(prediction["score"]) <= 1.0:
                raise HeldoutContractError(
                    f"{cell.exp_id}/{scene_id}: prediction violates frozen threshold"
                )
            rebuilt_scene_predictions.append(
                PredictionPoint(
                    x_m=float(prediction["x_m"]),
                    y_m=float(prediction["y_m"]),
                    score=float(prediction["score"]),
                    distance_from_shore_km=(
                        float(shore) if shore is not None else None
                    ),
                )
            )
        reconstructed_predictions[scene_id] = rebuilt_scene_predictions
        for key in aggregate_sums:
            aggregate_sums[key] += int(scene_aggregate[key])
            for name in slice_sums:
                slice_sums[name][key] += int(validated_slices[name][key])

    if any(aggregate_sums[key] != aggregate[key] for key in aggregate_sums):
        raise HeldoutContractError(
            f"{cell.exp_id}: per-scene metrics do not aggregate exactly"
        )
    dark_counts = slice_sums["dark"]
    dark_recall = (
        dark_counts["tp"] / (dark_counts["tp"] + dark_counts["fn"])
        if dark_counts["tp"] + dark_counts["fn"]
        else 0.0
    )
    near_counts = slice_sums["near_shore"]
    near_precision = near_counts["tp"] / (near_counts["tp"] + near_counts["fp"]) if near_counts["tp"] + near_counts["fp"] else 0.0
    near_recall = near_counts["tp"] / (near_counts["tp"] + near_counts["fn"]) if near_counts["tp"] + near_counts["fn"] else 0.0
    near_f1 = 2.0 * near_precision * near_recall / (near_precision + near_recall) if near_precision + near_recall else 0.0
    if (
        int(metrics["dark_support"]) != dark_counts["tp"] + dark_counts["fn"]
        or not math.isclose(float(metrics["dark_recall"]), dark_recall, rel_tol=1e-12, abs_tol=1e-12)
        or int(metrics["near_shore_support"]) != near_counts["tp"] + near_counts["fn"]
        or not math.isclose(float(metrics["near_shore_f1"]), near_f1, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise HeldoutContractError(
            f"{cell.exp_id}: slice aggregates disagree with per-scene evidence"
        )

    # Self-consistent forged counts are not enough. Re-run the immutable
    # scorer from the consumed canonical GT and serialized predictions, then
    # require every metric and match to equal its independently recomputed
    # value exactly.
    from src.eval.scorer import score_dataset

    try:
        recomputed = score_dataset(ground_truth_by_scene, reconstructed_predictions)
    except ValueError as exc:
        raise HeldoutContractError(
            f"{cell.exp_id}: final predictions cannot be rescored"
        ) from exc
    expected_dark = recomputed.slices["dark"]
    expected_near = recomputed.slices["near_shore"]
    expected_metrics = {
        **_metric_payload(recomputed.aggregate),
        "dark_recall": float(expected_dark.recall),
        "dark_support": int(expected_dark.tp + expected_dark.fn),
        "near_shore_f1": float(expected_near.f1),
        "near_shore_support": int(expected_near.tp + expected_near.fn),
    }
    if dict(metrics) != expected_metrics or dict(per_scene) != _per_scene_payload(
        recomputed
    ):
        raise HeldoutContractError(
            f"{cell.exp_id}: final evidence differs from an independent frozen-scorer pass"
        )
    return dict(payload)


def score_final_cell(
    *,
    cell: object,
    record: Mapping[str, object],
    runs_root: Path,
    raw_root: Path,
    stats: Mapping[str, object],
    det_cfg: Mapping[str, object],
    gt_by_scene: Mapping[str, Sequence[object]],
    eval_scenes: Sequence[str],
    device: str,
    campaign_git_sha: str,
    evaluator_git_sha: str,
    detector_sha256: str,
    cohort_sha256: str,
    test_result_sha256: str,
    grid_audit: Mapping[str, object],
    authorization_sha256: str | None,
    final_access: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Score and immutably publish one checkpoint in the consumed final run."""

    import torch

    if device.startswith("cuda"):
        from scripts.h100.precision import assert_sitecustomize_active

        strict_fp32 = assert_sitecustomize_active(torch)
    else:
        strict_fp32 = {"device": "cpu-test-only"}
    from src.eval.infer_scene import infer_scene
    from src.eval.scorer import score_dataset
    from src.eval.threshold import apply_threshold
    from src.train.lit_modules import HeatmapLitModule

    exp_id = str(cell.exp_id)
    threshold = float(record["best_dev"]["threshold"])
    checkpoint_binding = record["best_checkpoint"]
    checkpoint = _validated_final_checkpoint(
        record=record, runs_root=runs_root, exp_id=exp_id
    )
    module = HeatmapLitModule.load_from_checkpoint(
        str(checkpoint), map_location=device, load_weights=False
    ).eval()
    try:
        pred_by_scene = {
            scene_id: infer_scene(
                module,
                raw_root / scene_id,
                stats=stats,
                tau=det_cfg["decode"]["candidate_floor"],
                d_nms_m=det_cfg["decode"]["d_nms_m"],
                tile_px=det_cfg["eval"]["tile_px"],
                tile_stride_px=det_cfg["eval"]["tile_stride_px"],
                batch_size=det_cfg["eval"]["infer_batch"],
                device=device,
                precision=det_cfg["schedule"]["precision"],
            )
            for scene_id in eval_scenes
        }
        thresholded = apply_threshold(pred_by_scene, threshold)
        result = score_dataset(gt_by_scene, thresholded)
        aggregate = _metric_payload(result.aggregate)
        dark = result.slices["dark"]
        near = result.slices["near_shore"]
        metrics = {
            **aggregate,
            "dark_recall": float(dark.recall),
            "dark_support": int(dark.tp + dark.fn),
            "near_shore_f1": float(near.f1),
            "near_shore_support": int(near.tp + near.fn),
        }
        payload: dict[str, object] = {
            "final_result_schema": FINAL_CELL_RESULT_SCHEMA,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "exp_id": exp_id,
            "init": str(cell.init),
            "label_frac": float(cell.fraction),
            "seed": int(cell.seed),
            "study_design": FINAL_STUDY_DESIGN,
            "interpretation": (
                AMENDED_FINAL_INTERPRETATION
                if authorization_sha256 is not None
                else LEGACY_FINAL_INTERPRETATION
            ),
            "threshold": threshold,
            "threshold_source": "best-dev-checkpoint-bound",
            "dev_epoch": int(record["best_dev"]["epoch"]),
            "checkpoint": {
                "relative_path": str(checkpoint_binding["relative_path"]),
                "sha256": str(checkpoint_binding["sha256"]),
                "epoch": int(checkpoint_binding["epoch"]),
            },
            "campaign_git_sha": campaign_git_sha,
            "evaluator_git_sha": evaluator_git_sha,
            "detector_sha256": detector_sha256,
            "cohort_sha256": cohort_sha256,
            "test_result_sha256": test_result_sha256,
            "owner_authorization_sha256": authorization_sha256,
            "final_access": dict(final_access) if final_access is not None else None,
            "test_monotonicity": {
                "ok": bool(grid_audit["monotonicity_ok"]),
                "tolerance": float(grid_audit["monotonicity_tolerance"]),
                "violations": list(grid_audit["violations"]),
            },
            "inference_precision": str(det_cfg["schedule"]["precision"]),
            "strict_fp32": strict_fp32,
            "eval_scene_ids": list(eval_scenes),
            "metrics": metrics,
            "per_scene": _per_scene_payload(result),
            "thresholded_predictions": {
                scene_id: [
                    {
                        "x_m": float(prediction.x_m),
                        "y_m": float(prediction.y_m),
                        "score": float(prediction.score),
                        "distance_from_shore_km": (
                            float(prediction.distance_from_shore_km)
                            if prediction.distance_from_shore_km is not None
                            else None
                        ),
                    }
                    for prediction in thresholded[scene_id]
                ]
                for scene_id in eval_scenes
            },
        }
        output = runs_root / exp_id / FINAL_CELL_RESULT_FILENAME
        _write_once_lock(output, payload)
        return payload
    finally:
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run_parallel_final_workers(
    *,
    repo: Path,
    runs_root: Path,
    selected: Sequence[object],
    normalized_gt_bytes: bytes,
    worker_count: int,
) -> list[dict[str, object]]:
    """Run one cell/GPU with fail-stop scheduling and no worker retries."""

    if worker_count != 8 or len(selected) != 32:
        raise HeldoutContractError(
            "amended parallel final evaluation requires exactly 8 GPUs and 32 cells"
        )
    lock_sha256 = sha256_file(runs_root / "final_eval.lock")
    consumption_sha256 = sha256_file(
        runs_root / ".h100" / FINAL_CONSUMPTION_FILENAME
    )
    lock_payload = _regular_json(runs_root / "final_eval.lock", "final-eval lock")
    cohort = _regular_json(
        runs_root / ".h100" / COHORT_FILENAME, "training cohort"
    )
    splits = json.loads((repo / "data/splits.json").read_text(encoding="utf-8"))[
        "splits"
    ]
    eval_scene_ids = tuple(sorted(map(str, splits["eval_final"])))
    try:
        normalized_gt = json.loads(normalized_gt_bytes)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HeldoutContractError(
            "normalized final ground truth is invalid before worker launch"
        ) from exc
    if (
        not isinstance(normalized_gt, Mapping)
        or list(sorted(normalized_gt)) != list(eval_scene_ids)
        or any(not isinstance(normalized_gt.get(scene_id), list) for scene_id in eval_scene_ids)
    ):
        raise HeldoutContractError(
            "normalized final ground truth scene inventory is invalid"
        )
    ground_truth_by_scene = _ground_truth_points_from_normalized(
        normalized_gt, expected_scene_ids=eval_scene_ids
    )
    owner_amendment = lock_payload.get("owner_amendment")
    grid = lock_payload.get("grid")
    final_data_view = lock_payload.get("final_data_view")
    test_bindings = lock_payload.get("test_results")
    if (
        lock_payload.get("lock_schema") != 3
        or not isinstance(owner_amendment, Mapping)
        or not isinstance(grid, Mapping)
        or not isinstance(final_data_view, Mapping)
        or not isinstance(test_bindings, list)
        or len(eval_scene_ids) != 50
    ):
        raise HeldoutContractError("final worker controller bindings are malformed")
    test_by_id = {
        str(item["exp_id"]): item
        for item in test_bindings
        if isinstance(item, Mapping) and isinstance(item.get("exp_id"), str)
    }
    if len(test_by_id) != 32:
        raise HeldoutContractError("final worker TEST bindings are incomplete")
    final_access = {
        "lock_sha256": lock_sha256,
        "consumption_sha256": consumption_sha256,
        "data_view_sha256": str(final_data_view.get("sha256")),
    }
    logs_root = runs_root / "logs" / "h100-final"
    logs_root.mkdir(parents=True, exist_ok=True)
    pending = list(selected)
    running: dict[int, tuple[subprocess.Popen[bytes], object, object, Path]] = {}
    completed: dict[str, dict[str, object]] = {}
    failures: list[tuple[str, int]] = []

    def launch(gpu: int, cell: object) -> None:
        result_path = runs_root / str(cell.exp_id) / FINAL_CELL_RESULT_FILENAME
        if os.path.lexists(result_path):
            raise HeldoutContractError(
                f"final worker result already exists; refusing retry: {result_path}"
            )
        log_path = logs_root / f"{cell.exp_id}.log"
        try:
            log_fd = os.open(
                log_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise HeldoutContractError(
                f"final worker log already exists; refusing retry: {log_path}"
            ) from exc
        log_stream = os.fdopen(log_fd, "wb")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        command = [
            sys.executable,
            "-B",
            "-m",
            "src.eval.final_worker",
            "--repo",
            str(repo),
            "--runs-root",
            str(runs_root),
            "--exp-id",
            str(cell.exp_id),
            "--lock-sha256",
            lock_sha256,
            "--consumption-sha256",
            consumption_sha256,
            "--device",
            "cuda",
        ]
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdin=subprocess.PIPE,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
            )
            if process.stdin is None:
                raise HeldoutContractError("final worker stdin pipe was not created")
            process.stdin.write(normalized_gt_bytes)
            process.stdin.close()
        except Exception:
            # A pipe error can occur after Popen succeeded.  Never leave a
            # worker outside ``running``: it could otherwise consume a GPU or
            # publish an immutable result after the controller fail-stopped.
            if process is not None:
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log_stream.close()
            try:
                log_path.chmod(0o444)
            except OSError:
                pass
            raise
        running[gpu] = (process, cell, log_stream, log_path)
        print(f"[final eval] GPU {gpu}: launched {cell.exp_id}", flush=True)

    while pending or running:
        if not failures:
            for gpu in range(worker_count):
                if not pending or gpu in running:
                    continue
                cell = pending.pop(0)
                try:
                    launch(gpu, cell)
                except Exception as exc:
                    failures.append((str(cell.exp_id), -1))
                    print(
                        f"[final eval] STOP: could not launch {cell.exp_id}: {exc}; "
                        "no additional cells will launch",
                        flush=True,
                    )
                    break
        if not running:
            break
        finished = []
        for gpu, (process, cell, log_stream, log_path) in running.items():
            code = process.poll()
            if code is None:
                continue
            log_stream.close()
            log_path.chmod(0o444)
            finished.append(gpu)
            if code != 0:
                failures.append((str(cell.exp_id), int(code)))
                print(
                    f"[final eval] STOP: {cell.exp_id} failed with exit {code}; "
                    "no additional cells will launch",
                    flush=True,
                )
                continue
            try:
                result_path = (
                    runs_root / str(cell.exp_id) / FINAL_CELL_RESULT_FILENAME
                )
                payload = _regular_json(result_path, f"{cell.exp_id} final result")
                if result_path.stat().st_mode & 0o222:
                    raise HeldoutContractError(
                        f"{cell.exp_id}: final worker result is writable"
                    )
                test_binding = test_by_id.get(str(cell.exp_id))
                if not isinstance(test_binding, Mapping):
                    raise HeldoutContractError(
                        f"{cell.exp_id}: final worker TEST binding is absent"
                    )
                completed[str(cell.exp_id)] = _validate_final_cell_payload(
                    payload,
                    cell=cell,
                    record=cohort_record(cohort, str(cell.exp_id)),
                    eval_scene_ids=eval_scene_ids,
                    campaign_git_sha=str(lock_payload["campaign_git_sha"]),
                    evaluator_git_sha=str(lock_payload["evaluator_git_sha"]),
                    detector_sha256=str(lock_payload["detector_sha256"]),
                    cohort_sha256=str(lock_payload["cohort_sha256"]),
                    test_result_sha256=str(test_binding.get("sha256")),
                    grid_audit={
                        "monotonicity_ok": grid.get("monotonicity_ok"),
                        "monotonicity_tolerance": grid.get(
                            "monotonicity_tolerance"
                        ),
                        "violations": grid.get("violations"),
                    },
                    authorization_sha256=str(owner_amendment.get("sha256")),
                    final_access=final_access,
                    ground_truth_by_scene=ground_truth_by_scene,
                )
            except (KeyError, OSError, TypeError, ValueError, HeldoutContractError) as exc:
                failures.append((str(cell.exp_id), -2))
                print(
                    f"[final eval] STOP: invalid result from {cell.exp_id}: {exc}; "
                    "no additional cells will launch",
                    flush=True,
                )
                continue
            print(f"[final eval] GPU {gpu}: completed {cell.exp_id}", flush=True)
        for gpu in finished:
            del running[gpu]
        if running and not finished:
            time.sleep(0.5)

    if failures or pending or len(completed) != 32:
        raise HeldoutContractError(
            "final evaluation is permanently incomplete after label consumption; "
            f"failed={failures}, completed={len(completed)}, pending={len(pending)}; "
            "do not retry or delete the lock"
        )
    return [completed[str(cell.exp_id)] for cell in selected]


def _summary_row(
    *,
    cell: object,
    payload: Mapping[str, object],
    grid_audit: Mapping[str, object],
    campaign_git_sha: str,
    evaluator_git_sha: str,
    detector_sha256: str,
    cohort_sha256: str,
    authorization_sha256: str | None,
    final_result_sha256: str,
) -> dict[str, object]:
    checkpoint = payload["checkpoint"]
    metrics = payload["metrics"]
    if not isinstance(checkpoint, Mapping) or not isinstance(metrics, Mapping):
        raise HeldoutContractError(f"{cell.exp_id}: final result is malformed")
    final_access = payload.get("final_access")
    if final_access is not None and not isinstance(final_access, Mapping):
        raise HeldoutContractError(f"{cell.exp_id}: final access binding is malformed")
    return {
        "exp_id": cell.exp_id,
        "init": cell.init,
        "label_frac": cell.fraction,
        "seed": payload["seed"],
        "study_design": payload["study_design"],
        "threshold": payload["threshold"],
        "threshold_source": "best-dev-checkpoint-bound",
        "dev_epoch": payload["dev_epoch"],
        "checkpoint_relative_path": checkpoint["relative_path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_epoch": checkpoint["epoch"],
        "cohort_sha256": cohort_sha256,
        "test_result_sha256": payload["test_result_sha256"],
        "git_sha": campaign_git_sha,
        "campaign_git_sha": campaign_git_sha,
        "evaluator_git_sha": evaluator_git_sha,
        "owner_authorization_sha256": authorization_sha256,
        "final_result_sha256": final_result_sha256,
        "final_lock_sha256": (
            final_access.get("lock_sha256") if final_access is not None else None
        ),
        "final_consumption_sha256": (
            final_access.get("consumption_sha256")
            if final_access is not None
            else None
        ),
        "final_data_view_sha256": (
            final_access.get("data_view_sha256")
            if final_access is not None
            else None
        ),
        "test_monotonicity_ok": grid_audit["monotonicity_ok"],
        "interpretation": payload["interpretation"],
        "detector_sha256": detector_sha256,
        "inference_precision": payload["inference_precision"],
        **dict(metrics),
    }


def owner_authorization_evidence(
    *,
    owner: str,
    created_utc: str | None,
    evaluator_git_sha: str,
    campaign_git_sha: str,
    detector_sha256: str,
    cohort_sha256: str,
    grid_audit: Mapping[str, object],
    cells: Sequence[object],
    runs_root: Path,
) -> dict[str, object]:
    """Rebuild the exact owner receipt from current immutable evidence."""

    selected = [str(cell.exp_id) for cell in cells]
    if len(selected) != 32 or len(set(selected)) != 32:
        raise HeldoutContractError("owner amendment scope is not the exact 32-cell grid")
    if grid_audit.get("monotonicity_ok") is not False:
        raise HeldoutContractError(
            "post-TEST owner amendment requires a failed monotonicity diagnostic"
        )
    campaign = _campaign_binding(
        runs_root=runs_root,
        campaign_git_sha=campaign_git_sha,
        detector_sha256=detector_sha256,
        cohort_sha256=cohort_sha256,
        expected_ids=selected,
    )
    controls = _phase5_control_bindings(
        runs_root=runs_root,
        campaign_git_sha=campaign_git_sha,
        campaign_id=str(campaign["campaign_id"]),
    )
    return build_authorization(
        owner=owner,
        campaign=campaign,
        evaluator_git_sha=evaluator_git_sha,
        cohort={
            "relative_path": ".h100/TRAINING_COHORT.json",
            "sha256": cohort_sha256,
        },
        phase5_controls=controls,
        grid={
            "relative_path": "summary/grid.csv",
            "sha256": str(grid_audit["sha256"]),
            "monotonicity_tolerance": float(
                grid_audit["monotonicity_tolerance"]
            ),
            "monotonicity_ok": False,
            "violations": list(grid_audit["violations"]),
        },
        test_results=_test_result_bindings(cells=cells, runs_root=runs_root),
        selected_cells=selected,
        created_utc=created_utc,
    )



def main(argv: Sequence[str] | None = None) -> int:
    import pandas as pd
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i-am-sure", action="store_true")
    parser.add_argument(
        "--owner-amendment",
        type=Path,
        help=(
            "immutable hash-bound authorization for the disclosed all-32 "
            "post-TEST monotonicity exception"
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 8),
        default=1,
        help="one legacy process or the amended one-process-per-H100 controller",
    )
    parser.add_argument(
        "--hardware-json",
        type=Path,
        help="strict eight-H100 allocation receipt (required with --workers 8)",
    )
    parser.add_argument(
        "--data-view-json",
        type=Path,
        help="immutable verified Phase-6 data-view receipt (required with --workers 8)",
    )
    parser.add_argument(
        "--data-package-id",
        help="exact content-addressed Phase-6 package ID (required with --workers 8)",
    )
    args = parser.parse_args(argv)

    if not args.i_am_sure:
        raise SystemExit(
            "REFUSING: verified final evaluation is once-only; pass --i-am-sure "
            "only after the complete TEST cohort has been reviewed"
        )
    repo = args.repo.absolute()
    runs_root = args.runs_root.absolute()
    if (
        repo.is_symlink()
        or runs_root.is_symlink()
        or not repo.is_dir()
        or (runs_root.exists() and not runs_root.is_dir())
        or repo.resolve() != repo
        or runs_root.resolve() != runs_root
    ):
        raise HeldoutContractError(
            "repo and runs root must be existing canonical non-symlink directories"
        )
    lockfile = runs_root / "final_eval.lock"
    if lockfile.exists() or lockfile.is_symlink():
        raise SystemExit(
            f"REFUSING: {lockfile} exists; STOP and consult a human, never delete it"
        )

    data_path = _repo_path(repo, args.data_config)
    detector_path = _repo_path(repo, args.detector_config)
    data_cfg = yaml.safe_load(data_path.read_text())
    det_cfg = yaml.safe_load(detector_path.read_text())
    detector_sha256 = hashlib.sha256(detector_path.read_bytes()).hexdigest()
    cells = load_cells(repo)
    cohort_path = runs_root / ".h100" / COHORT_FILENAME
    raw_authorization: dict[str, object] | None = None
    authorization_path: Path | None = None
    if args.owner_amendment is not None:
        authorization_path = args.owner_amendment.absolute()
        canonical_authorization = (
            runs_root / ".h100" / AUTHORIZATION_FILENAME
        )
        if authorization_path != canonical_authorization:
            raise HeldoutContractError(
                f"owner amendment must use the canonical path {canonical_authorization}"
            )
        raw_authorization = _regular_json(
            authorization_path, "owner final-eval authorization"
        )
        campaign_binding = raw_authorization.get("campaign")
        campaign_git_sha = (
            campaign_binding.get("git_sha")
            if isinstance(campaign_binding, Mapping)
            else None
        )
        if not isinstance(campaign_git_sha, str):
            raise HeldoutContractError(
                "owner amendment lacks its campaign Git SHA"
            )
        evaluator_git_sha = _require_clean_evaluator(
            repo, campaign_git_sha=campaign_git_sha
        )
    else:
        campaign_git_sha = _git_sha(repo)
        evaluator_git_sha = campaign_git_sha

    # First barrier: validate every training marker/checkpoint byte without
    # touching TEST or verified-final data.
    cohort, cohort_sha256 = validate_training_cohort(
        path=cohort_path,
        cells=cells,
        runs_root=runs_root,
        git_sha=campaign_git_sha,
        detector_sha256=detector_sha256,
        candidate_floor=float(det_cfg["decode"]["candidate_floor"]),
    )

    # Reading the frozen split manifest is safe. Every TEST result must then
    # validate before eval-final raster checks, lock creation, or validation.csv.
    splits_path = _repo_path(repo, data_cfg["paths"]["splits"])
    splits = json.loads(splits_path.read_text())["splits"]
    test_scene_ids = tuple(sorted(map(str, splits["test"])))
    test_results = validate_complete_test_cohort(
        cells=cells,
        runs_root=runs_root,
        cohort=cohort,
        cohort_sha256=cohort_sha256,
        test_scene_ids=test_scene_ids,
    )

    # Third barrier: validate every scientific binding in the reportable TEST
    # grid and independently recompute the predeclared diagnostic. The default
    # path remains green-only. The sole amended path must bind the truthful
    # failure to an immutable owner receipt before final resources are touched.
    arms = yaml.safe_load((repo / "configs/arms.yaml").read_text())["arms"]
    fraction_counts = training_fraction_counts(
        repo=repo,
        fractions=tuple(sorted({float(cell.fraction) for cell in cells})),
    )
    grid_path = runs_root / "summary" / "grid.csv"
    grid_audit = validate_grid_contract(
        grid_path,
        cells=cells,
        test_results=test_results,
        cohort=cohort,
        runs_root=runs_root,
        arms=arms,
        fraction_counts=fraction_counts,
        git_sha=campaign_git_sha,
        detector_sha256=detector_sha256,
        precision=str(det_cfg["schedule"]["precision"]),
    )

    authorization_payload: dict[str, object] | None = None
    authorization_sha256: str | None = None
    if raw_authorization is None:
        if grid_audit["violations"]:
            first = grid_audit["violations"][0]
            raise HeldoutContractError(
                "grid.csv monotonicity STOP condition is false for "
                f"{first['init']}"
            )
        selected = [
            cell
            for cell in cells
            if float(cell.fraction) in LEGACY_EVAL_FRACS
        ]
        if len(selected) != 24 or len({cell.exp_id for cell in selected}) != 24:
            raise HeldoutContractError(
                "legacy verified final evaluation requires exactly 24 cells"
            )
    else:
        if authorization_path is None:
            raise AssertionError("owner amendment path disappeared")
        expected_authorization = owner_authorization_evidence(
            owner=raw_authorization.get("owner"),
            created_utc=raw_authorization.get("created_utc"),
            evaluator_git_sha=evaluator_git_sha,
            campaign_git_sha=campaign_git_sha,
            detector_sha256=detector_sha256,
            cohort_sha256=cohort_sha256,
            grid_audit=grid_audit,
            cells=cells,
            runs_root=runs_root,
        )
        authorization_payload, authorization_sha256 = validate_authorization(
            authorization_path,
            expected=expected_authorization,
        )
        selected = [
            cell
            for cell in cells
            if float(cell.fraction) in AMENDED_EVAL_FRACS
        ]
        if (
            len(selected) != 32
            or len({cell.exp_id for cell in selected}) != 32
            or [cell.exp_id for cell in selected]
            != authorization_payload["selected_cells"]
        ):
            raise HeldoutContractError(
                "owner-amended verified final evaluation requires the exact 32 cells"
            )
    if args.workers == 8 and authorization_payload is None:
        raise HeldoutContractError(
            "the eight-GPU final controller requires the all-32 owner amendment"
        )
    if authorization_payload is not None and args.workers != 8:
        raise HeldoutContractError(
            "the all-32 owner amendment requires the dedicated eight-GPU controller"
        )
    if args.workers == 8 and args.device != "cuda":
        raise HeldoutContractError(
            "the eight-GPU final controller requires --device cuda"
        )
    hardware_binding: dict[str, object] | None = None
    data_view_binding: dict[str, object] | None = None
    if args.workers == 8:
        if args.hardware_json is None:
            raise HeldoutContractError(
                "the eight-GPU final controller requires --hardware-json"
            )
        hardware_path = args.hardware_json.absolute()
        hardware = _regular_json(hardware_path, "final allocation hardware")
        if hardware_path.stat().st_mode & 0o222:
            raise HeldoutContractError(
                "final allocation hardware receipt must be immutable"
            )
        observed_hardware_class = _validated_h100_hardware_class(hardware)
        backend = hardware.get("backend")
        accepted_ready = _regular_json(
            runs_root / ".h100" / "H100_READY.json", "H100_READY"
        )
        accepted_class = _validated_h100_hardware_class(
            accepted_ready.get("hardware")
        )
        if observed_hardware_class != accepted_class:
            raise HeldoutContractError(
                "final allocation hardware class differs from the accepted campaign"
            )
        hardware_binding = {
            "relative_path": f".h100/{hardware_path.name}",
            "sha256": sha256_file(hardware_path),
            "strict_fp32": dict(backend),
            "hardware_class": observed_hardware_class,
        }
        expected_hardware_path = runs_root / str(hardware_binding["relative_path"])
        if hardware_path != expected_hardware_path:
            raise HeldoutContractError(
                f"final allocation hardware must use canonical path {expected_hardware_path}"
            )
        if args.data_view_json is None:
            raise HeldoutContractError(
                "the eight-GPU final controller requires --data-view-json"
            )
        if not isinstance(args.data_package_id, str) or not args.data_package_id:
            raise HeldoutContractError(
                "the eight-GPU final controller requires --data-package-id"
            )
        data_view_path = args.data_view_json.absolute()
        expected_data_view_path = (
            runs_root / ".h100" / FINAL_DATA_VIEW_FILENAME
        )
        if data_view_path != expected_data_view_path:
            raise HeldoutContractError(
                f"final data view must use canonical path {expected_data_view_path}"
            )
        try:
            from scripts.handoff.final_eval_package import validate_staged_view

            staged_view, staged_view_sha256 = validate_staged_view(
                data_view_path,
                expected_repo=repo,
                expected_package_id=args.data_package_id,
                expected_campaign_git_sha=campaign_git_sha,
                expected_evaluator_git_sha=evaluator_git_sha,
                expected_splits_sha256=sha256_file(splits_path),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise HeldoutContractError(
                f"final data-view validation failed before final access: {exc}"
            ) from exc
        data_view_binding = {
            "relative_path": f".h100/{FINAL_DATA_VIEW_FILENAME}",
            "sha256": staged_view_sha256,
            "package": staged_view["package"],
            "source": staged_view["source"],
            "validation_labels": staged_view["view"]["validation_labels"],
            "scenes": staged_view["view"]["scenes"],
        }
    selected_records: dict[str, Mapping[str, object]] = {}
    for cell in selected:
        record = cohort_record(cohort, cell.exp_id)
        _validated_final_checkpoint(
            record=record, runs_root=runs_root, exp_id=cell.exp_id
        )
        selected_records[cell.exp_id] = record

    eval_scenes = tuple(sorted(map(str, splits["eval_final"])))
    if len(eval_scenes) != 50 or len(set(eval_scenes)) != 50:
        raise HeldoutContractError("verified final split must contain exactly 50 scenes")
    raw_root = _repo_path(repo, data_cfg["paths"]["raw_xview3"]) / "GRD"
    absent = [
        scene_id
        for scene_id in eval_scenes
        if not (raw_root / scene_id / "VH_dB.tif").is_file()
        or not (raw_root / scene_id / "VV_dB.tif").is_file()
    ]
    if absent:
        raise HeldoutContractError(
            f"{len(absent)} eval_final scenes are not extracted (first: {absent[:3]})"
        )

    out = runs_root / "summary" / "final_verified.csv"
    consumed_receipt = runs_root / ".h100" / FINAL_CONSUMPTION_FILENAME
    normalized_gt_path = runs_root / ".h100" / FINAL_NORMALIZED_GT_FILENAME
    complete_receipt = runs_root / ".h100" / FINAL_COMPLETE_FILENAME
    occupied = [
        path
        for path in (
            out,
            consumed_receipt,
            normalized_gt_path,
            complete_receipt,
            *(
                runs_root / cell.exp_id / "final_verified_metrics.json"
                for cell in selected
            ),
            *(
                runs_root / "logs" / "h100-final" / f"{cell.exp_id}.log"
                for cell in selected
            ),
        )
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        raise HeldoutContractError(
            "verified-final output namespace is not empty: "
            + ", ".join(map(str, occupied[:5]))
        )

    test_bindings = _test_result_bindings(cells=cells, runs_root=runs_root)
    grid_lock_binding = {
        "relative_path": "summary/grid.csv",
        "sha256": str(grid_audit["sha256"]),
        "monotonicity_tolerance": float(
            grid_audit["monotonicity_tolerance"]
        ),
        "monotonicity_ok": bool(grid_audit["monotonicity_ok"]),
        "violations": list(grid_audit["violations"]),
    }
    if authorization_payload is None:
        lock_payload: dict[str, object] = {
            "lock_schema": 2,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "policy": "all-32-test-complete-before-once-only-final-access",
            "git_sha": campaign_git_sha,
            "detector_sha256": detector_sha256,
            "cohort_sha256": cohort_sha256,
            "grid": {
                "relative_path": "summary/grid.csv",
                "sha256": str(grid_audit["sha256"]),
            },
            "test_results": test_bindings,
            "selected_cells": [cell.exp_id for cell in selected],
            "eval_scene_count": len(eval_scenes),
            "inference_precision": det_cfg["schedule"]["precision"],
        }
    else:
        lock_payload = {
            "lock_schema": 3,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "policy": authorization_payload["policy"],
            "campaign_git_sha": campaign_git_sha,
            "evaluator_git_sha": evaluator_git_sha,
            "detector_sha256": detector_sha256,
            "cohort_sha256": cohort_sha256,
            "grid": grid_lock_binding,
            "owner_amendment": {
                "relative_path": f".h100/{AUTHORIZATION_FILENAME}",
                "sha256": authorization_sha256,
                "owner": authorization_payload["owner"],
                "decision": authorization_payload["decision"],
            },
            "phase5_controls": authorization_payload["phase5_controls"],
            "test_results": test_bindings,
            "selected_cells": [cell.exp_id for cell in selected],
            "eval_scene_count": len(eval_scenes),
            "inference_precision": det_cfg["schedule"]["precision"],
            "allocation_hardware": hardware_binding,
            "final_data_view": data_view_binding,
        }
    _write_once_lock(
        lockfile,
        lock_payload,
    )
    print(f"lockfile written: {lockfile}; verified-final access is now consumed")

    from src.eval.ground_truth import ground_truth_from_labels

    validation_path = (
        _repo_path(repo, data_cfg["paths"]["raw_xview3"])
        / "labels"
        / "validation.csv"
    )
    if validation_path.is_symlink() or not validation_path.is_file():
        raise HeldoutContractError(
            "verified-final validation.csv is absent or is a symlink"
        )
    validation_bytes = validation_path.read_bytes()
    validation_sha256 = hashlib.sha256(validation_bytes).hexdigest()
    if data_view_binding is not None:
        validation_binding = data_view_binding.get("validation_labels")
        if (
            not isinstance(validation_binding, Mapping)
            or validation_binding.get("path")
            != "repo/data/raw/xview3/labels/validation.csv"
            or validation_binding.get("bytes") != len(validation_bytes)
            or validation_binding.get("sha256") != validation_sha256
        ):
            raise HeldoutContractError(
                "consumed validation.csv differs from the pre-lock opaque-byte binding"
            )
    labels = pd.read_csv(io.BytesIO(validation_bytes))
    if len(labels) != 19_224:
        raise HeldoutContractError(
            f"verified-final validation.csv must have 19,224 rows, got {len(labels)}"
        )
    labels["scene_id"] = labels["scene_id"].astype(str)
    if set(labels["scene_id"].unique()) != set(eval_scenes):
        raise HeldoutContractError(
            "verified-final label scene IDs differ from the frozen 50-scene split"
        )
    stats = json.loads(_repo_path(repo, data_cfg["paths"]["stats"]).read_text())
    gt_by_scene = {
        scene_id: ground_truth_from_labels(
            labels[labels["scene_id"] == scene_id].to_dict(orient="records")
        )
        for scene_id in eval_scenes
    }
    normalized_gt = {
        scene_id: [
            {
                "x_m": point.x_m,
                "y_m": point.y_m,
                "confidence": point.confidence,
                "source": point.source,
                "distance_from_shore_km": point.distance_from_shore_km,
            }
            for point in gt_by_scene[scene_id]
        ]
        for scene_id in eval_scenes
    }
    normalized_gt_bytes = json.dumps(
        normalized_gt,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    normalized_gt_text = normalized_gt_bytes.decode("utf-8") + "\n"
    _write_once_text(normalized_gt_path, normalized_gt_text)
    normalized_gt_sha256 = hashlib.sha256(normalized_gt_bytes).hexdigest()
    _write_once_lock(
        consumed_receipt,
        {
            "consumption_schema": 1,
            "status": "verified-final-ground-truth-consumed",
            "consumed_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "lock": {
                "relative_path": "final_eval.lock",
                "sha256": sha256_file(lockfile),
            },
            "validation_csv": {
                "row_count": len(labels),
                "scene_count": len(eval_scenes),
                "sha256": validation_sha256,
            },
            "normalized_ground_truth": {
                "relative_path": f".h100/{FINAL_NORMALIZED_GT_FILENAME}",
                "sha256": normalized_gt_sha256,
                "stored_sha256": sha256_file(normalized_gt_path),
            },
            "normalized_ground_truth_sha256": normalized_gt_sha256,
            "campaign_git_sha": campaign_git_sha,
            "evaluator_git_sha": evaluator_git_sha,
            "owner_authorization_sha256": authorization_sha256,
            "final_data_view": data_view_binding,
        },
    )
    print(
        "verified-final labels opened once; immutable consumption receipt -> "
        f"{consumed_receipt}"
    )

    if args.workers == 8:
        payloads = _run_parallel_final_workers(
            repo=repo,
            runs_root=runs_root,
            selected=selected,
            normalized_gt_bytes=normalized_gt_bytes,
            worker_count=args.workers,
        )
    else:
        payloads = []
        for cell in selected:
            exp_id = cell.exp_id
            record = selected_records[exp_id]
            print(f"[final eval] {exp_id}")
            payloads.append(
                score_final_cell(
                    cell=cell,
                    record=record,
                    runs_root=runs_root,
                    raw_root=raw_root,
                    stats=stats,
                    det_cfg=det_cfg,
                    gt_by_scene=gt_by_scene,
                    eval_scenes=eval_scenes,
                    device=args.device,
                    campaign_git_sha=campaign_git_sha,
                    evaluator_git_sha=evaluator_git_sha,
                    detector_sha256=detector_sha256,
                    cohort_sha256=cohort_sha256,
                    test_result_sha256=sha256_file(
                        runs_root / exp_id / TEST_RESULT_FILENAME
                    ),
                    grid_audit=grid_audit,
                    authorization_sha256=authorization_sha256,
                )
            )
    if len(payloads) != len(selected):
        raise HeldoutContractError("final result count differs from selected cells")
    final_result_bindings = [
        {
            "exp_id": str(cell.exp_id),
            "relative_path": f"{cell.exp_id}/{FINAL_CELL_RESULT_FILENAME}",
            "sha256": sha256_file(
                runs_root / str(cell.exp_id) / FINAL_CELL_RESULT_FILENAME
            ),
        }
        for cell in selected
    ]
    rows = []
    for cell, payload, result_binding in zip(
        selected, payloads, final_result_bindings, strict=True
    ):
        rows.append(
            _summary_row(
                cell=cell,
                payload=payload,
                grid_audit=grid_audit,
                campaign_git_sha=campaign_git_sha,
                evaluator_git_sha=evaluator_git_sha,
                detector_sha256=detector_sha256,
                cohort_sha256=cohort_sha256,
                authorization_sha256=authorization_sha256,
                final_result_sha256=str(result_binding["sha256"]),
            )
        )

    _write_once_text(out, pd.DataFrame(rows).to_csv(index=False))
    if authorization_payload is not None:
        if hardware_binding is None or data_view_binding is None:
            raise AssertionError("amended final completion lost hardware/data-view binding")
        _write_once_lock(
            complete_receipt,
            {
                "completion_schema": 1,
                "status": "all-32-verified-final-complete",
                "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "interpretation": (
                    AMENDED_FINAL_INTERPRETATION
                ),
                "seed": 0,
                "study_design": FINAL_STUDY_DESIGN,
                "campaign_git_sha": campaign_git_sha,
                "evaluator_git_sha": evaluator_git_sha,
                "owner_authorization_sha256": authorization_sha256,
                "lock_sha256": sha256_file(lockfile),
                "consumption_sha256": sha256_file(consumed_receipt),
                "data_view_sha256": str(data_view_binding["sha256"]),
                "summary": {
                    "relative_path": "summary/final_verified.csv",
                    "sha256": sha256_file(out),
                    "row_count": len(rows),
                },
                "final_results": final_result_bindings,
                "allocation_hardware": hardware_binding,
                "test_monotonicity": grid_lock_binding,
            },
        )
    print(f"FINAL verified-scene results -> {out}. Nothing is tuned after this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
