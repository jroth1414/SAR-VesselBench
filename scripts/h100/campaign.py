"""Eight-GPU strict-FP32 H100 queue around the unchanged Lightning trainer."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import yaml

from scripts.h100.build_venv import EXPECTED_PYTHON_VERSION
from scripts.h100.contracts import (
    EFFECTIVE_BATCH,
    EXTERNAL_CONTROLS_POLICY,
    EXPECTED_GPU_COUNT,
    EXPECTED_PRECISION,
    GRADIENT_ACCUMULATION,
    MICRO_BATCH,
    Cell,
    assert_empty_core_namespaces,
    atomic_write_json,
    load_cells,
    sha256_file,
    validate_gpu_inventory,
)
from src.analysis.curves import (
    GRID_COLUMNS,
    GRID_COUNT_COLUMNS,
    training_fraction_counts,
)
from scripts.h100.lightning_contract import validate_trainer_contract_evidence
from scripts.h100.precision import assert_sitecustomize_active
from scripts.h100.source_validation import validate_source_receipt
from scripts.h100.data_staging import (
    TRAINING_LABELS_EXPOSED_PATH,
    validate_data_view,
)
from src.eval.ground_truth_audit import audit_ground_truth_scope
from src.eval.heldout_contract import (
    COHORT_FILENAME,
    TEST_RESULT_FILENAME,
    create_training_cohort,
    validate_complete_test_cohort,
    validate_test_result,
    validate_training_cohort,
    validate_training_cohort_cell,
)
from src.eval.result_contract import ResultContractError, load_completion_marker

HOST_REQUEUE_EXIT_CODE = 75


def validate_campaign_start_state(
    *,
    repo: Path,
    runs_root: Path,
    manifest_path: Path,
    cohort_path: Path,
) -> bool:
    """Return whether this is a valid resume, or require a pristine first start."""

    if manifest_path.is_symlink():
        raise RuntimeError("H100 campaign manifest cannot be a symlink")
    if manifest_path.exists():
        if not manifest_path.is_file():
            raise RuntimeError("H100 campaign manifest is not a regular file")
        return True

    if cohort_path.exists() or cohort_path.is_symlink():
        raise RuntimeError(
            "training cohort exists without a campaign manifest; refusing first launch"
        )
    assert_empty_core_namespaces(repo, runs_root)
    return False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def reconcile_stale_open_attempt(
    provenance: dict[str, object],
    *,
    phase: str,
    recovered_utc: str,
) -> bool:
    """Close one crash-orphaned final attempt before a resumed launch.

    A controller or node crash cannot record a monotonic duration. The
    conservative, deterministic substitute is the timezone-aware wall interval
    from the persisted attempt start to the new controller's recovery
    observation. All earlier attempts must already be complete and the prior
    accumulated total must match them exactly; ambiguous provenance fails
    closed.
    """

    attempts = provenance.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError("occupied cell provenance has no attempts to reconcile")
    try:
        prior_seconds = float(provenance["accumulated_active_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("occupied cell active-time accounting is absent") from exc
    if not math.isfinite(prior_seconds) or prior_seconds < 0:
        raise RuntimeError("occupied cell active-time accounting is invalid")

    measured = 0.0
    open_indices: list[int] = []
    exit_fields = {"exit", "exit_code"}
    for index, raw in enumerate(attempts):
        if not isinstance(raw, dict):
            raise RuntimeError("occupied cell attempt provenance is not mutable JSON")
        if raw.get("attempt") != index + 1:
            raise RuntimeError(
                "occupied cell runtime attempts are not sequentially numbered"
            )
        completion_present = {
            key
            for key in ("finished_utc", "active_seconds", *exit_fields)
            if key in raw
        }
        if not completion_present:
            open_indices.append(index)
            continue
        if (
            "finished_utc" not in raw
            or "active_seconds" not in raw
            or not exit_fields & set(raw)
        ):
            raise RuntimeError("occupied cell has a partially closed runtime attempt")
        try:
            active_seconds = float(raw["active_seconds"])
            finished = datetime.fromisoformat(
                str(raw["finished_utc"]).replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("occupied cell has invalid closed-attempt timing") from exc
        if (
            not math.isfinite(active_seconds)
            or active_seconds <= 0
            or finished.tzinfo is None
        ):
            raise RuntimeError("occupied cell has invalid closed-attempt timing")
        measured += active_seconds
    if not math.isclose(prior_seconds, measured, rel_tol=1e-12, abs_tol=1e-9):
        raise RuntimeError("occupied cell accumulated active time is inconsistent")
    if not open_indices:
        return False
    if open_indices != [len(attempts) - 1]:
        raise RuntimeError("only the final runtime attempt may be open after a crash")

    current = attempts[-1]
    if current.get("phase") != phase or provenance.get("phase") != phase:
        raise RuntimeError("crash-orphaned attempt phase differs from the resumed phase")
    try:
        started = datetime.fromisoformat(
            str(current["started_utc"]).replace("Z", "+00:00")
        )
        recovered = datetime.fromisoformat(recovered_utc.replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("crash-orphaned attempt timestamps are invalid") from exc
    if started.tzinfo is None or recovered.tzinfo is None:
        raise RuntimeError("crash-orphaned attempt timestamps require timezones")
    active_seconds = (
        recovered.astimezone(timezone.utc) - started.astimezone(timezone.utc)
    ).total_seconds()
    if not math.isfinite(active_seconds) or active_seconds <= 0:
        raise RuntimeError("crash-orphaned attempt recovery interval is not positive")
    current.update(
        {
            "finished_utc": recovered.isoformat(),
            "exit": "controller-or-node-crash-recovered",
            "active_seconds": active_seconds,
            "active_seconds_basis": "started-to-recovery-observation",
            "recovered_from": "open-attempt-on-controller-start",
        }
    )
    provenance["accumulated_active_seconds"] = prior_seconds + active_seconds
    return True


def promote_hpc_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(
        run_dir.rglob("hpc_ckpt*.ckpt"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise RuntimeError(f"no Lightning HPC checkpoint found under {run_dir}")
    source = candidates[-1]
    destination = run_dir / "checkpoints/last.ckpt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".ckpt.promoting")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    return destination


def marker_for(cell: Cell, runs_root: Path) -> Path:
    return runs_root / cell.exp_id / "final_metrics.json"


def test_marker_for(cell: Cell, runs_root: Path) -> Path:
    return runs_root / cell.exp_id / TEST_RESULT_FILENAME


def expected_recipe(
    cell: Cell, *, git_sha: str, detector_sha256: str
) -> dict[str, object]:
    return {
        "exp_id": cell.exp_id,
        "git_sha": git_sha,
        "detector_sha256": detector_sha256,
        "precision": EXPECTED_PRECISION,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch": EFFECTIVE_BATCH,
    }


def load_test_scene_ids(repo: Path) -> tuple[str, ...]:
    try:
        payload = json.loads((repo / "data/splits.json").read_text(encoding="utf-8"))
        scene_ids = tuple(sorted(map(str, payload["splits"]["test"])))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("frozen TEST scene IDs are absent or malformed") from exc
    if len(scene_ids) != 16 or len(set(scene_ids)) != 16:
        raise RuntimeError("frozen TEST split must contain exactly 16 unique scenes")
    return scene_ids


def hardware_class(payload: Mapping[str, object]) -> dict:
    devices = validate_gpu_inventory(payload.get("devices", []))
    result = {
        "gpu_name": devices[0]["name"],
        "total_memory_bytes": int(devices[0]["total_memory_bytes"]),
        "compute_capability": list(devices[0]["compute_capability"]),
        "driver_version": str(payload.get("driver_version", "")),
        "torch": str(payload.get("torch", "")),
        "cuda_build": str(payload.get("cuda_build", "")),
    }
    if any(not str(result[key]).strip() for key in ("driver_version", "torch", "cuda_build")):
        raise RuntimeError("H100 hardware class lacks driver/torch/CUDA provenance")
    return result


def validate_runtime_provenance(
    provenance: Mapping[str, object],
    *,
    cell: Cell,
    campaign_id: str,
    git_sha: str,
    detector_sha256: str,
    venv_sha256: str,
    venv_build_sha256: str,
    base_python: Mapping[str, object],
    wheelhouse: Mapping[str, object],
    base_payload: Mapping[str, str],
    runtime_amendment: Mapping[str, str],
    acceptance_uuid: str,
    source_validation_sha256: str,
    h100_ready_sha256: str,
    external_controls_policy: str,
    strict_fp32: Mapping[str, str],
    accepted_hardware_class: Mapping[str, object],
    evaluation_ground_truth_sha256: str,
    training_cohort: Mapping[str, str] | None = None,
) -> None:
    if external_controls_policy != EXTERNAL_CONTROLS_POLICY:
        raise RuntimeError("runtime provenance external-controls policy is invalid")
    expected = {
        "schema": 2,
        "campaign_id": campaign_id,
        "exp_id": cell.exp_id,
        "git_sha": git_sha,
        "detector_sha256": detector_sha256,
        "venv_sha256": venv_sha256,
        "venv_build_sha256": venv_build_sha256,
        "base_python": dict(base_python),
        "wheelhouse": dict(wheelhouse),
        "base_payload": dict(base_payload),
        "runtime_amendment": dict(runtime_amendment),
        "acceptance_uuid": acceptance_uuid,
        "source_validation_sha256": source_validation_sha256,
        "h100_ready_sha256": h100_ready_sha256,
        "external_controls_policy": external_controls_policy,
        "strict_fp32": dict(strict_fp32),
        "accepted_hardware_class": dict(accepted_hardware_class),
        "precision": EXPECTED_PRECISION,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch": EFFECTIVE_BATCH,
    }
    phase = provenance.get("phase")
    if phase not in {"train", "score-test"}:
        raise RuntimeError(
            f"recipe-mismatched runtime provenance for {cell.exp_id}: "
            f"invalid phase {phase!r}"
        )
    if phase == "score-test" and training_cohort is None:
        raise RuntimeError("score-test runtime provenance lacks cohort expectation")
    expected.update(
        {
            "phase": phase,
            "training_cohort": (
                dict(training_cohort or {}) if phase == "score-test" else None
            ),
            "evaluation_ground_truth_sha256": evaluation_ground_truth_sha256,
        }
    )
    mismatches = {
        key: (value, provenance.get(key))
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    gpu_expected = {
        "gpu_name": accepted_hardware_class["gpu_name"],
        "gpu_total_memory_bytes": accepted_hardware_class["total_memory_bytes"],
        "compute_capability": accepted_hardware_class["compute_capability"],
        "driver_version": accepted_hardware_class["driver_version"],
        "torch": accepted_hardware_class["torch"],
        "cuda_build": accepted_hardware_class["cuda_build"],
    }
    mismatches.update(
        {
            key: (value, provenance.get(key))
            for key, value in gpu_expected.items()
            if provenance.get(key) != value
        }
    )
    attempts = provenance.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        mismatches["attempts"] = ("nonempty list", attempts)
    elif any(
        not isinstance(item, Mapping)
        or not str(item.get("gpu_uuid", "")).strip()
        for item in attempts
    ):
        mismatches["attempts.gpu_uuid"] = ("nonempty mapped attempts", attempts)
    elif (
        any(item.get("phase") not in {"train", "score-test"} for item in attempts)
        or attempts[-1].get("phase") != provenance.get("phase")
    ):
        mismatches["attempts.phase"] = (
            "valid phases ending at top-level phase",
            attempts,
        )
    if not str(provenance.get("gpu_uuid", "")).strip():
        mismatches["gpu_uuid"] = ("nonempty", provenance.get("gpu_uuid"))
    if mismatches:
        raise RuntimeError(f"recipe-mismatched runtime provenance for {cell.exp_id}: {mismatches}")


def validate_training_completion(
    cell: Cell,
    *,
    runs_root: Path,
    git_sha: str,
    detector_sha256: str,
    candidate_floor: float,
    seal: bool = False,
) -> dict[str, object]:
    run_dir = runs_root / cell.exp_id
    marker = marker_for(cell, runs_root)
    try:
        payload, _checkpoint = load_completion_marker(
            marker,
            candidate_floor=candidate_floor,
            expected_recipe=expected_recipe(
                cell,
                git_sha=git_sha,
                detector_sha256=detector_sha256,
            ),
        )
    except ResultContractError as exc:
        raise RuntimeError(
            f"{cell.exp_id}: invalid schema-2 training completion: {exc}"
        ) from exc
    try:
        validate_trainer_contract_evidence(payload.get("h100_runtime_contract"))
    except RuntimeError as exc:
        raise RuntimeError(
            f"{cell.exp_id}: invalid H100 trainer evidence: {exc}"
        ) from exc
    last = run_dir / "checkpoints/last.ckpt"
    if last.is_symlink() or not last.is_file() or last.stat().st_size <= 0:
        raise RuntimeError(f"{cell.exp_id}: training completion lacks durable last.ckpt")
    if seal:
        marker.chmod(0o444)
    if marker.stat().st_mode & 0o222:
        raise RuntimeError(f"{cell.exp_id}: training marker is not immutable")
    return payload


def validate_scored_completion(
    cell: Cell,
    *,
    runs_root: Path,
    git_sha: str,
    detector_sha256: str,
    repo: Path | None = None,
    candidate_floor: float | None = None,
    cohort: Mapping[str, object] | None = None,
    cohort_sha256: str | None = None,
    test_scene_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    source_root = repo or Path(__file__).resolve().parents[2]
    if candidate_floor is None:
        detector = yaml.safe_load(
            (source_root / "configs/detector.yaml").read_text(encoding="utf-8")
        )
        candidate_floor = float(detector["decode"]["candidate_floor"])
    training = validate_training_completion(
        cell,
        runs_root=runs_root,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        candidate_floor=candidate_floor,
    )
    if cohort is None or cohort_sha256 is None:
        raise RuntimeError("scored completion requires the frozen training cohort")
    canonical = runs_root / ".h100" / COHORT_FILENAME
    (
        validated_cohort,
        validated_sha256,
        _cohort_record,
        _cohort_training,
    ) = validate_training_cohort_cell(
        path=canonical,
        expected_sha256=cohort_sha256,
        cells=load_cells(source_root),
        runs_root=runs_root,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        candidate_floor=candidate_floor,
        exp_id=cell.exp_id,
    )
    if dict(cohort) != validated_cohort:
        raise RuntimeError("controller training-cohort payload drifted")
    if cohort_sha256 != validated_sha256:
        raise RuntimeError("controller training-cohort SHA-256 drifted")
    scene_ids = test_scene_ids or load_test_scene_ids(source_root)
    test = validate_test_result(
        path=test_marker_for(cell, runs_root),
        exp_id=cell.exp_id,
        cohort=validated_cohort,
        cohort_sha256=validated_sha256,
        test_scene_ids=scene_ids,
    )
    metrics = test["metrics"]
    payload = dict(training)
    payload.update(
        {
            "test_inference_precision": test["inference_precision"],
            "test_f1": metrics["f1"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_near_shore_f1": metrics["near_shore_f1"],
            "test_scored_at": test["scored_utc"],
            "test_result_sha256": sha256_file(test_marker_for(cell, runs_root)),
            "training_cohort_sha256": validated_sha256,
        }
    )
    return payload


def checkpoint_for_preemption(run_dir: Path) -> Path:
    wrapper_path = run_dir / "cell_wrapper.json"
    phase = None
    if wrapper_path.exists():
        phase = json.loads(wrapper_path.read_text()).get("phase")
    if phase in {"train-complete", "score-test", "score-test-complete"}:
        last = run_dir / "checkpoints/last.ckpt"
        best = run_dir / "checkpoints/best.ckpt"
        if any(
            path.is_symlink() or not path.is_file() or path.stat().st_size <= 0
            for path in (best, last)
        ):
            raise RuntimeError(
                f"completed/scoring preemption lacks durable checkpoints: {run_dir}"
            )
        return last
    if phase != "train":
        raise RuntimeError(f"preemption phase is absent or invalid: {run_dir}")
    return promote_hpc_checkpoint(run_dir)


def _validated_phase_runtime(
    provenance: Mapping[str, object],
    payload: Mapping[str, object],
    *,
    exp_id: str,
    phase: str,
) -> dict[str, object]:
    attempts = provenance.get("attempts")
    phase_runtime = provenance.get("phase_runtime")
    if not isinstance(attempts, list) or not isinstance(phase_runtime, Mapping):
        raise RuntimeError(f"{exp_id}: phase runtime provenance is absent")
    selected = [
        attempt
        for attempt in attempts
        if isinstance(attempt, Mapping) and attempt.get("phase") == phase
    ]
    record = phase_runtime.get(phase)
    if (
        not selected
        or not isinstance(record, Mapping)
        or selected[-1].get("exit_code") != 0
        or not str(selected[-1].get("finished_utc", "")).strip()
    ):
        raise RuntimeError(f"{exp_id}: {phase} phase is not durably complete")
    try:
        active_seconds = float(record["active_seconds"])
        elapsed_hours = float(record["elapsed_hours"])
        measured = sum(float(attempt["active_seconds"]) for attempt in selected)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{exp_id}: {phase} phase runtime is invalid") from exc
    if (
        not math.isfinite(active_seconds)
        or active_seconds <= 0
        or not math.isclose(active_seconds, measured)
        or not math.isclose(elapsed_hours, active_seconds / 3600.0)
        or record.get("attempts") != len(selected)
        or not str(record.get("completed_utc", "")).strip()
    ):
        raise RuntimeError(f"{exp_id}: {phase} phase runtime is inconsistent")
    if phase == "train":
        expected = {
            "final_metrics_sha256": sha256_file(Path(str(provenance["run_dir"])) / "final_metrics.json"),
            "best_checkpoint_sha256": payload["best_checkpoint"]["sha256"],
            "epochs_run": payload["epochs_run"],
            "best_dev_f1": payload["best_dev_f1"],
        }
    else:
        expected = {
            "test_metrics_sha256": payload["test_result_sha256"],
            "training_cohort_sha256": payload["training_cohort_sha256"],
            "test_f1": payload["test_f1"],
        }
    mismatches = {
        key: (value, record.get(key))
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{exp_id}: {phase} phase binding mismatch: {mismatches}")
    return dict(record)



def _recover_completed_phase_runtime(
    *,
    provenance_path: Path,
    provenance: dict[str, object],
    payload: Mapping[str, object],
    exp_id: str,
    phase: str,
    marker_path: Path,
) -> dict[str, object]:
    """Close only the exact marker-written/controller-crash provenance window."""

    attempts = provenance.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError(f"{exp_id}: no attempt is available for crash recovery")
    current = attempts[-1]
    phase_records = provenance.get("phase_runtime")
    completion_keys = {"finished_utc", "exit_code", "active_seconds"}
    if (
        not isinstance(current, dict)
        or current.get("phase") != phase
        or completion_keys & set(current)
        or not isinstance(phase_records, Mapping)
        or phase in phase_records
    ):
        raise RuntimeError(
            f"{exp_id}: {phase} provenance is not an exact completion crash window"
        )
    try:
        started = datetime.fromisoformat(str(current["started_utc"]))
        if started.tzinfo is None:
            raise ValueError("naive timestamp")
        finished = datetime.fromtimestamp(
            marker_path.stat().st_mtime,
            timezone.utc,
        )
        active_seconds = (finished - started.astimezone(timezone.utc)).total_seconds()
        prior_seconds = float(provenance.get("accumulated_active_seconds", 0.0))
        previous_measured = sum(
            float(attempt["active_seconds"])
            for attempt in attempts[:-1]
            if isinstance(attempt, Mapping)
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"{exp_id}: {phase} completion crash timing is invalid"
        ) from exc
    if (
        not math.isfinite(active_seconds)
        or active_seconds <= 0
        or not math.isfinite(prior_seconds)
        or prior_seconds < 0
        or not math.isclose(prior_seconds, previous_measured)
    ):
        raise RuntimeError(
            f"{exp_id}: {phase} completion crash timing is inconsistent"
        )

    finished_utc = finished.isoformat()
    current.update(
        {
            "finished_utc": finished_utc,
            "exit_code": 0,
            "active_seconds": active_seconds,
            "recovered_from": "validated-completion-marker",
        }
    )
    selected = [
        attempt
        for attempt in attempts
        if isinstance(attempt, Mapping) and attempt.get("phase") == phase
    ]
    phase_seconds = sum(float(attempt["active_seconds"]) for attempt in selected)
    phase_record: dict[str, object] = {
        "completed_utc": finished_utc,
        "active_seconds": phase_seconds,
        "elapsed_hours": phase_seconds / 3600.0,
        "attempts": len(selected),
        "controller_crash_recovered": True,
    }
    total_seconds = prior_seconds + active_seconds
    if phase == "train":
        phase_record.update(
            {
                "final_metrics_sha256": sha256_file(marker_path),
                "best_checkpoint_sha256": payload["best_checkpoint"]["sha256"],
                "epochs_run": payload["epochs_run"],
                "best_dev_f1": payload["best_dev_f1"],
            }
        )
        provenance.update(
            {
                "training_completed_utc": finished_utc,
                "epochs_run": payload["epochs_run"],
                "best_dev_f1": payload["best_dev_f1"],
                "h100_runtime_contract": payload["h100_runtime_contract"],
            }
        )
    elif phase == "score-test":
        phase_record.update(
            {
                "test_metrics_sha256": payload["test_result_sha256"],
                "training_cohort_sha256": payload["training_cohort_sha256"],
                "test_f1": payload["test_f1"],
            }
        )
        provenance.update(
            {
                "completed_utc": finished_utc,
                "elapsed_hours": total_seconds / 3600.0,
                "test_f1": payload["test_f1"],
                "test_scored_at": payload["test_scored_at"],
            }
        )
    else:
        raise RuntimeError(f"unsupported crash-recovery phase: {phase!r}")
    phase_runtime = dict(provenance.get("phase_runtime", {}))
    phase_runtime[phase] = phase_record
    provenance.update(
        {
            "phase_runtime": phase_runtime,
            "accumulated_active_seconds": total_seconds,
        }
    )
    atomic_write_json(provenance_path, provenance)
    return provenance

def existing_cell_state(
    cell: Cell,
    *,
    phase: str,
    repo: Path,
    runs_root: Path,
    campaign_id: str,
    git_sha: str,
    detector_sha256: str,
    candidate_floor: float,
    venv_sha256: str,
    venv_build_sha256: str,
    base_python: Mapping[str, object],
    wheelhouse: Mapping[str, object],
    base_payload: Mapping[str, str],
    runtime_amendment: Mapping[str, str],
    acceptance_uuid: str,
    source_validation_sha256: str,
    evaluation_ground_truth_sha256: str,
    h100_ready_sha256: str,
    external_controls_policy: str,
    strict_fp32: Mapping[str, str],
    accepted_hardware_class: Mapping[str, object],
    cohort: Mapping[str, object] | None,
    cohort_sha256: str | None,
    test_scene_ids: tuple[str, ...],
) -> str:
    if phase not in {"train", "score-test"}:
        raise RuntimeError(f"invalid controller phase: {phase!r}")
    run_dir = runs_root / cell.exp_id
    if not run_dir.exists():
        return "pending"
    test_path = test_marker_for(cell, runs_root)
    if phase == "train" and (test_path.exists() or test_path.is_symlink()):
        raise RuntimeError(f"{cell.exp_id}: TEST result exists before cohort freeze")
    provenance_path = run_dir / "runtime_provenance.json"
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise RuntimeError(f"occupied namespace lacks provenance: {run_dir}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    cohort_binding = (
        {
            "path": str(runs_root / ".h100" / COHORT_FILENAME),
            "sha256": str(cohort_sha256),
        }
        if cohort_sha256 is not None
        else None
    )
    validate_runtime_provenance(
        provenance,
        cell=cell,
        campaign_id=campaign_id,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        venv_sha256=venv_sha256,
        venv_build_sha256=venv_build_sha256,
        base_python=base_python,
        wheelhouse=wheelhouse,
        base_payload=base_payload,
        runtime_amendment=runtime_amendment,
        acceptance_uuid=acceptance_uuid,
        source_validation_sha256=source_validation_sha256,
        h100_ready_sha256=h100_ready_sha256,
        external_controls_policy=external_controls_policy,
        strict_fp32=strict_fp32,
        accepted_hardware_class=accepted_hardware_class,
        evaluation_ground_truth_sha256=evaluation_ground_truth_sha256,
        training_cohort=cohort_binding,
    )
    final = marker_for(cell, runs_root)
    if not final.exists() and not final.is_symlink():
        return "resume"
    training = validate_training_completion(
        cell,
        runs_root=runs_root,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        candidate_floor=candidate_floor,
        seal=True,
    )
    if phase == "train":
        try:
            _validated_phase_runtime(
                provenance,
                training,
                exp_id=cell.exp_id,
                phase="train",
            )
        except RuntimeError:
            provenance = _recover_completed_phase_runtime(
                provenance_path=provenance_path,
                provenance=provenance,
                payload=training,
                exp_id=cell.exp_id,
                phase="train",
                marker_path=final,
            )
            _validated_phase_runtime(
                provenance,
                training,
                exp_id=cell.exp_id,
                phase="train",
            )
        return "complete"
    if cohort is None or cohort_sha256 is None:
        raise RuntimeError("score-test state inspection requires the frozen cohort")
    if not test_path.exists() and not test_path.is_symlink():
        return "resume"
    scored = validate_scored_completion(
        cell,
        repo=repo,
        runs_root=runs_root,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        candidate_floor=candidate_floor,
        cohort=cohort,
        cohort_sha256=cohort_sha256,
        test_scene_ids=test_scene_ids,
    )
    try:
        finalized_cell_runtime(provenance, scored, cell.exp_id)
    except RuntimeError:
        provenance = _recover_completed_phase_runtime(
            provenance_path=provenance_path,
            provenance=provenance,
            payload=scored,
            exp_id=cell.exp_id,
            phase="score-test",
            marker_path=test_path,
        )
        finalized_cell_runtime(provenance, scored, cell.exp_id)
    return "complete"


def finalized_cell_runtime(
    provenance: Mapping[str, object], marker: Mapping[str, object], exp_id: str
) -> dict[str, object]:
    _validated_phase_runtime(
        provenance,
        marker,
        exp_id=exp_id,
        phase="train",
    )
    _validated_phase_runtime(
        provenance,
        marker,
        exp_id=exp_id,
        phase="score-test",
    )
    attempts = provenance.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RuntimeError(f"{exp_id}: finalized provenance has no attempts")
    try:
        active_seconds = float(provenance["accumulated_active_seconds"])
        elapsed_hours = float(provenance["elapsed_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"{exp_id}: finalized runtime is invalid") from exc
    measured = sum(
        float(attempt["active_seconds"])
        for attempt in attempts
        if isinstance(attempt, Mapping)
    )
    if (
        provenance.get("phase") != "score-test"
        or not math.isfinite(active_seconds)
        or active_seconds <= 0
        or not math.isclose(active_seconds, measured)
        or not math.isclose(elapsed_hours, active_seconds / 3600.0)
        or not str(provenance.get("completed_utc", "")).strip()
        or provenance.get("epochs_run") != marker.get("epochs_run")
        or provenance.get("best_dev_f1") != marker.get("best_dev_f1")
        or provenance.get("test_f1") != marker.get("test_f1")
        or provenance.get("h100_runtime_contract") != marker.get("h100_runtime_contract")
    ):
        raise RuntimeError(f"{exp_id}: finalized runtime/marker binding mismatch")
    return {
        "elapsed_hours": elapsed_hours,
        "attempts": len(attempts),
        "epochs_run": marker.get("epochs_run"),
        "test_f1": marker.get("test_f1"),
    }


def validate_failed_namespace(
    cell: Cell,
    *,
    runs_root: Path,
    campaign_id: str,
    git_sha: str,
    detector_sha256: str,
    venv_sha256: str,
    venv_build_sha256: str,
    base_python: Mapping[str, object],
    wheelhouse: Mapping[str, object],
    base_payload: Mapping[str, str],
    runtime_amendment: Mapping[str, str],
    acceptance_uuid: str,
    source_validation_sha256: str,
    h100_ready_sha256: str,
    external_controls_policy: str,
    strict_fp32: Mapping[str, str],
    accepted_hardware_class: Mapping[str, object],
    evaluation_ground_truth_sha256: str,
    training_cohort: Mapping[str, str] | None,
) -> None:
    run_dir = runs_root / cell.exp_id
    provenance_path = run_dir / "runtime_provenance.json"
    if not run_dir.is_dir() or not provenance_path.is_file():
        raise RuntimeError(f"known failed cell lacks its bound namespace: {run_dir}")
    provenance = json.loads(provenance_path.read_text())
    validate_runtime_provenance(
        provenance,
        cell=cell,
        campaign_id=campaign_id,
        git_sha=git_sha,
        detector_sha256=detector_sha256,
        venv_sha256=venv_sha256,
        venv_build_sha256=venv_build_sha256,
        base_python=base_python,
        wheelhouse=wheelhouse,
        base_payload=base_payload,
        runtime_amendment=runtime_amendment,
        acceptance_uuid=acceptance_uuid,
        source_validation_sha256=source_validation_sha256,
        h100_ready_sha256=h100_ready_sha256,
        external_controls_policy=external_controls_policy,
        strict_fp32=strict_fp32,
        accepted_hardware_class=accepted_hardware_class,
        evaluation_ground_truth_sha256=evaluation_ground_truth_sha256,
        training_cohort=training_cohort,
    )


def next_launches(
    cells: list[Cell],
    *,
    running_ids: set[str],
    complete_ids: set[str],
    free_gpus: list[int],
    failure_seen: bool,
    preemption_seen: bool,
) -> list[tuple[int, Cell]]:
    if failure_seen or preemption_seen:
        return []
    pending = [
        cell
        for cell in cells
        if cell.exp_id not in running_ids and cell.exp_id not in complete_ids
    ]
    return list(zip(free_gpus, pending[: len(free_gpus)], strict=False))


def fail_stop_launches(
    cells: list[Cell],
    *,
    allowed_ids: set[str],
    failed_ids: set[str],
    running_ids: set[str],
    complete_ids: set[str],
    free_gpus: list[int],
    preemption_seen: bool,
) -> list[tuple[int, Cell]]:
    """Resume only cells that were already running when fail-stop engaged."""

    if preemption_seen:
        return []
    pending = [
        cell
        for cell in cells
        if cell.exp_id in allowed_ids
        and cell.exp_id not in failed_ids
        and cell.exp_id not in running_ids
        and cell.exp_id not in complete_ids
    ]
    return list(zip(free_gpus, pending[: len(free_gpus)], strict=False))


def restore_fail_stop(
    prior: Mapping[str, object], cells: list[Cell]
) -> tuple[bool, set[str], set[str]]:
    record = prior.get("fail_stop")
    if not isinstance(record, Mapping) or set(record) != {
        "engaged",
        "failed",
        "allowed_to_finish",
    }:
        raise RuntimeError("existing campaign fail-stop record is absent or invalid")
    engaged = record.get("engaged")
    failed_raw = record.get("failed")
    allowed_raw = record.get("allowed_to_finish")
    if type(engaged) is not bool or not isinstance(failed_raw, list) or not isinstance(
        allowed_raw, list
    ):
        raise RuntimeError("existing campaign fail-stop record has invalid types")
    if any(not isinstance(value, str) for value in [*failed_raw, *allowed_raw]):
        raise RuntimeError("existing campaign fail-stop IDs must be strings")
    failed = set(failed_raw)
    allowed = set(allowed_raw)
    valid_ids = {cell.exp_id for cell in cells}
    if (
        len(failed) != len(failed_raw)
        or len(allowed) != len(allowed_raw)
        or not failed <= valid_ids
        or not allowed <= valid_ids
        or failed & allowed
        or engaged != bool(failed)
    ):
        raise RuntimeError("existing campaign fail-stop IDs/state are inconsistent")
    return engaged, failed, allowed


def validate_campaign_resume_bindings(
    prior: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    mismatches = {
        key: (value, prior.get(key))
        for key, value in expected.items()
        if prior.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"existing H100 campaign manifest binding mismatch: {mismatches}"
        )


def collect_and_validate_grid(
    *,
    repo: Path,
    runs_root: Path,
    cells: list[Cell],
    git_sha: str,
    detector_sha256: str,
) -> Path:
    try:
        expected_counts = training_fraction_counts(
            repo=repo,
            fractions=tuple(sorted({float(cell.fraction) for cell in cells})),
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "could not derive exact frozen TRAIN fraction counts"
        ) from exc
    arms = yaml.safe_load((repo / "configs/arms.yaml").read_text())["arms"]
    grid = runs_root / "summary/grid.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.analysis.curves",
            "collect",
            "--runs-root",
            str(runs_root),
            "--out-csv",
            str(grid),
        ],
        cwd=repo,
        check=True,
    )
    with grid.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    if fields != GRID_COLUMNS:
        raise RuntimeError(f"grid.csv columns mismatch: {fields}")
    expected_ids = {cell.exp_id for cell in cells}
    expected_cells = {cell.exp_id: cell for cell in cells}
    actual_ids = {row.get("exp_id") for row in rows}
    if len(rows) != 32 or actual_ids != expected_ids:
        raise RuntimeError("grid.csv is not the exact 32-cell core matrix")
    for row in rows:
        exp_id = str(row.get("exp_id", ""))
        if set(row) != set(GRID_COLUMNS) or any(
            not isinstance(value, str)
            or not value.strip()
            or value.strip().casefold()
            in {
                "nan",
                "+nan",
                "-nan",
                "inf",
                "+inf",
                "-inf",
                "infinity",
                "+infinity",
                "-infinity",
            }
            for value in row.values()
        ):
            raise RuntimeError(f"grid.csv contains an absent/NaN value for {exp_id}")
        cell = expected_cells[exp_id]
        try:
            fraction = float(row["label_frac"])
            seed = int(row["seed"])
            epochs_run = int(row["epochs_run"])
            metrics = {
                key: float(row[key])
                for key in ("dev_f1", "dev_threshold", "test_f1")
            }
            counts = {
                key: int(row[key])
                for key in GRID_COUNT_COLUMNS
                if row[key].isdecimal()
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"grid.csv numeric values are invalid for {exp_id}") from exc
        if len(counts) != len(GRID_COUNT_COLUMNS):
            raise RuntimeError(f"grid.csv count columns are not integers for {exp_id}")
        expected_role = str(arms[cell.init]["role"])
        if (
            row["init"] != cell.init
            or row["track"] != cell.track
            or row["role"] != expected_role
            or not math.isclose(
                fraction, float(cell.fraction), rel_tol=0.0, abs_tol=1e-12
            )
            or seed != cell.seed
            or row["precision"] != EXPECTED_PRECISION
            or epochs_run <= 0
        ):
            raise RuntimeError(f"grid.csv core identity mismatch for {exp_id}")
        if (
            any(not math.isfinite(value) for value in metrics.values())
            or any(not 0.0 <= value <= 1.0 for value in metrics.values())
        ):
            raise RuntimeError(f"grid.csv metrics are non-finite/out of range for {exp_id}")
        wanted_counts = expected_counts[float(cell.fraction)]
        if counts != wanted_counts:
            raise RuntimeError(
                f"grid.csv TRAIN fraction counts mismatch for {exp_id}: "
                f"{counts} != {wanted_counts}"
            )
        if row.get("monotonicity_ok", "").lower() != "true":
            raise RuntimeError(
                f"grid.csv monotonicity STOP for {exp_id}"
            )
        if row.get("git_sha") != git_sha or row.get("detector_sha256") != detector_sha256:
            raise RuntimeError(f"grid.csv provenance mismatch for {exp_id}")
    return grid


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        assert_sitecustomize_active()
        self.args = args
        self.repo = args.repo.resolve()
        self.runs_root = args.runs_root.resolve()
        self.cells = load_cells(self.repo)
        self.git_sha = args.expected_git_sha
        self.detector_sha256 = sha256_file(self.repo / "configs/detector.yaml")
        self.venv_sha256 = args.venv_sha256
        self.detector_config = yaml.safe_load(
            (self.repo / "configs/detector.yaml").read_text(encoding="utf-8")
        )
        self.data_config = yaml.safe_load(
            (self.repo / "configs/data.yaml").read_text(encoding="utf-8")
        )
        self.candidate_floor = float(self.detector_config["decode"]["candidate_floor"])
        self.test_scene_ids = load_test_scene_ids(self.repo)
        self.venv_build_sha256 = args.venv_build_sha256
        self.venv_root = args.venv_root.absolute()
        expected_python = self.venv_root / "bin/python"
        if (
            Path(sys.prefix).resolve() != self.venv_root.resolve()
            or Path(sys.executable).resolve() != expected_python.resolve()
        ):
            raise RuntimeError("campaign is not running under the accepted native venv")
        expected_acceptance = (self.runs_root / ".h100/H100_READY.json").absolute()
        if (
            args.acceptance_json.absolute() != expected_acceptance
            or args.acceptance_json.is_symlink()
            or not args.acceptance_json.is_file()
        ):
            raise RuntimeError(
                f"controller H100_READY is not canonical: {expected_acceptance}"
            )
        self.h100_ready_sha256 = args.acceptance_sha256
        if sha256_file(args.acceptance_json) != self.h100_ready_sha256:
            raise RuntimeError("controller H100_READY SHA-256 mismatch")
        self.acceptance = json.loads(args.acceptance_json.read_text())
        if self.acceptance.get("schema") != 2 or self.acceptance.get("status") != "ready":
            raise RuntimeError("campaign requires a schema-2 ready H100 acceptance receipt")
        self.external_controls_policy = str(
            self.acceptance.get("external_controls_policy", "")
        )
        if self.external_controls_policy != EXTERNAL_CONTROLS_POLICY:
            raise RuntimeError("campaign H100_READY external-controls policy is invalid")
        accepted_venv = self.acceptance.get("venv")
        if not isinstance(accepted_venv, Mapping) or (
            accepted_venv.get("path") != str(self.venv_root)
            or accepted_venv.get("sha256") != self.venv_sha256
            or accepted_venv.get("venv_build_sha256") != self.venv_build_sha256
        ):
            raise RuntimeError("campaign native venv differs from acceptance")
        self.base_python = dict(accepted_venv.get("base_python", {}))
        if (
            not self.base_python
            or self.base_python.get("version") != EXPECTED_PYTHON_VERSION
        ):
            raise RuntimeError("campaign accepted base-Python identity is invalid")
        self.base_payload = dict(self.acceptance.get("base_payload", {}))
        self.runtime_amendment = dict(self.acceptance.get("runtime_amendment", {}))
        self.wheelhouse = dict(accepted_venv.get("wheelhouse", {}))
        wheelhouse_identity = self.wheelhouse.get("identity")
        base_extraction = self.wheelhouse.get("base_extraction")
        extraction_receipt = (
            base_extraction.get("receipt")
            if isinstance(base_extraction, Mapping)
            else None
        )
        if (
            not self.base_payload
            or not self.runtime_amendment
            or not isinstance(wheelhouse_identity, Mapping)
            or not str(wheelhouse_identity.get("sha256", ""))
            or not isinstance(extraction_receipt, Mapping)
            or extraction_receipt.get("package_id")
            != self.base_payload.get("package_id")
            or extraction_receipt.get("manifest_sha256")
            != self.base_payload.get("manifest_sha256")
            or extraction_receipt.get("wheelhouse") != wheelhouse_identity
        ):
            raise RuntimeError("campaign transfer/wheelhouse identities are invalid")
        self.acceptance_uuid = str(self.acceptance.get("acceptance_uuid", ""))
        if not self.acceptance_uuid:
            raise RuntimeError("campaign acceptance UUID is absent")
        accepted_source = self.acceptance.get("source")
        if not isinstance(accepted_source, Mapping):
            raise RuntimeError("campaign acceptance source binding is absent")
        if (
            accepted_source.get("git_sha") != self.git_sha
            or accepted_source.get("frozen_sha256", {}).get("configs/detector.yaml")
            != self.detector_sha256
        ):
            raise RuntimeError("campaign source differs from H100 acceptance")
        self.source_validation_sha256 = args.source_validation_sha256
        self.source_validation = validate_source_receipt(
            args.source_validation_json,
            expected_sha256=self.source_validation_sha256,
            expected_git_sha=self.git_sha,
            expected_hashes=accepted_source["frozen_sha256"],
            expected_base_payload=self.base_payload,
            expected_runtime_amendment=self.runtime_amendment,
        )
        if self.acceptance.get("source_validation", {}).get("sha256") != (
            self.source_validation_sha256
        ):
            raise RuntimeError("campaign source receipt differs from H100 acceptance")
        meta_root = self.runs_root / ".h100"
        self.cohort_path = meta_root / COHORT_FILENAME
        ground_truth_binding = self.acceptance.get("evaluation_ground_truth")
        if (
            not isinstance(ground_truth_binding, Mapping)
            or set(ground_truth_binding) != {"path", "sha256", "receipt"}
        ):
            raise RuntimeError("campaign evaluation-ground-truth binding is invalid")
        ground_truth_path = Path(str(ground_truth_binding["path"])).absolute()
        expected_ground_truth_path = (
            meta_root / "EVAL_GROUND_TRUTH_VALIDATED.json"
        ).absolute()
        if (
            ground_truth_path != expected_ground_truth_path
            or ground_truth_path.is_symlink()
            or not ground_truth_path.is_file()
            or ground_truth_path.stat().st_mode & 0o222
        ):
            raise RuntimeError(
                "campaign evaluation-ground-truth receipt is not canonical and immutable"
            )
        self.evaluation_ground_truth_sha256 = str(ground_truth_binding["sha256"])
        if sha256_file(ground_truth_path) != self.evaluation_ground_truth_sha256:
            raise RuntimeError("campaign evaluation-ground-truth SHA-256 mismatch")
        try:
            self.evaluation_ground_truth = json.loads(
                ground_truth_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("campaign evaluation-ground-truth receipt is invalid") from exc
        if self.evaluation_ground_truth != ground_truth_binding["receipt"]:
            raise RuntimeError("campaign evaluation-ground-truth receipt binding drifted")

        # This is the allocation's semantic data gate.  Before a cohort exists,
        # validation may inspect only the TRAIN+fixed-DEV8 view.  Once the
        # canonical all-32 cohort exists, a new allocation must receive the
        # independently staged TEST view instead.  Do this before opening any
        # label or raster through the repository symlinks.
        self.data_view_root = args.data_view_root.absolute()
        if self.data_view_root.is_symlink() or not self.data_view_root.is_dir():
            raise RuntimeError("campaign data-view root must be a regular directory")
        self.data_view_sha256 = args.data_view_receipt_sha256
        expected_data_view_phase = (
            "score-test"
            if self.cohort_path.exists() or self.cohort_path.is_symlink()
            else "train"
        )
        self.data_view = validate_data_view(
            self.data_view_root,
            repo=self.repo,
            runs_root=self.runs_root,
            expected_sha256=self.data_view_sha256,
            expected_git_sha=self.git_sha,
            expected_phase=expected_data_view_phase,
            expected_purpose="campaign",
            expected_base_package_id=str(self.base_payload["package_id"]),
            expected_base_manifest_sha256=str(self.base_payload["manifest_sha256"]),
            expected_runtime_package_id=str(self.runtime_amendment["package_id"]),
            expected_runtime_manifest_sha256=str(
                self.runtime_amendment["manifest_sha256"]
            ),
        )
        self.data_view_phase = str(self.data_view["phase"])
        if self.data_view_phase == "train":
            observed_dev8 = audit_ground_truth_scope(
                train_csv=self.data_view_root / TRAINING_LABELS_EXPOSED_PATH,
                splits_json=self.repo / str(self.data_config["paths"]["splits"]),
                scope="dev8",
            )
            expected_dev8 = self.evaluation_ground_truth.get("scopes", {}).get(
                "dev8", {}
            ).get("expected_counts")
            if observed_dev8 != expected_dev8:
                raise RuntimeError(
                    "campaign DEV8 ground truth differs from the source audit"
                )
        else:
            labels_binding = self.data_view.get("labels")
            source_audit_sha256 = hashlib.sha256(
                (
                    json.dumps(
                        self.evaluation_ground_truth,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
            if (
                not isinstance(labels_binding, Mapping)
                or labels_binding.get("source_audit_sha256")
                != source_audit_sha256
            ):
                raise RuntimeError(
                    "campaign TEST label view differs from the accepted source audit"
                )
        expected_source_validation = meta_root / "SOURCE_VALIDATED.json"
        if (
            args.source_validation_json.absolute() != expected_source_validation
            or args.source_validation_json.is_symlink()
            or not args.source_validation_json.is_file()
        ):
            raise RuntimeError(
                "controller source validation is not canonical: "
                f"{expected_source_validation}"
            )
        self.strict_fp32 = dict(self.acceptance.get("strict_fp32", {}))
        if set(self.strict_fp32.values()) != {"ieee"}:
            raise RuntimeError("campaign acceptance is not strict IEEE FP32")
        if args.hardware.get("backend") != self.strict_fp32:
            raise RuntimeError("current allocation strict backend differs from acceptance")
        self.accepted_hardware_class = hardware_class(self.acceptance["hardware"])
        self.allocation_hardware_class = hardware_class(args.hardware)
        if self.allocation_hardware_class != self.accepted_hardware_class:
            raise RuntimeError(
                "current allocation hardware class differs from H100 acceptance: "
                f"{self.allocation_hardware_class} != {self.accepted_hardware_class}"
            )
        meta_root.mkdir(parents=True, exist_ok=True)
        self.lock_handle = (meta_root / "campaign.lock").open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another H100 campaign controller holds campaign.lock") from exc
        self.running: dict[int, tuple[subprocess.Popen, Cell, float]] = {}
        self.phase = "train"
        self.training_complete_ids: set[str] = set()
        self.test_complete_ids: set[str] = set()
        self.complete_ids = self.training_complete_ids
        self.cohort: dict[str, object] | None = None
        self.cohort_sha256: str | None = None
        self.failure_seen = False
        self.failed_ids: set[str] = set()
        self.failure_allowed_ids: set[str] = set()
        self.preemption_seen = False
        self.preemption_forwarded = False
        self.events: list[dict] = []
        self.cell_runtime: dict[str, dict] = {}
        self.cell_phase_runtime: dict[str, dict] = {}
        self.request_dir = self.runs_root / ".h100/requeue-requests"
        self.manifest_path = self.runs_root / ".h100/campaign_manifest.json"
        signal.signal(signal.SIGUSR1, self._on_usr1)

    def _on_usr1(self, _signum, _frame) -> None:
        self.preemption_seen = True

    def record(self, event: str, **payload: object) -> None:
        item = {"utc": utc_now(), "event": event, **payload}
        self.events.append(item)
        print(json.dumps(item, sort_keys=True), flush=True)
        self.write_manifest()

    def write_manifest(self, status: str | None = None) -> None:
        payload = {
            "schema": 2,
            "campaign_id": self.args.campaign_id,
            "status": status or ("failed" if self.failure_seen else "running"),
            "phase": self.phase,
            "git_sha": self.git_sha,
            "detector_sha256": self.detector_sha256,
            "precision": EXPECTED_PRECISION,
            "micro_batch": MICRO_BATCH,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch": EFFECTIVE_BATCH,
            "venv_sha256": self.venv_sha256,
            "venv_build_sha256": self.venv_build_sha256,
            "base_python": self.base_python,
            "wheelhouse": self.wheelhouse,
            "base_payload": self.base_payload,
            "runtime_amendment": self.runtime_amendment,
            "acceptance_uuid": self.acceptance_uuid,
            "source_validation_sha256": self.source_validation_sha256,
            "evaluation_ground_truth_sha256": self.evaluation_ground_truth_sha256,
            "h100_ready_sha256": self.h100_ready_sha256,
            "external_controls_policy": self.external_controls_policy,
            "hardware": self.args.hardware,
            "accepted_hardware_class": self.accepted_hardware_class,
            "allocation_hardware_class": self.allocation_hardware_class,
            "strict_fp32": self.strict_fp32,
            "acceptance": {
                "uuid": self.acceptance_uuid,
                "created_utc": self.acceptance.get("created_utc"),
                "gates": self.acceptance.get("gates"),
                "frozen_sha256": self.acceptance.get("source", {}).get(
                    "frozen_sha256"
                ),
                "slurm_smoke": self.acceptance.get("slurm_smoke"),
                "scratch_free_bytes": self.acceptance.get("scratch_free_bytes"),
            },
            "source_validation": {
                "path": str(self.args.source_validation_json.absolute()),
                "sha256": self.source_validation_sha256,
                "receipt": self.source_validation,
            },
            "evaluation_ground_truth": {
                "path": str(self.runs_root / ".h100/EVAL_GROUND_TRUTH_VALIDATED.json"),
                "sha256": self.evaluation_ground_truth_sha256,
                "receipt": self.evaluation_ground_truth,
            },
            "data_view": {
                "path": str(self.data_view_root),
                "sha256": self.data_view_sha256,
                "phase": self.data_view_phase,
                "contract": self.data_view.get("contract"),
                "receipt": self.data_view,
            },
            "training_cohort": (
                {"path": str(self.cohort_path), "sha256": self.cohort_sha256}
                if self.cohort_sha256 is not None
                else None
            ),
            "throughput_projection": self.acceptance.get("projection"),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cell_order": [cell.exp_id for cell in self.cells],
            "complete": sorted(self.complete_ids),
            "training_complete": sorted(self.training_complete_ids),
            "test_complete": sorted(self.test_complete_ids),
            "running": {
                str(gpu): cell.exp_id
                for gpu, (_process, cell, _started) in self.running.items()
            },
            "events": self.events,
            "cell_runtime": self.cell_runtime,
            "cell_phase_runtime": self.cell_phase_runtime,
            "fail_stop": {
                "engaged": self.failure_seen,
                "failed": sorted(self.failed_ids),
                "allowed_to_finish": sorted(self.failure_allowed_ids),
            },
            "updated_utc": utc_now(),
        }
        atomic_write_json(self.manifest_path, payload)

    def initialize(self) -> None:
        self.runs_root.mkdir(parents=True, exist_ok=True)
        resumed = validate_campaign_start_state(
            repo=self.repo,
            runs_root=self.runs_root,
            manifest_path=self.manifest_path,
            cohort_path=self.cohort_path,
        )
        if self.cohort_path.exists() or self.cohort_path.is_symlink():
            self.cohort, self.cohort_sha256 = validate_training_cohort(
                path=self.cohort_path,
                cells=self.cells,
                runs_root=self.runs_root,
                git_sha=self.git_sha,
                detector_sha256=self.detector_sha256,
                candidate_floor=self.candidate_floor,
            )
            self.phase = "score-test"
            self.training_complete_ids = {cell.exp_id for cell in self.cells}
            self.complete_ids = self.test_complete_ids
        else:
            self.phase = "train"
            self.complete_ids = self.training_complete_ids
            leaked = [
                cell.exp_id
                for cell in self.cells
                if test_marker_for(cell, self.runs_root).exists()
                or test_marker_for(cell, self.runs_root).is_symlink()
            ]
            if leaked:
                raise RuntimeError(
                    "TEST results exist before the all-32 training cohort: "
                    + ", ".join(leaked)
                )
        if self.phase != self.data_view_phase:
            raise RuntimeError(
                "campaign phase differs from the validated allocation data view"
            )

        prior: dict[str, object] | None = None
        if self.manifest_path.exists():
            prior = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            bindings = {
                "campaign_id": self.args.campaign_id,
                "git_sha": self.git_sha,
                "detector_sha256": self.detector_sha256,
                "venv_sha256": self.venv_sha256,
                "venv_build_sha256": self.venv_build_sha256,
                "base_python": self.base_python,
                "wheelhouse": self.wheelhouse,
                "base_payload": self.base_payload,
                "runtime_amendment": self.runtime_amendment,
                "acceptance_uuid": self.acceptance_uuid,
                "source_validation_sha256": self.source_validation_sha256,
                "evaluation_ground_truth_sha256": self.evaluation_ground_truth_sha256,
                "h100_ready_sha256": self.h100_ready_sha256,
                "external_controls_policy": self.external_controls_policy,
                "strict_fp32": self.strict_fp32,
                "accepted_hardware_class": self.accepted_hardware_class,
            }
            validate_campaign_resume_bindings(prior, bindings)
            prior_phase = prior.get("phase")
            recovering_transition = (
                prior_phase == "train"
                and self.phase == "score-test"
                and set(prior.get("training_complete", []))
                == {cell.exp_id for cell in self.cells}
            )
            if prior_phase != self.phase and not recovering_transition:
                raise RuntimeError(
                    "existing campaign phase conflicts with canonical cohort state"
                )
            prior_cohort = prior.get("training_cohort")
            current_cohort = (
                {"path": str(self.cohort_path), "sha256": self.cohort_sha256}
                if self.cohort_sha256 is not None
                else None
            )
            if (
                recovering_transition
                and prior_cohort is not None
            ) or (
                not recovering_transition
                and prior_cohort != current_cohort
            ):
                raise RuntimeError(
                    "existing campaign training-cohort binding drifted"
                )
            self.events = list(prior.get("events", []))
            self.cell_runtime = dict(prior.get("cell_runtime", {}))
            self.cell_phase_runtime = dict(prior.get("cell_phase_runtime", {}))
            (
                self.failure_seen,
                self.failed_ids,
                self.failure_allowed_ids,
            ) = restore_fail_stop(prior, self.cells)

        cohort_binding = (
            {"path": str(self.cohort_path), "sha256": str(self.cohort_sha256)}
            if self.cohort_sha256 is not None
            else None
        )
        for cell in self.cells:
            if cell.exp_id in self.failed_ids:
                validate_failed_namespace(
                    cell,
                    runs_root=self.runs_root,
                    campaign_id=self.args.campaign_id,
                    git_sha=self.git_sha,
                    detector_sha256=self.detector_sha256,
                    venv_sha256=self.venv_sha256,
                    venv_build_sha256=self.venv_build_sha256,
                    base_python=self.base_python,
                    wheelhouse=self.wheelhouse,
                    base_payload=self.base_payload,
                    runtime_amendment=self.runtime_amendment,
                    acceptance_uuid=self.acceptance_uuid,
                    source_validation_sha256=self.source_validation_sha256,
                    h100_ready_sha256=self.h100_ready_sha256,
                    external_controls_policy=self.external_controls_policy,
                    strict_fp32=self.strict_fp32,
                    accepted_hardware_class=self.accepted_hardware_class,
                    evaluation_ground_truth_sha256=self.evaluation_ground_truth_sha256,
                    training_cohort=cohort_binding,
                )
                continue
            state = existing_cell_state(
                cell,
                phase=self.phase,
                repo=self.repo,
                runs_root=self.runs_root,
                campaign_id=self.args.campaign_id,
                git_sha=self.git_sha,
                detector_sha256=self.detector_sha256,
                candidate_floor=self.candidate_floor,
                venv_sha256=self.venv_sha256,
                venv_build_sha256=self.venv_build_sha256,
                base_python=self.base_python,
                wheelhouse=self.wheelhouse,
                base_payload=self.base_payload,
                runtime_amendment=self.runtime_amendment,
                acceptance_uuid=self.acceptance_uuid,
                source_validation_sha256=self.source_validation_sha256,
                evaluation_ground_truth_sha256=self.evaluation_ground_truth_sha256,
                h100_ready_sha256=self.h100_ready_sha256,
                external_controls_policy=self.external_controls_policy,
                strict_fp32=self.strict_fp32,
                accepted_hardware_class=self.accepted_hardware_class,
                cohort=self.cohort,
                cohort_sha256=self.cohort_sha256,
                test_scene_ids=self.test_scene_ids,
            )
            if state != "complete":
                continue
            self.complete_ids.add(cell.exp_id)
            provenance = json.loads(
                (self.runs_root / cell.exp_id / "runtime_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.cell_phase_runtime[cell.exp_id] = dict(
                provenance.get("phase_runtime", {})
            )
            if self.phase == "score-test":
                marker = validate_scored_completion(
                    cell,
                    repo=self.repo,
                    runs_root=self.runs_root,
                    git_sha=self.git_sha,
                    detector_sha256=self.detector_sha256,
                    candidate_floor=self.candidate_floor,
                    cohort=self.cohort,
                    cohort_sha256=self.cohort_sha256,
                    test_scene_ids=self.test_scene_ids,
                )
                self.cell_runtime[cell.exp_id] = finalized_cell_runtime(
                    provenance, marker, cell.exp_id
                )
        self.request_dir.mkdir(parents=True, exist_ok=True)
        for request in self.request_dir.glob("gpu-*.request"):
            request.unlink()
        self.record(
            "controller_started",
            resumed=resumed,
            phase=self.phase,
            recovered_cohort_transition=bool(
                prior is not None
                and prior.get("phase") == "train"
                and self.phase == "score-test"
            ),
        )
    def launch(self, gpu: int, cell: Cell) -> None:
        if self.phase == "score-test" and (
            self.cohort is None or self.cohort_sha256 is None
        ):
            raise RuntimeError("refusing to launch TEST scoring without frozen cohort")
        run_dir = self.runs_root / cell.exp_id
        run_dir.mkdir(parents=True, exist_ok=True)
        last_ckpt = run_dir / "checkpoints/last.ckpt"
        hardware = self.args.hardware["devices"][gpu]
        provenance_path = run_dir / "runtime_provenance.json"
        prior = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.exists()
            else {}
        )
        recovered_open_attempt = False
        if prior:
            recovered_open_attempt = reconcile_stale_open_attempt(
                prior,
                phase=self.phase,
                recovered_utc=utc_now(),
            )
        cohort_binding = (
            {"path": str(self.cohort_path), "sha256": str(self.cohort_sha256)}
            if self.phase == "score-test"
            else None
        )
        attempts = list(prior.get("attempts", []))
        attempts.append(
            {
                "attempt": len(attempts) + 1,
                "phase": self.phase,
                "training_cohort_sha256": self.cohort_sha256,
                "started_utc": utc_now(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "gpu_local_index": gpu,
                "gpu_uuid": hardware["uuid"],
                "gpu_name": hardware["name"],
                "gpu_total_memory_bytes": hardware["total_memory_bytes"],
                "compute_capability": hardware["compute_capability"],
                "driver_version": self.allocation_hardware_class["driver_version"],
                "torch": self.allocation_hardware_class["torch"],
                "cuda_build": self.allocation_hardware_class["cuda_build"],
                "resumed_from_last_ckpt": self.phase == "train" and last_ckpt.exists(),
            }
        )
        provenance = {
            **prior,
            "schema": 2,
            "campaign_id": self.args.campaign_id,
            "exp_id": cell.exp_id,
            "run_dir": str(run_dir),
            "phase": self.phase,
            "training_cohort": cohort_binding,
            "git_sha": self.git_sha,
            "detector_sha256": self.detector_sha256,
            "precision": EXPECTED_PRECISION,
            "micro_batch": MICRO_BATCH,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "effective_batch": EFFECTIVE_BATCH,
            "venv_sha256": self.venv_sha256,
            "venv_build_sha256": self.venv_build_sha256,
            "base_python": self.base_python,
            "wheelhouse": self.wheelhouse,
            "base_payload": self.base_payload,
            "runtime_amendment": self.runtime_amendment,
            "acceptance_uuid": self.acceptance_uuid,
            "source_validation_sha256": self.source_validation_sha256,
            "evaluation_ground_truth_sha256": self.evaluation_ground_truth_sha256,
            "h100_ready_sha256": self.h100_ready_sha256,
            "external_controls_policy": self.external_controls_policy,
            "strict_fp32": self.strict_fp32,
            "accepted_hardware_class": self.accepted_hardware_class,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
            "gpu_local_index": gpu,
            "gpu_name": hardware["name"],
            "gpu_uuid": hardware["uuid"],
            "gpu_total_memory_bytes": hardware["total_memory_bytes"],
            "compute_capability": hardware["compute_capability"],
            "driver_version": self.allocation_hardware_class["driver_version"],
            "torch": self.allocation_hardware_class["torch"],
            "cuda_build": self.allocation_hardware_class["cuda_build"],
            "started_utc": prior.get("started_utc", attempts[0]["started_utc"]),
            "attempts": attempts,
            "phase_runtime": dict(prior.get("phase_runtime", {})),
            "accumulated_active_seconds": float(
                prior.get("accumulated_active_seconds", 0.0)
            ),
        }
        atomic_write_json(provenance_path, provenance)
        log_path = self.runs_root / "logs/h100" / f"{cell.exp_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "scripts.h100.cell",
            "--repo",
            str(self.repo),
            "--runs-root",
            str(self.runs_root),
            "--phase",
            self.phase,
            "--exp-id",
            cell.exp_id,
            "--init",
            cell.init,
            "--label-frac",
            str(cell.fraction),
            "--git-sha",
            self.git_sha,
            "--detector-sha256",
            self.detector_sha256,
            "--candidate-floor",
            str(self.candidate_floor),
            "--workers",
            str(self.args.workers_per_gpu),
        ]
        if self.phase == "score-test":
            command.extend(
                [
                    "--training-cohort",
                    str(self.cohort_path),
                    "--training-cohort-sha256",
                    str(self.cohort_sha256),
                ]
            )
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "H100_REQUEUE_REQUEST_DIR": str(self.request_dir),
            "PATH": f"{self.repo / 'slurm/h100/shims'}:{os.environ['PATH']}",
        }
        output = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=self.repo,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        output.close()
        self.running[gpu] = (process, cell, time.monotonic())
        self.record(
            "cell_launched",
            gpu=gpu,
            exp_id=cell.exp_id,
            phase=self.phase,
            training_cohort_sha256=self.cohort_sha256,
            recovered_open_attempt=recovered_open_attempt,
        )
    def requeue_request_ready(self, gpu: int) -> bool:
        request = self.request_dir / f"gpu-{gpu}.request"
        try:
            return (
                request.is_file()
                and not request.is_symlink()
                and request.read_text(encoding="utf-8").strip()
                == os.environ["SLURM_JOB_ID"]
            )
        except (OSError, UnicodeError):
            return False

    def poll(self, *, preemption_barrier: bool = False) -> None:
        for gpu in list(self.running):
            process, cell, started = self.running[gpu]
            code = process.poll()
            if code is None:
                continue
            if (
                preemption_barrier
                and code == HOST_REQUEUE_EXIT_CODE
                and self.requeue_request_ready(gpu)
            ):
                continue
            elapsed = time.monotonic() - started
            del self.running[gpu]
            valid = False
            payload: dict[str, object] | None = None
            provenance_path = self.runs_root / cell.exp_id / "runtime_provenance.json"
            if code == 0:
                try:
                    if self.phase == "train":
                        payload = validate_training_completion(
                            cell,
                            runs_root=self.runs_root,
                            git_sha=self.git_sha,
                            detector_sha256=self.detector_sha256,
                            candidate_floor=self.candidate_floor,
                            seal=True,
                        )
                    else:
                        payload = validate_scored_completion(
                            cell,
                            repo=self.repo,
                            runs_root=self.runs_root,
                            git_sha=self.git_sha,
                            detector_sha256=self.detector_sha256,
                            candidate_floor=self.candidate_floor,
                            cohort=self.cohort,
                            cohort_sha256=self.cohort_sha256,
                            test_scene_ids=self.test_scene_ids,
                        )
                    provenance = json.loads(
                        provenance_path.read_text(encoding="utf-8")
                    )
                    validate_runtime_provenance(
                        provenance,
                        cell=cell,
                        campaign_id=self.args.campaign_id,
                        git_sha=self.git_sha,
                        detector_sha256=self.detector_sha256,
                        venv_sha256=self.venv_sha256,
                        venv_build_sha256=self.venv_build_sha256,
                        base_python=self.base_python,
                        wheelhouse=self.wheelhouse,
                        base_payload=self.base_payload,
                        runtime_amendment=self.runtime_amendment,
                        acceptance_uuid=self.acceptance_uuid,
                        source_validation_sha256=self.source_validation_sha256,
                        h100_ready_sha256=self.h100_ready_sha256,
                        external_controls_policy=self.external_controls_policy,
                        strict_fp32=self.strict_fp32,
                        accepted_hardware_class=self.accepted_hardware_class,
                        evaluation_ground_truth_sha256=self.evaluation_ground_truth_sha256,
                        training_cohort=(
                            {
                                "path": str(self.cohort_path),
                                "sha256": str(self.cohort_sha256),
                            }
                            if self.phase == "score-test"
                            else None
                        ),
                    )
                    valid = True
                except RuntimeError as exc:
                    self.record(
                        "invalid_completion_marker",
                        exp_id=cell.exp_id,
                        phase=self.phase,
                        error=str(exc),
                    )
            if valid:
                assert payload is not None
                provenance = json.loads(
                    provenance_path.read_text(encoding="utf-8")
                )
                finished = utc_now()
                provenance["attempts"][-1].update(
                    {
                        "finished_utc": finished,
                        "exit_code": code,
                        "active_seconds": elapsed,
                    }
                )
                total_seconds = (
                    float(provenance.get("accumulated_active_seconds", 0.0))
                    + elapsed
                )
                selected = [
                    attempt
                    for attempt in provenance["attempts"]
                    if attempt.get("phase") == self.phase
                ]
                phase_seconds = sum(
                    float(attempt["active_seconds"]) for attempt in selected
                )
                phase_record: dict[str, object] = {
                    "completed_utc": finished,
                    "active_seconds": phase_seconds,
                    "elapsed_hours": phase_seconds / 3600.0,
                    "attempts": len(selected),
                }
                if self.phase == "train":
                    phase_record.update(
                        {
                            "final_metrics_sha256": sha256_file(
                                marker_for(cell, self.runs_root)
                            ),
                            "best_checkpoint_sha256": payload["best_checkpoint"]["sha256"],
                            "epochs_run": payload["epochs_run"],
                            "best_dev_f1": payload["best_dev_f1"],
                        }
                    )
                    provenance.update(
                        {
                            "training_completed_utc": finished,
                            "epochs_run": payload["epochs_run"],
                            "best_dev_f1": payload["best_dev_f1"],
                            "h100_runtime_contract": payload["h100_runtime_contract"],
                        }
                    )
                    self.training_complete_ids.add(cell.exp_id)
                else:
                    phase_record.update(
                        {
                            "test_metrics_sha256": payload["test_result_sha256"],
                            "training_cohort_sha256": payload[
                                "training_cohort_sha256"
                            ],
                            "test_f1": payload["test_f1"],
                        }
                    )
                    provenance.update(
                        {
                            "completed_utc": finished,
                            "elapsed_hours": total_seconds / 3600.0,
                            "test_f1": payload["test_f1"],
                            "test_scored_at": payload["test_scored_at"],
                        }
                    )
                    self.test_complete_ids.add(cell.exp_id)
                phase_runtime = dict(provenance.get("phase_runtime", {}))
                phase_runtime[self.phase] = phase_record
                provenance.update(
                    {
                        "phase_runtime": phase_runtime,
                        "accumulated_active_seconds": total_seconds,
                    }
                )
                atomic_write_json(provenance_path, provenance)
                _validated_phase_runtime(
                    provenance,
                    payload,
                    exp_id=cell.exp_id,
                    phase=self.phase,
                )
                self.cell_phase_runtime[cell.exp_id] = phase_runtime
                if self.phase == "score-test":
                    self.cell_runtime[cell.exp_id] = finalized_cell_runtime(
                        provenance, payload, cell.exp_id
                    )
                self.record(
                    "cell_phase_complete",
                    gpu=gpu,
                    exp_id=cell.exp_id,
                    phase=self.phase,
                    elapsed_hours=phase_seconds / 3600.0,
                )
            else:
                first_failure = not self.failure_seen
                self.failure_seen = True
                self.failed_ids.add(cell.exp_id)
                self.failure_allowed_ids.discard(cell.exp_id)
                if first_failure:
                    self.failure_allowed_ids.update(
                        running_cell.exp_id
                        for _process, running_cell, _started in self.running.values()
                    )
                    self.record(
                        "fail_stop_engaged",
                        phase=self.phase,
                        failed_exp_id=cell.exp_id,
                        allowed_to_finish=sorted(self.failure_allowed_ids),
                    )
                self.record(
                    "cell_failed",
                    gpu=gpu,
                    exp_id=cell.exp_id,
                    phase=self.phase,
                    exit_code=code,
                )
    def _stop_children(self) -> None:
        for process, _cell, _started in self.running.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and any(
            process.poll() is None for process, _cell, _started in self.running.values()
        ):
            time.sleep(0.5)
        for process, _cell, _started in self.running.values():
            if process.poll() is None:
                process.terminate()
        for process, _cell, _started in self.running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    def freeze_training_cohort(self) -> None:
        if (
            self.phase != "train"
            or self.running
            or self.failure_seen
            or self.training_complete_ids
            != {cell.exp_id for cell in self.cells}
        ):
            raise RuntimeError(
                "training cohort freeze requires all 32 successful training cells"
            )
        leaked = [
            cell.exp_id
            for cell in self.cells
            if test_marker_for(cell, self.runs_root).exists()
            or test_marker_for(cell, self.runs_root).is_symlink()
        ]
        if leaked:
            raise RuntimeError(
                "TEST result appeared before cohort freeze: " + ", ".join(leaked)
            )
        create_training_cohort(
            cells=self.cells,
            runs_root=self.runs_root,
            output=self.cohort_path,
            git_sha=self.git_sha,
            detector_sha256=self.detector_sha256,
            candidate_floor=self.candidate_floor,
        )
        self.cohort, self.cohort_sha256 = validate_training_cohort(
            path=self.cohort_path,
            cells=self.cells,
            runs_root=self.runs_root,
            git_sha=self.git_sha,
            detector_sha256=self.detector_sha256,
            candidate_floor=self.candidate_floor,
        )
        self.phase = "score-test"
        self.complete_ids = self.test_complete_ids
        self.record(
            "training_cohort_frozen",
            path=str(self.cohort_path),
            sha256=self.cohort_sha256,
            cells=len(self.training_complete_ids),
        )


    def finalize_grid(self) -> int:
        if (
            self.phase != "score-test"
            or self.cohort is None
            or self.cohort_sha256 is None
            or self.running
            or self.training_complete_ids
            != {cell.exp_id for cell in self.cells}
            or self.test_complete_ids
            != {cell.exp_id for cell in self.cells}
        ):
            raise RuntimeError(
                "campaign grid finalization requires the frozen 32-cell TEST cohort"
            )
        try:
            validated_cohort, validated_sha256 = validate_training_cohort(
                path=self.cohort_path,
                cells=self.cells,
                runs_root=self.runs_root,
                git_sha=self.git_sha,
                detector_sha256=self.detector_sha256,
                candidate_floor=self.candidate_floor,
            )
            if (
                validated_cohort != self.cohort
                or validated_sha256 != self.cohort_sha256
            ):
                raise RuntimeError(
                    "training cohort drifted before final grid validation"
                )
            validate_complete_test_cohort(
                cells=self.cells,
                runs_root=self.runs_root,
                cohort=self.cohort,
                cohort_sha256=self.cohort_sha256,
                test_scene_ids=self.test_scene_ids,
            )
            grid = collect_and_validate_grid(
                repo=self.repo,
                runs_root=self.runs_root,
                cells=self.cells,
                git_sha=self.git_sha,
                detector_sha256=self.detector_sha256,
            )
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            self.failure_seen = True
            self.record("grid_validation_failed", error=str(exc))
            self.write_manifest(status="failed")
            return 1
        self.record(
            "grid_validated",
            path=str(grid),
            sha256=sha256_file(grid),
            rows=32,
        )
        self.write_manifest(status="complete")
        return 0

    def preempt_and_requeue(self) -> int:
        # A child may finish between the main-loop poll and SIGUSR1 forwarding.
        # Finalize valid completions first; invalid or unbound exits fail closed.
        self.poll(preemption_barrier=True)
        if self.failure_seen:
            self._stop_children()
            self.running.clear()
            self.record(
                "preemption_invalid_child_exit",
                failed_ids=sorted(self.failed_ids),
            )
            return 1
        if not self.running:
            if self.complete_ids == {cell.exp_id for cell in self.cells}:
                if self.phase == "train":
                    self.freeze_training_cohort()
                    self.write_manifest(status="host-requeue-required")
                    return HOST_REQUEUE_EXIT_CODE
                return self.finalize_grid()
            self.write_manifest(status="host-requeue-required")
            return HOST_REQUEUE_EXIT_CODE
        if not self.preemption_forwarded:
            forwarded = 0
            for process, _cell, _started in self.running.values():
                if process.poll() is not None:
                    continue
                try:
                    os.kill(process.pid, signal.SIGUSR1)
                    forwarded += 1
                except ProcessLookupError:
                    pass
            self.preemption_forwarded = True
            self.record("usr1_forwarded", count=forwarded)

        deadline = time.monotonic() + self.args.checkpoint_timeout
        while True:
            self.poll(preemption_barrier=True)
            if self.failure_seen:
                self._stop_children()
                self.running.clear()
                self.record(
                    "preemption_invalid_child_exit",
                    failed_ids=sorted(self.failed_ids),
                )
                return 1
            expected = list(self.running)
            if all(self.requeue_request_ready(gpu) for gpu in expected):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(1)
        missing = [
            gpu
            for gpu in self.running
            if not self.requeue_request_ready(gpu)
        ]
        if missing:
            self.failure_seen = True
            self._stop_children()
            self.running.clear()
            self.record("preemption_checkpoint_timeout", missing_gpus=missing)
            return 1

        # Every child may have completed successfully while the checkpoint
        # barrier was polling. A fully complete matrix must still pass the
        # same grid finalization as the normal controller loop; otherwise the
        # allocation requeues to launch the remaining cells.
        if not self.running:
            if self.complete_ids == {cell.exp_id for cell in self.cells}:
                if self.phase == "train":
                    self.freeze_training_cohort()
                    self.write_manifest(status="host-requeue-required")
                    return HOST_REQUEUE_EXIT_CODE
                return self.finalize_grid()
            self.write_manifest(status="host-requeue-required")
            return HOST_REQUEUE_EXIT_CODE

        for _gpu, (_process, cell, started) in self.running.items():
            checkpoint = checkpoint_for_preemption(self.runs_root / cell.exp_id)
            provenance_path = self.runs_root / cell.exp_id / "runtime_provenance.json"
            provenance = json.loads(provenance_path.read_text())
            elapsed = time.monotonic() - started
            provenance["accumulated_active_seconds"] = float(
                provenance.get("accumulated_active_seconds", 0.0)
            ) + elapsed
            provenance["attempts"][-1].update(
                {"finished_utc": utc_now(), "exit": "preempted", "active_seconds": elapsed}
            )
            atomic_write_json(provenance_path, provenance)
            self.record(
                "checkpoint_promoted",
                exp_id=cell.exp_id,
                last_ckpt=str(checkpoint),
                sha256=sha256_file(checkpoint),
            )
        self._stop_children()
        self.running.clear()
        self.write_manifest(status="host-requeue-required")
        return HOST_REQUEUE_EXIT_CODE

    def run(self) -> int:
        self.initialize()
        while True:
            self.poll()
            if self.preemption_seen:
                return self.preempt_and_requeue()
            running_ids = {cell.exp_id for _process, cell, _started in self.running.values()}
            free = [gpu for gpu in range(EXPECTED_GPU_COUNT) if gpu not in self.running]
            if self.failure_seen:
                launches = fail_stop_launches(
                    self.cells,
                    allowed_ids=self.failure_allowed_ids,
                    failed_ids=self.failed_ids,
                    running_ids=running_ids,
                    complete_ids=self.complete_ids,
                    free_gpus=free,
                    preemption_seen=self.preemption_seen,
                )
            else:
                launches = next_launches(
                    self.cells,
                    running_ids=running_ids,
                    complete_ids=self.complete_ids,
                    free_gpus=free,
                    failure_seen=False,
                    preemption_seen=self.preemption_seen,
                )
            for gpu, cell in launches:
                self.launch(gpu, cell)
            if self.failure_seen and not self.running:
                self.write_manifest(status="failed")
                return 1
            if self.complete_ids == {cell.exp_id for cell in self.cells} and not self.running:
                if self.phase == "train":
                    self.freeze_training_cohort()
                    self.write_manifest(status="host-requeue-required")
                    return HOST_REQUEUE_EXIT_CODE
                return self.finalize_grid()
            time.sleep(self.args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--venv-root", type=Path, required=True)
    parser.add_argument("--venv-sha256", required=True)
    parser.add_argument("--venv-build-sha256", required=True)
    parser.add_argument("--hardware-json", type=Path, required=True)
    parser.add_argument("--acceptance-json", type=Path, required=True)
    parser.add_argument("--source-validation-json", type=Path, required=True)
    parser.add_argument("--source-validation-sha256", required=True)
    parser.add_argument("--data-view-root", type=Path, required=True)
    parser.add_argument("--data-view-receipt-sha256", required=True)
    parser.add_argument("--acceptance-sha256", required=True)
    parser.add_argument("--workers-per-gpu", type=int, default=6)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--checkpoint-timeout", type=float, default=240.0)
    args = parser.parse_args()
    if args.workers_per_gpu * EXPECTED_GPU_COUNT > 48:
        parser.error("workers-per-gpu exceeds the 48-CPU allocation")
    args.hardware = json.loads(args.hardware_json.read_text())
    return Controller(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
