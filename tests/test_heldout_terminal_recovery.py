"""A cell without a Lightning curve validates only through a bound terminal recovery.

Two replacement32 cells ended in a zero-step terminal resume that re-ran the
last development evaluation and did not persist ``metrics.csv``. The marker
hash-binds ``terminal_recovery.json``; the generator accepts the missing curve
only when that record added no training and selected exactly the marker's
checkpoint and development result.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.analysis.heldout_results import EvidenceError, validate_evidence

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "results/h100/evidence"
ARMS = REPO / "configs/arms.yaml"
DETECTOR = REPO / "configs/detector.yaml"
SPLITS = REPO / "data/splits.json"
CELL = "vitrand-f10-s0"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_bytes((json.dumps(payload, indent=1) + "\n").encode("utf-8"))


def _recovered_evidence(tmp_path: Path, **overrides: object) -> Path:
    evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, evidence)
    # Re-binding a marker changes the cohort hash, which every TEST result
    # binds; dropping them all keeps the all-or-nothing TEST rule satisfied.
    for test_result in evidence.glob("*/test_metrics.json"):
        test_result.unlink()
    cell = evidence / CELL
    (cell / "metrics.csv").unlink()
    marker_path = cell / "final_metrics.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    record = {
        "schema": 1,
        "status": "terminal-dev-recovered",
        "exp_id": CELL,
        "metrics_csv_absent": True,
        "added_optimizer_steps": 0,
        "added_scheduler_steps": 0,
        "added_training_batches": 0,
        "selected_dev": marker["best_dev"],
        "selected_checkpoint": dict(marker["best_checkpoint"]),
    }
    record.update(overrides)
    _write_json(cell / "terminal_recovery.json", record)
    marker["terminal_recovery"] = {
        "relative_path": "terminal_recovery.json",
        "sha256": _sha(cell / "terminal_recovery.json"),
    }
    _write_json(marker_path, marker)
    cohort_path = evidence / "TRAINING_COHORT.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    for entry in cohort["cells"]:
        if entry["exp_id"] == CELL:
            entry["completion_marker"]["sha256"] = _sha(marker_path)
    _write_json(cohort_path, cohort)
    return evidence


def _validate(evidence: Path) -> dict:
    return validate_evidence(evidence, arms_config=ARMS, detector_config=DETECTOR, splits_config=SPLITS)


def test_bound_zero_step_terminal_recovery_replaces_the_curve(tmp_path: Path) -> None:
    validated = _validate(_recovered_evidence(tmp_path))
    assert validated["cells"][CELL]["curve"] == "absent-terminal-recovery"
    assert validated["cells"]["vitrand-f25-s0"]["curve"] == "lightning-csv"


def test_missing_curve_without_a_recovery_binding_refuses(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, evidence)
    (evidence / CELL / "metrics.csv").unlink()
    with pytest.raises(EvidenceError, match="training curve is absent"):
        _validate(evidence)


def test_recovery_that_added_training_refuses(tmp_path: Path) -> None:
    evidence = _recovered_evidence(tmp_path, added_training_batches=3)
    with pytest.raises(EvidenceError, match="terminal recovery"):
        _validate(evidence)


def test_recovery_that_selected_a_different_result_refuses(tmp_path: Path) -> None:
    evidence = _recovered_evidence(tmp_path, selected_dev={"f1": 0.5})
    with pytest.raises(EvidenceError, match="terminal recovery"):
        _validate(evidence)


def test_tampered_recovery_record_refuses(tmp_path: Path) -> None:
    evidence = _recovered_evidence(tmp_path)
    record_path = evidence / CELL / "terminal_recovery.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["runtime_seconds"] = 1.0
    _write_json(record_path, record)
    with pytest.raises(EvidenceError, match="terminal recovery"):
        _validate(evidence)
