"""The once-only 50-scene evaluation renders only from bound, complete results.

Each cell's ``final_verified_metrics.json`` must hash to its entry in
``FINAL_EVAL_COMPLETE.json``, bind the frozen cohort, its own TEST result and
its checkpoint-bound threshold, and agree with its counts. The 32 results are
all-or-nothing, like TEST.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from src.analysis.heldout_results import EvidenceError, render_tex, validate_evidence

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "results/h100/evidence"
ARMS = REPO / "configs/arms.yaml"
DETECTOR = REPO / "configs/detector.yaml"
SPLITS = REPO / "data/splits.json"
CELL = "sarmae-f50-s0"


def _validate(evidence: Path = EVIDENCE) -> dict:
    return validate_evidence(evidence, arms_config=ARMS, detector_config=DETECTOR, splits_config=SPLITS)


def _copy(tmp_path: Path) -> Path:
    destination = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, destination)
    return destination


def _rebind(evidence: Path, exp_id: str) -> None:
    """Re-hash one final result into FINAL_EVAL_COMPLETE so only the inner check can fire."""

    complete_path = evidence / "FINAL_EVAL_COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    data = (evidence / exp_id / "final_verified_metrics.json").read_bytes()
    complete["cell_result_sha256"][exp_id] = hashlib.sha256(data).hexdigest()
    complete_path.write_bytes(json.dumps(complete).encode("utf-8"))


def _edit_final(evidence: Path, exp_id: str, edit) -> None:
    path = evidence / exp_id / "final_verified_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    edit(payload)
    path.write_bytes(json.dumps(payload).encode("utf-8"))


def test_committed_final_results_validate_with_one_shared_ground_truth() -> None:
    final = _validate()["final_results"]
    assert len(final) == 32
    supports = {(r["tp"] + r["fn"], r["dark_support"], r["near_shore_support"]) for r in final.values()}
    assert supports == {(8642, 3644, 2686)}


def test_final_macros_render_when_the_evaluation_is_complete() -> None:
    tex = render_tex(_validate())
    assert "\\HevFinalEvaltrue" in tex
    assert "\\def\\HevFinalFCNNImageNetHundred{" in tex
    assert "\\def\\HevFinalPositives{8{,}642}" in tex


def test_tampered_final_result_refuses(tmp_path: Path) -> None:
    evidence = _copy(tmp_path)
    _edit_final(evidence, CELL, lambda p: p["metrics"].update(f1=0.99))
    with pytest.raises(EvidenceError, match="FINAL_EVAL_COMPLETE"):
        _validate(evidence)


def test_final_result_with_inconsistent_counts_refuses_even_when_rehashed(tmp_path: Path) -> None:
    evidence = _copy(tmp_path)
    _edit_final(evidence, CELL, lambda p: p["metrics"].update(f1=0.99))
    _rebind(evidence, CELL)
    with pytest.raises(EvidenceError, match="TP/FP/FN"):
        _validate(evidence)


def test_final_result_bound_to_another_threshold_refuses(tmp_path: Path) -> None:
    evidence = _copy(tmp_path)
    _edit_final(evidence, CELL, lambda p: p["best_dev"].update(threshold=0.5))
    _rebind(evidence, CELL)
    with pytest.raises(EvidenceError, match="does not bind"):
        _validate(evidence)


def test_partial_final_results_refuse(tmp_path: Path) -> None:
    evidence = _copy(tmp_path)
    (evidence / CELL / "final_verified_metrics.json").unlink()
    with pytest.raises(EvidenceError, match="all-or-nothing"):
        _validate(evidence)


def test_no_final_results_renders_the_sealed_branch(tmp_path: Path) -> None:
    evidence = _copy(tmp_path)
    for path in evidence.glob("*/final_verified_metrics.json"):
        path.unlink()
    (evidence / "FINAL_EVAL_COMPLETE.json").unlink()
    validated = _validate(evidence)
    assert validated["final_results"] == {}
    assert "\\HevFinalEvalfalse" in render_tex(validated)
