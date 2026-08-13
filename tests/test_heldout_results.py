"""Fail-closed held-out evidence generator tests.

The committed generated artifacts must be exactly reproducible from the
committed evidence tree, and any tampered evidence byte must refuse to
generate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.analysis.heldout_results import (
    EvidenceError,
    render_tex,
    validate_evidence,
)

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "results/h100/evidence"
ARMS = REPO / "configs/arms.yaml"
DETECTOR = REPO / "configs/detector.yaml"
SPLITS = REPO / "data/splits.json"
GENERATED = REPO / "docs/results/generated"


def _validated() -> dict:
    return validate_evidence(
        EVIDENCE, arms_config=ARMS, detector_config=DETECTOR, splits_config=SPLITS
    )


def _copy_evidence(tmp_path: Path) -> Path:
    destination = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, destination)
    return destination


def _revalidate(evidence: Path) -> dict:
    return validate_evidence(
        evidence, arms_config=ARMS, detector_config=DETECTOR, splits_config=SPLITS
    )


def test_committed_evidence_validates_and_reproduces_the_committed_tex() -> None:
    validated = _validated()
    assert len(validated["cells"]) == 32
    committed = (GENERATED / "heldout_results.tex").read_text(encoding="utf-8")
    assert render_tex(validated) == committed


def test_committed_manifest_binds_the_committed_inputs() -> None:
    manifest = json.loads(
        (GENERATED / "heldout_generated_manifest.json").read_text(encoding="utf-8")
    )
    validated = _validated()
    assert manifest["inputs"]["cohort_sha256"] == validated["campaign"]["cohort_sha256"]
    assert manifest["inputs"]["detector_sha256"] == validated["campaign"]["detector_sha256"]
    assert manifest["summary"]["cells"] == 32


def test_every_dev_value_is_finite_and_bound_to_a_checkpoint_hash() -> None:
    validated = _validated()
    for exp_id, cell in validated["cells"].items():
        assert 0.0 < cell["dev_f1"] < 1.0, exp_id
        assert len(cell["checkpoint_sha256"]) == 64, exp_id
        assert 0.0 < cell["threshold"] < 1.0, exp_id


def test_tampered_marker_byte_refuses_to_generate(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    marker = evidence / "vitrand-f10-s0" / "final_metrics.json"
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["best_dev"]["f1"] = 0.999
    marker.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceError, match="cohort binding"):
        _revalidate(evidence)


def test_tampered_training_curve_refuses_to_generate(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    curve = evidence / "vitrand-f10-s0" / "metrics.csv"
    text = curve.read_text(encoding="utf-8")
    lines = text.splitlines()
    header = lines[0].split(",")
    f1_column = header.index("dev_f1")
    for index in range(1, len(lines)):
        row = lines[index].split(",")
        if row[f1_column].strip():
            row[f1_column] = "0.9999"
            lines[index] = ",".join(row)
            break
    curve.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="training curve"):
        _revalidate(evidence)


def test_missing_cell_refuses_to_generate(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    shutil.rmtree(evidence / "sarmae-f50-s0")
    with pytest.raises(EvidenceError):
        _revalidate(evidence)


def test_partial_or_unbound_test_results_refuse_to_generate(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    bogus = {
        "test_result_schema": 1,
        "status": "test-complete",
        "scored_utc": "2026-01-01T00:00:00+00:00",
        "exp_id": "vitrand-f10-s0",
        "cohort_sha256": "0" * 64,
        "completion_marker_sha256": "0" * 64,
        "git_sha": "0" * 40,
        "detector_sha256": "0" * 64,
        "inference_precision": "32-true",
        "threshold_source": {
            "kind": "best-dev-checkpoint-bound",
            "threshold": 0.5,
            "dev_epoch": 0,
            "checkpoint_relative_path": "vitrand-f10-s0/checkpoints/best.ckpt",
            "checkpoint_sha256": "0" * 64,
            "checkpoint_epoch": 0,
        },
        "metrics": {},
        "per_scene": {},
    }
    (evidence / "vitrand-f10-s0" / "test_metrics.json").write_text(
        json.dumps(bogus), encoding="utf-8"
    )
    with pytest.raises(EvidenceError):
        _revalidate(evidence)


def test_tampered_test_threshold_refuses_to_generate(tmp_path: Path) -> None:
    """A test result claiming a threshold other than the cohort-bound one
    must fail even though its own metrics are internally TP/FP/FN-consistent."""

    evidence = _copy_evidence(tmp_path)
    result_path = evidence / "vitrand-f10-s0" / "test_metrics.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["threshold_source"]["threshold"] = 0.01
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceError, match="binding mismatch"):
        _revalidate(evidence)


def test_committed_test_results_are_all_present_and_bound() -> None:
    validated = _validated()
    assert len(validated["test_results"]) == 32
    for exp_id, test in validated["test_results"].items():
        assert 0.0 < test["f1"] < 1.0, exp_id
        assert test["inference_precision"] == "32-true", exp_id


def test_dev_to_test_macros_render_with_real_values() -> None:
    validated = _validated()
    tex = render_tex(validated)
    assert "\\HevTestCompletetrue" in tex
    assert "\\HevTestFViTRandomTen{0.7042" in tex or "\\HevTestFViTRandomTen{0.70" in tex
