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
GENERATED = REPO / "docs/results/generated"


def _validated() -> dict:
    return validate_evidence(EVIDENCE, arms_config=ARMS, detector_config=DETECTOR)


def _copy_evidence(tmp_path: Path) -> Path:
    destination = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, destination)
    return destination


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
        validate_evidence(evidence, arms_config=ARMS, detector_config=DETECTOR)


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
        validate_evidence(evidence, arms_config=ARMS, detector_config=DETECTOR)


def test_missing_cell_refuses_to_generate(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    shutil.rmtree(evidence / "sarmae-f50-s0")
    with pytest.raises(EvidenceError):
        validate_evidence(evidence, arms_config=ARMS, detector_config=DETECTOR)


def test_partial_or_unbound_test_results_refuse_to_generate(tmp_path: Path) -> None:
    evidence = _copy_evidence(tmp_path)
    bogus = {
        "exp_id": "vitrand-f10-s0",
        "cohort_sha256": "0" * 64,
        "completion_marker_sha256": "0" * 64,
        "detector_sha256": "0" * 64,
        "metrics": {},
        "per_scene": {},
    }
    (evidence / "vitrand-f10-s0" / "test_metrics.json").write_text(
        json.dumps(bogus), encoding="utf-8"
    )
    with pytest.raises(EvidenceError):
        validate_evidence(evidence, arms_config=ARMS, detector_config=DETECTOR)


def test_dev_macros_flip_to_dashes_never_silently(tmp_path: Path) -> None:
    validated = _validated()
    tex = render_tex(validated)
    assert "\\HevTestCompletefalse" in tex
    assert tex.count("\\textemdash") >= 32
    assert "\\HevDevFViTImageNetHundred{0.9399}" in tex
    assert "\\HevDevFCNNImageNetHundred{0.9387}" in tex
