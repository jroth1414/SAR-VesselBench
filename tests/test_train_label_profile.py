"""The training-label profile counts the frozen train split and binds the audited inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.analysis.heldout_results import EvidenceError, render_tex, validate_evidence
from src.eval.ground_truth_audit import profile_split_labels, scene_ids_sha256

REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "results/h100/evidence"
ARMS = REPO / "configs/arms.yaml"
DETECTOR = REPO / "configs/detector.yaml"
SPLITS = REPO / "data/splits.json"
CSV_HEADER = "scene_id,is_vessel,confidence,distance_from_shore_km\n"


def _validate(evidence: Path = EVIDENCE) -> dict:
    return validate_evidence(evidence, arms_config=ARMS, detector_config=DETECTOR, splits_config=SPLITS)


def test_profile_counts_positives_and_near_shore_positives_of_the_train_split(tmp_path: Path) -> None:
    labels = tmp_path / "train.csv"
    labels.write_bytes((CSV_HEADER + "\n".join([
        "a,True,HIGH,0.5",       # near-shore positive
        "a,True,MEDIUM,12.0",    # offshore positive
        "a,True,LOW,0.1",        # ignore
        "a,False,HIGH,1.0",      # near-shore background
        "b,True,HIGH,",          # positive, unknown distance (not near shore)
        "z,True,HIGH,0.2",       # not a train scene
    ]) + "\n").encode("utf-8"))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"splits": {
        "train": ["a", "b"], "dev": [f"d{i}" for i in range(8)], "test": ["t"], "eval_final": ["z"],
    }}), encoding="utf-8")

    profile = profile_split_labels(train_csv=labels, splits_json=splits, split="train")

    train = profile["train"]
    assert train["scene_count"] == 2 and train["scene_ids_sha256"] == scene_ids_sha256(["a", "b"])
    assert (train["positive"], train["near_shore_positive"]) == (3, 1)
    assert (train["ignore"], train["background"], train["near_shore_background"]) == (1, 1, 1)


def test_committed_profile_validates_and_renders_the_label_gap() -> None:
    validated = _validate()
    assert validated["train_labels"] == {"positive": 7678, "near_shore_positive": 27}
    tex = render_tex(validated)
    assert "\\def\\HevTrainNearShorePositives{27}" in tex
    assert "\\def\\HevTrainNearShorePct{0.4}" in tex
    assert "\\def\\HevFinalNearShorePct{31}" in tex


@pytest.mark.parametrize(
    ("path", "value"),
    [(("inputs", "train_csv", "sha256"), "0" * 64), (("train", "scene_ids_sha256"), "0" * 64)],
)
def test_profile_that_does_not_bind_the_audited_inputs_refuses(tmp_path: Path, path, value) -> None:
    evidence = tmp_path / "evidence"
    shutil.copytree(EVIDENCE, evidence)
    profile_path = evidence / "TRAIN_LABEL_PROFILE.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    node = profile
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(EvidenceError, match="training label profile"):
        _validate(evidence)
