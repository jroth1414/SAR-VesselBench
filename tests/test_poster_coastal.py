"""Near-shore poster analysis: distance bins, the training-label binding and crop choice."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.analysis.heldout_results import EvidenceError
from src.analysis.poster_coastal import (
    BIN_LABELS,
    CROP_PX,
    choose_crops,
    coastal_numbers,
    distance_bin,
    training_bin_counts,
)
from src.analysis.poster_figures import macro_name
from src.eval.scorer import GroundTruthPoint, PredictionPoint, score_points


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(None, None), (0.0, 0), (0.99, 0), (1.0, 1), (2.0, 2), (4.99, 2), (5.0, 3), (10.0, 3), (10.01, 4),
     (9999.99, 4)],  # xView3 codes "beyond 10 km" as 9999.99
)
def test_distance_bins_follow_the_xview3_cap(distance, expected) -> None:
    assert distance_bin(distance) == expected


def _train_csv(path: Path) -> Path:
    path.write_bytes(
        b"scene_id,is_vessel,confidence,distance_from_shore_km\n"
        b"a,True,HIGH,0.5\n"
        b"a,True,MEDIUM,9999.99\n"
        b"a,True,LOW,0.5\n"  # ignored: low confidence
        b"a,False,HIGH,0.5\n"  # background
        b"b,True,HIGH,0.5\n"  # not a training scene
    )
    return path


def test_training_counts_need_the_audited_file(tmp_path: Path) -> None:
    csv_path = _train_csv(tmp_path / "train.csv")
    audit = {"inputs": {"train_csv": {"sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}}}
    assert training_bin_counts(csv_path, ["a"], audit).tolist() == [1, 0, 0, 0, 1]
    audit["inputs"]["train_csv"]["sha256"] = "0" * 64
    with pytest.raises(EvidenceError, match="audit"):
        training_bin_counts(csv_path, ["a"], audit)


def _scene(offset_px: float, near_shore: bool, found: bool):
    distance = 0.5 if near_shore else 9999.99
    gt = [GroundTruthPoint((offset_px + 50 * i) * 10.0, offset_px * 10.0, distance_from_shore_km=distance)
          for i in range(5)]
    predictions = [PredictionPoint(p.x_m, p.y_m, 0.9, distance) for p in gt] if found else []
    return gt, predictions, score_points(gt, predictions)


def test_crops_pick_a_found_offshore_window_and_a_missed_coastal_one() -> None:
    per_scene = {"open-sea": _scene(2000, near_shore=False, found=True),
                 "harbor": _scene(3000, near_shore=True, found=False)}
    crops = choose_crops(per_scene)
    assert [(kind, scene) for kind, scene, *_ in crops] == [("offshore", "open-sea"), ("near shore", "harbor")]
    for _kind, _scene_id, x0, y0 in crops:
        assert x0 >= 0 and y0 >= 0 and CROP_PX == 800


def test_coastal_numbers_render_as_macros() -> None:
    hit, total = np.array([1, 2, 3, 4, 5]), np.array([10, 10, 10, 10, 10])
    numbers = coastal_numbers(np.array([1, 0, 0, 0, 3]), total, {"bins": {"best": (hit, total)}}, "best",
                              [("offshore", "s1", 0, 0), ("near shore", "s2", 0, 0)])
    assert numbers["train_beyond_ten_km_pct"] == 75.0
    assert numbers["best_recall_zero_to_one_km"] == 0.1 and numbers["verified_vessels_beyond_ten_km"] == 10
    assert numbers["example_near_shore_scene"] == "s2"
    assert all(macro_name(key).isalpha() for key in numbers)
    assert len(BIN_LABELS) == 5
