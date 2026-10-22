from __future__ import annotations

import numpy as np
import pytest

from src.eval.ground_truth import (
    classify_label,
    ground_truth_from_labels,
    normalize_confidence,
    parse_vessel_boolean,
)
from src.eval.infer_scene import ground_truth_from_labels as infer_scene_converter
from src.eval.scorer import PredictionPoint, score_points


def row(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "detect_scene_column": 12,
        "detect_scene_row": 34,
        "confidence": "HIGH",
        "is_vessel": True,
        "source": "ais",
        "distance_from_shore_km": 3.5,
    }
    result.update(overrides)
    return result


@pytest.mark.parametrize("confidence", ["HIGH", " high ", "medium", " MeDiUm "])
@pytest.mark.parametrize("is_vessel", [True, np.bool_(True), "true", " TRUE "])
def test_explicit_high_medium_vessels_are_positive(confidence, is_vessel):
    label = row(confidence=confidence, is_vessel=is_vessel)
    assert classify_label(label) == "positive"
    points = ground_truth_from_labels([label])
    assert len(points) == 1
    assert points[0].confidence == confidence.strip().upper()


@pytest.mark.parametrize("confidence", ["HIGH", "medium"])
@pytest.mark.parametrize("is_vessel", [False, np.bool_(False), "false", " FALSE "])
def test_explicit_high_medium_non_vessels_are_background(confidence, is_vessel):
    label = row(confidence=confidence, is_vessel=is_vessel)
    assert classify_label(label) == "background"
    assert ground_truth_from_labels([label]) == []


@pytest.mark.parametrize("is_vessel", [None, float("nan"), True, False, 1, "yes"])
def test_low_is_always_retained_as_ignore_without_parsing_vessel(is_vessel):
    label = row(confidence=" low ", is_vessel=is_vessel)
    assert classify_label(label) == "ignore"
    point = ground_truth_from_labels([label])[0]
    assert point.confidence == "LOW"
    result = score_points([point], [PredictionPoint(120, 340, 0.9)])
    assert result.aggregate.ignored_predictions == 1
    assert result.aggregate.tp == result.aggregate.fp == result.aggregate.fn == 0


@pytest.mark.parametrize("value", [None, float("nan"), "", "unknown", 1])
def test_missing_or_unknown_confidence_is_rejected(value):
    with pytest.raises(ValueError, match="confidence"):
        normalize_confidence(value)
    with pytest.raises(ValueError, match="invalid evaluation label at row 0"):
        ground_truth_from_labels([row(confidence=value)])


@pytest.mark.parametrize(
    "value",
    [None, float("nan"), 0, 1, np.int64(1), "", "yes", "no", "unknown"],
)
def test_high_medium_missing_or_noncanonical_vessel_is_rejected(value):
    with pytest.raises(ValueError, match="is_vessel"):
        parse_vessel_boolean(value)
    with pytest.raises(ValueError, match="is_vessel"):
        ground_truth_from_labels([row(is_vessel=value)])


def test_source_and_shore_survive_conversion_with_nan_becoming_none():
    points = ground_truth_from_labels(
        [
            row(source="manual", distance_from_shore_km="1.25"),
            row(
                confidence="LOW",
                is_vessel=None,
                source=float("nan"),
                distance_from_shore_km=float("nan"),
            ),
        ]
    )
    assert points[0].source == "Manual"
    assert points[0].distance_from_shore_km == pytest.approx(1.25)
    assert points[1].source is None
    assert points[1].distance_from_shore_km is None


def test_infer_scene_reexports_the_single_shared_converter():
    assert infer_scene_converter is ground_truth_from_labels


def test_background_rows_never_reach_the_frozen_scorer_as_positives():
    points = ground_truth_from_labels(
        [
            row(detect_scene_column=0, detect_scene_row=0, is_vessel=True),
            row(detect_scene_column=50, detect_scene_row=0, is_vessel=False),
        ]
    )
    result = score_points(points, [PredictionPoint(500, 0, 0.9, 3.0)])
    assert result.aggregate.tp == 0
    assert result.aggregate.fp == 1
    assert result.aggregate.fn == 1
