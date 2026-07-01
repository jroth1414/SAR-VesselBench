from __future__ import annotations

from src.eval.scorer import GroundTruthPoint, PredictionPoint
from src.eval.threshold import apply_threshold, select_f1_threshold


def test_select_f1_threshold_uses_dev_scores_only():
    threshold = select_f1_threshold(
        {"scene": [GroundTruthPoint(0, 0)]},
        {
            "scene": [
                PredictionPoint(0, 0, 0.9),
                PredictionPoint(500, 0, 0.2, distance_from_shore_km=5.0),
            ]
        },
    )

    assert threshold == 0.9


def test_select_f1_threshold_tie_breaks_to_highest_threshold():
    threshold = select_f1_threshold(
        {
            "scene": [
                GroundTruthPoint(0, 0),
                GroundTruthPoint(500, 0, confidence="LOW"),
            ]
        },
        {
            "scene": [
                PredictionPoint(0, 0, 0.9),
                PredictionPoint(500, 0, 0.8),
            ]
        },
    )

    assert threshold == 0.9


def test_apply_threshold_filters_with_frozen_threshold_without_reselecting():
    filtered = apply_threshold(
        {
            "scene": [
                PredictionPoint(0, 0, 0.4),
                PredictionPoint(100, 0, 0.9),
            ]
        },
        0.5,
    )

    assert filtered == {"scene": [PredictionPoint(100, 0, 0.9)]}
