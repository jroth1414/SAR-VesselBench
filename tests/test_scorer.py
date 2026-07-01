from __future__ import annotations

import pytest

from src.eval.scorer import GroundTruthPoint, PredictionPoint, score_dataset, score_points


def test_appendix_a_worked_example():
    gt = [GroundTruthPoint(0, 0), GroundTruthPoint(500, 0)]
    pred = [
        PredictionPoint(10, 0, 0.9),
        PredictionPoint(180, 0, 0.8, distance_from_shore_km=3.0),
        PredictionPoint(900, 0, 0.7, distance_from_shore_km=3.0),
    ]

    result = score_points(gt, pred)

    assert result.aggregate.tp == 1
    assert result.aggregate.fp == 2
    assert result.aggregate.fn == 1
    assert result.aggregate.precision == pytest.approx(1 / 3)
    assert result.aggregate.recall == pytest.approx(1 / 2)
    assert result.aggregate.f1 == pytest.approx(0.4)


def test_exact_hit_scores_true_positive():
    result = score_points(
        [GroundTruthPoint(10, 20)],
        [PredictionPoint(10, 20, 1.0)],
    )

    assert result.aggregate.tp == 1
    assert result.aggregate.fp == 0
    assert result.aggregate.fn == 0
    assert result.aggregate.f1 == pytest.approx(1.0)


def test_hit_at_199_m_is_true_positive():
    result = score_points(
        [GroundTruthPoint(0, 0)],
        [PredictionPoint(199, 0, 0.5)],
    )

    assert result.aggregate.tp == 1
    assert result.aggregate.fp == 0
    assert result.aggregate.fn == 0


def test_miss_at_201_m_is_false_positive_and_false_negative():
    result = score_points(
        [GroundTruthPoint(0, 0)],
        [PredictionPoint(201, 0, 0.5, distance_from_shore_km=4.0)],
    )

    assert result.aggregate.tp == 0
    assert result.aggregate.fp == 1
    assert result.aggregate.fn == 1


def test_two_predictions_on_one_gt_make_one_tp_one_fp():
    result = score_points(
        [GroundTruthPoint(0, 0)],
        [
            PredictionPoint(10, 0, 0.9),
            PredictionPoint(20, 0, 0.8, distance_from_shore_km=4.0),
        ],
    )

    assert result.aggregate.tp == 1
    assert result.aggregate.fp == 1
    assert result.aggregate.fn == 0


def test_score_order_priority_beats_nearest_lower_score_prediction():
    result = score_points(
        [GroundTruthPoint(0, 0)],
        [
            PredictionPoint(10, 0, 0.1, distance_from_shore_km=4.0),
            PredictionPoint(150, 0, 0.9),
        ],
    )

    assert result.matches[0].prediction_index == 1
    assert result.matches[0].outcome == "tp"
    assert result.matches[1].prediction_index == 0
    assert result.matches[1].outcome == "fp"


def test_low_confidence_gt_ignores_nearby_prediction():
    result = score_points(
        [GroundTruthPoint(0, 0, confidence="LOW")],
        [PredictionPoint(10, 0, 0.9)],
    )

    assert result.aggregate.tp == 0
    assert result.aggregate.fp == 0
    assert result.aggregate.fn == 0
    assert result.aggregate.ignored_predictions == 1
    assert result.aggregate.f1 == pytest.approx(0.0)


def test_low_confidence_gt_does_not_hide_real_positive_match():
    result = score_points(
        [
            GroundTruthPoint(0, 0, confidence="LOW"),
            GroundTruthPoint(100, 0, confidence="HIGH"),
        ],
        [PredictionPoint(90, 0, 0.9)],
    )

    assert result.aggregate.tp == 1
    assert result.aggregate.fp == 0
    assert result.aggregate.fn == 0
    assert result.aggregate.ignored_predictions == 0


def test_dark_and_near_shore_slice_math():
    gt = [
        GroundTruthPoint(0, 0, source="Manual", distance_from_shore_km=1.5),
        GroundTruthPoint(500, 0, source="AIS", distance_from_shore_km=5.0),
        GroundTruthPoint(1000, 0, source="Manual", distance_from_shore_km=3.0),
    ]
    pred = [
        PredictionPoint(10, 0, 0.9),
        PredictionPoint(500, 0, 0.8),
    ]

    result = score_points(gt, pred)

    assert result.aggregate.tp == 2
    assert result.aggregate.fn == 1
    assert result.slices["dark"].tp == 1
    assert result.slices["dark"].fn == 1
    assert result.slices["dark"].recall == pytest.approx(0.5)
    assert result.slices["near_shore"].tp == 1
    assert result.slices["near_shore"].fn == 0
    assert result.slices["near_shore"].f1 == pytest.approx(1.0)


def test_near_shore_unmatched_false_positive_counts_against_slice_f1():
    gt = [GroundTruthPoint(0, 0, distance_from_shore_km=1.0)]
    pred = [
        PredictionPoint(0, 0, 0.9),
        PredictionPoint(500, 0, 0.8, distance_from_shore_km=1.0),
    ]

    result = score_points(gt, pred)

    assert result.slices["near_shore"].tp == 1
    assert result.slices["near_shore"].fp == 1
    assert result.slices["near_shore"].fn == 0
    assert result.slices["near_shore"].f1 == pytest.approx(2 / 3)


def test_missing_prediction_shore_distance_raises_for_unmatched_false_positive():
    gt = [GroundTruthPoint(0, 0, distance_from_shore_km=1.0)]
    pred = [PredictionPoint(500, 0, 0.8)]

    with pytest.raises(ValueError, match="distance_from_shore_km"):
        score_points(gt, pred)


def test_score_dataset_does_not_match_points_across_scenes():
    result = score_dataset(
        {"scene_with_gt": [GroundTruthPoint(0, 0)]},
        {
            "different_scene": [
                PredictionPoint(0, 0, 0.9, distance_from_shore_km=5.0)
            ]
        },
    )

    assert result.aggregate.tp == 0
    assert result.aggregate.fp == 1
    assert result.aggregate.fn == 1


def test_score_dataset_counts_missing_prediction_scene_as_false_negative():
    result = score_dataset({"scene": [GroundTruthPoint(0, 0)]}, {})

    assert result.aggregate.tp == 0
    assert result.aggregate.fp == 0
    assert result.aggregate.fn == 1


def test_score_dataset_counts_prediction_only_scene_as_false_positive():
    result = score_dataset(
        {},
        {"scene": [PredictionPoint(0, 0, 0.9, distance_from_shore_km=5.0)]},
    )

    assert result.aggregate.tp == 0
    assert result.aggregate.fp == 1
    assert result.aggregate.fn == 0
