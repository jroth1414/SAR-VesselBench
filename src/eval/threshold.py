"""Dev-set operating point selection for scored point predictions."""

from __future__ import annotations

from typing import Iterable, Mapping

from src.eval.scorer import GroundTruthPoint, PredictionPoint, score_dataset


def select_f1_threshold(
    dev_ground_truth_by_scene: Mapping[str, Iterable[GroundTruthPoint]],
    dev_predictions_by_scene: Mapping[str, Iterable[PredictionPoint]],
) -> float:
    """Select the F1-maximizing threshold on dev predictions only.

    Ties are resolved toward the highest threshold to prefer the more
    conservative operating point.
    """

    scores = sorted(
        {
            prediction.score
            for predictions in dev_predictions_by_scene.values()
            for prediction in predictions
        },
        reverse=True,
    )
    if not scores:
        return 1.0

    best_threshold = scores[0]
    best_f1 = -1.0
    for threshold in scores:
        filtered = apply_threshold(dev_predictions_by_scene, threshold)
        f1 = score_dataset(dev_ground_truth_by_scene, filtered).aggregate.f1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def apply_threshold(
    predictions_by_scene: Mapping[str, Iterable[PredictionPoint]],
    threshold: float,
) -> dict[str, list[PredictionPoint]]:
    """Apply a frozen threshold without inspecting ground truth."""

    return {
        scene_id: [
            prediction for prediction in predictions if prediction.score >= threshold
        ]
        for scene_id, predictions in predictions_by_scene.items()
    }
