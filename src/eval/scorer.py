"""Distance-matched point scorer for the xView3 study.

This module is intentionally small: every reported study number flows through
this scorer, and DEVPLAN Phase 2 freezes it before model work begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GroundTruthPoint:
    """A vessel annotation in scene/world coordinates measured in meters."""

    x_m: float
    y_m: float
    confidence: str = "HIGH"
    source: str | None = None
    distance_from_shore_km: float | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence.upper() == "LOW"

    @property
    def is_dark(self) -> bool:
        return self.source == "Manual"

    @property
    def is_near_shore(self) -> bool:
        return (
            self.distance_from_shore_km is not None
            and self.distance_from_shore_km <= 2.0
        )


@dataclass(frozen=True)
class PredictionPoint:
    """A point prediction in scene/world coordinates measured in meters."""

    x_m: float
    y_m: float
    score: float
    distance_from_shore_km: float | None = None


@dataclass(frozen=True)
class Match:
    prediction_index: int
    ground_truth_index: int | None
    distance_m: float | None
    outcome: str


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    fn: int
    ignored_predictions: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2.0 * self.precision * self.recall / denom if denom else 0.0


@dataclass(frozen=True)
class ScoreResult:
    aggregate: Metrics
    slices: Mapping[str, Metrics]
    matches: Sequence[Match]


@dataclass(frozen=True)
class DatasetScoreResult:
    aggregate: Metrics
    slices: Mapping[str, Metrics]
    scene_results: Mapping[str, ScoreResult]


def score_points(
    ground_truth: Iterable[GroundTruthPoint],
    predictions: Iterable[PredictionPoint],
    *,
    tolerance_m: float = 200.0,
    low_conf_ignore: bool = True,
) -> ScoreResult:
    """Score predictions with greedy confidence-ordered distance matching.

    HIGH/MEDIUM GT are positives. LOW GT are ignored by default: they do not
    count as false negatives, and a prediction within tolerance of a LOW GT is
    ignored rather than counted as a true positive or false positive.
    """

    gt = list(ground_truth)
    pred = list(predictions)
    positive_indices = [
        index for index, point in enumerate(gt) if not point.is_low_confidence
    ]
    low_indices = [index for index, point in enumerate(gt) if point.is_low_confidence]

    unmatched_positive = set(positive_indices)
    matches: list[Match] = []

    sorted_predictions = sorted(
        enumerate(pred), key=lambda item: item[1].score, reverse=True
    )

    for pred_index, prediction in sorted_predictions:
        nearest_positive = _nearest_index(
            prediction, gt, unmatched_positive, tolerance_m=tolerance_m
        )
        if nearest_positive is not None:
            gt_index, distance = nearest_positive
            unmatched_positive.remove(gt_index)
            matches.append(Match(pred_index, gt_index, distance, "tp"))
            continue

        nearest_low = (
            _nearest_index(prediction, gt, low_indices, tolerance_m=tolerance_m)
            if low_conf_ignore
            else None
        )
        if nearest_low is not None:
            gt_index, distance = nearest_low
            matches.append(Match(pred_index, gt_index, distance, "ignored"))
            continue

        matches.append(Match(pred_index, None, None, "fp"))

    aggregate = Metrics(
        tp=sum(match.outcome == "tp" for match in matches),
        fp=sum(match.outcome == "fp" for match in matches),
        fn=len(unmatched_positive),
        ignored_predictions=sum(match.outcome == "ignored" for match in matches),
    )

    slices = {
        "dark": _slice_metrics(
            gt, pred, matches, positive_indices, lambda point: point.is_dark
        ),
        "near_shore": _slice_metrics(
            gt,
            pred,
            matches,
            positive_indices,
            lambda point: point.is_near_shore,
            prediction_predicate=_prediction_is_near_shore,
        ),
    }
    return ScoreResult(aggregate=aggregate, slices=slices, matches=tuple(matches))


def score_dataset(
    ground_truth_by_scene: Mapping[str, Iterable[GroundTruthPoint]],
    predictions_by_scene: Mapping[str, Iterable[PredictionPoint]],
    *,
    tolerance_m: float = 200.0,
    low_conf_ignore: bool = True,
) -> DatasetScoreResult:
    """Score every scene independently, then aggregate metrics.

    Matching across scenes is invalid even when coordinates coincide, so callers
    should use this wrapper for split-level metrics.
    """

    scene_ids = sorted(set(ground_truth_by_scene) | set(predictions_by_scene))
    scene_results = {
        scene_id: score_points(
            ground_truth_by_scene.get(scene_id, ()),
            predictions_by_scene.get(scene_id, ()),
            tolerance_m=tolerance_m,
            low_conf_ignore=low_conf_ignore,
        )
        for scene_id in scene_ids
    }
    return DatasetScoreResult(
        aggregate=_sum_metrics(result.aggregate for result in scene_results.values()),
        slices={
            "dark": _sum_metrics(result.slices["dark"] for result in scene_results.values()),
            "near_shore": _sum_metrics(
                result.slices["near_shore"] for result in scene_results.values()
            ),
        },
        scene_results=scene_results,
    )


def _nearest_index(
    prediction: PredictionPoint,
    ground_truth: Sequence[GroundTruthPoint],
    candidate_indices: Iterable[int],
    *,
    tolerance_m: float,
) -> tuple[int, float] | None:
    best: tuple[int, float] | None = None
    for gt_index in candidate_indices:
        gt_point = ground_truth[gt_index]
        distance = hypot(prediction.x_m - gt_point.x_m, prediction.y_m - gt_point.y_m)
        if distance <= tolerance_m and (best is None or distance < best[1]):
            best = (gt_index, distance)
    return best


def _slice_metrics(
    ground_truth: Sequence[GroundTruthPoint],
    predictions: Sequence[PredictionPoint],
    matches: Sequence[Match],
    positive_indices: Sequence[int],
    predicate: Callable[[GroundTruthPoint], bool],
    *,
    prediction_predicate: Callable[[PredictionPoint], bool] | None = None,
) -> Metrics:
    slice_gt = {
        gt_index for gt_index in positive_indices if predicate(ground_truth[gt_index])
    }
    matched_slice_gt = {
        match.ground_truth_index
        for match in matches
        if match.outcome == "tp" and match.ground_truth_index in slice_gt
    }
    tp = len(matched_slice_gt)
    fn = len(slice_gt - matched_slice_gt)
    fp = (
        sum(
            prediction_predicate(predictions[match.prediction_index])
            for match in matches
            if match.outcome == "fp"
        )
        if prediction_predicate is not None
        else 0
    )

    return Metrics(tp=tp, fp=fp, fn=fn)


def _prediction_is_near_shore(prediction: PredictionPoint) -> bool:
    if prediction.distance_from_shore_km is None:
        raise ValueError(
            "PredictionPoint.distance_from_shore_km is required for unmatched "
            "false positives when computing the near_shore slice."
        )
    return prediction.distance_from_shore_km <= 2.0


def _sum_metrics(metrics: Iterable[Metrics]) -> Metrics:
    items = list(metrics)
    return Metrics(
        tp=sum(item.tp for item in items),
        fp=sum(item.fp for item in items),
        fn=sum(item.fn for item in items),
        ignored_predictions=sum(item.ignored_predictions for item in items),
    )
