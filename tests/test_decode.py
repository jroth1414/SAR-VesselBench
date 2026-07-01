from __future__ import annotations

import numpy as np
import pytest

from src.eval.decode import DecodedPoint, decode_heatmap, distance_nms, find_peaks


def test_empty_heatmap_decodes_to_no_points():
    heatmap = np.zeros((4, 4), dtype=float)

    assert decode_heatmap(heatmap, threshold=0.5) == []


def test_plateau_handling_collapses_equal_peak_to_centroid():
    heatmap = np.zeros((5, 5), dtype=float)
    heatmap[1, 1] = 0.9
    heatmap[1, 2] = 0.9

    peaks = find_peaks(heatmap, threshold=0.5)

    assert len(peaks) == 1
    assert peaks[0].row == pytest.approx(1.0)
    assert peaks[0].col == pytest.approx(1.5)
    assert peaks[0].score == pytest.approx(0.9)


def test_non_maximum_plateau_is_suppressed():
    heatmap = np.zeros((5, 5), dtype=float)
    heatmap[1, 1] = 0.8
    heatmap[1, 2] = 0.8
    heatmap[2, 2] = 0.9

    peaks = find_peaks(heatmap, threshold=0.5)

    assert len(peaks) == 1
    assert peaks[0].row == pytest.approx(2.0)
    assert peaks[0].col == pytest.approx(2.0)


def test_distance_nms_suppresses_lower_score_neighbor():
    candidates = [
        DecodedPoint(row=0, col=0, score=0.5),
        DecodedPoint(row=3, col=4, score=0.9),
        DecodedPoint(row=100, col=100, score=0.4),
    ]

    kept = distance_nms(candidates, d_nms_m=10)

    assert kept == [
        DecodedPoint(row=3, col=4, score=0.9),
        DecodedPoint(row=100, col=100, score=0.4),
    ]


def test_decode_applies_output_stride_before_nms():
    heatmap = np.zeros((8, 8), dtype=float)
    heatmap[1, 1] = 0.8
    heatmap[3, 1] = 0.9

    points = decode_heatmap(
        heatmap,
        threshold=0.5,
        output_stride_m=10.0,
        d_nms_m=15.0,
    )

    assert points == [
        DecodedPoint(row=30.0, col=10.0, score=0.9),
        DecodedPoint(row=10.0, col=10.0, score=0.8),
    ]


def test_decode_rejects_non_2d_heatmaps():
    with pytest.raises(ValueError, match="2D"):
        decode_heatmap(np.zeros((1, 2, 3)), threshold=0.5)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_decode_rejects_non_finite_heatmaps(bad_value):
    heatmap = np.zeros((4, 4), dtype=float)
    heatmap[1, 1] = bad_value

    with pytest.raises(ValueError, match="finite"):
        decode_heatmap(heatmap, threshold=0.5)
