"""Heatmap peak decoding and distance NMS."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class DecodedPoint:
    row: float
    col: float
    score: float


def decode_heatmap(
    heatmap: np.ndarray,
    *,
    threshold: float,
    output_stride_m: float = 1.0,
    d_nms_m: float = 120.0,
) -> list[DecodedPoint]:
    """Decode a 2D heatmap into peak points after distance NMS."""

    array = np.asarray(heatmap, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {array.shape}")
    _validate_finite(array)

    peaks = find_peaks(array, threshold=threshold)
    candidates = [
        DecodedPoint(
            row=peak.row * output_stride_m,
            col=peak.col * output_stride_m,
            score=peak.score,
        )
        for peak in peaks
    ]
    return distance_nms(candidates, d_nms_m=d_nms_m)


def find_peaks(heatmap: np.ndarray, *, threshold: float) -> list[DecodedPoint]:
    """Find local maxima, collapsing equal-valued plateaus to one centroid."""

    array = np.asarray(heatmap, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"heatmap must be 2D, got shape {array.shape}")
    _validate_finite(array)
    if array.size == 0:
        return []

    visited = np.zeros(array.shape, dtype=bool)
    peaks: list[DecodedPoint] = []

    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            if visited[row, col] or array[row, col] < threshold:
                continue
            plateau = _collect_plateau(array, row, col, visited)
            value = array[row, col]
            if _is_local_maximum(array, plateau, value):
                rows = np.array([cell[0] for cell in plateau], dtype=float)
                cols = np.array([cell[1] for cell in plateau], dtype=float)
                peaks.append(
                    DecodedPoint(
                        row=float(rows.mean()),
                        col=float(cols.mean()),
                        score=float(value),
                    )
                )

    return sorted(peaks, key=lambda point: point.score, reverse=True)


def distance_nms(
    candidates: Iterable[DecodedPoint], *, d_nms_m: float
) -> list[DecodedPoint]:
    """Greedy score-ordered distance NMS using scipy.spatial.cKDTree."""

    ordered = sorted(candidates, key=lambda point: point.score, reverse=True)
    if not ordered:
        return []

    coords = np.array([[point.row, point.col] for point in ordered], dtype=float)
    tree = cKDTree(coords)
    suppressed = np.zeros(len(ordered), dtype=bool)
    kept: list[DecodedPoint] = []

    for index, point in enumerate(ordered):
        if suppressed[index]:
            continue
        kept.append(point)
        for neighbor in tree.query_ball_point(coords[index], r=d_nms_m):
            if neighbor != index:
                suppressed[neighbor] = True

    return kept


def _collect_plateau(
    array: np.ndarray, start_row: int, start_col: int, visited: np.ndarray
) -> list[tuple[int, int]]:
    value = array[start_row, start_col]
    queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
    visited[start_row, start_col] = True
    plateau: list[tuple[int, int]] = []

    while queue:
        row, col = queue.popleft()
        plateau.append((row, col))
        for n_row, n_col in _neighbors8(row, col, array.shape):
            if not visited[n_row, n_col] and array[n_row, n_col] == value:
                visited[n_row, n_col] = True
                queue.append((n_row, n_col))

    return plateau


def _is_local_maximum(
    array: np.ndarray, plateau: Sequence[tuple[int, int]], value: float
) -> bool:
    plateau_set = set(plateau)
    for row, col in plateau:
        for n_row, n_col in _neighbors8(row, col, array.shape):
            if (n_row, n_col) not in plateau_set and array[n_row, n_col] > value:
                return False
    return True


def _neighbors8(
    row: int, col: int, shape: tuple[int, int]
) -> Iterable[tuple[int, int]]:
    for d_row in (-1, 0, 1):
        for d_col in (-1, 0, 1):
            if d_row == 0 and d_col == 0:
                continue
            n_row = row + d_row
            n_col = col + d_col
            if 0 <= n_row < shape[0] and 0 <= n_col < shape[1]:
                yield n_row, n_col


def _validate_finite(array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise ValueError("heatmap must contain only finite values")
