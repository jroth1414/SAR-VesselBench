"""Paired scene bootstrap for micro-F1 and micro-F1 differences between cells.

Each resample draws evaluation scenes with replacement. Every cell uses the
same draw, so a difference between two cells is paired. Micro-F1 is
recomputed from the summed TP/FP/FN of the drawn scenes, the way the frozen
scorer aggregates. The intervals describe scene-sampling uncertainty only;
they do not include training-seed variation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

COUNT_KEYS = ("tp", "fp", "fn")


def scene_counts(per_scene: Mapping[str, Mapping[str, object]]) -> tuple[list[str], np.ndarray]:
    """Sorted scene IDs and a (scenes, 3) array of aggregate TP/FP/FN."""

    scenes = sorted(per_scene)
    rows = [[float(per_scene[s]["aggregate"][k]) for k in COUNT_KEYS] for s in scenes]  # type: ignore[index]
    return scenes, np.asarray(rows, dtype=float)


def micro_f1(counts: np.ndarray) -> np.ndarray | float:
    """Micro-F1 from (..., 3) TP/FP/FN; zero where there is nothing to score."""

    tp, fp, fn = counts[..., 0], counts[..., 1], counts[..., 2]
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0)
    return float(f1) if np.ndim(f1) == 0 else f1


class PairedBootstrap:
    def __init__(self, scenes: Sequence[str], *, n_resamples: int = 10_000, seed: int = 0) -> None:
        self.scenes = list(scenes)
        rng = np.random.default_rng(seed)
        self.index = rng.integers(0, len(self.scenes), size=(n_resamples, len(self.scenes)))

    def check(self, scenes: Sequence[str]) -> None:
        if list(scenes) != self.scenes:
            raise ValueError("cells were scored on different scene lists")

    def f1(self, counts: np.ndarray) -> np.ndarray:
        return micro_f1(counts[self.index].sum(axis=1))  # type: ignore[return-value]

    def difference(self, a: np.ndarray, b: np.ndarray, *, level: float = 0.95) -> dict[str, float]:
        """Point estimate and percentile interval of F1(a) - F1(b)."""

        draws = self.f1(a) - self.f1(b)
        tail = 100.0 * (1.0 - level) / 2.0
        return {
            "point": float(micro_f1(a.sum(axis=0)) - micro_f1(b.sum(axis=0))),
            "low": float(np.percentile(draws, tail)),
            "high": float(np.percentile(draws, 100.0 - tail)),
        }
