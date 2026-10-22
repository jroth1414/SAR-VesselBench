"""Paired scene bootstrap: deterministic, paired, and centred on the point estimate."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.scene_bootstrap import PairedBootstrap, micro_f1, scene_counts


def _per_scene(rows: dict[str, tuple[int, int, int]]) -> dict:
    return {s: {"aggregate": {"tp": tp, "fp": fp, "fn": fn}} for s, (tp, fp, fn) in rows.items()}


def test_identical_cells_differ_by_exactly_zero() -> None:
    scenes, counts = scene_counts(_per_scene({"a": (5, 1, 2), "b": (9, 3, 0), "c": (1, 0, 4)}))
    interval = PairedBootstrap(scenes, n_resamples=500, seed=0).difference(counts, counts)
    assert interval == {"point": 0.0, "low": 0.0, "high": 0.0}


def test_a_cell_better_on_every_scene_has_an_interval_above_zero() -> None:
    _, worse = scene_counts(_per_scene({s: (5, 5, 5) for s in "abcdef"}))
    scenes, better = scene_counts(_per_scene({s: (9, 1, 1) for s in "abcdef"}))
    interval = PairedBootstrap(scenes, n_resamples=500, seed=0).difference(better, worse)
    assert 0.0 < interval["low"] <= interval["point"] <= interval["high"]
    assert interval["point"] == pytest.approx(micro_f1(better.sum(0)) - micro_f1(worse.sum(0)))


def test_resampling_is_seeded_and_shared_across_cells() -> None:
    scenes, a = scene_counts(_per_scene({"a": (5, 1, 2), "b": (2, 3, 6), "c": (7, 0, 1)}))
    _, b = scene_counts(_per_scene({"a": (4, 2, 3), "b": (3, 1, 5), "c": (6, 1, 2)}))
    first = PairedBootstrap(scenes, n_resamples=300, seed=7).difference(a, b)
    second = PairedBootstrap(scenes, n_resamples=300, seed=7).difference(a, b)
    assert first == second


def test_scene_lists_must_match() -> None:
    scenes, _ = scene_counts(_per_scene({"a": (1, 0, 0), "b": (1, 0, 0)}))
    with pytest.raises(ValueError, match="scene"):
        PairedBootstrap(scenes, n_resamples=10).check(["a", "c"])
    assert isinstance(micro_f1(np.array([2.0, 1.0, 1.0])), float)
