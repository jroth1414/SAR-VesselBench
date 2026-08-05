"""LocateAnything corrected near-shore scoring regressions."""

from __future__ import annotations

import pytest

from src.eval.scorer import GroundTruthPoint, score_points
from src.references.locateanything_zs import (
    SceneShoreDistanceCache,
    chip_scene_metadata,
)


class _OffsetTransform:
    def __mul__(self, pixel):
        col, row = pixel
        return col + 1_000.0, row + 2_000.0


class _FakeShore:
    available = True

    def __init__(self, distance_km: float) -> None:
        self.distance_km = distance_km
        self.lookups = []

    def lookup_km(self, x_geo: float, y_geo: float) -> float:
        self.lookups.append((x_geo, y_geo))
        return self.distance_km


def _sidecar() -> dict[str, object]:
    return {
        "scene_id": "scene-a",
        "row0": 200,
        "col0": 100,
        "chip_size": 800,
        "gsd_m": 10.0,
    }


def test_unmatched_canonical_near_shore_prediction_adds_false_positive(
    tmp_path, monkeypatch
):
    shore = _FakeShore(distance_km=1.0)
    cache = SceneShoreDistanceCache(tmp_path)
    monkeypatch.setattr(
        cache,
        "_load_scene",
        lambda scene_id: (_OffsetTransform(), shore),
    )

    predictions = cache.predictions_for_centers(
        [(50.0, 60.0)],
        sidecar=_sidecar(),
        expected_scene_id="scene-a",
    )

    assert [(point.x_m, point.y_m) for point in predictions] == [(500.0, 600.0)]
    assert shore.lookups == [(1_150.0, 2_260.0)]
    scored = score_points(
        [GroundTruthPoint(0.0, 0.0, distance_from_shore_km=5.0)],
        predictions,
        tolerance_m=200.0,
    )
    assert scored.aggregate.fp == 1
    assert scored.slices["near_shore"].fp == 1


@pytest.mark.parametrize(
    "mutation",
    [
        {"scene_id": "../scene-a"},
        {"row0": True},
        {"row0": -1},
        {"col0": 1.5},
        {"chip_size": True},
        {"chip_size": 0},
        {"chip_size": 799},
        {"gsd_m": float("nan")},
        {"gsd_m": 20.0},
    ],
)
def test_malformed_chip_metadata_fails_closed(mutation):
    sidecar = _sidecar()
    sidecar.update(mutation)

    with pytest.raises(RuntimeError, match="R3 chip sidecar"):
        chip_scene_metadata(sidecar, expected_scene_id="scene-a")


@pytest.mark.parametrize("field", ["scene_id", "row0", "col0", "chip_size", "gsd_m"])
def test_missing_chip_metadata_fails_closed(field):
    sidecar = _sidecar()
    sidecar.pop(field)

    with pytest.raises(RuntimeError, match=field):
        chip_scene_metadata(sidecar, expected_scene_id="scene-a")


def test_chip_scene_mismatch_fails_closed():
    with pytest.raises(RuntimeError, match="does not match"):
        chip_scene_metadata(_sidecar(), expected_scene_id="scene-b")
