"""Poster v2 visuals: scoreboard tallies, reference checks, window choice, map positions, heatmap tiling."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.heldout_results import EvidenceError
from src.analysis.poster_extras import (
    _land_polygons,
    choose_dark_window,
    reference_numbers,
    scene_positions,
    scoreboard,
)

REPO = Path(__file__).resolve().parents[1]


def _intervals(sign_by_key: dict[str, int]) -> dict[str, dict[str, float]]:
    out = {}
    for metric in ("test", "final"):
        for track in ("vit", "cnn"):
            for fraction in (10, 25, 50, 100):
                key = f"{metric}_{track}_sar_minus_opt_{fraction}"
                sign = sign_by_key.get(key, 0)
                low, high = {1: (0.01, 0.05), -1: (-0.05, -0.01), 0: (-0.01, 0.01)}[sign]
                out[key] = {"point": (low + high) / 2, "low": low, "high": high}
    return out


def test_scoreboard_marks_a_win_only_when_the_interval_clears_zero() -> None:
    grid = scoreboard(_intervals({"test_cnn_sar_minus_opt_10": 1, "final_vit_sar_minus_opt_25": -1}), "opt")
    assert grid.shape == (4, 4)
    assert grid[0, 2] == 1  # 12 scenes, CNN test
    assert grid[1, 1] == -1  # 28 scenes, ViT verified
    assert int(np.abs(grid).sum()) == 2


def test_scoreboard_matches_the_answer_strip_on_the_evidence() -> None:
    from src.analysis.heldout_results import validate_evidence
    from src.analysis.poster_figures import scene_details

    evidence = REPO / "results/h100/evidence"
    validated = validate_evidence(evidence, arms_config=REPO / "configs/arms.yaml",
                                  detector_config=REPO / "configs/detector.yaml",
                                  splits_config=REPO / "data/splits.json")
    intervals = scene_details(evidence, validated)["intervals"]
    optical, imagenet = scoreboard(intervals, "opt"), scoreboard(intervals, "imagenet")
    assert ((optical > 0).sum(), (optical < 0).sum()) == (6, 2)
    assert ((imagenet > 0).sum(), (imagenet < 0).sum()) == (3, 7)


def test_reference_numbers_come_from_the_checksummed_files() -> None:
    numbers = reference_numbers(REPO / "results/h100/references", expected_test_positives=1165)
    assert round(numbers["yolo_test_f1"], 3) == 0.896
    assert round(numbers["zero_shot_f1"], 3) == 0.122 and numbers["zero_shot_prompt"] == "boat"


def test_reference_numbers_reject_an_edited_file(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "references"
    shutil.copytree(REPO / "results/h100/references", root)
    path = root / "yolo26-f100/final_metrics.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(EvidenceError, match="SHA256SUMS"):
        reference_numbers(root, expected_test_positives=1165)


def _row(scene: str, x: int, y: int, source: str, distance: str = "9999.99", confidence: str = "HIGH") -> dict:
    return {"scene_id": scene, "detect_scene_column": str(x), "detect_scene_row": str(y), "source": source,
            "is_vessel": "True", "confidence": confidence, "distance_from_shore_km": distance}


def test_dark_window_needs_both_kinds_of_vessel_offshore() -> None:
    rows = {
        "near": [_row("near", 100 + i * 20, 100, s, distance="1.0") for i, s in enumerate(["ais"] * 3 + ["manual"] * 3)],
        "far": [_row("far", 500 + i * 30, 400, s) for i, s in enumerate(["ais"] * 3 + ["manual"] * 4)],
    }
    scene, x0, y0 = choose_dark_window(rows, width=400, height=200)
    assert scene == "far" and x0 <= 500 and y0 <= 400
    with pytest.raises(EvidenceError):
        choose_dark_window({"near": rows["near"]}, width=400, height=200)


def test_scene_positions_average_label_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scene_id", "detect_lat", "detect_lon"])
        writer.writeheader()
        writer.writerows([{"scene_id": "a", "detect_lat": "10", "detect_lon": "2"},
                          {"scene_id": "a", "detect_lat": "12", "detect_lon": "4"},
                          {"scene_id": "b", "detect_lat": "0", "detect_lon": "0"}])
    assert scene_positions([path], {"a"}) == {"a": (3.0, 11.0)}


def test_land_polygons_must_be_the_pinned_file(tmp_path: Path) -> None:
    path = tmp_path / "land.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    with pytest.raises(EvidenceError, match="pinned"):
        list(_land_polygons(path))


def test_window_heatmap_reproduces_the_scene_tiling(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    torch = pytest.importorskip("torch")
    from src.analysis.poster_heatmaps import window_heatmap

    cache = tmp_path / "cache"
    cache.mkdir()
    for band in ("VH", "VV"):
        with rasterio.open(cache / f"s_{band}_dB.tif", "w", driver="GTiff", width=1200, height=1000, count=1,
                           dtype="float32", nodata=-32768.0) as dataset:
            dataset.write(np.full((1000, 1200), -20.0, dtype="float32"), 1)

    class Constant(torch.nn.Module):  # logit 0 everywhere -> sigmoid 0.5
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, x.shape[2] // 4, x.shape[3] // 4)

    stats = json.loads((REPO / "data/stats.json").read_text(encoding="utf-8"))
    heat = window_heatmap(Constant(), tmp_path, cache, "s", 300, 100, 400, stats, "cpu")
    assert heat.shape == (100, 100)  # an 800 px window maps to 200 stride-4 cells; 400 px to 100
    assert np.allclose(heat, 0.5)
