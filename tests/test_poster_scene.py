"""Scene-and-chip poster figure: reads a VH raster, caches its overview, writes both formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from src.analysis.poster_scene import CHIP_PX, DECIMATION, figure_scene_context, read_context  # noqa: E402


def _scene(tmp_path: Path) -> Path:
    scene = tmp_path / "scene"
    scene.mkdir()
    data = np.full((1200, 1600), -20.0, dtype="float32")
    data[:100, :] = -32768.0  # outside the swath
    with rasterio.open(scene / "VH_dB.tif", "w", driver="GTiff", width=1600, height=1200, count=1,
                       dtype="float32", nodata=-32768.0) as dataset:
        dataset.write(data, 1)
    return scene


def test_context_reads_the_chip_and_caches_the_overview(tmp_path: Path) -> None:
    overview, patch, size = read_context(_scene(tmp_path), tmp_path / "cache", chip=(200, 400))
    assert size == (1600, 1200)
    assert overview.shape == (1200 // DECIMATION, 1600 // DECIMATION)
    assert np.isnan(overview[0]).all() and np.isfinite(overview[-1]).all()
    assert patch.shape == (CHIP_PX, CHIP_PX)
    assert list((tmp_path / "cache").glob("*.npy"))


def test_figure_is_written_in_both_formats(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    paths = figure_scene_context(_scene(tmp_path), tmp_path / "cache", tmp_path / "out", chip=(200, 400))
    assert sorted(p.suffix for p in paths) == [".pdf", ".png"]
