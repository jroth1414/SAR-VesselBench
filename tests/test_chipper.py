from __future__ import annotations

import json

import numpy as np
import pytest

# rasterio is in the GPU lockfiles but deliberately not in requirements-ci.txt
# (CI is the minimal guard environment); these tests run wherever the data
# pipeline actually runs.
rasterio = pytest.importorskip("rasterio")

from rasterio.transform import from_origin  # noqa: E402

from src.data.chipper import (  # noqa: E402
    chip_scene,
    count_positives,
    iter_chip_origins,
    project_labels,
)


def test_iter_chip_origins_covers_scene_with_clamped_edges():
    origins = iter_chip_origins(2000, 1500, chip_size=800, overlap=100)

    rows = sorted({r for r, _ in origins})
    cols = sorted({c for _, c in origins})
    assert rows == [0, 700, 1200]  # last row clamped to 2000 - 800
    assert cols == [0, 700]  # last col clamped to 1500 - 800

    covered_rows = max(r + 800 for r, _ in origins)
    covered_cols = max(c + 800 for _, c in origins)
    assert covered_rows == 2000 and covered_cols == 1500


def test_iter_chip_origins_small_scene_single_chip():
    assert iter_chip_origins(500, 500, chip_size=800, overlap=100) == [(0, 0)]


def test_iter_chip_origins_rejects_bad_geometry():
    with pytest.raises(ValueError):
        iter_chip_origins(1000, 1000, chip_size=800, overlap=800)


def test_project_labels_into_chip_and_overlap():
    labels = [
        {"detect_scene_row": 750, "detect_scene_column": 100, "confidence": "HIGH"},
        {"detect_scene_row": 5000, "detect_scene_column": 5000, "confidence": "HIGH"},
    ]
    # Chip at origin (0, 0) and the overlapping chip at (700, 0): a label at
    # row 750 falls in both (overlap regions legitimately duplicate labels).
    first = project_labels(labels, row0=0, col0=0, chip_size=800)
    second = project_labels(labels, row0=700, col0=0, chip_size=800)

    assert len(first) == 1 and first[0]["chip_row"] == 750.0
    assert len(second) == 1 and second[0]["chip_row"] == 50.0


def test_count_positives_follows_confidence_protocol():
    chip_labels = [
        {"is_vessel": True, "confidence": "HIGH"},
        {"is_vessel": True, "confidence": "MEDIUM"},
        {"is_vessel": True, "confidence": "LOW"},  # ignore region, not positive
        {"is_vessel": False, "confidence": "HIGH"},  # fixed infrastructure
    ]
    n_vessels, has_low_conf = count_positives(chip_labels)
    assert n_vessels == 2
    assert has_low_conf is True


def _write_scene(scene_dir, height, width, *, nodata_rows=0):
    """Synthetic dB-like VH/VV rasters with optional nodata band at the top."""

    scene_dir.mkdir(parents=True)
    transform = from_origin(400000, 4600000, 10.0, 10.0)
    rng = np.random.default_rng(0)
    for name, base in (("VH_dB.tif", -25.0), ("VV_dB.tif", -15.0)):
        data = (base + rng.normal(0, 2, size=(height, width))).astype(np.float32)
        if nodata_rows:
            data[:nodata_rows] = -32768.0
        with rasterio.open(
            scene_dir / name,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:32634",
            transform=transform,
            nodata=-32768.0,
        ) as ds:
            ds.write(data, 1)


def test_chip_scene_end_to_end(tmp_path):
    scene_dir = tmp_path / "raw" / "aaaa000000000000t"
    _write_scene(scene_dir, 96, 96)
    labels = [
        {
            "detect_scene_row": 10,
            "detect_scene_column": 20,
            "is_vessel": True,
            "confidence": "HIGH",
            "source": "Manual",
            "vessel_length_m": 42.0,
            "distance_from_shore_km": 12.0,
        }
    ]

    records = chip_scene(
        scene_dir,
        tmp_path / "chips",
        scene_labels=labels,
        chip_size=64,
        overlap=16,
        max_nodata_frac=0.95,
    )

    # 96px with 64px chips / stride 48 -> origins {0, 32} per axis.
    assert {(record.row0, record.col0) for record in records} == {
        (0, 0), (0, 32), (32, 0), (32, 32),
    }

    for record in records:
        chips = np.load(record.chip_path)
        assert chips.shape == (2, 64, 64)
        assert chips.dtype == np.float16

    tagged = [record for record in records if record.n_vessels == 1]
    assert [(record.row0, record.col0) for record in tagged] == [(0, 0)]
    sidecar = json.loads(
        (tmp_path / "chips" / "aaaa000000000000t_r0_c0.json").read_text()
    )
    assert sidecar["labels"][0]["chip_row"] == 10.0
    assert sidecar["labels"][0]["source"] == "Manual"
    assert sidecar["labels"][0]["vessel_length_m"] == 42.0


def test_chip_scene_drops_mostly_nodata_chips(tmp_path):
    scene_dir = tmp_path / "raw" / "bbbb000000000000t"
    # Top 62 of 64 rows are nodata -> the (0, 0) chip is ~97% nodata.
    _write_scene(scene_dir, 64, 64, nodata_rows=62)

    records = chip_scene(
        scene_dir,
        tmp_path / "chips",
        chip_size=64,
        overlap=16,
        max_nodata_frac=0.95,
    )
    assert records == []

    kept = chip_scene(
        scene_dir,
        tmp_path / "chips_kept",
        chip_size=64,
        overlap=16,
        max_nodata_frac=0.99,
    )
    assert len(kept) == 1
    chips = np.load(kept[0].chip_path)
    assert np.isnan(chips[0, :62]).all()  # nodata stored as NaN
    assert np.isfinite(chips[0, 62:].astype(np.float32)).all()
