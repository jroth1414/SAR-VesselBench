from __future__ import annotations

import numpy as np
import pytest

from src.data.to_centroids import (
    box_to_center,
    mask_to_centroid,
    parse_voc_centroids,
    rbox_to_center,
)


def test_box_to_center_is_exact():
    center = box_to_center(10, 20, 30, 60)
    assert center.x == pytest.approx(20.0)
    assert center.y == pytest.approx(40.0)


def test_degenerate_box_raises():
    with pytest.raises(ValueError):
        box_to_center(30, 20, 10, 60)


def test_rbox_to_center_within_one_px():
    # A 20x10 box rotated 30 degrees around (50, 40).
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    half_extents = np.array([[-10, -5], [10, -5], [10, 5], [-10, 5]], dtype=float)
    corners = half_extents @ rotation.T + np.array([50.0, 40.0])

    center = rbox_to_center(corners)
    assert abs(center.x - 50.0) <= 1.0
    assert abs(center.y - 40.0) <= 1.0


def test_mask_to_centroid_within_one_px():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:21, 30:41] = True  # centroid at (col 35, row 15)

    center = mask_to_centroid(mask)
    assert abs(center.x - 35.0) <= 1.0
    assert abs(center.y - 15.0) <= 1.0


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        mask_to_centroid(np.zeros((8, 8), dtype=bool))


def test_parse_voc_centroids(tmp_path):
    xml = """<?xml version="1.0"?>
    <annotation>
      <object><name>ship</name>
        <bndbox><xmin>100</xmin><ymin>200</ymin><xmax>120</xmax><ymax>240</ymax></bndbox>
      </object>
      <object><name>ship</name>
        <bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox>
      </object>
    </annotation>
    """
    path = tmp_path / "sub_image.xml"
    path.write_text(xml)

    centroids = parse_voc_centroids(path)
    assert len(centroids) == 2
    assert centroids[0].x == pytest.approx(110.0)
    assert centroids[0].y == pytest.approx(220.0)
    assert centroids[1].x == pytest.approx(5.0)
    assert centroids[1].y == pytest.approx(5.0)
