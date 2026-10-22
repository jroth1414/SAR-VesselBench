"""Reduce labeled supervised sources to center points (DEVPLAN P1.4).

The detector trains on center points, so every labeled source is reduced to
centroids: box -> center, rotated box -> center, instance mask -> centroid.
This is what lets LS-SSDD feed the same heatmap head as xView3 (for both the
ViT Arm-4 and CNN Arm-8 supervised backbones) with no box-format
harmonization. LS-SSDD ships PASCAL-VOC XML per 800x800 sub-image.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Centroid:
    """A target center in sub-image pixel coordinates (x=col, y=row)."""

    x: float
    y: float
    source_kind: str  # "box" | "rbox" | "mask"


def box_to_center(xmin: float, ymin: float, xmax: float, ymax: float) -> Centroid:
    """Axis-aligned box -> its center point."""

    if xmax < xmin or ymax < ymin:
        raise ValueError(f"degenerate box ({xmin=}, {ymin=}, {xmax=}, {ymax=})")
    return Centroid(x=(xmin + xmax) / 2.0, y=(ymin + ymax) / 2.0, source_kind="box")


def rbox_to_center(corners: np.ndarray) -> Centroid:
    """Rotated box given as 4 (x, y) corners -> mean of the corners."""

    array = np.asarray(corners, dtype=float)
    if array.shape != (4, 2):
        raise ValueError(f"rotated box must be 4 (x, y) corners, got {array.shape}")
    center = array.mean(axis=0)
    return Centroid(x=float(center[0]), y=float(center[1]), source_kind="rbox")


def mask_to_centroid(mask: np.ndarray) -> Centroid:
    """Binary instance mask -> centroid of foreground pixels."""

    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {array.shape}")
    rows, cols = np.nonzero(array)
    if rows.size == 0:
        raise ValueError("mask has no foreground pixels")
    return Centroid(x=float(cols.mean()), y=float(rows.mean()), source_kind="mask")


def parse_voc_centroids(xml_path: str | Path) -> list[Centroid]:
    """Parse a PASCAL-VOC annotation XML (LS-SSDD format) into centroids."""

    root = ET.parse(str(xml_path)).getroot()
    centroids: list[Centroid] = []
    for obj in root.iter("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        values = {
            key: float(bndbox.findtext(key))  # type: ignore[arg-type]
            for key in ("xmin", "ymin", "xmax", "ymax")
        }
        centroids.append(box_to_center(**values))
    return centroids
