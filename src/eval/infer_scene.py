"""Tiled whole-scene inference (DEVPLAN P2.2 unit contract + P3.5 dev eval).

This module is the SINGLE site performing
``heatmap-px -> chip-px (x4) -> meters (x10) -> scene-meters``:
tiles a scene's [VH, VV, VH-VV] input at 512 px / stride 384, max-pastes the
sigmoid heatmaps onto a scene-level stride-4 canvas, then decodes ONCE
globally with ``output_stride_m = 40.0`` (4 px x 10 m GSD) and distance NMS —
so decoded (row, col) are already scene meters on the same grid as
``detect_scene_row/column * 10``.

Frozen-scorer adapters live here too:
- ``source`` case normalization ('manual' -> 'Manual', 'ais/manual' ->
  'AIS/Manual', 'ais' -> 'AIS'): the real CSVs are lowercase but the FROZEN
  scorer's dark test is ``source == "Manual"`` (BLOCKER-4 finding).
- Prediction-side ``distance_from_shore_km`` (required by the near-shore
  slice for unmatched FPs) is derived from the scene's 500 m bathymetry
  raster: land = elevation >= 0, distance transform in km, sampled at each
  prediction. Scenes without bathymetry yield None (the scorer then raises
  only if an unmatched FP actually needs it).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from src.eval.decode import DecodedPoint, decode_heatmap
from src.eval.scorer import GroundTruthPoint, PredictionPoint

GSD_M = 10.0
OUTPUT_STRIDE_PX = 4
OUTPUT_STRIDE_M = OUTPUT_STRIDE_PX * GSD_M  # 40.0 — the P2.2 contract
SOURCE_CASE = {"ais": "AIS", "ais/manual": "AIS/Manual", "manual": "Manual"}
SUPPORTED_PRECISIONS = {"16-mixed", "32-true"}


def _autocast_enabled(precision: str, device_type: str) -> bool:
    """Whether the shared recipe enables model-forward autocast."""

    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"unsupported shared inference precision: {precision}")
    return device_type == "cuda" and precision == "16-mixed"


def normalize_source(source: object) -> str | None:
    if source is None or (isinstance(source, float) and math.isnan(source)):
        return None
    return SOURCE_CASE.get(str(source).lower(), str(source))


def ground_truth_from_labels(rows: Sequence[Mapping[str, object]]) -> list[GroundTruthPoint]:
    """xView3 label rows (one scene) -> frozen-scorer GroundTruthPoints."""

    points = []
    for row in rows:
        shore = row.get("distance_from_shore_km")
        if isinstance(shore, float) and math.isnan(shore):
            shore = None
        points.append(
            GroundTruthPoint(
                x_m=float(row["detect_scene_column"]) * GSD_M,
                y_m=float(row["detect_scene_row"]) * GSD_M,
                confidence=str(row.get("confidence") or "HIGH"),
                source=normalize_source(row.get("source")),
                distance_from_shore_km=float(shore) if shore is not None else None,
            )
        )
    return points


class ShoreDistance:
    """distance_from_shore_km lookup from the scene's 500 m bathymetry."""

    def __init__(self, scene_dir: Path) -> None:
        import rasterio
        from scipy.ndimage import distance_transform_edt

        self.available = False
        bathy_path = scene_dir / "bathymetry.tif"
        if not bathy_path.exists():
            return
        with rasterio.open(bathy_path) as ds:
            data = ds.read(1)
            self.transform = ds.transform
            self.shape = data.shape
        valid = data != -32768.0
        land = (data >= 0.0) & valid
        if not land.any():
            # open-ocean scene: seed the transform from the raster edge so
            # distances are finite but large
            self.dist_km = np.full(self.shape, 9999.99, dtype=np.float32)
        else:
            cell_km = abs(self.transform.a) / 1000.0
            self.dist_km = (
                distance_transform_edt(~land).astype(np.float32) * cell_km
            )
        self.available = True

    def lookup_km(self, x_geo: float, y_geo: float) -> float | None:
        if not self.available:
            return None
        col, row = ~self.transform * (x_geo, y_geo)
        row = int(np.clip(row, 0, self.shape[0] - 1))
        col = int(np.clip(col, 0, self.shape[1] - 1))
        return float(self.dist_km[row, col])


@torch.no_grad()
def infer_scene(
    model: torch.nn.Module,
    scene_dir: str | Path,
    *,
    stats: Mapping[str, object],
    tau: float,
    d_nms_m: float = 120.0,
    tile_px: int = 512,
    tile_stride_px: int = 384,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
    precision: str,
) -> list[PredictionPoint]:
    """Run tiled inference over one scene -> scene-meter PredictionPoints."""

    import rasterio

    from src.data.datasets import _channel_stats

    scene_dir = Path(scene_dir)
    mean, std = _channel_stats(dict(stats))
    device = torch.device(device)
    model = model.to(device).eval()

    from src.data.transforms import normalize

    # Read each polarization ONCE into RAM (~2.6 GB float32 per band for the
    # largest scenes) and slice windows from memory: per-window reads against
    # the striped compressed GeoTIFFs are ~100x slower and made the every-5-
    # epoch dev eval unusable.
    with rasterio.open(scene_dir / "VH_dB.tif") as vh_ds:
        height, width = vh_ds.height, vh_ds.width
        geo_transform = vh_ds.transform
        vh_full = vh_ds.read(1)
    with rasterio.open(scene_dir / "VV_dB.tif") as vv_ds:
        vv_full = vv_ds.read(1)

    canvas = np.zeros(
        (height // OUTPUT_STRIDE_PX + 1, width // OUTPUT_STRIDE_PX + 1),
        dtype=np.float16,
    )

    origins = _tile_origins(height, width, tile_px, tile_stride_px)
    batch_tiles: list[np.ndarray] = []
    batch_origins: list[tuple[int, int]] = []

    def flush() -> None:
        if not batch_tiles:
            return
        tensor = torch.from_numpy(np.stack(batch_tiles)).to(device)
        with torch.autocast(
            device.type,
            dtype=torch.float16,
            enabled=_autocast_enabled(precision, device.type),
        ):
            heat = torch.sigmoid(model(tensor)).squeeze(1).float().cpu().numpy()
        for (row0, col0), tile_heat in zip(batch_origins, heat):
            out_r = row0 // OUTPUT_STRIDE_PX
            out_c = col0 // OUTPUT_STRIDE_PX
            window = canvas[
                out_r : out_r + tile_heat.shape[0],
                out_c : out_c + tile_heat.shape[1],
            ]
            np.maximum(window, tile_heat[: window.shape[0], : window.shape[1]], out=window)
        batch_tiles.clear()
        batch_origins.clear()

    for row0, col0 in origins:
        vh = vh_full[row0 : row0 + tile_px, col0 : col0 + tile_px]
        if (vh == -32768.0).mean() > 0.98:
            continue  # pure nodata tile
        vv = vv_full[row0 : row0 + tile_px, col0 : col0 + tile_px]
        tile = np.stack([vh, vv, vh - vv]).astype(np.float32)
        tile[np.stack([vh, vv, np.zeros_like(vh)]) == -32768.0] = np.nan
        tile[2][np.isnan(tile[0]) | np.isnan(tile[1])] = np.nan
        batch_tiles.append(normalize(tile, mean, std))
        batch_origins.append((row0, col0))
        if len(batch_tiles) >= batch_size:
            flush()
    flush()
    del vh_full, vv_full

    decoded: list[DecodedPoint] = decode_heatmap(
        canvas.astype(np.float32),
        threshold=tau,
        output_stride_m=OUTPUT_STRIDE_M,
        d_nms_m=d_nms_m,
    )

    shore = ShoreDistance(scene_dir)
    predictions = []
    for point in decoded:
        x_geo, y_geo = geo_transform * (point.col / GSD_M, point.row / GSD_M)
        predictions.append(
            PredictionPoint(
                x_m=point.col,
                y_m=point.row,
                score=point.score,
                distance_from_shore_km=shore.lookup_km(x_geo, y_geo),
            )
        )
    return predictions


def _tile_origins(
    height: int, width: int, tile: int, stride: int
) -> list[tuple[int, int]]:
    def axis(extent: int) -> list[int]:
        if extent <= tile:
            return [0]
        origins = list(range(0, extent - tile, stride))
        origins.append(extent - tile)
        return origins

    return [(r, c) for r in axis(height) for c in axis(width)]


def dev_f1(
    model: torch.nn.Module,
    *,
    scene_ids: Sequence[str],
    raw_root: str | Path,
    labels_csv: str | Path,
    stats_path: str | Path,
    tau: float,
    **infer_kwargs,
) -> dict:
    """Dev-scene scoring through the FROZEN scorer, P2.2b-style.

    ``tau`` is the LOW candidate floor for decoding; the operating point is
    then the F1-maximizing threshold selected on these dev predictions by
    ``src/eval/threshold.py`` (the single home for threshold selection —
    P2.2b/P5.4), reported alongside the resulting metrics.
    """

    import pandas as pd

    from src.eval.scorer import score_dataset
    from src.eval.threshold import apply_threshold, select_f1_threshold

    stats = json.loads(Path(stats_path).read_text())
    labels = pd.read_csv(labels_csv)

    gt_by_scene = {}
    pred_by_scene = {}
    for scene_id in scene_ids:
        rows = labels[labels["scene_id"] == scene_id].to_dict(orient="records")
        gt_by_scene[scene_id] = ground_truth_from_labels(rows)
        pred_by_scene[scene_id] = infer_scene(
            model, Path(raw_root) / scene_id, stats=stats, tau=tau, **infer_kwargs
        )

    threshold = select_f1_threshold(gt_by_scene, pred_by_scene)
    result = score_dataset(gt_by_scene, apply_threshold(pred_by_scene, threshold))
    return {
        "f1": result.aggregate.f1,
        "precision": result.aggregate.precision,
        "recall": result.aggregate.recall,
        "tp": result.aggregate.tp,
        "fp": result.aggregate.fp,
        "fn": result.aggregate.fn,
        "threshold": threshold,
        "n_candidates": sum(len(p) for p in pred_by_scene.values()),
    }
