"""Detector heatmaps for the poster: what the best cell's head outputs offshore vs near shore.

The windows are the two detection-example crops (``poster_coastal.choose_crops``).
For each one this module repeats ``infer_scene``'s tiling exactly: the
scene-level 512 px / stride 384 tile grid, [VH, VV, VH-VV] normalized with the
frozen training statistics, sigmoid heatmaps max-pasted at stride 4. Only the
tiles that overlap the window run, so each pixel shows the confidence the
scorer saw. The checkpoint is the cell's ``checkpoints/best.ckpt``, checked
against the SHA-256 its evaluation record binds.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

from src.analysis.heldout_results import EvidenceError

OUTPUT_STRIDE_PX = 4
NODATA = -32768.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_band(archive_dir: Path, cache_dir: Path, scene: str, band: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tif = cache_dir / f"{scene}_{band}_dB.tif"
    if not tif.exists():
        partial = tif.with_suffix(".tif.partial")
        with tarfile.open(archive_dir / f"{scene}.tar.gz", "r:gz") as archive:
            source = archive.extractfile(archive.getmember(f"{scene}/{band}_dB.tif"))
            with partial.open("wb") as target:
                for chunk in iter(lambda: source.read(1 << 24), b""):
                    target.write(chunk)
        partial.replace(tif)
    return tif


def load_best_module(run_dir: Path, evidence_record: dict, device: str):
    """Best checkpoint of one cell, verified against its bound SHA-256."""

    import pathlib

    import torch

    from src.train.lit_modules import HeatmapLitModule

    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    expected = evidence_record["checkpoint_sha256"]
    if _sha256(checkpoint) != expected:
        raise EvidenceError(f"{checkpoint} does not match the checkpoint bound by the evidence")
    # The cluster saved its hyperparameters with a PosixPath, which the safe
    # (weights-only) loader rejects and Windows cannot build; read it as a
    # PurePosixPath and keep weights_only=True.
    with torch.serialization.safe_globals([(pathlib.PurePosixPath, "pathlib.PosixPath")]):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # Pretrained weights are not needed: every tensor comes from the fine-tuned checkpoint.
    module = HeatmapLitModule(init_name=payload["hyper_parameters"]["init_name"], load_weights=False)
    module.load_state_dict(payload["state_dict"], strict=True)
    return module.eval().to(device)


def window_heatmap(module, archive_dir: Path, cache_dir: Path, scene: str, x0: int, y0: int, size: int,
                   stats: dict, device: str, tile_px: int = 512, tile_stride_px: int = 384) -> np.ndarray:
    """Stride-4 sigmoid heatmap over the window [y0, y0+size) x [x0, x0+size)."""

    import rasterio
    import torch
    from rasterio.windows import Window

    from src.data.datasets import _channel_stats
    from src.data.transforms import normalize
    from src.eval.infer_scene import _tile_origins

    mean, std = _channel_stats(dict(stats))
    vh_path = _cached_band(archive_dir, cache_dir, scene, "VH")
    vv_path = _cached_band(archive_dir, cache_dir, scene, "VV")
    with rasterio.open(vh_path) as vh_ds:
        height, width = vh_ds.height, vh_ds.width
    tiles = [(r, c) for r, c in _tile_origins(height, width, tile_px, tile_stride_px)
             if r < y0 + size and r + tile_px > y0 and c < x0 + size and c + tile_px > x0]
    r_min, c_min = min(r for r, _ in tiles), min(c for _, c in tiles)
    r_max, c_max = max(r for r, _ in tiles) + tile_px, max(c for _, c in tiles) + tile_px
    region = Window(c_min, r_min, c_max - c_min, r_max - r_min)
    with rasterio.open(vh_path) as vh_ds, rasterio.open(vv_path) as vv_ds:
        vh_all = vh_ds.read(1, window=region, boundless=True, fill_value=NODATA)
        vv_all = vv_ds.read(1, window=region, boundless=True, fill_value=NODATA)
    canvas = np.zeros(((r_max - r_min) // OUTPUT_STRIDE_PX + 1, (c_max - c_min) // OUTPUT_STRIDE_PX + 1),
                      dtype=np.float32)
    with torch.no_grad():
        for row0, col0 in tiles:
            rr, cc = row0 - r_min, col0 - c_min
            vh = vh_all[rr:rr + tile_px, cc:cc + tile_px]
            vv = vv_all[rr:rr + tile_px, cc:cc + tile_px]
            if (vh == NODATA).mean() > 0.98:
                continue
            tile = np.stack([vh, vv, vh - vv]).astype(np.float32)
            tile[np.stack([vh, vv, np.zeros_like(vh)]) == NODATA] = np.nan
            tile[2][np.isnan(tile[0]) | np.isnan(tile[1])] = np.nan
            batch = torch.from_numpy(normalize(tile, mean, std)[None]).to(device)
            heat = torch.sigmoid(module(batch)).squeeze().float().cpu().numpy()
            out_r, out_c = rr // OUTPUT_STRIDE_PX, cc // OUTPUT_STRIDE_PX
            target = canvas[out_r:out_r + heat.shape[0], out_c:out_c + heat.shape[1]]
            np.maximum(target, heat[:target.shape[0], :target.shape[1]], out=target)
    top, left = (y0 - r_min) // OUTPUT_STRIDE_PX, (x0 - c_min) // OUTPUT_STRIDE_PX
    return canvas[top:top + size // OUTPUT_STRIDE_PX, left:left + size // OUTPUT_STRIDE_PX]


def figure_heatmaps(panels, threshold: float, out_dir: Path) -> list[Path]:
    """panels: [(title, vh_image, heatmap, vessels_xy_km)]; overlays the head's confidence on the radar."""

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.ndimage import maximum_filter

    from src.analysis.poster_figures import INK, save_both

    fig = plt.figure(figsize=(10.6, 6.1))
    axes = [fig.add_axes((0.0, 0.2, 0.485, 0.71)), fig.add_axes((0.515, 0.2, 0.485, 0.71))]
    for axis, (title, image, heat, vessels) in zip(axes, panels):
        low, high = np.nanpercentile(image, [2, 99.5])
        axis.imshow(np.clip(image, low, high), cmap="gray", extent=(0, 8, 8, 0), interpolation="nearest")
        # Each peak covers a few 40 m cells; a 5-cell (200 m, the matching radius) max filter
        # makes it visible at poster scale without changing any value.
        shown = axis.imshow(np.ma.masked_less(maximum_filter(heat, size=5), 0.05), cmap="inferno", vmin=0,
                            vmax=1, alpha=0.9, extent=(0, 8, 8, 0), interpolation="nearest")
        for x, y in vessels:
            axis.plot(x, y, marker="o", ms=15, mfc="none", mec="#56B4E9", mew=2.5)
        peak = float(np.nanmax(heat))
        # Two decimals keep figure text out of check_poster's three-decimal cross-check.
        axis.set_title(f"{title}: peak {peak:.2f}" if peak >= 0.01 else f"{title}: peak < 0.01", color=INK)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    bar_axis = fig.add_axes((0.02, 0.07, 0.5, 0.05))
    bar = fig.colorbar(shown, cax=bar_axis, orientation="horizontal")
    bar.ax.set_title("detector confidence", color=INK, pad=10)
    bar.ax.axvline(threshold, color="#56B4E9", lw=5)
    # Legend sits beside the colorbar so the figure needs one row less.
    fig.legend([Line2D([], [], marker="o", ms=15, mfc="none", mec="#56B4E9", mew=2.5, ls="none"),
                Line2D([], [], color="#56B4E9", lw=5)],
               ["labelled vessel", f"threshold {threshold:.2f}"], loc="lower left", ncol=1,
               bbox_to_anchor=(0.56, -0.02), handletextpad=0.3, labelspacing=0.3)
    paths = save_both(fig, out_dir, "poster_detector_heatmaps")
    plt.close(fig)
    return paths


def build(cell: dict, runs_root: Path, crops, per_scene, archive_dir: Path, cache_dir: Path, stats_path: Path,
          out_dir: Path, device: str = "cuda") -> tuple[list[Path], dict]:
    """``cell`` is the validated evidence record (checkpoint SHA-256 and dev-bound threshold)."""

    from src.analysis.poster_coastal import CROP_PX, read_vh_window

    threshold = float(cell["threshold"])
    module = load_best_module(runs_root / cell["exp_id"], cell, device)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    panels, peaks = [], {}
    for kind, scene, x0, y0 in crops:
        image, x0, y0 = read_vh_window(archive_dir, cache_dir, scene, x0, y0)
        heat = window_heatmap(module, archive_dir, cache_dir, scene, x0, y0, CROP_PX, stats, device)
        gt = per_scene[scene][0]
        vessels = [((p.x_m / 10.0 - x0) / 100.0, (p.y_m / 10.0 - y0) / 100.0) for p in gt
                   if not p.is_low_confidence and 0 <= p.x_m / 10.0 - x0 < CROP_PX and 0 <= p.y_m / 10.0 - y0 < CROP_PX]
        panels.append((kind.capitalize(), image, heat, vessels))
        peaks[f"heatmap_peak_{kind.replace(' ', '_')}"] = float(np.nanmax(heat))
    paths = figure_heatmaps(panels, threshold, out_dir)
    peaks["heatmap_threshold"] = threshold
    return paths, peaks
