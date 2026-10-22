"""Poster-scale scene-and-chip figure: one Sentinel-1 scene beside one 8 km chip.

The report's static ``scene_context.png`` carries type sized for a page, which
is unreadable on the poster. This redraws the same view (scene
835f7629c3a3a9abt, chip at row 10500, column 12600) from the scene's xView3
``VH_dB.tif`` with poster-size type. The decimated scene overview is cached as
``.npy`` because reading it scans the whole 1.3 GB raster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CONTEXT_SCENE = "835f7629c3a3a9abt"
CONTEXT_CHIP = (10500, 12600)  # (row, column) of the report figure's 800-pixel chip
CHIP_PX = 800
PIXEL_M = 10.0
DECIMATION = 20  # overview pixels are 200 m
BOX_COLOR = "#D55E00"  # Okabe-Ito vermillion: not a role or detection-outcome hue


def read_context(scene_dir: Path, cache_dir: Path, chip: tuple[int, int] = CONTEXT_CHIP):
    """Scene overview (NaN outside the swath), the chip, and the raster size in pixels."""

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{scene_dir.name}_VH_overview_{DECIMATION}.npy"
    with rasterio.open(scene_dir / "VH_dB.tif") as dataset:
        size = (dataset.width, dataset.height)
        if cached.exists():
            overview = np.load(cached)
        else:
            masked = dataset.read(1, masked=True, resampling=Resampling.average,
                                  out_shape=(dataset.height // DECIMATION, dataset.width // DECIMATION))
            overview = masked.astype(float).filled(np.nan)
            np.save(cached, overview)
        row, col = chip
        patch = dataset.read(1, window=Window(col, row, CHIP_PX, CHIP_PX), masked=True).astype(float).filled(np.nan)
    return overview, patch, size


def figure_scene_context(scene_dir: Path, cache_dir: Path, out_dir: Path, chip: tuple[int, int] = CONTEXT_CHIP):
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle

    from src.analysis.poster_figures import INK, save_both

    overview, patch, (width_px, height_px) = read_context(scene_dir, cache_dir, chip)
    km_w, km_h = width_px * PIXEL_M / 1000, height_px * PIXEL_M / 1000
    chip_km = CHIP_PX * PIXEL_M / 1000
    cmap = plt.get_cmap("gray").with_extremes(bad="white")  # outside the swath blends into the poster
    fig, (scene_ax, chip_ax) = plt.subplots(1, 2, figsize=(9.6, 4.9),
                                            gridspec_kw={"width_ratios": [km_w / km_h, 1.0], "wspace": 0.25})
    low, high = np.nanpercentile(overview, [2, 99.5])
    scene_ax.imshow(np.clip(overview, low, high), cmap=cmap, extent=(0, km_w, km_h, 0), interpolation="nearest")
    low, high = np.nanpercentile(patch, [2, 99.5])
    chip_ax.imshow(np.clip(patch, low, high), cmap=cmap, extent=(0, chip_km, chip_km, 0), interpolation="nearest")
    row, col = chip
    x0, y0 = col * PIXEL_M / 1000, row * PIXEL_M / 1000
    scene_ax.add_patch(Rectangle((x0, y0), chip_km, chip_km, fill=False, ec=BOX_COLOR, lw=3.5))
    for corner in (0, chip_km):  # lead lines from the box to the chip's left corners
        fig.add_artist(ConnectionPatch(xyA=(x0 + chip_km, y0 + corner), coordsA=scene_ax.transData,
                                       xyB=(0, corner), coordsB=chip_ax.transData, color=BOX_COLOR, lw=2.5))
    scene_ax.set_title(f"Scene, {km_w:.0f} × {km_h:.0f} km", color=INK)
    chip_ax.set_title(f"Chip, {chip_km:.0f} × {chip_km:.0f} km", color=INK)
    for axis in (scene_ax, chip_ax):
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    for spine in chip_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BOX_COLOR)
        spine.set_linewidth(3.5)
    fig.subplots_adjust(left=0.01, right=0.985, bottom=0.015, top=0.86)
    paths = save_both(fig, out_dir, "poster_scene_context")
    plt.close(fig)
    return paths
