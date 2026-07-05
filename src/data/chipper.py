"""Scene -> 800x800 chips + label projection (DEVPLAN P1.3).

Reads the xView3 GRD per-scene GeoTIFFs (VH_dB, VV_dB) in windows and emits
800x800 float16 chips with per-chip JSON label sidecars and a per-scene
parquet manifest.

Pixel pipeline note (documented deviation from the P1.3 sketch): the plan's
``log10(clip(raw, 1, None))`` assumed linear-amplitude rasters, but the
original xView3 GRD products are distributed ALREADY IN dB (``VH_dB.tif`` /
``VV_dB.tif``, float32, nodata -32768). Applying a second log would corrupt
the pixels, so chips store the product's dB values directly as float16 with
nodata as NaN — the same "store float16 log-scale values, normalize at load
time with train-split stats" contract the plan intends.

Land fraction: xView3 ships no shoreline vectors; the 500 m ``bathymetry.tif``
(elevation >= 0 -> land) is the documented proxy where present.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

CHIP_SIZE_PX = 800
CHIP_OVERLAP_PX = 100  # stride 700
NODATA_SENTINEL = -32768.0

# Label CSV columns carried into every chip sidecar (Appendix B schema).
SIDECAR_LABEL_FIELDS = (
    "detect_id",
    "is_vessel",
    "is_fishing",
    "vessel_length_m",
    "confidence",
    "source",
    "distance_from_shore_km",
    "top",
    "left",
    "bottom",
    "right",
)


@dataclass(frozen=True)
class ChipRecord:
    """One manifest row (P1.3): chip path + provenance + label summary."""

    chip_path: str
    scene_id: str
    row0: int
    col0: int
    n_vessels: int
    has_low_conf: bool
    land_frac: float | None


def iter_chip_origins(
    height: int,
    width: int,
    *,
    chip_size: int = CHIP_SIZE_PX,
    overlap: int = CHIP_OVERLAP_PX,
) -> list[tuple[int, int]]:
    """Grid of (row0, col0) chip origins covering the scene.

    Stride is ``chip_size - overlap``; the final row/column of chips is
    clamped to the scene edge so every pixel is covered without stepping
    outside the raster. Scenes smaller than one chip yield a single origin
    at (0, 0) (the chip is then padded by the reader).
    """

    if chip_size <= 0 or not 0 <= overlap < chip_size:
        raise ValueError(f"invalid chip geometry ({chip_size=}, {overlap=})")
    stride = chip_size - overlap

    def axis_origins(extent: int) -> list[int]:
        if extent <= chip_size:
            return [0]
        last = extent - chip_size
        origins = list(range(0, last, stride))
        origins.append(last)
        return origins

    return [(r, c) for r in axis_origins(height) for c in axis_origins(width)]


def project_labels(
    scene_labels: Iterable[Mapping[str, object]],
    *,
    row0: int,
    col0: int,
    chip_size: int = CHIP_SIZE_PX,
) -> list[dict[str, object]]:
    """Project scene-pixel labels into chip-pixel coordinates.

    ``detect_scene_row``/``detect_scene_column`` are full-scene pixel
    coordinates on the dB raster grid (10 m GSD). A label lands in this chip
    when its point falls inside [origin, origin + chip_size). Labels in
    overlap regions legitimately appear in multiple chips.
    """

    projected: list[dict[str, object]] = []
    for label in scene_labels:
        scene_row = float(label["detect_scene_row"])  # type: ignore[arg-type]
        scene_col = float(label["detect_scene_column"])  # type: ignore[arg-type]
        chip_row = scene_row - row0
        chip_col = scene_col - col0
        if not (0 <= chip_row < chip_size and 0 <= chip_col < chip_size):
            continue
        entry: dict[str, object] = {
            "chip_row": chip_row,
            "chip_col": chip_col,
            "scene_row": scene_row,
            "scene_col": scene_col,
        }
        for field_name in SIDECAR_LABEL_FIELDS:
            if field_name in label:
                value = label[field_name]
                if isinstance(value, float) and math.isnan(value):
                    value = None
                entry[field_name] = value
        projected.append(entry)
    return projected


def count_positives(chip_labels: Sequence[Mapping[str, object]]) -> tuple[int, bool]:
    """(n_vessels, has_low_conf) per the Appendix B protocol.

    Detection positives are ``is_vessel`` truthy with confidence HIGH/MEDIUM;
    LOW-confidence labels become ignore regions, flagged so the sampler can
    see them.
    """

    n_vessels = 0
    has_low_conf = False
    for label in chip_labels:
        confidence = str(label.get("confidence", "")).upper()
        if confidence == "LOW":
            has_low_conf = True
            continue
        if bool(label.get("is_vessel")) and confidence in ("HIGH", "MEDIUM"):
            n_vessels += 1
    return n_vessels, has_low_conf


def read_chip(dataset, row0: int, col0: int, chip_size: int = CHIP_SIZE_PX) -> np.ndarray:
    """Read one float16 chip from an open rasterio dataset, nodata -> NaN."""

    from rasterio.windows import Window

    window = Window(col0, row0, chip_size, chip_size)
    data = dataset.read(1, window=window, boundless=True, fill_value=NODATA_SENTINEL)
    array = data.astype(np.float32)
    nodata = dataset.nodata if dataset.nodata is not None else NODATA_SENTINEL
    array[(array == nodata) | ~np.isfinite(array)] = np.nan
    return array.astype(np.float16)


def land_fraction(bathymetry_path: Path, chip_bounds: tuple[float, float, float, float]) -> float | None:
    """Fraction of the chip footprint with bathymetry elevation >= 0 (land).

    Returns None when the bathymetry raster is missing or does not intersect.
    """

    import rasterio
    from rasterio.windows import from_bounds

    if not bathymetry_path.exists():
        return None
    with rasterio.open(bathymetry_path) as bathy:
        try:
            window = from_bounds(*chip_bounds, transform=bathy.transform)
        except Exception:
            return None
        data = bathy.read(
            1, window=window, boundless=True, fill_value=NODATA_SENTINEL
        ).astype(np.float32)
        valid = data != NODATA_SENTINEL
        if not valid.any():
            return None
        return float((data[valid] >= 0.0).mean())


def chip_scene(
    scene_dir: Path,
    out_dir: Path,
    *,
    scene_labels: Sequence[Mapping[str, object]] = (),
    chip_size: int = CHIP_SIZE_PX,
    overlap: int = CHIP_OVERLAP_PX,
    max_nodata_frac: float = 0.95,
) -> list[ChipRecord]:
    """Chip one scene directory (VH_dB.tif + VV_dB.tif) to ``out_dir``.

    Writes ``<scene_id>_r{row0}_c{col0}.npy`` (float16, shape (2, chip, chip),
    channels [VH, VV], nodata as NaN) plus a JSON sidecar per chip, and
    returns the manifest rows. Chips above ``max_nodata_frac`` are dropped.
    """

    import rasterio

    scene_id = scene_dir.name
    vh_path = scene_dir / "VH_dB.tif"
    vv_path = scene_dir / "VV_dB.tif"
    for path in (vh_path, vv_path):
        if not path.exists():
            raise FileNotFoundError(f"{scene_id}: missing {path.name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[ChipRecord] = []

    with rasterio.open(vh_path) as vh, rasterio.open(vv_path) as vv:
        if (vh.height, vh.width) != (vv.height, vv.width):
            raise ValueError(f"{scene_id}: VH/VV raster shapes differ")
        for row0, col0 in iter_chip_origins(
            vh.height, vh.width, chip_size=chip_size, overlap=overlap
        ):
            vh_chip = read_chip(vh, row0, col0, chip_size)
            nodata_frac = float(np.isnan(vh_chip).mean())
            if nodata_frac > max_nodata_frac:
                continue
            vv_chip = read_chip(vv, row0, col0, chip_size)

            chip_labels = project_labels(
                scene_labels, row0=row0, col0=col0, chip_size=chip_size
            )
            n_vessels, has_low_conf = count_positives(chip_labels)

            window_bounds = rasterio.windows.bounds(
                rasterio.windows.Window(col0, row0, chip_size, chip_size),
                vh.transform,
            )
            frac_land = land_fraction(scene_dir / "bathymetry.tif", window_bounds)

            stem = f"{scene_id}_r{row0}_c{col0}"
            chip_path = out_dir / f"{stem}.npy"
            np.save(chip_path, np.stack([vh_chip, vv_chip]))
            sidecar = {
                "scene_id": scene_id,
                "row0": row0,
                "col0": col0,
                "chip_size": chip_size,
                "gsd_m": vh.res[0],
                "nodata_frac": nodata_frac,
                "land_frac": frac_land,
                "labels": chip_labels,
            }
            (out_dir / f"{stem}.json").write_text(json.dumps(sidecar, indent=1))

            records.append(
                ChipRecord(
                    chip_path=str(chip_path),
                    scene_id=scene_id,
                    row0=row0,
                    col0=col0,
                    n_vessels=n_vessels,
                    has_low_conf=has_low_conf,
                    land_frac=frac_land,
                )
            )
    return records


def scene_registry_entry(scene_dir: Path, n_labels: int, has_shoreline: bool | None) -> dict[str, object]:
    """Per-scene metadata row for the splits builder (P1.5 stratification)."""

    import rasterio
    import rasterio.warp

    with rasterio.open(scene_dir / "VH_dB.tif") as ds:
        bounds = ds.bounds
        lon, lat = rasterio.warp.transform(
            ds.crs,
            "EPSG:4326",
            [(bounds.left + bounds.right) / 2],
            [(bounds.top + bounds.bottom) / 2],
        )
        return {
            "scene_id": scene_dir.name,
            "height": ds.height,
            "width": ds.width,
            "crs": str(ds.crs),
            "center_lon": lon[0],
            "center_lat": lat[0],
            "n_labels": n_labels,
            "has_shoreline": has_shoreline,
        }


def load_scene_labels(labels_csv: Path, scene_id: str) -> list[dict[str, object]]:
    """Rows of the xView3 label CSV for one scene, as plain dicts."""

    import pandas as pd

    table = pd.read_csv(labels_csv)
    subset = table[table["scene_id"] == scene_id]
    return subset.to_dict(orient="records")


def main(argv: Sequence[str] | None = None) -> int:
    import pandas as pd
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="scene ids to chip (default: every directory under raw GRD)",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="xView3 label CSV; omitted -> imagery-only chips (empty sidecars)",
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="also chip eval_final (…v) scenes; by default they are only "
        "registered — their pixels are read at final eval (ground rule 4)",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    raw_root = Path(config["paths"]["raw_xview3"]) / "GRD"
    chips_root = Path(config["paths"]["chips"])
    manifest_root = Path(config["paths"]["manifests"])
    manifest_root.mkdir(parents=True, exist_ok=True)
    chip_cfg = config["chip"]

    scene_dirs = (
        [raw_root / scene for scene in args.scenes]
        if args.scenes
        else sorted(d for d in raw_root.iterdir() if d.is_dir())
    )

    from src.data.splits import scene_pool

    labels = pd.read_csv(args.labels_csv) if args.labels_csv else None
    registry_rows = []
    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        scene_labels: list[dict[str, object]] = []
        if labels is not None:
            scene_labels = labels[labels["scene_id"] == scene_id].to_dict(
                orient="records"
            )
        if scene_pool(scene_id) == "train_pool" or args.include_eval:
            records = chip_scene(
                scene_dir,
                chips_root / scene_id,
                scene_labels=scene_labels,
                chip_size=chip_cfg["size_px"],
                overlap=chip_cfg["overlap_px"],
                max_nodata_frac=chip_cfg["max_nodata_frac"],
            )
            pd.DataFrame([record.__dict__ for record in records]).to_parquet(
                manifest_root / f"{scene_id}.parquet"
            )
            print(f"{scene_id}: {len(records)} chips")
        else:
            print(f"{scene_id}: eval_final — registered, not chipped")
        has_shoreline = None
        if scene_labels:
            shore_km = config["splits"]["shore_km_threshold"]
            has_shoreline = any(
                label.get("distance_from_shore_km") is not None
                and not (
                    isinstance(label["distance_from_shore_km"], float)
                    and math.isnan(label["distance_from_shore_km"])
                )
                and float(label["distance_from_shore_km"]) < shore_km
                for label in scene_labels
            )
        registry_rows.append(
            scene_registry_entry(scene_dir, len(scene_labels), has_shoreline)
        )

    registry_path = manifest_root / "scenes.parquet"
    if registry_path.exists():
        existing = pd.read_parquet(registry_path)
        merged = pd.concat(
            [
                existing[~existing["scene_id"].isin([r["scene_id"] for r in registry_rows])],
                pd.DataFrame(registry_rows),
            ],
            ignore_index=True,
        )
    else:
        merged = pd.DataFrame(registry_rows)
    merged.to_parquet(registry_path)
    print(f"scene registry: {len(merged)} scenes -> {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
