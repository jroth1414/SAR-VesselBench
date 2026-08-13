"""Train-time augmentation on dB chips.

Random 512 crop from the 800 chip (vessel-biased: 70% of crops centered
within 128 px of a vessel when one exists), flips, 90-degree rotations, and
intensity jitter of +-1 dB. Nothing that breaks SAR statistics (no blur, no
elastic).

Unit note (documented): the plan writes "intensity jitter +-0.1 in log
space"; chips are stored in dB = 10*log10, so +-0.1 log10-units == +-1 dB.
Jitter is applied in dB BEFORE normalization.
"""

from __future__ import annotations

import numpy as np

CROP_PX = 512
VESSEL_BIASED_FRAC = 0.7
VESSEL_CENTER_RADIUS_PX = 128
INTENSITY_JITTER_DB = 1.0


def random_crop_origin(
    rng: np.random.Generator,
    chip_size: int,
    crop: int,
    vessel_rowcols: list[tuple[float, float]],
    *,
    vessel_biased_frac: float = VESSEL_BIASED_FRAC,
    center_radius: int = VESSEL_CENTER_RADIUS_PX,
) -> tuple[int, int]:
    """(row0, col0) of the crop window inside the chip."""

    max_origin = chip_size - crop
    if max_origin <= 0:
        return 0, 0
    if vessel_rowcols and rng.random() < vessel_biased_frac:
        row, col = vessel_rowcols[rng.integers(len(vessel_rowcols))]
        center_row = row + rng.uniform(-center_radius, center_radius)
        center_col = col + rng.uniform(-center_radius, center_radius)
        row0 = int(np.clip(center_row - crop / 2, 0, max_origin))
        col0 = int(np.clip(center_col - crop / 2, 0, max_origin))
        return row0, col0
    return int(rng.integers(max_origin + 1)), int(rng.integers(max_origin + 1))


def apply_flips_rot90(
    rng: np.random.Generator,
    image: np.ndarray,
    rowcols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Random H/V flip + k*90-degree rotation of (C, H, W) image + points.

    ``rowcols`` is (N, 2) in crop pixels. Returns the transformed image,
    points, and the op record (for reproducibility/debugging).
    """

    size = image.shape[-1]
    points = rowcols.copy().astype(np.float64).reshape(-1, 2)
    ops = {
        "hflip": int(rng.random() < 0.5),
        "vflip": int(rng.random() < 0.5),
        "rot90": int(rng.integers(4)),
    }
    if ops["hflip"]:
        image = image[..., ::-1]
        points[:, 1] = size - 1 - points[:, 1]
    if ops["vflip"]:
        image = image[..., ::-1, :]
        points[:, 0] = size - 1 - points[:, 0]
    for _ in range(ops["rot90"]):
        image = np.rot90(image, axes=(-2, -1))
        points = np.stack([size - 1 - points[:, 1], points[:, 0]], axis=1)
    return np.ascontiguousarray(image), points, ops


def intensity_jitter(
    rng: np.random.Generator, image_db: np.ndarray, *, jitter_db: float = INTENSITY_JITTER_DB
) -> np.ndarray:
    """Additive per-crop intensity offset in dB (applied before normalization)."""

    return image_db + rng.uniform(-jitter_db, jitter_db)


def normalize(image_db: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(x - mean) / std per channel; NaN (nodata) becomes 0 after normalization."""

    normalized = (image_db - mean[:, None, None]) / std[:, None, None]
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
