from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.data.datasets import nested_fraction_scenes  # noqa: E402
from src.data.transforms import apply_flips_rot90, random_crop_origin  # noqa: E402
from src.train.losses import (  # noqa: E402
    gaussian_radius_centers,
    penalty_reduced_focal_loss,
    render_target,
)


def test_render_target_peak_and_ignore_mask():
    heatmap, mask = render_target(128, [(30.0, 40.0)], [(90.0, 100.0)])
    assert float(heatmap.max()) == pytest.approx(1.0)
    assert divmod(int(heatmap.argmax()), 128) == (30, 40)
    assert mask[90, 100] == 0.0 and mask[90, 103] == 0.0  # radius 3 disk
    assert mask[90, 104] == 1.0


def test_render_target_fractional_center_still_peaks_at_one():
    # CenterNet quantizes centers to integer pixels; a fractional center must
    # still produce a peak of EXACTLY 1.0 (else the focal positive mask is
    # empty and no object ever gets a positive loss term).
    heatmap, _ = render_target(64, [(30.94, 40.37)])
    assert float(heatmap.max()) == pytest.approx(1.0)
    assert divmod(int(heatmap.argmax()), 64) == (30, 40)


def test_focal_loss_prefers_correct_prediction():
    heatmap, mask = render_target(64, [(20.0, 20.0)])
    good = torch.full((1, 64, 64), -4.0)
    good[0, 20, 20] = 6.0
    flat = torch.zeros(1, 64, 64)
    assert penalty_reduced_focal_loss(good, heatmap[None], mask[None]) < (
        penalty_reduced_focal_loss(flat, heatmap[None], mask[None])
    )


def test_gaussian_radius_centers_protocol():
    labels = [
        {"chip_row": 40.0, "chip_col": 80.0, "is_vessel": True, "confidence": "MEDIUM"},
        {"chip_row": 8.0, "chip_col": 8.0, "is_vessel": None, "confidence": "LOW"},
        {"chip_row": 12.0, "chip_col": 12.0, "is_vessel": False, "confidence": "HIGH"},
    ]
    positives, ignores = gaussian_radius_centers(labels, input_stride=4)
    assert positives == [(10.0, 20.0)]  # only the MEDIUM vessel, /4
    assert ignores == [(2.0, 2.0)]  # LOW -> ignore; non-vessel HIGH -> background


def test_flips_rot90_keep_points_on_pixels():
    rng = np.random.default_rng(0)
    image = np.zeros((1, 64, 64), dtype=np.float32)
    image[0, 10, 20] = 7.0
    points = np.array([[10.0, 20.0]])
    for _ in range(20):
        out_img, out_pts, _ = apply_flips_rot90(rng, image.copy(), points.copy())
        row, col = int(round(out_pts[0, 0])), int(round(out_pts[0, 1]))
        assert out_img[0, row, col] == 7.0  # point tracks the pixel exactly


def test_vessel_biased_crop_contains_vessel_most_of_the_time():
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(200):
        row0, col0 = random_crop_origin(rng, 800, 512, [(400.0, 400.0)])
        hits += (row0 <= 400 < row0 + 512) and (col0 <= 400 < col0 + 512)
    assert hits >= 150  # 70% biased path + random-path chance


def test_nested_fraction_scenes_nest_and_are_deterministic():
    scenes = [f"{i:016x}t" for i in range(100)]
    f10 = nested_fraction_scenes(scenes, 0.10, frac_seed=0)
    f25 = nested_fraction_scenes(scenes, 0.25, frac_seed=0)
    f50 = nested_fraction_scenes(scenes, 0.50, frac_seed=0)
    assert len(f10) == 10 and len(f25) == 25 and len(f50) == 50
    assert set(f10) <= set(f25) <= set(f50)
    assert f10 == nested_fraction_scenes(scenes, 0.10, frac_seed=0)
    assert f10 != nested_fraction_scenes(scenes, 0.10, frac_seed=1)
    with pytest.raises(ValueError):
        nested_fraction_scenes(scenes, 0.0, frac_seed=0)
