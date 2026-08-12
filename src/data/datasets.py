"""Active xView3 training dataset shared by all eight study arms.

Scene membership is asserted at construction, and the fixed seeded permutation
makes the 10%, 25%, 50%, and 100% training-scene subsets strictly nested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.transforms import (
    CROP_PX,
    apply_flips_rot90,
    intensity_jitter,
    normalize,
    random_crop_origin,
)
from src.train.losses import gaussian_radius_centers, render_target

OUTPUT_STRIDE = 4


def nested_fraction_scenes(
    train_scenes: Sequence[str], label_frac: float, *, frac_seed: int
) -> list[str]:
    """First ceil(frac * n) scenes of ONE fixed seeded permutation (nesting)."""

    if not 0.0 < label_frac <= 1.0:
        raise ValueError(f"label_frac must be in (0, 1], got {label_frac}")
    ordered = sorted(train_scenes)
    rng = np.random.default_rng(frac_seed)
    rng.shuffle(ordered)
    keep = int(np.ceil(label_frac * len(ordered)))
    return sorted(ordered[:keep])


def _channel_stats(stats: dict) -> tuple[np.ndarray, np.ndarray]:
    vh = stats["channels"]["VH"]
    vv = stats["channels"]["VV"]
    mean = np.array([vh["mean"], vv["mean"], vh["mean"] - vv["mean"]], dtype=np.float32)
    std = np.array(
        [vh["std"], vv["std"], float(np.hypot(vh["std"], vv["std"]))],
        dtype=np.float32,
    )
    return mean, std


class FineTuneDataset(Dataset):
    """800px labeled xView3 chips -> augmented 512 crops + stride-4 targets."""

    def __init__(
        self,
        *,
        chips_root: str | Path,
        splits_path: str | Path,
        stats_path: str | Path,
        split: str,
        label_frac: float = 1.0,
        frac_seed: int = 0,
        crop: int = CROP_PX,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        splits = json.loads(Path(splits_path).read_text())["splits"]
        if split not in ("train", "dev", "test"):
            raise ValueError(
                f"FineTuneDataset refuses split {split!r} — eval_final is "
                "touched exactly once by final_eval.py (ground rule 4)"
            )
        scenes = splits[split]
        if split == "train" and label_frac < 1.0:
            scenes = nested_fraction_scenes(scenes, label_frac, frac_seed=frac_seed)
        self.scenes = set(scenes)

        chips_root = Path(chips_root)
        self.chip_paths: list[Path] = []
        self.foreground: list[bool] = []
        for scene_id in sorted(self.scenes):
            scene_dir = chips_root / scene_id
            if not scene_dir.exists():
                raise FileNotFoundError(
                    f"{split}: no chips for scene {scene_id} under {chips_root}"
                )
            for chip_path in sorted(scene_dir.glob("*.npy")):
                if chip_path.name.split("_r")[0] not in self.scenes:
                    raise AssertionError(
                        f"membership violation: {chip_path} not in split {split}"
                    )
                self.chip_paths.append(chip_path)
                sidecar = json.loads(chip_path.with_suffix(".json").read_text())
                self.foreground.append(
                    any(
                        bool(l.get("is_vessel"))
                        and str(l.get("confidence") or "").upper() in ("HIGH", "MEDIUM")
                        for l in sidecar["labels"]
                    )
                )

        stats = json.loads(Path(stats_path).read_text())
        self.mean, self.std = _channel_stats(stats)
        self.crop = crop
        self.augment = augment
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def _worker_rng(self) -> np.random.Generator:
        # Lazy per-worker generator: seeded from (run seed, worker seed) once,
        # then its state ADVANCES across epochs so augmentations differ per
        # epoch while staying deterministic for a fixed run seed.
        if self._rng is None:
            self._rng = np.random.default_rng(
                (self.seed, torch.initial_seed() % 2**31)
            )
        return self._rng

    def __len__(self) -> int:
        return len(self.chip_paths)

    def __getitem__(self, index: int) -> dict:
        rng = self._worker_rng()
        chip_path = self.chip_paths[index]
        chips = np.load(chip_path).astype(np.float32)  # (2, H, W) dB, NaN nodata
        sidecar = json.loads(chip_path.with_suffix(".json").read_text())
        labels = sidecar["labels"]

        vessel_rowcols = [
            (float(l["chip_row"]), float(l["chip_col"]))
            for l in labels
            if bool(l.get("is_vessel"))
            and str(l.get("confidence") or "").upper() in ("HIGH", "MEDIUM")
        ]

        if self.augment:
            row0, col0 = random_crop_origin(
                rng, chips.shape[-1], self.crop, vessel_rowcols
            )
        else:
            row0 = col0 = max(0, (chips.shape[-1] - self.crop) // 2)
        window = chips[:, row0 : row0 + self.crop, col0 : col0 + self.crop]

        crop_labels = []
        for label in labels:
            row = float(label["chip_row"]) - row0
            col = float(label["chip_col"]) - col0
            if 0 <= row < self.crop and 0 <= col < self.crop:
                crop_labels.append({**label, "chip_row": row, "chip_col": col})

        points = np.array(
            [[l["chip_row"], l["chip_col"]] for l in crop_labels], dtype=np.float64
        ).reshape(-1, 2)
        if self.augment:
            window, points, _ = apply_flips_rot90(rng, window, points)
            window = intensity_jitter(rng, window)
        for label, (row, col) in zip(crop_labels, points):
            label["chip_row"], label["chip_col"] = float(row), float(col)

        image = np.concatenate([window, window[0:1] - window[1:2]], axis=0)
        image = normalize(image, self.mean, self.std)

        positives, ignores = gaussian_radius_centers(
            crop_labels, input_stride=OUTPUT_STRIDE
        )
        heatmap, mask = render_target(self.crop // OUTPUT_STRIDE, positives, ignores)

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "heatmap": heatmap,
            "mask": mask,
            "n_positives": len(positives),
        }
