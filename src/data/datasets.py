"""Training datasets (DEVPLAN P3.4/P3.5): active xView3 + legacy LS-SSDD.

``FineTuneDataset`` is shared by all eight current arms. The historical
``SupervisedSARDataset`` emits the same sample contract and is retained only
to reproduce superseded LS-SSDD experiments; no current arm consumes it.

Fairness guards baked in:
- Scene membership is asserted at construction (ground rule 3): a dataset
  refuses chips whose scene is not in its split, and refuses eval_final.
- Label fractions NEST (10% of scenes is a subset of 25%, etc.): the seeded
  permutation uses the frozen data-config seed, NOT the per-run seed, so the
  same scene subsets are reused by every arm and both tracks.

Normalization note (documented): the frozen ``data/stats.json`` holds
per-polarization (VH, VV) stats. The third channel (VH-VV) is normalized with
the derived mean (muVH - muVV) and std sqrt(sVH^2 + sVV^2); the independence
approximation only scales one channel by a constant shared by every arm, so
it can not become a between-arm confound.

Legacy LS-SSDD note: sub-images are single-channel 8-bit JPGs. The
gray value stands in for both polarizations (VH = VV = x, so VH-VV = 0), and
per-dataset mean/std (computed once over the frozen internal train split and
cached to ``data/lsssdd_stats.json``) normalizes it to the same N(0,1) scale
as the xView3 input.
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


class SupervisedSARDataset(Dataset):
    """LS-SSDD 800px sub-images -> the same sample dict (Arms 4/8 pretraining)."""

    def __init__(
        self,
        *,
        lsssdd_root: str | Path,
        split_path: str | Path,
        part: str,
        crop: int = CROP_PX,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        root = Path(lsssdd_root)
        license_note = root / "LICENSE.note"
        if not license_note.exists():
            raise FileNotFoundError(
                f"{license_note} missing — refusing to use LS-SSDD without its "
                "recorded license (P1.2 gate)"
            )
        split = json.loads(Path(split_path).read_text())
        if part not in ("train", "val"):
            raise ValueError(f"part must be train/val, got {part!r}")
        names = set(split[part])

        image_paths = {
            path.name: path
            for path in root.rglob("*.jpg")
            if path.name in names
        }
        missing = names - set(image_paths)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} LS-SSDD sub-images in the frozen split are "
                f"missing on disk (first: {sorted(missing)[:3]})"
            )
        self.items = [image_paths[name] for name in sorted(names)]
        annotation_dirs = [d for d in root.rglob("Annotations_sub") if d.is_dir()]
        self.annotation_dir = annotation_dirs[0] if annotation_dirs else None

        self.stats = self._load_or_compute_stats(root, split["train"], image_paths)
        self.crop = crop
        self.augment = augment
        self.seed = seed
        self._rng: np.random.Generator | None = None
        self.foreground = [
            len(self._centroids(path)) > 0 for path in self.items
        ]

    def _worker_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng(
                (self.seed, torch.initial_seed() % 2**31)
            )
        return self._rng

    def _load_or_compute_stats(self, root: Path, train_names, image_paths) -> dict:
        cache = Path("data/lsssdd_stats.json")
        if cache.exists():
            return json.loads(cache.read_text())
        from PIL import Image

        rng = np.random.default_rng(0)
        sample_names = list(train_names)
        rng.shuffle(sample_names)
        values = []
        for name in sample_names[:500]:
            if name in image_paths:
                values.append(np.asarray(Image.open(image_paths[name]).convert("L"), dtype=np.float32))
        stack = np.stack(values)
        stats = {"mean": float(stack.mean()), "std": float(stack.std())}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(stats, indent=1), newline="\n")
        return stats

    def _centroids(self, image_path: Path) -> list[tuple[float, float]]:
        from src.data.to_centroids import parse_voc_centroids

        if self.annotation_dir is None:
            return []
        xml_path = self.annotation_dir / f"{image_path.stem}.xml"
        if not xml_path.exists():
            return []
        return [(c.y, c.x) for c in parse_voc_centroids(xml_path)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        rng = self._worker_rng()
        image_path = self.items[index]
        gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
        gray = (gray - self.stats["mean"]) / max(self.stats["std"], 1e-6)

        centroids = self._centroids(image_path)  # (row, col) in 800px coords
        if self.augment:
            row0, col0 = random_crop_origin(
                rng, gray.shape[-1], self.crop, centroids
            )
        else:
            row0 = col0 = max(0, (gray.shape[-1] - self.crop) // 2)
        window = gray[None, row0 : row0 + self.crop, col0 : col0 + self.crop]

        points = np.array(
            [
                (row - row0, col - col0)
                for row, col in centroids
                if 0 <= row - row0 < self.crop and 0 <= col - col0 < self.crop
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        if self.augment:
            window, points, _ = apply_flips_rot90(rng, window, points)

        # Single-pol stand-in for the fixed channel rep: VH = VV = x -> VH-VV = 0.
        image = np.concatenate([window, window, np.zeros_like(window)], axis=0)

        positives = [(row / OUTPUT_STRIDE, col / OUTPUT_STRIDE) for row, col in points]
        heatmap, mask = render_target(self.crop // OUTPUT_STRIDE, positives, ())

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "heatmap": heatmap,
            "mask": mask,
            "n_positives": len(positives),
        }
