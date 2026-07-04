"""Scene-level split builder + frozen data artifacts (DEVPLAN P1.5).

Owns three frozen-at-sprint-1-merge artifacts:

- ``data/splits.json``   — scene-level train/dev/test/eval_final membership,
  stratified by coarse region (k-means bins over scene-center lon/lat) and
  shoreline presence. Stored as per-split lists so the ``test_split_disjoint``
  guard checks real disjointness.
- ``data/stats.json``    — per-polarization mean/std computed ONCE over the
  train-split chips only, reused unchanged for every label fraction, both
  tracks, and all seeds.
- ``data/lsssdd_split.json`` — the seeded LS-SSDD internal train/val
  partition consumed identically by the Arm-4 and Arm-8 pretrainings.

Scene-pool convention: xView3 scene ids end in ``t`` (train pool — split
75/15/10 here) or ``v`` (the human-verified validation scenes -> eval_final,
excluded from everything, touched exactly once).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

SPLIT_NAMES = ("train", "dev", "test", "eval_final")


def scene_pool(scene_id: str) -> str:
    """'train_pool' for ...t scenes, 'eval_final' for ...v scenes."""

    if scene_id.endswith("t"):
        return "train_pool"
    if scene_id.endswith("v"):
        return "eval_final"
    raise ValueError(f"unrecognized scene-id suffix (not t/v): {scene_id}")


def region_bins(lonlats: np.ndarray, *, n_bins: int, seed: int) -> np.ndarray:
    """Cluster scene centers into coarse region bins (seeded k-means)."""

    from scipy.cluster.vq import kmeans2

    points = np.asarray(lonlats, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"lonlats must be (n, 2), got {points.shape}")
    k = min(n_bins, len(points))
    if k <= 1:
        return np.zeros(len(points), dtype=int)
    _, labels = kmeans2(points, k, seed=seed, minit="++")
    return labels


def build_splits(
    scene_records: Sequence[Mapping[str, object]],
    *,
    fractions: Mapping[str, float],
    seed: int,
    n_region_bins: int = 6,
) -> dict[str, list[str]]:
    """Stratified scene-level split of the xView3 train pool.

    ``scene_records`` need ``scene_id``, ``center_lon``, ``center_lat`` and
    optionally ``has_shoreline``. Scenes whose id ends in ``v`` go straight to
    ``eval_final``. Within each (region-bin, shoreline) stratum the seeded
    order is split by the train/dev/test fractions (largest-remainder), so
    membership is deterministic in (records, fractions, seed).
    """

    keys = ("train_frac", "dev_frac", "test_frac")
    fracs = np.array([float(fractions[key]) for key in keys])
    if not np.isclose(fracs.sum(), 1.0):
        raise ValueError(f"train/dev/test fractions must sum to 1, got {fracs}")

    eval_final = sorted(
        str(record["scene_id"])
        for record in scene_records
        if scene_pool(str(record["scene_id"])) == "eval_final"
    )
    pool = sorted(
        (record for record in scene_records
         if scene_pool(str(record["scene_id"])) == "train_pool"),
        key=lambda record: str(record["scene_id"]),
    )
    if not pool:
        return {"train": [], "dev": [], "test": [], "eval_final": eval_final}

    lonlats = np.array(
        [[float(r["center_lon"]), float(r["center_lat"])] for r in pool]
    )
    bins = region_bins(lonlats, n_bins=n_region_bins, seed=seed)

    strata: dict[tuple[int, bool], list[str]] = {}
    for record, region in zip(pool, bins):
        shoreline = bool(record.get("has_shoreline") or False)
        strata.setdefault((int(region), shoreline), []).append(
            str(record["scene_id"])
        )

    rng = np.random.default_rng(seed)
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_NAMES}
    splits["eval_final"] = eval_final

    for stratum_key in sorted(strata):
        ids = sorted(strata[stratum_key])
        rng.shuffle(ids)
        counts = _largest_remainder(len(ids), fracs)
        cursor = 0
        for split_name, count in zip(("train", "dev", "test"), counts):
            splits[split_name].extend(ids[cursor : cursor + count])
            cursor += count

    # Tiny-pool fixup: keep dev and test non-empty when there are >= 3 scenes,
    # taking from train (the largest split) deterministically.
    if len(pool) >= 3:
        for needy in ("dev", "test"):
            if not splits[needy] and len(splits["train"]) > 1:
                splits[needy].append(splits["train"].pop())

    for name in SPLIT_NAMES:
        splits[name] = sorted(splits[name])
    return splits


def _largest_remainder(total: int, fractions: np.ndarray) -> list[int]:
    raw = fractions * total
    counts = np.floor(raw).astype(int)
    remainder_order = np.argsort(-(raw - counts))
    for index in remainder_order[: total - counts.sum()]:
        counts[index] += 1
    return counts.tolist()


def build_lsssdd_split(
    sub_image_names: Iterable[str], *, train_frac: float, seed: int
) -> dict[str, list[str]]:
    """Seeded internal train/val partition of LS-SSDD sub-images (P1.5)."""

    names = sorted(set(str(name) for name in sub_image_names))
    if not names:
        raise ValueError("no LS-SSDD sub-image names supplied")
    rng = np.random.default_rng(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_train = int(round(train_frac * len(shuffled)))
    n_train = min(max(n_train, 1), len(shuffled) - 1) if len(shuffled) > 1 else 1
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:]),
    }


def compute_channel_stats(chip_paths: Sequence[Path]) -> dict[str, object]:
    """Per-polarization mean/std over float16 chips (NaN = nodata, excluded).

    Accumulates in float64. Chip layout is (2, H, W) with channels [VH, VV]
    (see chipper.py). Returns the stats plus the contributing scene ids so
    ``test_splits`` can assert they are a subset of the train split.
    """

    channel_names = ("VH", "VV")
    count = np.zeros(2, dtype=np.int64)
    total = np.zeros(2, dtype=np.float64)
    total_sq = np.zeros(2, dtype=np.float64)
    scenes: set[str] = set()

    for chip_path in chip_paths:
        chips = np.load(chip_path).astype(np.float64)
        if chips.ndim != 3 or chips.shape[0] != 2:
            raise ValueError(f"{chip_path}: expected (2, H, W) chip")
        scenes.add(Path(chip_path).name.split("_r")[0])
        for channel in range(2):
            values = chips[channel]
            valid = np.isfinite(values)
            count[channel] += int(valid.sum())
            total[channel] += float(values[valid].sum())
            total_sq[channel] += float((values[valid] ** 2).sum())

    if (count == 0).any():
        raise ValueError("no valid pixels accumulated for at least one channel")
    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 0.0)
    return {
        "channels": {
            name: {"mean": float(mean[i]), "std": float(np.sqrt(variance[i]))}
            for i, name in enumerate(channel_names)
        },
        "n_pixels": [int(n) for n in count],
        "scenes": sorted(scenes),
    }


def scene_split_map(splits: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Invert per-split lists to the scene_id -> split mapping."""

    mapping: dict[str, str] = {}
    for split_name, scene_ids in splits.items():
        for scene_id in scene_ids:
            if scene_id in mapping:
                raise ValueError(
                    f"{scene_id} appears in both {mapping[scene_id]} and {split_name}"
                )
            mapping[str(scene_id)] = str(split_name)
    return mapping


def main(argv: Sequence[str] | None = None) -> int:
    import pandas as pd
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument(
        "--build-stats",
        action="store_true",
        help="also compute data/stats.json over the train-split chips",
    )
    parser.add_argument(
        "--lsssdd-root",
        default=None,
        help="LS-SSDD sub-image root; when given, also writes data/lsssdd_split.json",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="overwrite existing artifacts (they freeze at sprint-1 merge; "
        "overwriting a frozen artifact is a STOP)",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    seed = int(config["seed"])
    paths = config["paths"]

    splits_path = Path(paths["splits"])
    if splits_path.exists() and not args.allow_overwrite:
        raise SystemExit(
            f"{splits_path} already exists and freezes at sprint-1 merge; "
            "pass --allow-overwrite only if it is not yet frozen (see DEVPLAN)."
        )

    registry = pd.read_parquet(Path(paths["manifests"]) / "scenes.parquet")
    splits = build_splits(
        registry.to_dict(orient="records"),
        fractions=config["splits"],
        seed=seed,
        n_region_bins=int(config["splits"]["n_region_bins"]),
    )
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(
        json.dumps({"meta": {"seed": seed, "fractions": config["splits"]}, "splits": splits}, indent=1)
    )
    print({name: len(ids) for name, ids in splits.items()}, "->", splits_path)

    if args.build_stats:
        chips_root = Path(paths["chips"])
        train_chips = [
            path
            for scene_id in splits["train"]
            for path in sorted((chips_root / scene_id).glob("*.npy"))
        ]
        if not train_chips:
            raise SystemExit("no train-split chips found; run the chipper first")
        stats = compute_channel_stats(train_chips)
        stats_path = Path(paths["stats"])
        if stats_path.exists() and not args.allow_overwrite:
            raise SystemExit(f"{stats_path} already exists (frozen artifact)")
        stats_path.write_text(json.dumps(stats, indent=1))
        print(f"stats over {len(train_chips)} train chips -> {stats_path}")

    if args.lsssdd_root:
        names = sorted(
            path.name
            for path in Path(args.lsssdd_root).rglob("*.jpg")
        )
        lsssdd = build_lsssdd_split(
            names, train_frac=float(config["lsssdd_split"]["train_frac"]), seed=seed
        )
        lsssdd_path = Path(paths["lsssdd_split"])
        if lsssdd_path.exists() and not args.allow_overwrite:
            raise SystemExit(f"{lsssdd_path} already exists (frozen artifact)")
        lsssdd_path.write_text(
            json.dumps({"meta": {"seed": seed}, **lsssdd}, indent=1)
        )
        print(
            f"lsssdd split: {len(lsssdd['train'])} train / {len(lsssdd['val'])} val -> {lsssdd_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
