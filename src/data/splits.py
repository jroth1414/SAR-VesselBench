"""Scene-level split builder + frozen data artifacts.

Owns three frozen-at-sprint-1-merge artifacts:

- ``data/splits.json``   — scene-level train/dev/test/eval_final membership,
  stratified by coarse region (k-means bins over scene-center lon/lat) and
  shoreline presence. Stored as per-split lists so the ``test_split_disjoint``
  guard checks real disjointness.
- ``data/stats.json``    — per-polarization mean/std computed ONCE over the
  train-split chips only, reused unchanged for every label fraction, both
  tracks, and all seeds.
- ``data/lsssdd_split.json`` — immutable historical provenance only; this
  module does not regenerate or consume it.

Scene-pool convention: xView3 scene ids end in ``t`` (train pool — split
75/15/10 here) or ``v`` (the human-verified validation scenes -> eval_final,
excluded from everything, touched exactly once).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

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


def scene_records_from_labels(
    labels_by_scene: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    shore_km_threshold: float = 5.0,
) -> list[dict[str, object]]:
    """Per-scene stratification records derived from the label CSV alone.

    Scene center is approximated by the mean of the scene's label lat/lons
    (documented proxy — avoids extracting 554 rasters just to read georefs;
    labels span each scene, so their centroid is a fine input for COARSE
    region binning). ``has_shoreline`` uses the plan's <5 km rule; the
    dataset's 9999.99 far-from-shore sentinel is naturally excluded.
    """

    records = []
    for scene_id, labels in labels_by_scene.items():
        lats = [float(l["detect_lat"]) for l in labels if _isnum(l.get("detect_lat"))]
        lons = [float(l["detect_lon"]) for l in labels if _isnum(l.get("detect_lon"))]
        if not lats or not lons:
            raise ValueError(f"{scene_id}: no usable label coordinates")
        records.append(
            {
                "scene_id": str(scene_id),
                "center_lon": float(np.mean(lons)),
                "center_lat": float(np.mean(lats)),
                "has_shoreline": any(
                    _isnum(l.get("distance_from_shore_km"))
                    and float(l["distance_from_shore_km"]) < shore_km_threshold
                    for l in labels
                ),
                "n_labels": len(labels),
            }
        )
    return sorted(records, key=lambda r: r["scene_id"])


def _isnum(value: object) -> bool:
    if value is None:
        return False
    try:
        return not np.isnan(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def select_scenes(
    scene_records: Sequence[Mapping[str, object]],
    *,
    n_select: int,
    seed: int,
    n_region_bins: int = 6,
) -> list[dict[str, object]]:
    """Stratified, seeded selection of the study's train-pool scenes (P1.3/P1.5).

    Proportional allocation over (region-bin, shoreline) strata with largest
    remainder, seeded shuffle inside each stratum. Deterministic in
    (records, n_select, seed).
    """

    pool = sorted(
        (dict(record) for record in scene_records),
        key=lambda record: str(record["scene_id"]),
    )
    if n_select >= len(pool):
        return pool

    lonlats = np.array(
        [[float(r["center_lon"]), float(r["center_lat"])] for r in pool]
    )
    bins = region_bins(lonlats, n_bins=n_region_bins, seed=seed)
    strata: dict[tuple[int, bool], list[dict[str, object]]] = {}
    for record, region in zip(pool, bins):
        key = (int(region), bool(record.get("has_shoreline") or False))
        strata.setdefault(key, []).append(record)

    stratum_keys = sorted(strata)
    sizes = np.array([len(strata[key]) for key in stratum_keys], dtype=float)
    quotas = _largest_remainder_quota(sizes / sizes.sum(), n_select, caps=sizes.astype(int))

    rng = np.random.default_rng(seed)
    selected: list[dict[str, object]] = []
    for key, quota in zip(stratum_keys, quotas):
        members = sorted(strata[key], key=lambda record: str(record["scene_id"]))
        rng.shuffle(members)
        selected.extend(members[:quota])
    return sorted(selected, key=lambda record: str(record["scene_id"]))


def _largest_remainder_quota(
    fractions: np.ndarray, total: int, *, caps: np.ndarray
) -> list[int]:
    raw = fractions * total
    quotas = np.minimum(np.floor(raw).astype(int), caps)
    while quotas.sum() < total:
        remainders = np.where(quotas < caps, raw - quotas, -np.inf)
        index = int(np.argmax(remainders))
        if not np.isfinite(remainders[index]):
            break
        quotas[index] += 1
    return quotas.tolist()


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
        "--from-labels",
        default=None,
        metavar="TRAIN_CSV",
        help="derive stratification records from the xView3 train label CSV "
        "(label-centroid scene centers) instead of the chipped scene registry",
    )
    parser.add_argument(
        "--validation-labels",
        default=None,
        metavar="VAL_CSV",
        help="validation label CSV; its scene ids form eval_final (with --from-labels)",
    )
    parser.add_argument(
        "--n-scenes",
        type=int,
        default=None,
        help="stratified-select this many train-pool scenes before splitting "
        "(the study scene-count decision; recorded in the artifact meta)",
    )
    parser.add_argument(
        "--build-stats",
        action="store_true",
        help="also compute data/stats.json over the train-split chips",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="only compute data/stats.json from the EXISTING splits.json "
        "(used after the long chipping job; does not touch splits.json)",
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

    if not args.stats_only:
        if splits_path.exists() and not args.allow_overwrite:
            raise SystemExit(
                f"{splits_path} already exists and freezes at sprint-1 merge; "
                "pass --allow-overwrite only if it is not yet frozen."
            )

        if args.from_labels:
            labels = pd.read_csv(args.from_labels)
            records = scene_records_from_labels(
                {
                    scene_id: group.to_dict(orient="records")
                    for scene_id, group in labels.groupby("scene_id")
                },
                shore_km_threshold=float(config["splits"]["shore_km_threshold"]),
            )
            if args.n_scenes:
                records = select_scenes(
                    records,
                    n_select=args.n_scenes,
                    seed=seed,
                    n_region_bins=int(config["splits"]["n_region_bins"]),
                )
            if args.validation_labels:
                val_scenes = sorted(
                    pd.read_csv(args.validation_labels)["scene_id"].unique()
                )
                records = records + [
                    {"scene_id": scene_id, "center_lon": 0.0, "center_lat": 0.0}
                    for scene_id in val_scenes
                ]
        else:
            registry = pd.read_parquet(Path(paths["manifests"]) / "scenes.parquet")
            records = registry.to_dict(orient="records")

        splits = build_splits(
            records,
            fractions=config["splits"],
            seed=seed,
            n_region_bins=int(config["splits"]["n_region_bins"]),
        )
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "seed": seed,
            "fractions": config["splits"],
            "n_scenes_selected": args.n_scenes,
            "selection_source": (
                f"labels:{args.from_labels}" if args.from_labels else "scene registry"
            ),
        }
        # newline="\n" everywhere an artifact is written: these files are
        # sha256-pinned, and Windows CRLF translation would silently change
        # the hash relative to the LF-normalized repo copy.
        splits_path.write_text(
            json.dumps({"meta": meta, "splits": splits}, indent=1), newline="\n"
        )
        print({name: len(ids) for name, ids in splits.items()}, "->", splits_path)
    else:
        splits = json.loads(splits_path.read_text())["splits"] if splits_path.exists() else None

    if args.build_stats or args.stats_only:
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
        stats_path.write_text(json.dumps(stats, indent=1), newline="\n")
        print(f"stats over {len(train_chips)} train chips -> {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
