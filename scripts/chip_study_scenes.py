"""Extract + chip every train/dev/test scene in the frozen split (P1.3 at scale).

Resumable driver for the one-time study-set chipping run: for each scene in
``data/splits.json`` (train, dev, test), extract its archive if needed, chip
it with labels, and write the per-scene manifest. Scenes with an existing
manifest are skipped, so the job can be killed and relaunched freely.
Extracted rasters are kept — whole-scene tiled inference (P3.5/P5.4) reads
them later. When every scene is done, computes ``data/stats.json`` over the
train-split chips (the frozen per-polarization stats) unless it already
exists.

Typical launch (detached, logged):
    python scripts/chip_study_scenes.py --archive-dir /path/to/xview3/train \
        --labels-csv /path/to/xview3/train.csv > runs/logs/chip_study.log 2>&1
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import tarfile
from pathlib import Path
from typing import Sequence


def log(message: str) -> None:
    stamp = dt.datetime.now().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    import pandas as pd
    import yaml

    from src.data.chipper import chip_scene, scene_registry_entry
    from src.data.splits import compute_channel_stats

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument(
        "--splits", nargs="*", default=["train", "dev", "test"],
        help="which splits to process (eval_final is deliberately not chipped)",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    paths = config["paths"]
    chip_cfg = config["chip"]
    shore_km = float(config["splits"]["shore_km_threshold"])
    raw_root = Path(paths["raw_xview3"]) / "GRD"
    chips_root = Path(paths["chips"])
    manifest_root = Path(paths["manifests"])
    manifest_root.mkdir(parents=True, exist_ok=True)

    splits = json.loads(Path(paths["splits"]).read_text())["splits"]
    scene_ids = sorted(
        scene_id for name in args.splits for scene_id in splits[name]
    )
    labels = pd.read_csv(args.labels_csv)
    log(f"{len(scene_ids)} scenes over splits {args.splits}; labels: {len(labels)} rows")

    failures: list[str] = []
    registry_rows: list[dict[str, object]] = []
    for index, scene_id in enumerate(scene_ids, start=1):
        manifest_path = manifest_root / f"{scene_id}.parquet"
        if manifest_path.exists():
            log(f"({index}/{len(scene_ids)}) {scene_id}: manifest exists — skip")
            continue
        try:
            scene_dir = raw_root / scene_id
            if not (scene_dir / "VH_dB.tif").exists():
                archive = Path(args.archive_dir) / f"{scene_id}.tar.gz"
                log(f"({index}/{len(scene_ids)}) {scene_id}: extracting {archive.name}")
                with tarfile.open(archive) as tar:
                    tar.extractall(raw_root, filter="data")

            scene_labels = labels[labels["scene_id"] == scene_id].to_dict(
                orient="records"
            )
            records = chip_scene(
                scene_dir,
                chips_root / scene_id,
                scene_labels=scene_labels,
                chip_size=chip_cfg["size_px"],
                overlap=chip_cfg["overlap_px"],
                max_nodata_frac=chip_cfg["max_nodata_frac"],
            )
            pd.DataFrame([record.__dict__ for record in records]).to_parquet(
                manifest_path
            )
            has_shoreline = any(
                pd.notna(l.get("distance_from_shore_km"))
                and float(l["distance_from_shore_km"]) < shore_km
                for l in scene_labels
            )
            registry_rows.append(
                scene_registry_entry(scene_dir, len(scene_labels), has_shoreline)
            )
            log(
                f"({index}/{len(scene_ids)}) {scene_id}: {len(records)} chips, "
                f"{len(scene_labels)} labels"
            )
        except Exception as error:  # noqa: BLE001 — keep the overnight job alive
            failures.append(scene_id)
            log(f"({index}/{len(scene_ids)}) {scene_id}: FAILED — {error!r}")

    if registry_rows:
        registry_path = manifest_root / "scenes.parquet"
        new_rows = pd.DataFrame(registry_rows)
        if registry_path.exists():
            existing = pd.read_parquet(registry_path)
            new_rows = pd.concat(
                [existing[~existing["scene_id"].isin(new_rows["scene_id"])], new_rows],
                ignore_index=True,
            )
        new_rows.to_parquet(registry_path)
        log(f"scene registry updated: {len(new_rows)} scenes")

    if failures:
        log(f"DONE WITH FAILURES ({len(failures)}): {failures} — investigate before stats")
        return 1

    stats_path = Path(paths["stats"])
    if stats_path.exists():
        log("stats.json already exists — leaving it untouched (frozen artifact)")
    else:
        train_chips = [
            path
            for scene_id in splits["train"]
            for path in sorted((chips_root / scene_id).glob("*.npy"))
        ]
        log(f"computing per-pol stats over {len(train_chips)} train chips ...")
        stats = compute_channel_stats(train_chips)
        stats_path.write_text(json.dumps(stats, indent=1), newline="\n")
        log(f"stats -> {stats_path}: {stats['channels']}")

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
