"""xView3 GRD acquisition -> data/raw/xview3/GRD/<scene_id>/.

Two acquisition paths, both writing a revision-pinned ``SOURCE.note``:

- ``--from-local DIR`` (primary in practice): register the original xView3
  per-scene ``<scene_id>.tar.gz`` archives (containing VH_dB/VV_dB/bathymetry/
  owiMask/owiWind* GeoTIFFs) already downloaded from the DIU distribution
  (https://iuu.xview.us). Extracts any scene not yet present.

- ``--from-hf``: selective ``huggingface_hub.snapshot_download`` from the
  SARFish mirror, GRD only, never SLC, with a size estimate + cap.

  *** FORMAT DISCREPANCY (surfaced 2026-07-04, treat as a STOP before using
  this path): the SARFish HF dataset hosts raw Sentinel-1 ``*.SAFE.zip``
  products keyed by S1 product id — NOT the xView3-format per-scene dB
  GeoTIFFs this pipeline (and the plan's chipper) assume — and its labels are
  NOT on HF (they come from the DIU download-links page, auth + country
  restricted). Using SARFish would require an S1 preprocessing step and a
  product-id -> scene-id mapping that this project never specified. ***
"""

from __future__ import annotations

import argparse
import datetime as dt
import tarfile
from pathlib import Path
from typing import Sequence

SARFISH_REPO = "ConnorLuckettDSTG/SARFish"


def register_local_archives(
    archive_dir: Path,
    out_root: Path,
    *,
    scene_ids: Sequence[str] | None = None,
) -> list[str]:
    """Extract per-scene tar.gz archives into the raw GRD layout."""

    archives = sorted(archive_dir.glob("*.tar.gz"))
    if scene_ids:
        wanted = set(scene_ids)
        archives = [a for a in archives if a.name.removesuffix(".tar.gz") in wanted]
    if not archives:
        raise SystemExit(f"no scene archives found under {archive_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    registered: list[str] = []
    for archive in archives:
        scene_id = archive.name.removesuffix(".tar.gz")
        scene_dir = out_root / scene_id
        if (scene_dir / "VH_dB.tif").exists():
            print(f"{scene_id}: already extracted")
        else:
            print(f"{scene_id}: extracting {archive.name}")
            with tarfile.open(archive) as tar:
                tar.extractall(out_root, filter="data")
        registered.append(scene_id)

    _write_source_note(
        out_root.parent / "SOURCE.note",
        source=f"local xView3 archives from {archive_dir}",
        detail=(
            "Original xView3 per-scene tar.gz distribution (DIU, "
            "https://iuu.xview.us), downloaded via aria2 with per-file "
            "integrity verification (see the aria2 logs beside the archives). "
            f"Registered scenes: {', '.join(registered)}"
        ),
    )
    return registered


def download_from_hf(
    out_root: Path,
    *,
    revision: str,
    partitions: Sequence[str],
    size_cap_gb: float,
) -> None:
    """Selective SARFish GRD pull (never SLC), size-capped, revision-pinned."""

    from huggingface_hub import HfApi, snapshot_download

    if not revision:
        raise SystemExit(
            "a pinned --revision is mandatory (P1.1): an unpinned source "
            "silently voids the frozen splits' reproducibility"
        )

    patterns = [f"GRD/{partition}/*" for partition in partitions]
    api = HfApi()
    files = api.list_repo_files(SARFISH_REPO, repo_type="dataset", revision=revision)
    info = api.repo_info(
        SARFISH_REPO, repo_type="dataset", revision=revision, files_metadata=True
    )
    selected = [
        sibling
        for sibling in info.siblings
        if any(sibling.rfilename.startswith(f"GRD/{p}/") for p in partitions)
    ]
    total_gb = sum(s.size or 0 for s in selected) / 2**30
    print(f"{len(selected)}/{len(files)} files selected, estimated {total_gb:.1f} GB")
    if total_gb > size_cap_gb:
        raise SystemExit(
            f"estimated pull {total_gb:.1f} GB exceeds the cap {size_cap_gb} GB "
            "(P1.1 budget guard) — reduce partitions/scenes or raise the cap"
        )

    snapshot_download(
        SARFISH_REPO,
        repo_type="dataset",
        revision=revision,
        allow_patterns=patterns,
        local_dir=out_root,
    )
    _write_source_note(
        out_root.parent / "SOURCE.note",
        source=f"hf://datasets/{SARFISH_REPO}@{revision}",
        detail=(
            f"partitions={list(partitions)}; SLC never fetched. NOTE: SARFish "
            "GRD files are raw Sentinel-1 SAFE.zip products, not xView3-format "
            "GeoTIFFs — see the module docstring STOP before chipping these."
        ),
    )


def _write_source_note(path: Path, *, source: str, detail: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    entry = f"[{stamp}] {source}\n  {detail}\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
    print(f"SOURCE.note updated: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-local", metavar="DIR", help="register local xView3 tar.gz archives")
    mode.add_argument("--from-hf", action="store_true", help="selective SARFish HF pull (see STOP in docstring)")
    parser.add_argument("--scenes", nargs="*", help="restrict to these scene ids")
    parser.add_argument("--revision", default=None, help="HF dataset revision pin (mandatory for --from-hf)")
    parser.add_argument(
        "--partitions", nargs="*", default=["train", "validation"],
        help="SARFish GRD partitions for --from-hf",
    )
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(config["paths"]["raw_xview3"]) / "GRD"

    if args.from_local:
        register_local_archives(Path(args.from_local), out_root, scene_ids=args.scenes)
    else:
        download_from_hf(
            out_root,
            revision=args.revision or config["download"].get("sarfish_revision") or "",
            partitions=args.partitions,
            size_cap_gb=float(config["download"]["size_cap_gb"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
