"""Chip QA galleries for human eyeballs (P1.5 / P3.5).

``--qa`` renders a 4x4 gallery of random chips (VH channel, dB grayscale)
with any sidecar label points overlaid -> runs/qa/chips.png. The full
prediction galleries (P7.2) land with sprint-9-analysis.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np


def render_chip_gallery(
    chips_root: Path,
    out_path: Path,
    *,
    n_rows: int = 4,
    n_cols: int = 4,
    seed: int = 0,
    prefer_labeled: bool = True,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chip_paths = sorted(chips_root.rglob("*.npy"))
    if not chip_paths:
        raise SystemExit(f"no chips under {chips_root}; run the chipper first")

    rng = random.Random(seed)
    if prefer_labeled:
        labeled = [
            path
            for path in chip_paths
            if json.loads(path.with_suffix(".json").read_text())["labels"]
        ]
        pool = labeled if len(labeled) >= n_rows * n_cols else chip_paths
    else:
        pool = chip_paths
    sample = rng.sample(pool, min(n_rows * n_cols, len(pool)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for axis, chip_path in zip(np.asarray(axes).ravel(), sample):
        vh = np.load(chip_path)[0].astype(np.float32)
        axis.imshow(vh, cmap="gray", vmin=-35.0, vmax=-5.0)
        sidecar = json.loads(chip_path.with_suffix(".json").read_text())
        for label in sidecar["labels"]:
            color = "red" if str(label.get("confidence", "")).upper() == "LOW" else "lime"
            axis.plot(label["chip_col"], label["chip_row"], "o", mfc="none", mec=color, ms=10)
        axis.set_title(chip_path.stem, fontsize=6)
        axis.axis("off")
    for axis in np.asarray(axes).ravel()[len(sample):]:
        axis.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"gallery ({len(sample)} chips) -> {out_path}")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--qa", action="store_true", help="render runs/qa/chips.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.qa:
        raise SystemExit("only --qa is implemented until sprint-9-analysis")

    config = yaml.safe_load(Path(args.config).read_text())
    render_chip_gallery(
        Path(config["paths"]["chips"]),
        Path("runs/qa/chips.png"),
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
