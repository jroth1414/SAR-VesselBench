"""Chip QA galleries for human eyeballs (P1.5 / P3.5).

``--qa`` renders a 4x4 gallery of random chips (VH channel, dB grayscale)
with any sidecar label points overlaid -> runs/qa/chips.png.
``--pred-gallery CKPT`` additionally overlays a checkpoint's decoded
predictions on labeled dev-split chips -> runs/qa/pred_gallery.png (the
P3.5 smoke acceptance artifact). The full per-arm galleries (P7.2) land
with sprint-9-analysis.
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


def render_pred_gallery(
    checkpoint: Path,
    *,
    config: dict,
    out_path: Path,
    n_rows: int = 4,
    n_cols: int = 4,
    seed: int = 0,
    tau: float = 0.05,
) -> Path:
    """Overlay decoded predictions (crosses) + labels (circles) on dev chips.

    Decodes at the LOW candidate floor (the operating threshold is dev-tuned,
    P2.2b) and samples center-crops that actually contain >= 1 positive.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    from src.data.datasets import FineTuneDataset
    from src.eval.decode import decode_heatmap
    from src.train.lit_modules import HeatmapLitModule

    module = HeatmapLitModule.load_from_checkpoint(str(checkpoint), map_location="cuda")
    module.eval()

    dataset = FineTuneDataset(
        chips_root=config["paths"]["chips"],
        splits_path=config["paths"]["splits"],
        stats_path=config["paths"]["stats"],
        split="dev",
        augment=False,
    )
    rng = random.Random(seed)
    labeled = [i for i, fg in enumerate(dataset.foreground) if fg]
    rng.shuffle(labeled)
    # keep only crops whose center 512 window still contains a positive
    samples = []
    for index in labeled:
        candidate = dataset[index]
        if candidate["n_positives"] > 0:
            samples.append(candidate)
        if len(samples) == n_rows * n_cols:
            break

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for axis, sample in zip(np.asarray(axes).ravel(), samples):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = module(sample["image"][None].cuda())
        heat = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        peaks = decode_heatmap(heat, threshold=tau, output_stride_m=4.0, d_nms_m=12.0)

        axis.imshow(sample["image"][0].numpy(), cmap="gray", vmin=-2, vmax=2)
        target = sample["heatmap"].numpy()
        for row, col in zip(*np.nonzero(target >= 1.0 - 1e-6)):
            axis.plot(col * 4, row * 4, "o", mfc="none", mec="lime", ms=12)
        for peak in peaks:
            axis.plot(peak.col, peak.row, "x", color="red", ms=8)
        axis.set_title(f"{len(peaks)} pred", fontsize=6)
        axis.axis("off")
    for axis in np.asarray(axes).ravel()[len(samples):]:
        axis.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"pred gallery -> {out_path}")
    return out_path


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--qa", action="store_true", help="render runs/qa/chips.png")
    parser.add_argument(
        "--pred-gallery",
        default=None,
        metavar="CKPT",
        help="also render runs/qa/pred_gallery.png from this checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.qa and not args.pred_gallery:
        raise SystemExit("nothing to do: pass --qa and/or --pred-gallery CKPT")

    config = yaml.safe_load(Path(args.config).read_text())
    if args.qa:
        render_chip_gallery(
            Path(config["paths"]["chips"]),
            Path("runs/qa/chips.png"),
            seed=args.seed,
        )
    if args.pred_gallery:
        render_pred_gallery(
            Path(args.pred_gallery),
            config=config,
            out_path=Path("runs/qa/pred_gallery.png"),
            seed=args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
