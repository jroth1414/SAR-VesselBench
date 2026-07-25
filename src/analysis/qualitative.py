"""Chip QA and prediction galleries for human review (P1.5 / P3.5 / P7.2).

``--qa`` renders a 4x4 gallery of random chips (VH channel, dB grayscale)
with any sidecar label points overlaid -> runs/qa/chips.png.

``--pred-gallery CKPT`` overlays decoded predictions on labeled center crops.
Both full Lightning checkpoints and the weights-only ``state_dict`` exports in
the Phase-7 analysis package are accepted. Weights-only exports require
``--init-name`` so the matching architecture can be rebuilt without loading the
original initialization checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

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
            color = (
                "red"
                if str(label.get("confidence", "")).upper() == "LOW"
                else "lime"
            )
            axis.plot(
                label["chip_col"],
                label["chip_row"],
                "o",
                mfc="none",
                mec=color,
                ms=10,
            )
        axis.set_title(chip_path.stem, fontsize=6)
        axis.axis("off")
    for axis in np.asarray(axes).ravel()[len(sample) :]:
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
    init_name: str | None = None,
    device: str = "auto",
    gallery_split: str = "dev",
) -> Path:
    """Overlay decoded predictions (crosses) + labels (circles) on chips.

    ``tau`` may be the LOW candidate floor for smoke QA or the run's frozen dev
    operating threshold for Phase-7 comparisons. Only scene directories
    present below the configured chip root are sampled, so a curated partial
    chip tree does not pretend to be a complete split.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from src.eval.decode import decode_heatmap

    resolved_device = _resolve_device(device)
    module = load_gallery_module(
        checkpoint,
        init_name=init_name,
        device=resolved_device,
    )
    chip_paths = select_gallery_chip_paths(
        Path(config["paths"]["chips"]),
        splits_path=Path(config["paths"]["splits"]),
        split=gallery_split,
        count=n_rows * n_cols,
        seed=seed,
    )
    stats = json.loads(
        Path(config["paths"]["stats"]).read_text(encoding="utf-8")
    )
    samples = [_prepare_gallery_sample(path, stats) for path in chip_paths]
    if not samples:
        raise SystemExit(
            f"no labeled center crops for split {gallery_split!r} under "
            f"{config['paths']['chips']}"
        )

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    for axis, sample in zip(np.asarray(axes).ravel(), samples):
        image = sample["image"][None].to(resolved_device)
        autocast = (
            torch.autocast("cuda", dtype=torch.float16)
            if resolved_device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            logits = module(image)
        heat = torch.sigmoid(logits)[0, 0].float().cpu().numpy()
        peaks = decode_heatmap(
            heat,
            threshold=tau,
            output_stride_m=4.0,  # heatmap cell -> input pixel for plotting
            d_nms_m=12.0,  # 12 input px = the frozen 120 m NMS radius
        )

        axis.imshow(sample["image"][0].numpy(), cmap="gray", vmin=-2, vmax=2)
        for row, col in sample["positives"]:
            axis.plot(col, row, "o", mfc="none", mec="lime", ms=12)
        for peak in peaks:
            axis.plot(peak.col, peak.row, "x", color="red", ms=8)
        axis.set_title(
            f"{sample['name']} | {len(peaks)} pred @ {tau:.3f}",
            fontsize=6,
        )
        axis.axis("off")
    for axis in np.asarray(axes).ravel()[len(samples) :]:
        axis.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"pred gallery -> {out_path}")
    return out_path


def _checkpoint_state(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Return ``(state_dict, hparams)`` for Lightning or weights-only payloads."""

    nested = payload.get("state_dict")
    if isinstance(nested, Mapping):
        hparams = payload.get("hyper_parameters")
        return nested, hparams if isinstance(hparams, Mapping) else {}
    if payload and all(isinstance(key, str) for key in payload):
        if any(key.startswith(("backbone.", "head.")) for key in payload):
            return payload, {}
    raise ValueError(
        "checkpoint is neither a Lightning checkpoint nor a weights-only "
        "HeatmapLitModule state_dict"
    )


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the gallery but is not available")
    return device


def load_gallery_module(
    checkpoint: Path,
    *,
    init_name: str | None,
    device,
    head_channels: int = 256,
):
    """Rebuild the detector architecture and strictly load exported weights.

    This deliberately avoids constructing the Lightning training module: a
    gallery needs only the shared backbone + head forward path, and the
    analysis package should remain usable in a lightweight inference
    environment.
    """

    import torch

    from src.models.heatmap_head import HeatmapHead
    from src.models.init_loaders import build_init

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"unsupported checkpoint payload type: {type(payload).__name__}"
        )
    state_dict, hparams = _checkpoint_state(payload)
    saved_init = hparams.get("init_name")
    if init_name is None:
        init_name = str(saved_init) if saved_init is not None else None
    elif saved_init is not None and init_name != saved_init:
        raise ValueError(
            f"--init-name {init_name!r} disagrees with checkpoint "
            f"hyperparameter {saved_init!r}"
        )
    if init_name is None:
        raise ValueError(
            "weights-only checkpoints require --init-name "
            "(for example, vit_random or bigearthnet_s1)"
        )

    saved_head_channels = hparams.get("head_channels")
    if saved_head_channels is not None:
        head_channels = int(saved_head_channels)

    class GalleryDetector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = build_init(init_name, load_weights=False)
            self.head = HeatmapHead(
                self.backbone.out_channels,
                self.backbone.out_stride,
                head_channels=head_channels,
            )

        def forward(self, image):
            return self.head(self.backbone(image))

    module = GalleryDetector()
    module.load_state_dict(state_dict, strict=True)
    module.to(device)
    module.eval()
    return module


def select_gallery_chip_paths(
    chips_root: Path,
    *,
    splits_path: Path,
    split: str,
    count: int,
    seed: int,
) -> list[Path]:
    """Select labeled center-crop chips from available directories in a split."""

    splits = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]
    if split not in ("dev", "test"):
        raise ValueError("prediction galleries are restricted to dev or test")
    allowed = set(splits[split])
    candidates = [
        path
        for path in sorted(chips_root.glob("*/*.npy"))
        if path.parent.name in allowed and _center_crop_has_positive(path)
    ]
    if not candidates:
        return []
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:count]


def _center_crop_has_positive(chip_path: Path, crop: int = 512) -> bool:
    sidecar = json.loads(
        chip_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    chip_size = int(sidecar.get("chip_size", 800))
    offset = max(0, (chip_size - crop) // 2)
    for label in sidecar["labels"]:
        row = float(label["chip_row"]) - offset
        col = float(label["chip_col"]) - offset
        if (
            bool(label.get("is_vessel"))
            and str(label.get("confidence") or "").upper() in ("HIGH", "MEDIUM")
            and 0 <= row < crop
            and 0 <= col < crop
        ):
            return True
    return False


def _prepare_gallery_sample(
    chip_path: Path,
    stats: dict,
    crop: int = 512,
) -> dict[str, Any]:
    import torch

    from src.data.datasets import _channel_stats
    from src.data.transforms import normalize

    chips = np.load(chip_path).astype(np.float32)
    offset = max(0, (chips.shape[-1] - crop) // 2)
    window = chips[:, offset : offset + crop, offset : offset + crop]
    sidecar = json.loads(
        chip_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    positives = []
    for label in sidecar["labels"]:
        row = float(label["chip_row"]) - offset
        col = float(label["chip_col"]) - offset
        if (
            bool(label.get("is_vessel"))
            and str(label.get("confidence") or "").upper() in ("HIGH", "MEDIUM")
            and 0 <= row < crop
            and 0 <= col < crop
        ):
            positives.append((row, col))

    image = np.concatenate([window, window[0:1] - window[1:2]], axis=0)
    mean, std = _channel_stats(stats)
    image = normalize(image, mean, std)
    return {
        "image": torch.from_numpy(image.astype(np.float32)),
        "positives": positives,
        "name": chip_path.stem,
    }


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
    parser.add_argument(
        "--init-name",
        default=None,
        help="required for weights-only exports; rebuilds the matching architecture",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="inference device: auto, cuda, cuda:N, or cpu",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="prediction threshold; use the run's frozen dev threshold for P7 galleries",
    )
    parser.add_argument(
        "--gallery-split",
        choices=("dev", "test"),
        default="dev",
        help="sample only available scene directories belonging to this split",
    )
    parser.add_argument("--chips-root", default=None, help="override config paths.chips")
    parser.add_argument(
        "--splits-path",
        default=None,
        help="override config paths.splits",
    )
    parser.add_argument("--stats-path", default=None, help="override config paths.stats")
    parser.add_argument("--out-path", default=None, help="prediction-gallery output PNG")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.qa and not args.pred_gallery:
        raise SystemExit("nothing to do: pass --qa and/or --pred-gallery CKPT")

    config = yaml.safe_load(Path(args.config).read_text())
    for key, value in (
        ("chips", args.chips_root),
        ("splits", args.splits_path),
        ("stats", args.stats_path),
    ):
        if value is not None:
            config["paths"][key] = value
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
            out_path=Path(args.out_path or "runs/qa/pred_gallery.png"),
            seed=args.seed,
            tau=args.threshold,
            init_name=args.init_name,
            device=args.device,
            gallery_split=args.gallery_split,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
