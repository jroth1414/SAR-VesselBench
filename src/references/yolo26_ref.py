"""Arm R2 — YOLO26 detector reference (DEVPLAN P4.5). OFF the controlled curves.

Pipeline (three subcommands):

- ``export``: chips -> YOLO-format dataset. Images are the study's fixed
  [VH, VV, VH-VV] tensor, normalized by the FROZEN train stats, clipped to
  +-3 sigma and scaled to uint8 JPG (so YOLO sees the same physics as the
  study arms). Boxes are synthesized from points+lengths per the plan:
  square side ``max(6 px, vessel_length_m / 10)`` centered on the point
  (10 m GSD); positives are HIGH/MEDIUM ``is_vessel``; missing lengths fall
  back to the 6 px floor. LOW-confidence labels are OMITTED — ultralytics
  has no ignore-region mechanism (documented reference-arm protocol
  difference; the study arms zero the loss there instead).
- ``train``: YOLO26 (COCO init) via ultralytics on the exported dataset.
- ``score``: tiled whole-scene prediction (800 px windows, stride 700),
  box centers -> scene meters -> our distance NMS -> the SACRED scorer,
  with the operating threshold selected on dev by src/eval/threshold.py
  (P2.2b) and applied frozen to test.

Reference-arm freedoms used (Leaf tier, reported separately): YOLO's own
augmentation/optimizer defaults; val split = dev-split chips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.runtime.reference import (
    ReferenceRuntimeInputs,
    add_runtime_provenance_arguments,
    begin_reference_execution,
    finish_reference_execution,
    load_runtime_inputs,
    publish_reference_result,
    result_directory,
)

GSD_M = 10.0
MIN_BOX_PX = 6.0
CLIP_SIGMA = 3.0
RESULT_SCHEMA = 2
EXPECTED_R2_BEST_SHA256 = (
    "15520cb6cff9d4b01ed5c4a7e039fab763e8e5b0ca5b8e6bffd591ef0d7b8064"
)
EXPECTED_GT_COUNTS = {
    "dev": {"positive": 1479, "background": 804, "ignore": 441},
    "test": {"positive": 1165, "background": 420, "ignore": 325},
}


def _sha256_file(path: Path, chunk_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _metrics_payload(metrics) -> dict[str, int | float]:
    return {
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "ignored_predictions": metrics.ignored_predictions,
    }


def _score_payload(result) -> dict[str, object]:
    return {
        "aggregate": _metrics_payload(result.aggregate),
        "slices": {
            name: _metrics_payload(metrics)
            for name, metrics in sorted(result.slices.items())
        },
        "per_scene": {
            scene_id: {
                "aggregate": _metrics_payload(scene.aggregate),
                "slices": {
                    name: _metrics_payload(metrics)
                    for name, metrics in sorted(scene.slices.items())
                },
            }
            for scene_id, scene in sorted(result.scene_results.items())
        },
    }


def _label_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    from src.eval.ground_truth import classify_label

    counts = {"positive": 0, "background": 0, "ignore": 0}
    for row in rows:
        counts[classify_label(row)] += 1
    return counts


def _known_r2_checkpoint(weights: Path) -> tuple[Path, str]:
    expected = Path("runs/yolo26-f100/weights/best.pt")
    if weights.resolve() != expected.resolve():
        raise RuntimeError(
            "R2 correction must rescore the preserved runs/yolo26-f100/weights/best.pt"
        )
    if weights.is_symlink() or not weights.is_file():
        raise RuntimeError(f"R2 preserved checkpoint is absent or unsafe: {weights}")
    actual_sha256 = _sha256_file(weights)
    if actual_sha256 != EXPECTED_R2_BEST_SHA256:
        raise RuntimeError(
            "R2 preserved best.pt SHA-256 mismatch: "
            f"{actual_sha256} != {EXPECTED_R2_BEST_SHA256}"
        )
    return weights, actual_sha256


def _chip_to_uint8(chips: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """(2, H, W) float dB chip -> (H, W, 3) uint8 [VH, VV, VH-VV]."""

    image = np.concatenate([chips, chips[0:1] - chips[1:2]], axis=0).astype(np.float32)
    normalized = (image - mean[:, None, None]) / std[:, None, None]
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = (normalized + CLIP_SIGMA) / (2 * CLIP_SIGMA)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)


def yolo_label_lines(labels: Sequence[dict], chip_px: int) -> list[str]:
    """Sidecar labels -> YOLO txt lines (class cx cy w h, normalized)."""

    from src.eval.ground_truth import classify_label

    lines = []
    for label in labels:
        if classify_label(label) != "positive":
            continue
        length_m = label.get("vessel_length_m")
        side_px = MIN_BOX_PX
        if length_m is not None and np.isfinite(float(length_m)):
            side_px = max(MIN_BOX_PX, float(length_m) / GSD_M)
        cx = float(label["chip_col"]) / chip_px
        cy = float(label["chip_row"]) / chip_px
        side = min(side_px / chip_px, 1.0)
        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
            lines.append(f"0 {cx:.6f} {cy:.6f} {side:.6f} {side:.6f}")
    return lines


def export_dataset(
    *,
    chips_root: Path,
    splits_path: Path,
    stats_path: Path,
    out_root: Path,
    workers: int = 8,
) -> None:
    import multiprocessing as mp

    splits = json.loads(splits_path.read_text())["splits"]
    stats = json.loads(stats_path.read_text())
    vh, vv = stats["channels"]["VH"], stats["channels"]["VV"]
    mean = np.array([vh["mean"], vv["mean"], vh["mean"] - vv["mean"]], dtype=np.float32)
    std = np.array(
        [vh["std"], vv["std"], float(np.hypot(vh["std"], vv["std"]))], dtype=np.float32
    )

    jobs = []
    for part, scene_split in (("train", "train"), ("val", "dev")):
        (out_root / "images" / part).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / part).mkdir(parents=True, exist_ok=True)
        for scene_id in splits[scene_split]:
            for chip_path in sorted((chips_root / scene_id).glob("*.npy")):
                jobs.append((str(chip_path), part))
    print(f"exporting {len(jobs)} chips with {workers} workers -> {out_root}")

    with mp.Pool(
        workers,
        initializer=_export_init,
        initargs=(mean, std, str(out_root)),
    ) as pool:
        done = 0
        for _ in pool.imap_unordered(_export_one, jobs, chunksize=64):
            done += 1
            if done % 10000 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    dataset_yaml = out_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {out_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: vessel",
                "",
            ]
        ),
        newline="\n",
    )
    print(f"dataset config -> {dataset_yaml}")


_EXPORT_CTX: dict = {}


def _export_init(mean: np.ndarray, std: np.ndarray, out_root: str) -> None:
    _EXPORT_CTX["mean"] = mean
    _EXPORT_CTX["std"] = std
    _EXPORT_CTX["out_root"] = Path(out_root)


def _export_one(job: tuple[str, str]) -> None:
    import cv2

    chip_path, part = job
    chip_path = Path(chip_path)
    out_root = _EXPORT_CTX["out_root"]
    image_path = out_root / "images" / part / f"{chip_path.stem}.jpg"
    label_path = out_root / "labels" / part / f"{chip_path.stem}.txt"
    if image_path.exists() and label_path.exists():
        return

    chips = np.load(chip_path).astype(np.float32)
    image = _chip_to_uint8(chips, _EXPORT_CTX["mean"], _EXPORT_CTX["std"])
    cv2.imwrite(str(image_path), image[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])

    sidecar = json.loads(chip_path.with_suffix(".json").read_text())
    lines = yolo_label_lines(sidecar["labels"], sidecar["chip_size"])
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), newline="\n")


def train(out_root: Path, *, model_name: str, epochs: int, batch: int, imgsz: int) -> None:
    from ultralytics import YOLO

    model = YOLO(model_name)  # COCO-pretrained checkpoint download
    model.train(
        data=str(out_root / "dataset.yaml"),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        # ABSOLUTE path: ultralytics' global settings.json rewrites relative
        # project dirs under its own runs_dir (cost us a 33 h rerun scare).
        project=str(Path("runs").resolve()),
        name="yolo26-f100",
        exist_ok=True,
        single_cls=True,
        seed=0,
    )


def score(
    *,
    weights: Path,
    data_cfg: dict,
    det_cfg: dict,
    runtime_inputs: ReferenceRuntimeInputs,
    result_dir: Path,
    device: str = "cuda",
) -> dict:
    """Dev-threshold-selected scoring through the SACRED scorer (P4.5).

    Tiles each scene 800 px / stride 700 (the chip grid), converts each
    tile to the same uint8 representation YOLO trained on, takes YOLO box
    CENTERS as candidate points with their confidences, dedups across tile
    overlaps with our distance NMS, selects the F1-max threshold on the DEV
    scenes (P2.2b), applies it frozen to the TEST scenes, and atomically writes
    fresh corrected-campaign metrics plus provenance.
    """

    import pandas as pd
    import rasterio
    import torch
    from ultralytics import YOLO

    from src.eval.decode import DecodedPoint, distance_nms
    from src.eval.infer_scene import GSD_M, ShoreDistance, ground_truth_from_labels
    from src.eval.scorer import PredictionPoint, score_dataset
    from src.eval.threshold import apply_threshold, select_f1_threshold

    execution = begin_reference_execution(
        runtime_inputs,
        reference_precision="float32",
        device=device,
        torch_module=torch,
    )
    weights, checkpoint_sha256 = _known_r2_checkpoint(weights)
    model = YOLO(str(weights))
    splits = json.loads(Path(data_cfg["paths"]["splits"]).read_text())["splits"]
    stats = json.loads(Path(data_cfg["paths"]["stats"]).read_text())
    vh_s, vv_s = stats["channels"]["VH"], stats["channels"]["VV"]
    mean = np.array([vh_s["mean"], vv_s["mean"], vh_s["mean"] - vv_s["mean"]], dtype=np.float32)
    std = np.array(
        [vh_s["std"], vv_s["std"], float(np.hypot(vh_s["std"], vv_s["std"]))],
        dtype=np.float32,
    )
    labels = pd.read_csv(Path(data_cfg["paths"]["raw_xview3"]) / "labels" / "train.csv")
    raw_root = Path(data_cfg["paths"]["raw_xview3"]) / "GRD"
    d_nms_m = det_cfg["decode"]["d_nms_m"]

    def predict_scene(scene_id: str) -> list[PredictionPoint]:
        scene_dir = raw_root / scene_id
        with rasterio.open(scene_dir / "VH_dB.tif") as ds:
            vh_full = ds.read(1)
            height, width = ds.height, ds.width
            transform = ds.transform
        with rasterio.open(scene_dir / "VV_dB.tif") as ds:
            vv_full = ds.read(1)

        candidates: list[DecodedPoint] = []
        stride, tile = 700, 800
        rows = list(range(0, max(height - tile, 1), stride)) + [max(height - tile, 0)]
        cols = list(range(0, max(width - tile, 1), stride)) + [max(width - tile, 0)]
        batch_imgs, batch_origins = [], []

        def flush() -> None:
            if not batch_imgs:
                return
            results = model.predict(
                batch_imgs,
                imgsz=tile,
                conf=0.01,
                verbose=False,
                device=device,
                half=False,
            )
            for (row0, col0), result in zip(batch_origins, results):
                boxes = result.boxes
                for (cx, cy), conf in zip(
                    boxes.xywh[:, :2].tolist(), boxes.conf.tolist()
                ):
                    candidates.append(
                        DecodedPoint(
                            row=(row0 + cy) * GSD_M,
                            col=(col0 + cx) * GSD_M,
                            score=float(conf),
                        )
                    )
            batch_imgs.clear()
            batch_origins.clear()

        for row0 in rows:
            for col0 in cols:
                vh = vh_full[row0 : row0 + tile, col0 : col0 + tile]
                if (vh == -32768.0).mean() > 0.98:
                    continue
                vv = vv_full[row0 : row0 + tile, col0 : col0 + tile]
                chip = np.stack([vh, vv]).astype(np.float32)
                chip[chip == -32768.0] = np.nan
                image = _chip_to_uint8(chip, mean, std)
                batch_imgs.append(image[..., ::-1])  # ultralytics expects BGR
                batch_origins.append((row0, col0))
                if len(batch_imgs) >= 8:
                    flush()
        flush()
        del vh_full, vv_full

        kept = distance_nms(candidates, d_nms_m=d_nms_m)
        shore = ShoreDistance(scene_dir)
        predictions = []
        for point in kept:
            x_geo, y_geo = transform * (point.col / GSD_M, point.row / GSD_M)
            predictions.append(
                PredictionPoint(
                    x_m=point.col,
                    y_m=point.row,
                    score=point.score,
                    distance_from_shore_km=shore.lookup_km(x_geo, y_geo),
                )
            )
        return predictions

    def collect(scene_ids: list[str]):
        gt, preds = {}, {}
        counts = {"positive": 0, "background": 0, "ignore": 0}
        for scene_id in scene_ids:
            print(f"  {scene_id}", flush=True)
            rows = labels[labels["scene_id"] == scene_id].to_dict(orient="records")
            for category, count in _label_counts(rows).items():
                counts[category] += count
            gt[scene_id] = ground_truth_from_labels(rows)
            preds[scene_id] = predict_scene(scene_id)
        return gt, preds, counts

    print("dev scenes (threshold selection):")
    dev_gt, dev_pred, dev_counts = collect(sorted(splits["dev"]))
    threshold = select_f1_threshold(dev_gt, dev_pred)
    dev_result = score_dataset(dev_gt, apply_threshold(dev_pred, threshold))

    print("test scenes (frozen threshold):")
    test_gt, test_pred, test_counts = collect(sorted(splits["test"]))
    test_result = score_dataset(test_gt, apply_threshold(test_pred, threshold))

    observed_counts = {"dev": dev_counts, "test": test_counts}
    if observed_counts != EXPECTED_GT_COUNTS:
        raise RuntimeError(
            "R2 corrected GT count gate failed: "
            f"{observed_counts} != {EXPECTED_GT_COUNTS}"
        )

    provenance = finish_reference_execution(
        execution,
        device=device,
        torch_module=torch,
    )
    payload = {
        "result_schema": RESULT_SCHEMA,
        "exp_id": "yolo26-f100",
        "reference": "R2",
        "source_git_sha": provenance["git_sha"],
        "scored_at": provenance["finished_utc"],
        "inference_precision": "float32",
        "training_disposition": "preserved-best-pt-rescore-only",
        "checkpoint": {
            "relative_path": "weights/best.pt",
            "sha256": checkpoint_sha256,
        },
        "ground_truth_contract": {
            "version": 2,
            "positive": "is_vessel=true and confidence in {HIGH,MEDIUM}",
            "background": "is_vessel=false and confidence in {HIGH,MEDIUM}",
            "ignore": "confidence=LOW",
            "dev_counts": dev_counts,
            "test_counts": test_counts,
        },
        "threshold_source": {
            "split": "dev",
            "checkpoint_sha256": checkpoint_sha256,
        },
        "threshold": threshold,
        "dev": _score_payload(dev_result),
        "test": _score_payload(test_result),
        # Stable flat aliases retained for reference-summary consumers.
        "dev_f1": dev_result.aggregate.f1,
        "test_f1": test_result.aggregate.f1,
        "test_precision": test_result.aggregate.precision,
        "test_recall": test_result.aggregate.recall,
        "test_near_shore_f1": test_result.slices["near_shore"].f1,
    }
    out, provenance_path = publish_reference_result(
        result_dir,
        metrics=payload,
        provenance=provenance,
    )
    print(json.dumps(payload, indent=1))
    print(f"runtime provenance -> {provenance_path}")
    torch.cuda.empty_cache()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "train", "score"])
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--out-root", default="data/yolo")
    parser.add_argument("--model", default="yolo26m.pt")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/yolo26-f100/weights/best.pt"),
        help="preserved R2 best.pt used by the corrected score command",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    add_runtime_provenance_arguments(parser)
    args = parser.parse_args(argv)

    config = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(args.out_root)

    if args.command == "export":
        export_dataset(
            chips_root=Path(config["paths"]["chips"]),
            splits_path=Path(config["paths"]["splits"]),
            stats_path=Path(config["paths"]["stats"]),
            out_root=out_root,
            workers=args.workers,
        )
    elif args.command == "train":
        train(
            out_root,
            model_name=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
        )
    else:
        import yaml as _yaml

        repo = Path(__file__).resolve().parents[2]
        try:
            runtime_inputs = load_runtime_inputs(args, repo=repo, required=True)
            corrected_result_dir = result_directory(args, "yolo26-f100")
        except RuntimeError as exc:
            parser.error(str(exc))
        assert runtime_inputs is not None
        det_cfg = _yaml.safe_load(Path("configs/detector.yaml").read_text())
        weights = args.weights
        if not weights.exists():
            raise SystemExit(f"{weights} missing — train first")
        score(
            weights=weights,
            data_cfg=config,
            det_cfg=det_cfg,
            runtime_inputs=runtime_inputs,
            result_dir=corrected_result_dir,
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
