"""Near-shore analysis for the poster: where vessels are vs. where detectors find them.

Three inputs live outside the evidence tree, and each is checked before use:

* the xView3 training labels (``train.csv``) must hash to the binding in the
  evidence tree's ground-truth audit receipt;
* the verified labels (``validation.csv``, the public xView3 validation
  labels behind the 50 verified scenes) must reproduce every cell's TP/FP/FN
  when the frozen scorer re-scores the cell's stored thresholded predictions;
* example imagery is read from the xView3 validation archives
  (``<scene>.tar.gz`` holding ``<scene>/VH_dB.tif``).

xView3 records ``distance_from_shore_km`` only up to 10 km and codes larger
distances as 9999.99, so the last bin is "> 10 km".
"""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.analysis.heldout_results import EvidenceError
from src.eval.ground_truth import classify_label, ground_truth_from_labels
from src.eval.scorer import PredictionPoint, ScoreResult, score_points

BIN_LABELS = ("0–1", "1–2", "2–5", "5–10", ">10")
BIN_EDGES = (1.0, 2.0, 5.0, 10.0)
LABEL_CAP_KM = 10.0
CROP_PX = 800  # 8 km at 10 m pixels
FOUND, MISSED, FALSE_ALARM = "#56B4E9", "#E69F00", "#CC79A7"  # Okabe-Ito, distinct under CVD


def distance_bin(distance_km: float | None) -> int | None:
    if distance_km is None:
        return None
    if distance_km > LABEL_CAP_KM:  # includes the 9999.99 code for "beyond 10 km"
        return len(BIN_EDGES)
    for index, edge in enumerate(BIN_EDGES):
        if distance_km < edge or (edge == LABEL_CAP_KM and distance_km <= edge):
            return index
    return len(BIN_EDGES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_bin_counts(train_csv: Path, train_scene_ids: Sequence[str], audit: Mapping[str, object]) -> np.ndarray:
    """Positive training labels per distance bin; the CSV must match the audit binding."""

    if _sha256(train_csv) != audit["inputs"]["train_csv"]["sha256"]:  # type: ignore[index]
        raise EvidenceError("training labels do not hash to the audit receipt's binding")
    wanted = set(train_scene_ids)
    counts = np.zeros(len(BIN_LABELS), dtype=int)
    with train_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["scene_id"] in wanted and classify_label(row) == "positive":
                raw = (row.get("distance_from_shore_km") or "").strip()
                bin_index = distance_bin(float(raw) if raw else None)
                if bin_index is not None:
                    counts[bin_index] += 1
    return counts


def rescore_verified(
    evidence_root: Path, cell_ids: Mapping[tuple, str], verified_csv: Path
) -> dict[str, object]:
    """Re-score every cell's verified-set predictions; fail unless counts reproduce."""

    rows_by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    with verified_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            rows_by_scene[row["scene_id"]].append(row)
    bins: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    scenes: dict[tuple, dict[str, tuple[list, list, ScoreResult]]] = {}
    for key, exp_id in cell_ids.items():
        payload = json.loads((evidence_root / exp_id / "final_verified_metrics.json").read_text(encoding="utf-8"))
        hit = np.zeros(len(BIN_LABELS), dtype=int)
        total = np.zeros(len(BIN_LABELS), dtype=int)
        tp = fp = fn = 0
        per_scene = {}
        for scene, stored in payload["thresholded_predictions"].items():
            gt = ground_truth_from_labels(rows_by_scene[scene])
            predictions = [PredictionPoint(p["x_m"], p["y_m"], p["score"], p["distance_from_shore_km"]) for p in stored]
            result = score_points(gt, predictions)
            tp, fp, fn = tp + result.aggregate.tp, fp + result.aggregate.fp, fn + result.aggregate.fn
            matched = {m.ground_truth_index for m in result.matches if m.outcome == "tp"}
            for index, point in enumerate(gt):
                bin_index = None if point.is_low_confidence else distance_bin(point.distance_from_shore_km)
                if bin_index is not None:
                    total[bin_index] += 1
                    hit[bin_index] += index in matched
            per_scene[scene] = (gt, predictions, result)
        expected = payload["metrics"]
        if (tp, fp, fn) != (expected["tp"], expected["fp"], expected["fn"]):
            raise EvidenceError(f"{exp_id}: verified labels do not reproduce the stored TP/FP/FN")
        bins[key] = (hit, total)
        scenes[key] = per_scene
    return {"bins": bins, "scenes": scenes}


BIN_WORDS = ("zero_to_one_km", "one_to_two_km", "two_to_five_km", "five_to_ten_km", "beyond_ten_km")


def coastal_numbers(train_counts: np.ndarray, verified_counts: np.ndarray, rescored: Mapping[str, object],
                    best_key: tuple, crops: Sequence[tuple[str, str, int, int]]) -> dict[str, object]:
    """Poster numbers for the near-shore block (keys are letters and underscores only)."""

    hit, total = rescored["bins"][best_key]  # type: ignore[index]
    numbers: dict[str, object] = {
        "train_beyond_ten_km_pct": 100.0 * train_counts[-1] / train_counts.sum(),
        "verified_beyond_ten_km_pct": 100.0 * verified_counts[-1] / verified_counts.sum(),
    }
    for word, h, t in zip(BIN_WORDS, hit, total):
        numbers[f"best_recall_{word}"] = float(h / t)
        numbers[f"verified_vessels_{word}"] = int(t)
    for kind, scene, _x0, _y0 in crops:
        numbers[f"example_{kind.replace(' ', '_')}_scene"] = scene
    return numbers


def _windows(per_scene: Mapping[str, tuple[list, list, ScoreResult]]):
    """Candidate 8 km windows centred on positives, with hit/miss counts."""

    for scene, (gt, _predictions, result) in per_scene.items():
        matched = {m.ground_truth_index for m in result.matches if m.outcome == "tp"}
        positives = [(i, p) for i, p in enumerate(gt) if not p.is_low_confidence]
        for i, centre in positives:
            cx, cy = centre.x_m / 10.0, centre.y_m / 10.0
            inside = [(j, p) for j, p in positives
                      if abs(p.x_m / 10.0 - cx) < CROP_PX / 2 and abs(p.y_m / 10.0 - cy) < CROP_PX / 2]
            hits = sum(1 for j, _ in inside if j in matched)
            near = sum(1 for _, p in inside if p.is_near_shore)
            yield scene, cx, cy, len(inside), hits, near


def choose_crops(per_scene: Mapping[str, tuple[list, list, ScoreResult]]) -> list[tuple[str, str, int, int]]:
    """An offshore window where most vessels are found and a near-shore window where most are missed."""

    windows = list(_windows(per_scene))
    offshore = max((w for w in windows if w[5] == 0 and w[3] >= 4), key=lambda w: (w[4] / w[3], w[4]))
    coastal = max((w for w in windows if w[5] >= 0.8 * w[3]), key=lambda w: (w[3] - w[4], w[3]))
    crops = []
    for kind, (scene, cx, cy, *_rest) in (("offshore", offshore), ("near shore", coastal)):
        crops.append((kind, scene, int(cx - CROP_PX / 2), int(cy - CROP_PX / 2)))
    return crops


def read_vh_window(archive_dir: Path, cache_dir: Path, scene: str, x0: int, y0: int) -> tuple[np.ndarray, int, int]:
    """Read an 8 km VH window, extracting the scene's VH raster from its archive once."""

    import rasterio
    from rasterio.windows import Window

    cache_dir.mkdir(parents=True, exist_ok=True)
    tif = cache_dir / f"{scene}_VH_dB.tif"
    if not tif.exists():
        partial = tif.with_suffix(".tif.partial")
        with tarfile.open(archive_dir / f"{scene}.tar.gz", "r:gz") as archive:
            source = archive.extractfile(archive.getmember(f"{scene}/VH_dB.tif"))
            with partial.open("wb") as target:
                for chunk in iter(lambda: source.read(1 << 24), b""):
                    target.write(chunk)
        partial.replace(tif)
    with rasterio.open(tif) as dataset:
        x0 = max(0, min(x0, dataset.width - CROP_PX))
        y0 = max(0, min(y0, dataset.height - CROP_PX))
        image = dataset.read(1, window=Window(x0, y0, CROP_PX, CROP_PX)).astype(float)
    return np.where(np.isfinite(image), image, np.nan), x0, y0


# --------------------------------------------------------------------------- figures
def figure_recall_by_distance(train_counts, verified_counts, rescored, best_key, out_dir: Path, style) -> list[Path]:
    import matplotlib.pyplot as plt

    from src.analysis.poster_figures import GRID, INK, save_both

    x = np.arange(len(BIN_LABELS))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.6, 6.6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.2]})
    top.bar(x - 0.2, 100 * train_counts / train_counts.sum(), width=0.38, color="#9AA3B2",
            label="training labels (automated)")
    top.bar(x + 0.2, 100 * verified_counts / verified_counts.sum(), width=0.38, color="#14213D",
            label="verified labels (human)")
    top.set_ylabel("% of vessels")
    top.set_title("Where the vessels are", color=INK)
    top.legend(loc="upper left", fontsize=style["legend"])
    recalls = np.array([h / np.maximum(t, 1) for h, t in rescored["bins"].values()])
    bottom.fill_between(x, recalls.min(axis=0), recalls.max(axis=0), color=GRID, label="range, all 32 cells")
    hit, total = rescored["bins"][best_key]
    bottom.plot(x, hit / total, color=style["best_color"], marker="D", lw=4.5, markersize=15,
                markeredgecolor="white", markeredgewidth=2.5, label=style["best_label"])
    bottom.set_ylim(0, 1.0)  # headroom keeps the legend clear of the curve
    bottom.set_yticks([0, 0.5, 1.0])
    bottom.set_ylabel("Recall")
    bottom.set_title("Where the detectors find them", color=INK)
    bottom.set_xticks(x)
    bottom.set_xticklabels(BIN_LABELS)
    bottom.set_xlabel("Distance to shore (km)")
    bottom.legend(loc="upper left", fontsize=style["legend"])
    fig.tight_layout(h_pad=1.5)
    paths = save_both(fig, out_dir, "poster_recall_by_distance")
    plt.close(fig)
    return paths


def figure_detection_examples(crops, per_scene, archive_dir: Path, cache_dir: Path, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from src.analysis.poster_figures import INK, save_both

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.6))
    for axis, (kind, scene, x0, y0) in zip(axes, crops):
        image, x0, y0 = read_vh_window(archive_dir, cache_dir, scene, x0, y0)
        low, high = np.nanpercentile(image, [2, 99.5])
        axis.imshow(np.clip(image, low, high), cmap="gray", extent=(0, 8, 8, 0), interpolation="nearest")
        gt, predictions, result = per_scene[scene]
        matched = {m.ground_truth_index for m in result.matches if m.outcome == "tp"}
        false_alarms = [m.prediction_index for m in result.matches if m.outcome == "fp"]

        def local(point):
            return (point.x_m / 10.0 - x0) / 100.0, (point.y_m / 10.0 - y0) / 100.0

        for index, point in enumerate(gt):
            u, v = local(point)
            if point.is_low_confidence or not (0 <= u < 8 and 0 <= v < 8):
                continue
            if index in matched:
                axis.plot(u, v, marker="o", ms=17, mfc="none", mec=FOUND, mew=3)
            else:
                axis.plot(u, v, marker="s", ms=15, mfc="none", mec=MISSED, mew=3)
        for index in false_alarms:
            u, v = local(predictions[index])
            if 0 <= u < 8 and 0 <= v < 8:
                axis.plot(u, v, marker="x", ms=14, mew=3.5, color=FALSE_ALARM)
        axis.set_title(f"{kind.capitalize()}", color=INK)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    handles = [Line2D([], [], marker="o", ms=15, mfc="none", mec=FOUND, mew=3, ls="none"),
               Line2D([], [], marker="s", ms=13, mfc="none", mec=MISSED, mew=3, ls="none"),
               Line2D([], [], marker="x", ms=13, mew=3.5, color=FALSE_ALARM, ls="none")]
    fig.legend(handles, ["found", "missed", "false alarm"], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.0)
    paths = save_both(fig, out_dir, "poster_detection_examples")
    plt.close(fig)
    return paths
