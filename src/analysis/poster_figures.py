"""Poster-scale figures and numbers for the AIPR 2026 poster.

One source feeds both poster formats. Every figure is written twice, as a
vector PDF for the tikzposter build and as a 300-dpi PNG for the PowerPoint
build. Every number the poster quotes is written to ``poster_numbers.json``
and, as LaTeX macros, to ``poster_macros.tex``. Values come only through
``heldout_results.validate_evidence`` and the per-scene records of the files
it validated:

* 95% intervals come from a paired scene bootstrap (``scene_bootstrap``);
  they cover scene sampling, not training-seed variation;
* the optional rerun reference is a second validated cohort (the August
  tree, e.g. ``git archive 481200e results/h100/evidence``) and measures how
  far test F1 moves when unchanged cells are retrained;
* the near-shore label gap uses the evidence tree's bound
  ``TRAIN_LABEL_PROFILE.json``;
* the distance-to-shore figure and the example crops (``poster_coastal``)
  read local xView3 labels and imagery, each checked before use.

The report-scale renderers in ``heldout_results`` stay untouched; this module
reuses only their role palette, markers and labels.

Usage:
  python -m src.analysis.poster_figures --output-dir docs/poster/generated \
      [--rerun-reference <August evidence tree>] \
      [--train-labels train.csv --verified-labels validation.csv \
       --verified-imagery <validation archives> --imagery-cache <dir>] \
      [--context-scene <xView3 GRD dir>/835f7629c3a3a9abt]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.analysis.heldout_results import (
    FRACTION_SCENES,
    FRACTIONS,
    ROLE_COLOR,
    ROLE_LABEL,
    ROLE_MARKER,
    ROLE_ORDER,
    EvidenceError,
    validate_evidence,
)
from src.analysis.scene_bootstrap import PairedBootstrap, micro_f1, scene_counts

TRACKS = ("vit", "cnn")
TRACK_TITLE = {"vit": "ViT-B/16", "cnn": "ConvNeXt-V2-Base"}
PRETRAINED = ("optical", "sar", "imagenet")
METRIC_FILE = {"test": "test_metrics.json", "final": "final_verified_metrics.json"}
NEAR_SHORE_KM = 2.0
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 10_000, 0
# Training curves behind the ViT-SAR dip at 28 scenes (quoted in Limits), with
# SatDINO as the same-budget comparison.
DYNAMICS_CELLS = (("vit", "sar", 10), ("vit", "sar", 25), ("vit", "sar", 50), ("vit", "optical", 25))
INK = "#1F2328"
MUTED = "#6B7280"
GRID = "#E5E7EB"
OPTICAL_TEXT = "#9A6700"
# Poster type scale (points), sized against ~30 pt body text on the poster.
TITLE_PT, LABEL_PT, TICK_PT, LEGEND_PT, ANNOT_PT = 30, 27, 24, 24, 23


# --------------------------------------------------------------------------- style
def poster_style() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": TICK_PT,
        "axes.titlesize": TITLE_PT,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": LABEL_PT,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 1.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 1.4,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "legend.fontsize": LEGEND_PT,
        "legend.frameon": False,
        "lines.linewidth": 4.5,
        "lines.markersize": 17,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
    }


def save_both(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    # In the PDF, dpi sets only the resolution of embedded imagery (lines and
    # text stay vector); the default 100 dpi printed SAR crops at ~100 ppi.
    fig.savefig(paths[0], dpi=300, metadata={"CreationDate": None})
    fig.savefig(paths[1], dpi=300)
    return paths


def role_line_kwargs(role: str) -> dict[str, object]:
    return {
        "color": ROLE_COLOR[role],
        "marker": ROLE_MARKER[role],
        "linestyle": (0, (4, 2.5)) if role == "floor" else "-",
        "markeredgecolor": "white",
        "markeredgewidth": 2.5,
        "label": ROLE_LABEL[role],
        "zorder": 2 if role == "floor" else 3,
        "clip_on": False,
    }


def fraction_axis(axis, *, label: bool) -> None:
    axis.set_xticks(range(len(FRACTIONS)))
    axis.set_xticklabels([str(FRACTION_SCENES[f]) for f in FRACTIONS])
    axis.set_xlim(-0.25, len(FRACTIONS) - 1 + 0.25)
    if label:
        axis.set_xlabel("Training scenes")


# --------------------------------------------------------------------------- data
def _cell_ids(validated: Mapping[str, object]) -> dict[tuple[str, str, int], str]:
    return {
        (c["track"], c["role"], int(c["label_fraction"])): exp_id
        for exp_id, c in validated["cells"].items()  # type: ignore[union-attr]
    }


def grids(validated: Mapping[str, object]) -> dict[str, dict[tuple[str, str, int], float]]:
    cells = validated["cells"]
    tests = validated["test_results"]
    final = validated["final_results"]
    out: dict[str, dict[tuple[str, str, int], float]] = {"dev": {}, "test": {}, "final": {}}
    for exp_id, cell in cells.items():  # type: ignore[union-attr]
        key = (cell["track"], cell["role"], int(cell["label_fraction"]))
        out["dev"][key] = float(cell["dev_f1"])
        out["test"][key] = float(tests[exp_id]["f1"])  # type: ignore[index]
        out["final"][key] = float(final[exp_id]["f1"])  # type: ignore[index]
    return out


def rerun_variation(current: Mapping[str, object], reference: Mapping[str, object]) -> dict[str, float]:
    """Test-F1 differences for cells whose recipe did not change between cohorts.

    The CNN-optical arm changed its input adapter, so it is excluded.
    """

    deltas = []
    for exp_id, cell in current["cells"].items():  # type: ignore[union-attr]
        if cell["track"] == "cnn" and cell["role"] == "optical":
            continue
        a = float(current["test_results"][exp_id]["f1"])  # type: ignore[index]
        b = float(reference["test_results"][exp_id]["f1"])  # type: ignore[index]
        deltas.append(abs(a - b))
    return {"cells": len(deltas), "mean": sum(deltas) / len(deltas), "max": max(deltas)}


def scene_details(evidence_root: Path, validated: Mapping[str, object]) -> dict[str, object]:
    """Bootstrap intervals and near-shore breakdowns from the validated per-scene records."""

    ids = _cell_ids(validated)
    details: dict[str, object] = {"intervals": {}, "offshore": {}}
    for metric, filename in METRIC_FILE.items():
        payloads = {key: json.loads((evidence_root / exp_id / filename).read_text(encoding="utf-8"))
                    for key, exp_id in ids.items()}
        counts = {}
        boot = None
        for key, payload in payloads.items():
            scenes, array = scene_counts(payload["per_scene"])
            if boot is None:
                boot = PairedBootstrap(scenes, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
            boot.check(scenes)
            counts[key] = array
        intervals = details["intervals"]
        for track in TRACKS:
            for role in PRETRAINED:
                intervals[f"{metric}_{track}_{role}_gain12"] = boot.difference(
                    counts[(track, role, 10)], counts[(track, "floor", 10)])
            for fraction in FRACTIONS:
                intervals[f"{metric}_{track}_sar_minus_opt_{fraction}"] = boot.difference(
                    counts[(track, "sar", fraction)], counts[(track, "optical", fraction)])
                intervals[f"{metric}_{track}_sar_minus_imagenet_{fraction}"] = boot.difference(
                    counts[(track, "sar", fraction)], counts[(track, "imagenet", fraction)])
        intervals[f"{metric}_vit_imagenet12_minus_random111"] = boot.difference(
            counts[("vit", "imagenet", 10)], counts[("vit", "floor", 100)])
        if metric == "final":
            for key, payload in payloads.items():
                total = np.array([sum(s["aggregate"][k] for s in payload["per_scene"].values())
                                  for k in ("tp", "fp", "fn")], dtype=float)
                near = np.array([sum(s["slices"]["near_shore"][k] for s in payload["per_scene"].values())
                                 for k in ("tp", "fp", "fn")], dtype=float)
                predictions = [p for scene in payload["thresholded_predictions"].values() for p in scene]
                details["offshore"][key] = {
                    "offshore_f1": float(micro_f1(total - near)),
                    "near_shore_recall": float(near[0] / (near[0] + near[2])),
                    "predictions": len(predictions),
                    "near_shore_predictions": sum(
                        1 for p in predictions if p["distance_from_shore_km"] <= NEAR_SHORE_KM),
                }
    details["curves"] = {key: training_curve(evidence_root / ids[key] / "metrics.csv") for key in DYNAMICS_CELLS}
    return details


def training_curve(path: Path) -> dict[str, object]:
    """Dev evaluations (epoch, F1, precision, recall) and mean training loss per epoch."""

    dev: list[tuple[int, float, float, float]] = []
    losses: dict[int, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not (row.get("epoch") or "").strip():
                continue  # learning-rate-only rows carry no epoch
            epoch = int(float(row["epoch"]))
            if (row.get("dev_f1") or "").strip():
                dev.append((epoch, float(row["dev_f1"]), float(row["dev_precision"]), float(row["dev_recall"])))
            if (row.get("train_loss") or "").strip():
                losses.setdefault(epoch, []).append(float(row["train_loss"]))
    return {"dev": dev, "loss": {e: sum(v) / len(v) for e, v in sorted(losses.items())}}


# --------------------------------------------------------------------------- numbers
def poster_numbers(
    validated: Mapping[str, object],
    rerun: Mapping[str, float] | None,
    details: Mapping[str, object],
    coastal: Mapping[str, float] | None = None,
) -> dict[str, object]:
    g = grids(validated)
    ids = _cell_ids(validated)
    final = validated["final_results"]
    first = next(iter(final.values()))  # type: ignore[union-attr]
    vessels = int(first["tp"] + first["fn"])
    n: dict[str, object] = {
        "cells": len(validated["cells"]),  # type: ignore[arg-type]
        "gpu_hours": round(float(validated["campaign"]["gpu_hours"]), 1),  # type: ignore[index]
        "test_scenes": 16,
        "final_scenes": 50,
        "final_vessels": vessels,
        "final_dark": int(first["dark_support"]),
        "final_near_shore": int(first["near_shore_support"]),
        "final_near_shore_pct": 100.0 * int(first["near_shore_support"]) / vessels,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    }
    train = validated.get("train_labels")
    if train:
        n["train_positive"] = int(train["positive"])
        n["train_near_shore"] = int(train["near_shore_positive"])
        n["train_near_shore_pct"] = 100.0 * int(train["near_shore_positive"]) / int(train["positive"])

    def gains(metric: str, fraction: int) -> list[float]:
        return [g[metric][(t, r, fraction)] - g[metric][(t, "floor", fraction)] for t in TRACKS for r in PRETRAINED]

    intervals: Mapping[str, Mapping[str, float]] = details["intervals"]  # type: ignore[assignment]
    gain_keys = [k for k in intervals if k.endswith("_gain12")]
    n["gain12_comparisons"] = len(gain_keys)
    n["gain12_ci_above_zero"] = sum(1 for k in gain_keys if intervals[k]["low"] > 0)
    # Head-to-head tallies: a win needs the whole 95% interval on one side of zero.
    for rival, tag in (("opt", "optical"), ("imagenet", "imagenet")):
        keys = [k for k in intervals if f"_sar_minus_{rival}_" in k]
        n[f"sar_vs_{tag}_comparisons"] = len(keys)
        n[f"sar_vs_{tag}_sar_wins"] = sum(1 for k in keys if intervals[k]["low"] > 0)
        n[f"sar_vs_{tag}_rival_wins"] = sum(1 for k in keys if intervals[k]["high"] < 0)
    for metric in ("test", "final"):
        n[f"{metric}_gain12_min"] = min(gains(metric, 10))
        n[f"{metric}_gain12_max"] = max(gains(metric, 10))
        n[f"{metric}_f1_min"] = min(g[metric].values())
        n[f"{metric}_f1_max"] = max(g[metric].values())
        for track in TRACKS:
            for fraction in FRACTIONS:
                key = f"{metric}_{track}_sar_minus_opt_{fraction}"
                n[key] = g[metric][(track, "sar", fraction)] - g[metric][(track, "optical", fraction)]
                n[f"{key}_ci"] = intervals[key]
            n[f"{metric}_{track}_random_111"] = g[metric][(track, "floor", 100)]
        n[f"{metric}_vit_imagenet12"] = g[metric][("vit", "imagenet", 10)]
        n[f"{metric}_vit_imagenet12_minus_random111_ci"] = intervals[f"{metric}_vit_imagenet12_minus_random111"]
    n["cnn_optical_test_gain_min"] = min(g["test"][("cnn", "optical", f)] - g["test"][("cnn", "floor", f)] for f in FRACTIONS)
    n["cnn_optical_test_gain_max"] = max(g["test"][("cnn", "optical", f)] - g["test"][("cnn", "floor", f)] for f in FRACTIONS)
    tests = {e: float(t["f1"]) for e, t in validated["test_results"].items()}  # type: ignore[union-attr]
    finals = {e: float(r["f1"]) for e, r in final.items()}  # type: ignore[union-attr]
    n["mean_test_minus_final"] = sum(tests[e] - finals[e] for e in finals) / len(finals)
    for key in ("precision", "recall", "dark_recall", "near_shore_f1"):
        values = [float(r[key]) for r in final.values()]  # type: ignore[union-attr]
        n[f"final_{key}_min"], n[f"final_{key}_max"] = min(values), max(values)
    offshore: Mapping[tuple, Mapping[str, float]] = details["offshore"]  # type: ignore[assignment]
    n["final_offshore_f1_min"] = min(o["offshore_f1"] for o in offshore.values())
    n["final_offshore_f1_max"] = max(o["offshore_f1"] for o in offshore.values())
    n["final_near_shore_recall_max"] = max(o["near_shore_recall"] for o in offshore.values())
    best_key = max(offshore, key=lambda k: finals[ids[k]])
    best = offshore[best_key]
    n["best_cell"] = ids[best_key]
    n["best_cell_label"] = (f"{ROLE_LABEL[best_key[1]]} {best_key[0].upper()}, "
                            f"{FRACTION_SCENES[best_key[2]]} scenes")
    n["best_cell_test"] = tests[ids[best_key]]
    n["best_cell_final"] = finals[ids[best_key]]
    n["best_cell_offshore_f1"] = best["offshore_f1"]
    n["best_cell_predictions"] = int(best["predictions"])
    n["best_cell_near_shore_predictions"] = int(best["near_shore_predictions"])
    n["best_cell_near_shore_f1"] = float(final[ids[best_key]]["near_shore_f1"])  # type: ignore[index]
    if coastal is not None:
        n.update(coastal)
    cells = validated["cells"]
    for label, key in (("sarmae_f10", ("vit", "sar", 10)), ("sarmae_f25", ("vit", "sar", 25)),
                       ("sarmae_f50", ("vit", "sar", 50)), ("satdino_f25", ("vit", "optical", 25))):
        n[f"{label}_best_epoch"] = int(cells[ids[key]]["best_epoch"])  # type: ignore[index]
        n[f"{label}_epochs_run"] = int(cells[ids[key]]["epochs_run"])  # type: ignore[index]
    curve = details["curves"][("vit", "sar", 25)]  # type: ignore[index]
    first_eval, last_eval = curve["dev"][0], curve["dev"][-1]
    n["sarmae_f25_dev_precision_first"], n["sarmae_f25_dev_precision_last"] = first_eval[2], last_eval[2]
    n["sarmae_f25_dev_recall_first"], n["sarmae_f25_dev_recall_last"] = first_eval[3], last_eval[3]
    losses = curve["loss"]
    n["sarmae_f25_loss_first"], n["sarmae_f25_loss_last"] = losses[min(losses)], losses[max(losses)]
    n["vit_sar_test_drop_12_28"] = g["test"][("vit", "sar", 10)] - g["test"][("vit", "sar", 25)]
    n["cnn_sar_test_drop_56_111"] = g["test"][("cnn", "sar", 50)] - g["test"][("cnn", "sar", 100)]
    if rerun is not None:
        n["rerun_cells"] = int(rerun["cells"])
        n["rerun_mean"] = rerun["mean"]
        n["rerun_max"] = rerun["max"]
    return n


def _fmt(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.3f}"


def _signed(value: float) -> str:
    return f"{'+' if value >= 0 else '−'}{abs(value):.3f}"


def number_strings(numbers: Mapping[str, object]) -> dict[str, str]:
    """Display strings shared by both poster builds (LaTeX and PowerPoint)."""

    out: dict[str, str] = {}
    for key, value in numbers.items():
        if isinstance(value, Mapping):  # a bootstrap interval
            out[key] = f"[{_signed(value['low'])}, {_signed(value['high'])}]"
        elif ("minus_opt" in key or "gain" in key) and isinstance(value, float):
            out[key] = _signed(value)
        elif key.endswith("_pct"):
            out[key] = f"{value:.1f}%" if value < 10 else f"{value:.0f}%"
        else:
            out[key] = _fmt(value)
    out["gpu_hours"] = f"{float(numbers['gpu_hours']):.1f}"
    return out


DIGIT_WORDS = {"1": "One", "10": "Ten", "12": "Twelve", "25": "TwentyFive", "28": "TwentyEight",
               "50": "Fifty", "56": "FiftySix", "100": "Hundred", "111": "OneEleven"}


def macro_name(key: str) -> str:
    """TeX control sequences take letters only, so digit runs become words."""

    words = []
    for part in key.replace("-", "_").split("_"):
        part = re.sub(r"\d+", lambda m: DIGIT_WORDS[m.group(0)], part)
        words.append(part[:1].upper() + part[1:])
    name = "PN" + "".join(words)
    if not name.isalpha():
        raise ValueError(f"macro name is not letters-only: {name}")
    return name


def write_macros(strings: Mapping[str, str], path: Path) -> None:

    def tex(value: str) -> str:
        return (value.replace("−", "$-$").replace("×", "$\\times$").replace(",", "{,}")
                .replace("_", "\\_").replace("%", "\\%"))

    lines = ["% GENERATED by src.analysis.poster_figures -- do not edit."]
    lines += [f"\\def\\{macro_name(k)}{{{tex(v)}}}" for k, v in sorted(strings.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- figures
def figure_label_efficiency(g, out_dir: Path) -> list[Path]:
    """2x2: rows = test (16 scenes) and verified (50 scenes); columns = tracks."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.8), sharex=True)
    rows = (("test", "Test F1 (16 scenes)", (0.64, 0.90)),
            ("final", "Verified F1 (50 scenes)", (0.40, 0.59)))
    for r, (metric, ylabel, ylim) in enumerate(rows):
        for c, track in enumerate(TRACKS):
            axis = axes[r][c]
            for role in ROLE_ORDER:
                ys = [g[metric][(track, role, f)] for f in FRACTIONS]
                axis.plot(range(len(FRACTIONS)), ys, **role_line_kwargs(role))
            axis.set_ylim(*ylim)
            fraction_axis(axis, label=(r == 1))
            if r == 0:
                axis.set_title(TRACK_TITLE[track], color=INK)
            if c == 0:
                axis.set_ylabel(ylabel)
            else:
                axis.tick_params(labelleft=True)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.035),
               handlelength=2.6, columnspacing=1.6)
    fig.tight_layout(h_pad=2.2, w_pad=2.4, rect=(0, 0, 1, 0.965))
    paths = save_both(fig, out_dir, "poster_label_efficiency")
    plt.close(fig)
    return paths


def figure_sar_minus_optical(details: Mapping[str, object], out_dir: Path) -> list[Path]:
    """SAR minus optical F1 per budget with 95% paired scene-bootstrap intervals."""

    import matplotlib.pyplot as plt

    intervals: Mapping[str, Mapping[str, float]] = details["intervals"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.9), sharey=True)
    styles = (("test", "Test (16 scenes)", "-", "o", -0.09), ("final", "Verified (50 scenes)", (0, (1.2, 1.4)), "s", 0.09))
    for axis, track in zip(axes, TRACKS):
        axis.axhline(0, color=MUTED, lw=1.6, zorder=1)
        for metric, label, linestyle, marker, offset in styles:
            spans = [intervals[f"{metric}_{track}_sar_minus_opt_{f}"] for f in FRACTIONS]
            xs = np.arange(len(FRACTIONS)) + offset
            ys = np.array([s["point"] for s in spans])
            err = np.array([[s["point"] - s["low"] for s in spans], [s["high"] - s["point"] for s in spans]])
            axis.errorbar(xs, ys, yerr=err, color=INK, linestyle=linestyle, marker=marker,
                          markerfacecolor=INK if metric == "test" else "white", markeredgecolor=INK,
                          markeredgewidth=2.5, elinewidth=2.4, capsize=7, capthick=2.4, label=label,
                          zorder=3, clip_on=False)
        axis.set_title(TRACK_TITLE[track], color=INK)
        axis.set_ylim(-0.18, 0.14)
        fraction_axis(axis, label=True)
    # Direction labels sit outside the data area so they never cover a point.
    axes[1].text(1.03, 0.94, "SAR\nahead", transform=axes[1].transAxes, va="top", ha="left",
                 fontsize=ANNOT_PT, color=ROLE_COLOR["sar"], fontweight="bold", linespacing=1.0)
    axes[1].text(1.03, 0.06, "optical\nahead", transform=axes[1].transAxes, va="bottom", ha="left",
                 fontsize=ANNOT_PT, color=OPTICAL_TEXT, fontweight="bold", linespacing=1.0)
    axes[0].set_ylabel("SAR − optical F1")
    axes[1].tick_params(labelleft=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06),
               handlelength=2.4, columnspacing=2.0)
    fig.tight_layout(w_pad=2.4, rect=(0, 0, 1, 0.94))
    paths = save_both(fig, out_dir, "poster_sar_minus_optical")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- main
def _coastal(args, validated, details, kwargs) -> tuple[dict[str, object], tuple]:
    """Distance-to-shore figure, example crops and their numbers (needs local xView3 data)."""

    import tempfile

    from src.analysis import poster_coastal as pc

    ids = _cell_ids(validated)
    root = args.evidence_root
    audit = json.loads((root / "EVAL_GROUND_TRUTH_VALIDATED.json").read_text(encoding="utf-8"))
    splits = json.loads(kwargs["splits_config"].read_text(encoding="utf-8"))
    train_counts = pc.training_bin_counts(args.train_labels, list(map(str, splits["splits"]["train"])), audit)
    rescored = pc.rescore_verified(root, ids, args.verified_labels)
    finals = {e: float(r["f1"]) for e, r in validated["final_results"].items()}  # type: ignore[union-attr]
    best_key = max(ids, key=lambda k: finals[ids[k]])
    verified_counts = next(iter(rescored["bins"].values()))[1]
    style = {"legend": LEGEND_PT - 2, "best_color": ROLE_COLOR[best_key[1]],
             "best_label": "best cell"}  # the poster captions name it
    pc.figure_recall_by_distance(train_counts, verified_counts, rescored, best_key, args.output_dir, style)
    crops = pc.choose_crops(rescored["scenes"][best_key])
    cache = args.imagery_cache or Path(tempfile.gettempdir()) / "xview3-vh-cache"
    pc.figure_detection_examples(crops, rescored["scenes"][best_key], args.verified_imagery, cache, args.output_dir)
    return pc.coastal_numbers(train_counts, verified_counts, rescored, best_key, crops), best_key


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-root", default=Path("results/h100/evidence"), type=Path)
    parser.add_argument("--rerun-reference", type=Path)
    parser.add_argument("--output-dir", default=Path("docs/poster/generated"), type=Path)
    parser.add_argument("--arms-config", default=Path("configs/arms.yaml"), type=Path)
    parser.add_argument("--detector-config", default=Path("configs/detector.yaml"), type=Path)
    parser.add_argument("--splits-config", default=Path("data/splits.json"), type=Path)
    # Near-shore block: inputs outside the evidence tree, each checked before use (see poster_coastal).
    parser.add_argument("--train-labels", type=Path, help="xView3 train.csv (hash-bound by the audit receipt)")
    parser.add_argument("--verified-labels", type=Path, help="xView3 validation.csv behind the 50 verified scenes")
    parser.add_argument("--verified-imagery", type=Path, help="directory of <scene>.tar.gz validation archives")
    parser.add_argument("--imagery-cache", type=Path, help="where extracted VH rasters are kept between runs")
    parser.add_argument("--context-scene", type=Path,
                        help="xView3 scene directory with VH_dB.tif for the scene-and-chip figure (poster_scene)")
    args = parser.parse_args(argv)
    coastal_inputs = (args.train_labels, args.verified_labels, args.verified_imagery)
    if any(coastal_inputs) and not all(coastal_inputs):
        parser.error("--train-labels, --verified-labels and --verified-imagery go together")
    kwargs = {"arms_config": args.arms_config, "detector_config": args.detector_config,
              "splits_config": args.splits_config}
    validated = validate_evidence(args.evidence_root, **kwargs)
    if not validated["final_results"] or not validated["test_results"] or not validated["train_labels"]:
        raise EvidenceError("the poster needs the complete TEST and final evaluations and the label profile")
    rerun = None
    if args.rerun_reference is not None:
        rerun = rerun_variation(validated, validate_evidence(args.rerun_reference, **kwargs))
    details = scene_details(args.evidence_root, validated)

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(poster_style())

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    figure_label_efficiency(grids(validated), out)
    figure_sar_minus_optical(details, out)
    figures = 2
    coastal = None
    if args.train_labels is not None:
        coastal, best_key = _coastal(args, validated, details, kwargs)
        figures += 2
    if args.context_scene is not None:
        import tempfile

        from src.analysis.poster_scene import figure_scene_context

        figure_scene_context(args.context_scene, args.imagery_cache or Path(tempfile.gettempdir()) / "xview3-vh-cache",
                             out)
        figures += 1
    numbers = poster_numbers(validated, rerun, details, coastal)
    strings = number_strings(numbers)
    (out / "poster_numbers.json").write_text(
        json.dumps({"values": numbers, "display": strings}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    write_macros(strings, out / "poster_macros.tex")
    print(json.dumps({"figures": figures, "numbers": len(numbers), "rerun": rerun}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
