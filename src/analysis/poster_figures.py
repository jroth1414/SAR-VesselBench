"""Poster-scale figures and numbers for the AIPR 2026 poster.

One source feeds both poster formats. Every figure is written twice, as a
vector PDF for the tikzposter build and as a 300-dpi PNG for the PowerPoint
build. Every number the poster quotes is written to ``poster_numbers.json``
and, as LaTeX macros, to ``poster_macros.tex``. Values come only through
``heldout_results.validate_evidence``; the optional rerun reference is a
second validated cohort (the August tree, e.g. exported with
``git archive 481200e results/h100/evidence``).

The report-scale renderers in ``heldout_results`` stay untouched; this module
reuses only their role palette, markers and labels.

Usage:
  python -m src.analysis.poster_figures --output-dir docs/poster/generated \
      [--rerun-reference <August evidence tree>]
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

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

TRACKS = ("vit", "cnn")
TRACK_TITLE = {"vit": "ViT-B/16", "cnn": "ConvNeXt-V2-Base"}
PRETRAINED = ("optical", "sar", "imagenet")
INK = "#1F2328"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BAND = "#D9DDE3"
# Poster type scale (points). Body text on the poster is ~25 pt.
TITLE_PT, LABEL_PT, TICK_PT, LEGEND_PT, ANNOT_PT = 30, 27, 24, 24, 23
# Facts from the BigEarthNet-S2 diagnostic (DEV only, outside this evidence
# tree): docs/BES2_FIRST_CONV_RESET_ACADEMIC_REPORT.md on
# origin/sprint-8-final-eval-amendment and the 2026-09-14 owner amendment.
STEM_DIAGNOSTIC = {"weight_norm_ratio": 5.85, "dev_f1_before": 0.799, "dev_f1_after": 0.837}


# --------------------------------------------------------------------------- style
def register_roboto() -> bool:
    """Make the poster's text face available to matplotlib if Tectonic cached it."""

    import glob

    from matplotlib import font_manager

    cache = os.path.join(os.environ.get("LOCALAPPDATA", ""), "TectonicProject", "Tectonic", "cache")
    found = False
    for face in ("Roboto-Regular.otf", "Roboto-Bold.otf"):
        for path in glob.glob(os.path.join(cache, "**", face), recursive=True)[:1]:
            font_manager.fontManager.addfont(path)
            found = True
    return found


def poster_style() -> dict[str, object]:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Roboto", "Arial", "DejaVu Sans"],
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
    fig.savefig(paths[0], metadata={"CreationDate": None})
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


# --------------------------------------------------------------------------- numbers
def poster_numbers(validated: Mapping[str, object], rerun: Mapping[str, float] | None) -> dict[str, object]:
    g = grids(validated)
    final = validated["final_results"]
    first = next(iter(final.values()))  # type: ignore[union-attr]
    n: dict[str, object] = {
        "cells": len(validated["cells"]),  # type: ignore[arg-type]
        "gpu_hours": round(float(validated["campaign"]["gpu_hours"]), 1),  # type: ignore[index]
        "code_sha": str(validated["campaign"]["git_sha"])[:7],  # type: ignore[index]
        "test_scenes": 16,
        "final_scenes": 50,
        "final_vessels": int(first["tp"] + first["fn"]),
        "final_dark": int(first["dark_support"]),
        "final_near_shore": int(first["near_shore_support"]),
    }

    def gains(metric: str, fraction: int) -> list[float]:
        return [g[metric][(t, r, fraction)] - g[metric][(t, "floor", fraction)] for t in TRACKS for r in PRETRAINED]

    for metric in ("test", "final"):
        n[f"{metric}_gain12_min"] = min(gains(metric, 10))
        n[f"{metric}_gain12_max"] = max(gains(metric, 10))
        n[f"{metric}_f1_min"] = min(g[metric].values())
        n[f"{metric}_f1_max"] = max(g[metric].values())
        for track in TRACKS:
            for fraction in (10, 25, 50, 100):
                n[f"{metric}_{track}_sar_minus_opt_{fraction}"] = (
                    g[metric][(track, "sar", fraction)] - g[metric][(track, "optical", fraction)]
                )
            n[f"{metric}_{track}_random_111"] = g[metric][(track, "floor", 100)]
            best12 = max(PRETRAINED, key=lambda r: g[metric][(track, r, 10)])
            n[f"{metric}_{track}_best12_role"] = ROLE_LABEL[best12]
            n[f"{metric}_{track}_best12"] = g[metric][(track, best12, 10)]
    n["cnn_optical_test_gain_min"] = min(g["test"][("cnn", "optical", f)] - g["test"][("cnn", "floor", f)] for f in FRACTIONS)
    n["cnn_optical_test_gain_max"] = max(g["test"][("cnn", "optical", f)] - g["test"][("cnn", "floor", f)] for f in FRACTIONS)
    n["cnn_optical_final_gain_min"] = min(g["final"][("cnn", "optical", f)] - g["final"][("cnn", "floor", f)] for f in FRACTIONS)
    tests = [float(t["f1"]) for t in validated["test_results"].values()]  # type: ignore[union-attr]
    finals = {e: float(r["f1"]) for e, r in final.items()}  # type: ignore[union-attr]
    n["mean_test_minus_final"] = sum(
        float(validated["test_results"][e]["f1"]) - finals[e] for e in finals  # type: ignore[index]
    ) / len(finals)
    n["test_f1_min"], n["test_f1_max"] = min(tests), max(tests)
    for key in ("precision", "recall", "dark_recall", "near_shore_f1"):
        values = [float(r[key]) for r in final.values()]  # type: ignore[union-attr]
        n[f"final_{key}_min"], n[f"final_{key}_max"] = min(values), max(values)
    best = max(finals, key=finals.get)
    n["best_cell"] = best
    n["best_cell_test"] = float(validated["test_results"][best]["f1"])  # type: ignore[index]
    n["best_cell_final"] = finals[best]
    n["vit_sar_test_drop_12_28"] = g["test"][("vit", "sar", 10)] - g["test"][("vit", "sar", 25)]
    n["cnn_sar_test_drop_56_111"] = g["test"][("cnn", "sar", 50)] - g["test"][("cnn", "sar", 100)]
    n["stem_weight_norm_ratio"] = STEM_DIAGNOSTIC["weight_norm_ratio"]
    n["stem_dev_before"] = STEM_DIAGNOSTIC["dev_f1_before"]
    n["stem_dev_after"] = STEM_DIAGNOSTIC["dev_f1_after"]
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

    out = {key: _fmt(value) for key, value in numbers.items()}
    for key, value in numbers.items():
        if ("minus_opt" in key or "gain" in key) and isinstance(value, float):
            out[key] = _signed(value)
    out["stem_weight_norm_ratio"] = f"{float(numbers['stem_weight_norm_ratio']):.2f}×"
    out["gpu_hours"] = f"{float(numbers['gpu_hours']):.1f}"
    return out


DIGIT_WORDS = {"1": "One", "10": "Ten", "12": "Twelve", "25": "TwentyFive", "28": "TwentyEight",
               "50": "Fifty", "56": "FiftySix", "100": "Hundred", "111": "OneEleven"}


def macro_name(key: str) -> str:
    """TeX control sequences take letters only, so digit runs become words."""

    import re

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
        return value.replace("−", "$-$").replace("×", "$\\times$").replace(",", "{,}").replace("_", "\\_")

    lines = ["% GENERATED by src.analysis.poster_figures -- do not edit."]
    lines += [f"\\def\\{macro_name(k)}{{{tex(v)}}}" for k, v in sorted(strings.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- figures
def figure_label_efficiency(g, out_dir: Path) -> list[Path]:
    """2x2: rows = test (16 scenes) and verified (50 scenes); columns = tracks."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 11.0), sharex=True)
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


def figure_sar_minus_optical(g, rerun: Mapping[str, float] | None, out_dir: Path) -> list[Path]:
    """SAR minus optical F1 per budget, test and verified, with the rerun band."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.1), sharey=True)
    styles = (("test", "Test (16 scenes)", "-", "o"), ("final", "Verified (50 scenes)", (0, (1.2, 1.4)), "s"))
    for axis, track in zip(axes, TRACKS):
        if rerun is not None:
            axis.axhspan(-rerun["max"], rerun["max"], color=BAND, zorder=0, lw=0)
        axis.axhline(0, color=MUTED, lw=1.6, zorder=1)
        for metric, label, linestyle, marker in styles:
            ys = [g[metric][(track, "sar", f)] - g[metric][(track, "optical", f)] for f in FRACTIONS]
            axis.plot(range(len(FRACTIONS)), ys, color=INK, linestyle=linestyle, marker=marker,
                      markerfacecolor=INK if metric == "test" else "white", markeredgecolor=INK,
                      markeredgewidth=2.5, label=label, zorder=3, clip_on=False)
        axis.set_title(TRACK_TITLE[track], color=INK)
        axis.set_ylim(-0.10, 0.10)
        fraction_axis(axis, label=True)
    # Direction labels sit outside the data area so they never cover a point.
    axes[1].text(1.03, 0.92, "SAR\nahead", transform=axes[1].transAxes, va="top", ha="left",
                 fontsize=ANNOT_PT, color=ROLE_COLOR["sar"], fontweight="bold", linespacing=1.0)
    axes[1].text(1.03, 0.08, "optical\nahead", transform=axes[1].transAxes, va="bottom", ha="left",
                 fontsize=ANNOT_PT, color="#9A6700", fontweight="bold", linespacing=1.0)
    axes[0].set_ylabel("SAR − optical F1")
    axes[1].tick_params(labelleft=True)
    handles, labels = axes[0].get_legend_handles_labels()
    if rerun is not None:
        handles.append(Patch(facecolor=BAND, edgecolor="none"))
        labels.append(f"Rerun variation (±{rerun['max']:.3f})")
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06),
               handlelength=2.4, columnspacing=1.4)
    fig.tight_layout(w_pad=2.4, rect=(0, 0, 1, 0.94))
    paths = save_both(fig, out_dir, "poster_sar_minus_optical")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evidence-root", default=Path("results/h100/evidence"), type=Path)
    parser.add_argument("--rerun-reference", type=Path)
    parser.add_argument("--output-dir", default=Path("docs/poster/generated"), type=Path)
    parser.add_argument("--arms-config", default=Path("configs/arms.yaml"), type=Path)
    parser.add_argument("--detector-config", default=Path("configs/detector.yaml"), type=Path)
    parser.add_argument("--splits-config", default=Path("data/splits.json"), type=Path)
    args = parser.parse_args(argv)
    kwargs = {"arms_config": args.arms_config, "detector_config": args.detector_config,
              "splits_config": args.splits_config}
    validated = validate_evidence(args.evidence_root, **kwargs)
    if not validated["final_results"] or not validated["test_results"]:
        raise EvidenceError("the poster needs the complete TEST and final evaluations")
    rerun = None
    if args.rerun_reference is not None:
        rerun = rerun_variation(validated, validate_evidence(args.rerun_reference, **kwargs))

    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib

    matplotlib.use("Agg")
    register_roboto()
    matplotlib.rcParams.update(poster_style())

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    g = grids(validated)
    figure_label_efficiency(g, out)
    figure_sar_minus_optical(g, rerun, out)
    numbers = poster_numbers(validated, rerun)
    strings = number_strings(numbers)
    (out / "poster_numbers.json").write_text(
        json.dumps({"values": numbers, "display": strings}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    write_macros(strings, out / "poster_macros.tex")
    print(json.dumps({"figures": 2, "numbers": len(numbers), "rerun": rerun}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
