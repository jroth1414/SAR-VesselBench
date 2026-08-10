"""Generate the paper's result macros, tables, and figures from V100 campaign artifacts.

Reads per-run final_metrics.json, runtime_provenance.json, and metrics/metrics.csv
from the extracted campaign archive, cross-checks every value against the audited
development-score record, verifies the claims made in the manuscript prose, and
emits docs/aipr2026/generated/*.tex plus the figure PDFs. No result number is
hand-copied anywhere in the paper source.

Usage:
    python tools/make_results.py [--runs-root PATH]
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_DIR = Path(__file__).resolve().parent.parent
REPO_DOCS_FIGURES = PAPER_DIR.parent / "figures"
DEFAULT_RUNS = Path(r"D:\v100-archive\run-data\xview3-v100-diagnostic-fresh34-48e10534\runs")

TRACKS = ("Vit", "Cnn")
ROLES = ("Random", "Optical", "Sar", "ImageNet")
FRACS = ((10, "Ten", 12), (25, "TwentyFive", 28), (50, "Fifty", 56), (100, "Hundred", 111))
EXP_PREFIX = {
    ("Vit", "Random"): "vitrand",
    ("Vit", "Optical"): "satdino",
    ("Vit", "Sar"): "sarmae",
    ("Vit", "ImageNet"): "vitin1k",
    ("Cnn", "Random"): "cnnrand",
    ("Cnn", "Optical"): "beS2",
    ("Cnn", "Sar"): "beS1",
    ("Cnn", "ImageNet"): "cnnin1k",
}
ROLE_LABEL = {"Random": "Random", "Optical": "Optical", "Sar": "SAR", "ImageNet": "ImageNet"}
TRACK_LABEL = {"Vit": "ViT", "Cnn": "CNN"}

# Audited best-development F1 record (docs/aipr2026 v100 development-score audit).
EXPECTED_F1 = {
    ("Vit", "Random"): (0.866, 0.861, 0.835, 0.833),
    ("Vit", "Optical"): (0.871, 0.888, 0.850, 0.858),
    ("Vit", "Sar"): (0.880, 0.897, 0.852, 0.859),
    ("Vit", "ImageNet"): (0.891, 0.883, 0.854, 0.864),
    ("Cnn", "Random"): (0.754, 0.771, 0.799, 0.823),
    ("Cnn", "Optical"): (0.749, 0.740, 0.778, 0.797),
    ("Cnn", "Sar"): (0.849, 0.858, 0.844, 0.853),
    ("Cnn", "ImageNet"): (0.882, 0.905, 0.861, 0.862),
}
AUDITED_CORE_GPU_HOURS = 1435.591
AUDITED_REF_GPU_HOURS = 42.178
AUDITED_HARDWARE = "Tesla V100-SXM2-32GB"
CODE_REV = "48e10534"
DETECTOR_HASH = "c42ae65b"
VIT_PARAMS = 89_996_801
CNN_PARAMS = 93_988_865

# Figure identity: role -> (color, marker, linestyle). Palette validated with the
# dataviz six-checks script on a white surface; gray is the deliberate neutral
# baseline (secondary-encoded by dash + x marker), not a categorical slot.
ROLE_STYLE = {
    "ImageNet": ("#2a78d6", "o", "-"),
    "Sar": ("#eb6834", "s", "-"),
    "Optical": ("#1baf7a", "^", "-"),
    "Random": ("#555555", "x", "--"),
}


def fail(msg):
    sys.exit(f"make_results: CHECK FAILED: {msg}")


def load_cell(runs, track, role, fpct):
    exp = f"{EXP_PREFIX[(track, role)]}-f{fpct}-s0"
    d = runs / exp
    fm = json.loads((d / "final_metrics.json").read_text())
    rp = json.loads((d / "runtime_provenance.json").read_text())
    with open(d / "metrics" / "metrics.csv", newline="") as fh:
        evals = [r for r in csv.DictReader(fh) if r.get("dev_f1")]
    if not evals:
        fail(f"{exp}: no dev evaluations in metrics.csv")
    best_row = max(evals, key=lambda r: float(r["dev_f1"]))
    if abs(float(best_row["dev_f1"]) - fm["best_dev_f1"]) > 5e-4:
        fail(f"{exp}: metrics.csv best {best_row['dev_f1']} != marker {fm['best_dev_f1']}")
    if rp["hardware"] != AUDITED_HARDWARE:
        fail(f"{exp}: hardware {rp['hardware']}")
    return {
        "exp": exp,
        "f1": fm["best_dev_f1"],
        "epochs_run": fm["epochs_run"],
        "best_epoch": int(float(best_row["epoch"])),
        "threshold": fm["last_dev"]["threshold"],
        "precision": fm["last_dev"]["precision"],
        "recall": fm["last_dev"]["recall"],
        "gpu_hours": rp["gpu_hours"],
        "finished_utc": rp["finished_utc"],
    }


def r3(x):
    return round(x, 3)


def fmt3(x):
    return f"{x:.3f}"


def macro(name, value):
    return f"\\def\\{name}{{{value}}}"


def num(name, value, signed=False):
    s = f"{value:+.3f}" if signed else f"{value:.3f}"
    return macro(name, f"\\ensuremath{{{s}}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    args = ap.parse_args()
    runs = args.runs_root

    cells = {}
    for track in TRACKS:
        for role in ROLES:
            for fpct, _, _ in FRACS:
                cells[(track, role, fpct)] = load_cell(runs, track, role, fpct)

    # --- audit cross-check: every value must match the recorded development audit
    for (track, role), expected in EXPECTED_F1.items():
        for (fpct, _, _), exp_val in zip(FRACS, expected):
            got = r3(cells[(track, role, fpct)]["f1"])
            if got != exp_val:
                fail(f"{track}/{role}/f{fpct}: {got} != audited {exp_val}")

    core_hours = sum(c["gpu_hours"] for c in cells.values())
    if abs(core_hours - AUDITED_CORE_GPU_HOURS) > 0.01:
        fail(f"core GPU-hours {core_hours:.3f} != audited {AUDITED_CORE_GPU_HOURS}")

    yolo = json.loads((runs / "yolo26-f100" / "final_metrics.json").read_text())
    loc = json.loads((runs / "locateanything-zs" / "final_metrics.json").read_text())
    ref_hours = 0.0
    finished = [c["finished_utc"] for c in cells.values()]
    for ref in ("yolo26-f100", "locateanything-zs"):
        rp = json.loads((runs / ref / "runtime_provenance.json").read_text())
        if rp["hardware"] != AUDITED_HARDWARE:
            fail(f"{ref}: hardware {rp['hardware']}")
        ref_hours += rp["gpu_hours"]
        finished.append(rp["finished_utc"])
    if abs(ref_hours - AUDITED_REF_GPU_HOURS) > 0.01:
        fail(f"reference GPU-hours {ref_hours:.3f} != audited {AUDITED_REF_GPU_HOURS}")

    from datetime import datetime

    done = max(datetime.fromisoformat(t) for t in finished)
    completed_utc = done.strftime("%Y-%m-%d %H:%M UTC")

    def F(track, role, fpct):
        return cells[(track, role, fpct)]["f1"]

    # --- verify every comparative claim made in the manuscript prose
    leaders = {
        (t, f): max(ROLES, key=lambda r: F(t, r, f)) for t in TRACKS for f, _, _ in FRACS
    }
    imagenet_leads = sum(1 for v in leaders.values() if v == "ImageNet")
    if imagenet_leads != 7 or leaders[("Vit", 25)] != "Sar":
        fail(f"leader pattern changed: {leaders}")
    for t in TRACKS:
        for f, _, _ in FRACS:
            if F(t, "Sar", f) <= F(t, "Optical", f):
                fail(f"SAR-optical not positive at {t}/f{f}")
    for f, _, _ in FRACS:
        if F("Cnn", "Optical", f) >= F("Cnn", "Random", f):
            fail(f"CNN optical gain not negative at f{f}")
        for role in ("Sar", "ImageNet"):
            if F("Cnn", role, f) <= F("Cnn", "Random", f):
                fail(f"CNN {role} does not beat random at f{f}")
    scarce = {(r, f): F("Vit", r, f) for r in ROLES for f in (10, 25)}
    if min(scarce, key=scarce.get) != ("Random", 25) or max(scarce, key=scarce.get) != ("Sar", 25):
        fail("ViT scarce-budget range endpoints changed")
    full = {r: F("Vit", r, 100) for r in ROLES}
    if min(full, key=full.get) != "Random" or max(full, key=full.get) != "ImageNet":
        fail("ViT 111-scene range endpoints changed")
    for r in ROLES:
        if max(F("Vit", r, 10), F("Vit", r, 25)) <= F("Vit", r, 100):
            fail(f"ViT {r} does not end below its scarce-label peak")
    floor_vit, floor_cnn = F("Vit", "Random", 100), F("Cnn", "Random", 100)
    for r in ("Optical", "Sar", "ImageNet"):
        if F("Vit", r, 10) < floor_vit:
            fail(f"ViT {r} does not cross the floor at 12 scenes")
    for r in ("Sar", "ImageNet"):
        if F("Cnn", r, 10) < floor_cnn:
            fail(f"CNN {r} does not cross the floor at 12 scenes")
    if any(F("Cnn", "Optical", f) >= floor_cnn for f, _, _ in FRACS):
        fail("CNN optical unexpectedly crosses the floor")
    for t, r in (("Vit", "Random"), ("Vit", "Optical"), ("Vit", "Sar"), ("Vit", "ImageNet"), ("Cnn", "ImageNet")):
        if F(t, r, 25) - F(t, r, 50) <= 0:
            fail(f"25->50 decline missing for {t}/{r}")
    sar_opt_vit = [F("Vit", "Sar", f) - F("Vit", "Optical", f) for f, _, _ in FRACS]
    if max(sar_opt_vit) != max(sar_opt_vit[0], sar_opt_vit[1]) or sar_opt_vit[3] >= 0.005:
        fail("ViT SAR-optical no longer peaks in the scarce regime")

    # --- macros
    lines = [
        "% GENERATED by tools/make_results.py -- do not edit by hand.",
        f"% Source: V100 campaign archive, code revision {CODE_REV}.",
        macro("ResultProvenance", "v100-development-selection"),
        macro("ResultCodeRev", CODE_REV),
        macro("ResultDetectorHash", DETECTOR_HASH),
        macro("ResultPrecision", "32-true"),
        macro("ResultCompletedUTC", completed_utc),
        macro("ResultCoreGPUHours", f"{core_hours:,.1f}"),
        macro("ResultReferenceGPUHours", f"{ref_hours:.1f}"),
        macro("ResultTotalGPUHours", f"{core_hours + ref_hours:,.1f}"),
        macro("ResultVitParameterCount", f"{VIT_PARAMS:,}".replace(",", "{,}")),
        macro("ResultCnnParameterCount", f"{CNN_PARAMS:,}".replace(",", "{,}")),
    ]
    for t in TRACKS:
        for r in ROLES:
            for fpct, fname, _ in FRACS:
                lines.append(num(f"DevF{t}{r}{fname}", r3(F(t, r, fpct))))
    for t in TRACKS:
        for r in ("Optical", "Sar", "ImageNet"):
            for fpct, fname, _ in FRACS:
                g = F(t, r, fpct) - F(t, "Random", fpct)
                lines.append(num(f"Gain{t}{r}{fname}", g, signed=True))
                lines.append(num(f"Gain{t}{r}{fname}Abs", abs(g)))
    for t in TRACKS:
        for fpct, fname, _ in FRACS:
            g = F(t, "Sar", fpct) - F(t, "Optical", fpct)
            lines.append(num(f"SarOpt{t}{fname}", g, signed=True))
            lines.append(num(f"SarOpt{t}{fname}Abs", abs(g)))
    lines.append(num("SarOptVitMaxAbs", max(sar_opt_vit)))
    for t in TRACKS:
        for r in ROLES:
            lines.append(num(f"MonoDrop{t}{r}", F(t, r, 25) - F(t, r, 50)))
    lines.append(num("FloorVit", r3(floor_vit)))
    lines.append(num("FloorCnn", r3(floor_cnn)))
    for t in TRACKS:
        th = [cells[(t, r, f)]["threshold"] for r in ROLES for f, _, _ in FRACS]
        lines.append(num(f"ThreshMin{t}", min(th)))
        lines.append(num(f"ThreshMax{t}", max(th)))
        ep = [cells[(t, r, f)]["best_epoch"] for r in ROLES for f, _, _ in FRACS]
        lines.append(macro(f"MedianBestEpoch{t}", f"{median(ep):.0f}"))
        gh = [cells[(t, r, f)]["gpu_hours"] for r in ROLES for f, _, _ in FRACS]
        lines.append(macro(f"MeanCellHours{t}", f"{sum(gh) / len(gh):.1f}"))
    lines.append(macro("CellsPrecOverRecall", str(sum(1 for c in cells.values() if c["precision"] > c["recall"]))))
    lines.append(macro("CellsEarlyStopped", str(sum(1 for c in cells.values() if c["epochs_run"] < 50))))

    gen = PAPER_DIR / "generated"
    gen.mkdir(exist_ok=True)
    (gen / "v100_results.tex").write_text("\n".join(lines) + "\n", newline="\n")

    # --- core table
    rows = []
    for t in TRACKS:
        for r in ROLES:
            for fpct, _, scenes in FRACS:
                c = cells[(t, r, fpct)]
                rows.append(
                    f"{TRACK_LABEL[t]} & {ROLE_LABEL[r]} & {scenes} & {fmt3(r3(c['f1']))} & "
                    f"{c['best_epoch']} & {c['epochs_run']} & {c['gpu_hours']:.1f} \\\\"
                )
        if t == "Vit":
            rows.append("\\midrule")
    core_table = "\n".join(
        [
            "% GENERATED by tools/make_results.py -- do not edit by hand.",
            "\\begin{table}[tp]",
            "\\caption{Complete 32-cell campaign record. F1 is the best development",
            "score recorded during training at that evaluation's swept threshold; the",
            "best epoch is recovered from the per-evaluation history; epochs and",
            "GPU-hours are measured per cell on one V100. Scene counts give the nested",
            "training subsets.}",
            "\\label{tab:core-results}",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{5pt}",
            "\\begin{tabular}{@{}llrrrrr@{}}",
            "\\toprule",
            "Track & Initialization & Scenes & Dev F1 & Best ep. & Epochs & GPU-h \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    (gen / "v100_core_table.tex").write_text(core_table + "\n", newline="\n")

    # --- reference table
    loc_rows = [
        f"LocateAnything-3B & zero-shot, prompt ``{p}'' & {fmt3(loc['per_prompt'][p]['f1'])} \\\\"
        for p in ("ship", "vessel", "boat")
    ]
    ref_table = "\n".join(
        [
            "% GENERATED by tools/make_results.py -- do not edit by hand.",
            "\\begin{table}[t]",
            "\\caption{External reference systems on the same fixed development",
            "scenes, each under its own published recipe. YOLO26 fine-tunes on the",
            "full 111-scene training split. These systems share no backbone,",
            "pretraining role, or training protocol with the controlled arms and",
            "stay off the label-efficiency curves.}",
            "\\label{tab:references}",
            "\\centering",
            "\\small",
            "\\begin{tabular}{@{}llr@{}}",
            "\\toprule",
            "System & Adaptation & Dev F1 \\\\",
            "\\midrule",
            f"YOLO26 & fine-tuned, 111 scenes & {fmt3(yolo['dev_f1'])} \\\\",
            *loc_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    (gen / "v100_reference_table.tex").write_text(ref_table + "\n", newline="\n")

    # --- figures
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": "#0b0b0b",
            "xtick.color": "#52514e",
            "ytick.color": "#52514e",
            "grid.color": "#e1e0d9",
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
        }
    )
    scenes_x = [s for _, _, s in FRACS]
    figs = PAPER_DIR / "figures"

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.35), sharey=True, layout="constrained")
    for ax, t in zip(axes, TRACKS):
        for r in ("Random", "Optical", "Sar", "ImageNet"):
            color, mark, ls = ROLE_STYLE[r]
            ax.plot(
                scenes_x,
                [F(t, r, f) for f, _, _ in FRACS],
                marker=mark,
                markersize=4,
                linewidth=1.2,
                linestyle=ls,
                color=color,
                label=ROLE_LABEL[r],
            )
        ax.set_title("ViT-B/16" if t == "Vit" else "ConvNeXt-V2-Base", fontsize=8)
        ax.set_xticks(scenes_x)
        ax.set_xlabel("Training scenes")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Development F1")
    axes[0].set_ylim(0.72, 0.92)
    axes[1].legend(loc="lower right", fontsize=7, handlelength=1.8)
    fig.savefig(figs / "label_efficiency.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.35), sharey=True, layout="constrained")
    for ax, t in zip(axes, TRACKS):
        for r in ("Optical", "Sar", "ImageNet"):
            color, mark, ls = ROLE_STYLE[r]
            ax.plot(
                scenes_x,
                [F(t, r, f) - F(t, "Random", f) for f, _, _ in FRACS],
                marker=mark,
                markersize=4,
                linewidth=1.2,
                color=color,
                label=ROLE_LABEL[r],
            )
        ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_title("ViT-B/16" if t == "Vit" else "ConvNeXt-V2-Base", fontsize=8)
        ax.set_xticks(scenes_x)
        ax.set_xlabel("Training scenes")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("$\\Delta$F1 vs. random floor")
    axes[1].legend(loc="upper right", fontsize=7, handlelength=1.8)
    fig.savefig(figs / "transfer_gains.pdf")
    plt.close(fig)

    for img in ("input_triptych.png", "heatmap_overlay.png"):
        shutil.copyfile(REPO_DOCS_FIGURES / img, figs / img)

    fig_label_eff = r"""% GENERATED by tools/make_results.py -- do not edit by hand.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/label_efficiency}
\caption{Development-F1 label-efficiency curves for all 32 cells, one panel per
architecture track. Solid lines are pretrained arms; the dashed gray line is
the track's random floor. Values are seed-0 point estimates; no uncertainty
ribbon is drawn.}
\label{fig:label-efficiency}
\end{figure}
"""
    (gen / "v100_fig_label_efficiency.tex").write_text(fig_label_eff, newline="\n")

    fig_gains = r"""% GENERATED by tools/make_results.py -- do not edit by hand.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/transfer_gains}
\caption{Transfer gain over the track-matched random floor
(Eq.~\ref{eq:transfer-gain}) at each tested budget. The optical CNN arm is
negative at every fraction; the ViT arms compress near zero.}
\label{fig:transfer-gains}
\end{figure}
"""
    (gen / "v100_fig_transfer_gains.tex").write_text(fig_gains, newline="\n")

    fig_qual = r"""% GENERATED by tools/make_results.py -- do not edit by hand.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/input_triptych}\\[4pt]
\includegraphics[width=\textwidth]{figures/heatmap_overlay}
\caption{Qualitative views from development scenes. Top: the fixed
three-channel input, identical for every arm. Bottom: a vessel-and-wake
target and the shared detector's heatmap response from a development-phase
probe run on the frozen splits (peak score 0.22), illustrating why the
protocol pairs a permissive candidate floor with a per-cell threshold
sweep. Illustrative only; not a scored sample.}
\label{fig:qualitative}
\end{figure}
"""
    (gen / "v100_fig_qualitative.tex").write_text(fig_qual, newline="\n")

    # machine-readable dump for provenance
    dump = {
        "runs_root": str(runs),
        "completed_utc": completed_utc,
        "core_gpu_hours": core_hours,
        "reference_gpu_hours": ref_hours,
        "cells": {c["exp"]: c for c in cells.values()},
        "references": {"yolo26-f100": yolo, "locateanything-zs": loc},
    }
    (gen / "paper_data.json").write_text(json.dumps(dump, indent=1), newline="\n")
    print("make_results: all audit cross-checks and prose-claim checks passed")
    print(f"make_results: wrote macros, 2 tables, 3 figure blocks, 2 PDFs under {PAPER_DIR}")


if __name__ == "__main__":
    main()
