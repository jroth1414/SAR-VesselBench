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
import numpy as np

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
        "history": [(int(float(r["epoch"])), float(r["dev_f1"])) for r in evals],
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
    ap.add_argument("--labels-train", type=Path, default=Path(r"D:\train.csv"))
    ap.add_argument("--labels-val", type=Path, default=Path(r"D:\validation.csv"))
    ap.add_argument("--splits", type=Path, default=PAPER_DIR.parent.parent / "data" / "splits.json")
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
            "\\scriptsize",
            "\\renewcommand{\\arraystretch}{0.95}",
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

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.8), sharey=True, layout="constrained")
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

    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.78), sharey=True, layout="constrained")
    for ax, t in zip(axes[:2], TRACKS):
        for r in ("Optical", "Sar", "ImageNet"):
            color, mark, ls = ROLE_STYLE[r]
            ax.plot(
                scenes_x,
                [F(t, r, f) - F(t, "Random", f) for f, _, _ in FRACS],
                marker=mark,
                markersize=3.5,
                linewidth=1.1,
                color=color,
                label=ROLE_LABEL[r],
            )
        ax.set_title("ViT-B/16" if t == "Vit" else "ConvNeXt-V2-B", fontsize=8)
    for t, ls, mark, fill in (("Vit", "--", "o", "none"), ("Cnn", "-", "o", "#0b0b0b")):
        axes[2].plot(
            scenes_x,
            [F(t, "Sar", f) - F(t, "Optical", f) for f, _, _ in FRACS],
            marker=mark,
            markersize=3.5,
            linewidth=1.1,
            linestyle=ls,
            color="#0b0b0b",
            markerfacecolor=fill,
            label=TRACK_LABEL[t],
        )
    axes[2].set_title("SAR $-$ optical", fontsize=8)
    axes[2].legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    for ax in axes:
        ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_xticks(scenes_x)
        ax.tick_params(axis="x", labelsize=6.5)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_xlabel("Training scenes")
    axes[0].set_ylabel("$\\Delta$F1")
    axes[1].legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    fig.savefig(figs / "transfer_gains.pdf")
    plt.close(fig)

    # operating points: PR scatter with iso-F1, threshold strip, cost scatter
    track_fill = {"Vit": True, "Cnn": False}
    track_marker = {"Vit": "o", "Cnn": "s"}
    fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.7), layout="constrained")
    ax = axes[0]
    rec = [c["recall"] for c in cells.values()]
    prc = [c["precision"] for c in cells.values()]
    lo = min(min(rec), min(prc)) - 0.03
    hi = max(max(rec), max(prc)) + 0.02
    rr = np.linspace(lo, hi, 300)
    for fval in (0.7, 0.8, 0.9):
        pp = fval * rr / np.maximum(2 * rr - fval, 1e-9)
        m = (rr > fval / 2 + 0.02) & (pp >= lo) & (pp <= hi)
        ax.plot(rr[m], pp[m], color="#e1e0d9", linewidth=0.7, zorder=1)
        if m.any():
            ax.annotate(f"F1={fval}", (rr[m][-1], pp[m][-1]), fontsize=5,
                        color="#898781", ha="right", va="bottom")
    ax.plot([lo, hi], [lo, hi], color="#c3c2b7", linewidth=0.6, linestyle=":", zorder=1)
    for t in TRACKS:
        for r0 in ROLES:
            color = ROLE_STYLE[r0][0]
            xs = [cells[(t, r0, f)]["recall"] for f, _, _ in FRACS]
            ys = [cells[(t, r0, f)]["precision"] for f, _, _ in FRACS]
            ax.scatter(xs, ys, s=12, marker=track_marker[t], zorder=3,
                       facecolors=color if track_fill[t] else "none",
                       edgecolors=color, linewidths=0.9)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax = axes[1]
    for i, r0 in enumerate(ROLES):
        color = ROLE_STYLE[r0][0]
        for t, dx in (("Vit", -0.17), ("Cnn", 0.17)):
            ys = [cells[(t, r0, f)]["threshold"] for f, _, _ in FRACS]
            ax.scatter([i + dx] * len(ys), ys, s=12, marker=track_marker[t],
                       facecolors=color if track_fill[t] else "none",
                       edgecolors=color, linewidths=0.9)
    ax.set_xticks(range(len(ROLES)))
    ax.set_xticklabels(["Rand", "Opt", "SAR", "IN"], fontsize=6.5)
    ax.set_ylabel("Threshold")
    ax.set_xlim(-0.6, len(ROLES) - 0.4)

    ax = axes[2]
    for t in TRACKS:
        for r0 in ROLES:
            color = ROLE_STYLE[r0][0]
            xs = [cells[(t, r0, f)]["gpu_hours"] for f, _, _ in FRACS]
            ys = [cells[(t, r0, f)]["f1"] for f, _, _ in FRACS]
            ax.scatter(xs, ys, s=12, marker=track_marker[t],
                       facecolors=color if track_fill[t] else "none",
                       edgecolors=color, linewidths=0.9)
    ax.set_xlabel("GPU-hours")
    ax.set_ylabel("Dev F1")
    for ax in axes:
        ax.grid(True, linewidth=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.savefig(figs / "operating_points.pdf")
    plt.close(fig)

    # training dynamics: dev-F1 evaluation history, scarce vs full budgets
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 1.75), sharey=True, layout="constrained")
    for ax, t in zip(axes, TRACKS):
        for r0 in ROLES:
            color = ROLE_STYLE[r0][0]
            for fpct, ls in ((10, "-"), (100, ":")):
                h = cells[(t, r0, fpct)]["history"]
                ax.plot([e for e, _ in h], [v for _, v in h], color=color,
                        linestyle=ls, linewidth=1.0)
                be, bf = max(h, key=lambda p: p[1])
                ax.plot([be], [bf], marker=".", color=color, markersize=5)
        ax.set_title("ViT-B/16" if t == "Vit" else "ConvNeXt-V2-Base", fontsize=8)
        ax.set_xlabel("Epoch")
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Development F1")
    from matplotlib.lines import Line2D

    axes[0].legend(
        handles=[Line2D([], [], color=ROLE_STYLE[r0][0], linewidth=1.2,
                        label=ROLE_LABEL[r0]) for r0 in ROLES],
        loc="lower right", fontsize=6.5, handlelength=1.6)
    axes[1].legend(
        handles=[Line2D([], [], color="#0b0b0b", linestyle="-", label="12 scenes"),
                 Line2D([], [], color="#0b0b0b", linestyle=":", label="111 scenes")],
        loc="lower right", fontsize=6.5, handlelength=1.6)
    fig.savefig(figs / "training_dynamics.pdf")
    plt.close(fig)

    # scene map: study-cohort geographic distribution
    def scene_centroids(csv_path):
        acc = {}
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    la, lo_ = float(row["detect_lat"]), float(row["detect_lon"])
                except (ValueError, KeyError):
                    continue
                acc.setdefault(row["scene_id"], []).append((la, lo_))
        return {s: (median(x for x, _ in v), median(y for _, y in v)) for s, v in acc.items()}

    splits = json.loads(args.splits.read_text())["splits"]
    cent_train = scene_centroids(args.labels_train)
    cent_val = scene_centroids(args.labels_val)
    split_style = [
        ("train", cent_train, "#c3c2b7", ".", 8, "Train (111)"),
        ("dev", cent_train, "#4a3aa7", "^", 14, "Development (23)"),
        ("test", cent_train, "#e34948", "s", 14, "Test (16)"),
        ("eval_final", cent_val, "#008300", "o", 14, "Verified final (50)"),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 1.42), layout="constrained")
    for key, cent, color, mark, size, label in split_style:
        missing = [s for s in splits[key] if s not in cent]
        if missing:
            fail(f"scene map: no label rows for {key} scenes {missing[:3]}")
        pts = [cent[s] for s in splits[key]]
        ax.scatter([lo_ for _, lo_ in pts], [la for la, _ in pts], s=size,
                   marker=mark, color=color, label=label, linewidths=0)
    ax.set_xlabel("Longitude ($^\\circ$)")
    ax.set_ylabel("Latitude ($^\\circ$)")
    ax.grid(True, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="best", fontsize=6.5, ncols=2, handletextpad=0.2, columnspacing=0.8)
    fig.savefig(figs / "scene_map.pdf")
    plt.close(fig)

    for img in ("input_triptych.png", "heatmap_overlay.png", "scene_context.png"):
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
\caption{Left, center: transfer gain over the track-matched random floor
(Eq.~\ref{eq:transfer-gain}) at each tested budget. The optical CNN arm is
negative at every fraction; the ViT arms compress near zero. Right: the
SAR--optical contrast (Eq.~\ref{eq:sar-optical-gap}) for both tracks; the
sign agrees at every fraction while the magnitude is architecture-dependent.}
\label{fig:transfer-gains}
\end{figure}
"""
    (gen / "v100_fig_transfer_gains.tex").write_text(fig_gains, newline="\n")

    fig_qual = r"""% GENERATED by tools/make_results.py -- do not edit by hand.
\begin{figure}[t]
\centering
\includegraphics[width=0.66\textwidth]{figures/input_triptych}\\[3pt]
\includegraphics[width=0.66\textwidth]{figures/heatmap_overlay}
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

    fig_op = r"""% GENERATED by tools/make_results.py -- do not edit by hand.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/operating_points}
\caption{Final-evaluation operating points for all 32 cells (filled circles:
ViT; open squares: CNN; colors as in Fig.~\ref{fig:label-efficiency}). Left:
precision--recall positions with iso-F1 contours; most cells sit above the
dotted $P=R$ diagonal. Center: swept threshold by initialization role.
Right: best-development F1 against measured GPU-hours per cell.}
\label{fig:operating-points}
\end{figure}
"""
    (gen / "v100_fig_operating_points.tex").write_text(fig_op, newline="\n")

    fig_dyn = r"""% GENERATED by tools/make_results.py -- do not edit by hand.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/training_dynamics}
\caption{Development-F1 evaluation history for the smallest (12 scenes,
solid) and largest (111 scenes, dotted) budgets, with the best-checkpoint
epoch marked. Pretrained arms separate from the random floor within the
first evaluations; the CNN random floor climbs for the full 50 epochs.}
\label{fig:training-dynamics}
\end{figure}
"""
    (gen / "v100_fig_training_dynamics.tex").write_text(fig_dyn, newline="\n")

    fig_map = r"""% GENERATED by tools/make_results.py -- do not edit by hand.
\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/scene_map}
\caption{Scene centroids of the frozen splits, computed from the label
records. Sentinel-1 revisits of the same fishing regions place development,
test, and verified scenes near training scenes, which is why
Sect.~\ref{sec:data-evaluation} reports in-region generalization.}
\label{fig:scene-map}
\end{figure}
"""
    (gen / "v100_fig_scene_map.tex").write_text(fig_map, newline="\n")

    fig_chip = r"""% GENERATED by tools/make_results.py -- do not edit by hand.
\begin{figure}[t]
\centering
\includegraphics[width=0.85\textwidth]{figures/scene_context}
\caption{Scale context for one study scene. Left: a full Sentinel-1
ground-range-detected scene ($29{,}658\times21{,}656$ pixels at 10 m
spacing). Right: the single $800\times800$ chip ($8\times8$ km) marked by
the red box; vessels occupy only a few pixels at this scale.}
\label{fig:chip-context}
\end{figure}
"""
    (gen / "v100_fig_chip_context.tex").write_text(fig_chip, newline="\n")

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
