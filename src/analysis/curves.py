"""Two-track label-efficiency curves + grid.csv bookkeeping (P5.4 / P7).

- ``collect``: sweep ``runs/<exp_id>/final_metrics.json`` for every manifest
  cell and (re)build ``runs/summary/grid.csv`` — one row per run with the
  dev-selected threshold's test/dev metrics, per-arm metadata from
  configs/arms.yaml, and a ``monotonicity_ok`` flag per arm (F1
  non-decreasing in label fraction within seed noise; a False is a STOP,
  DEVPLAN P5 acceptance).
- ``plot``: render the two-track label-efficiency figure (x = label
  fraction, log scale; y = F1; solid ViT / dashed CNN; one color per
  pretraining role; shaded seed bands when reruns exist) ->
  ``runs/summary/label_efficiency.png``. Renders whatever cells exist, so
  the partial Phase-4 figure and the full Phase-5 figure share one code
  path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

ROLE_COLORS = {
    "floor": "#888888",
    "optical": "#1f77b4",
    "sar": "#d62728",
    "supervised": "#2ca02c",
}
SEED_NOISE_TOLERANCE = 0.02  # monotone "within seed noise" slack (F1 points)


def collect(arms_config: dict, runs_root: Path, out_csv: Path) -> "object":
    import pandas as pd

    rows = []
    fracs = arms_config["label_fracs"]
    seeds_core = arms_config["seeds"]["core"]
    seeds_rerun = arms_config["seeds"]["reruns"]
    rerun_fracs = set(arms_config["seeds"]["rerun_fracs"])

    for init_name, meta in arms_config["arms"].items():
        for frac in fracs:
            seeds = list(seeds_core) + (
                list(seeds_rerun) if frac in rerun_fracs else []
            )
            for seed in seeds:
                exp_id = f"{meta['short']}-f{int(round(frac * 100))}-s{seed}"
                final = runs_root / exp_id / "final_metrics.json"
                if not final.exists():
                    continue
                payload = json.loads(final.read_text())
                dev = payload.get("last_dev") or {}
                rows.append(
                    {
                        "exp_id": exp_id,
                        "init": init_name,
                        "track": meta["track"],
                        "role": meta["role"],
                        "label_frac": frac,
                        "seed": seed,
                        "dev_f1": payload.get("best_dev_f1"),
                        "dev_threshold": dev.get("threshold"),
                        "test_f1": payload.get("test_f1"),  # written by P5.4 scoring
                        "epochs_run": payload.get("epochs_run"),
                    }
                )

    table = pd.DataFrame(rows)
    if not table.empty:
        table["monotonicity_ok"] = True
        metric = "test_f1" if table["test_f1"].notna().any() else "dev_f1"
        for init_name, group in table.groupby("init"):
            means = group.groupby("label_frac")[metric].mean().sort_index()
            drops = means.diff().fillna(0.0)
            if (drops < -SEED_NOISE_TOLERANCE).any():
                table.loc[table["init"] == init_name, "monotonicity_ok"] = False

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"{len(table)} run rows -> {out_csv}")
    if not table.empty and not table["monotonicity_ok"].all():
        bad = sorted(table.loc[~table["monotonicity_ok"], "init"].unique())
        print(f"WARNING: monotonicity violated for {bad} — STOP condition (DEVPLAN P5)")
    return table


def plot(table, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = "test_f1" if table["test_f1"].notna().any() else "dev_f1"
    fig, axis = plt.subplots(figsize=(7.5, 5.5))

    for (track, role), group in table.groupby(["track", "role"]):
        stats = (
            group.groupby("label_frac")[metric]
            .agg(["mean", "min", "max"])
            .sort_index()
        )
        style = "-" if track == "vit" else "--"
        color = ROLE_COLORS[role]
        label = f"{track.upper()} {role}"
        axis.plot(stats.index, stats["mean"], style, color=color, marker="o", label=label)
        if (stats["max"] - stats["min"]).max() > 0:
            axis.fill_between(stats.index, stats["min"], stats["max"], color=color, alpha=0.12)

    axis.set_xscale("log")
    axis.set_xticks(sorted(table["label_frac"].unique()))
    axis.set_xticklabels(
        [f"{int(f * 100)}%" for f in sorted(table["label_frac"].unique())]
    )
    axis.set_xlabel("labeled xView3 training scenes (fraction, log scale)")
    axis.set_ylabel(f"{'test' if metric == 'test_f1' else 'dev'} F1 (frozen scorer)")
    axis.set_title("Label efficiency by pretraining role — solid ViT, dashed CNN")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncols=2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_png}")


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "plot", "all"], nargs="?", default="all")
    parser.add_argument("--arms-config", default="configs/arms.yaml")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out-csv", default="runs/summary/grid.csv")
    parser.add_argument("--out-png", default="runs/summary/label_efficiency.png")
    args = parser.parse_args(argv)

    arms_config = yaml.safe_load(Path(args.arms_config).read_text())
    table = collect(arms_config, Path(args.runs_root), Path(args.out_csv))
    if args.command in ("plot", "all"):
        if table.empty:
            print("no grid rows yet — nothing to plot")
        else:
            plot(table, Path(args.out_png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
