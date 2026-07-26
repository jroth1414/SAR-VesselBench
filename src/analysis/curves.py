"""Two-track label-efficiency curves + grid.csv bookkeeping (P5.4 / P7).

- ``collect``: sweep ``runs/<exp_id>/final_metrics.json`` for every manifest
  cell and (re)build ``runs/summary/grid.csv`` — one row per run with the
  dev-selected threshold's test/dev metrics, per-arm metadata from
  configs/arms.yaml, and a ``monotonicity_ok`` flag per arm (F1
  non-decreasing within the fixed diagnostic tolerance; a False is a STOP,
  DEVPLAN P5 acceptance).
- ``plot``: render the two-track label-efficiency figure (x = label
  fraction, log scale; y = F1; solid ViT / dashed CNN; one color per
  pretraining role) ->
  ``runs/summary/label_efficiency.png``. Renders whatever cells exist, so
  the partial Phase-4 figure and the full Phase-5 figure share one code
  path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import yaml

ROLE_COLORS = {
    "floor": "#888888",
    "optical": "#1f77b4",
    "sar": "#d62728",
    "imagenet": "#2ca02c",
}
MONOTONICITY_TOLERANCE = 0.02  # fixed F1-point diagnostic; not uncertainty
DETECTOR_PATH = Path(__file__).resolve().parents[2] / "configs" / "detector.yaml"
EXPECTED_DETECTOR_SHA256 = hashlib.sha256(DETECTOR_PATH.read_bytes()).hexdigest()
EXPECTED_PRECISION = yaml.safe_load(DETECTOR_PATH.read_text())["schedule"]["precision"]


def metric_column(table) -> str:
    """Choose one complete point-estimate column for the whole figure."""

    if "test_f1" in table.columns and table["test_f1"].notna().all():
        return "test_f1"
    if "dev_f1" in table.columns and table["dev_f1"].notna().all():
        return "dev_f1"
    raise ValueError(
        "grid rows do not share a complete test_f1 or dev_f1 column"
    )


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
                if (
                    payload.get("precision") != EXPECTED_PRECISION
                    or payload.get("detector_sha256") != EXPECTED_DETECTOR_SHA256
                ):
                    raise ValueError(
                        f"{exp_id}: completion marker does not match shared recipe"
                    )
                dev = payload.get("last_dev") or {}
                rows.append(
                    {
                        "exp_id": exp_id,
                        "init": init_name,
                        "track": meta["track"],
                        "role": meta["role"],
                        "label_frac": frac,
                        "seed": seed,
                        "precision": payload.get("precision"),
                        "detector_sha256": payload.get("detector_sha256"),
                        "git_sha": payload.get("git_sha"),
                        "dev_f1": payload.get("best_dev_f1"),
                        "dev_threshold": dev.get("threshold"),
                        "test_f1": payload.get("test_f1"),  # written by P5.4 scoring
                        "epochs_run": payload.get("epochs_run"),
                    }
                )

    table = pd.DataFrame(rows)
    if not table.empty:
        if table.duplicated(["init", "label_frac"]).any():
            raise ValueError(
                "duplicate arm/fraction rows violate the seed-0 point-estimate design"
            )
        table["monotonicity_ok"] = True
        metric = metric_column(table)
        for init_name, group in table.groupby("init"):
            points = group.sort_values("label_frac")[metric]
            drops = points.diff().fillna(0.0)
            if (drops < -MONOTONICITY_TOLERANCE).any():
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

    if table.duplicated(["track", "role", "label_frac"]).any():
        raise ValueError(
            "duplicate track/role/fraction rows cannot be plotted as point estimates"
        )
    metric = metric_column(table)
    fig, axis = plt.subplots(figsize=(7.5, 5.5))

    for (track, role), group in table.groupby(["track", "role"]):
        points = group.sort_values("label_frac")
        style = "-" if track == "vit" else "--"
        color = ROLE_COLORS[role]
        label = f"{track.upper()} {role}"
        axis.plot(
            points["label_frac"], points[metric], style,
            color=color, marker="o", label=label,
        )

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
