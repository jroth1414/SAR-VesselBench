"""Phase-7 slice tables from exported grid metrics.

This module consumes the small, git-shareable artifacts under ``results/`` or
the live ``runs/`` tree. It does not rescore scenes, tune thresholds, or touch
the once-only final-eval path.

Outputs:
- ``slice_metrics.csv``: one row per study run with frozen-threshold dev/test
  gaps, dark-vessel support/recall, near-shore F1, and optional oracle fields
  when future exports include them.
- ``role_deltas.csv``: per track/fraction/seed deltas versus that track's
  random floor, plus SAR-vs-optical and ImageNet-vs-SAR contrasts.
- ``threshold_transfer.png``: best-dev versus frozen-threshold test F1, with
  large transfer gaps flagged.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _manifest_rows(arms_config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fracs = arms_config["label_fracs"]
    seeds_core = arms_config["seeds"]["core"]
    seeds_rerun = arms_config["seeds"].get("reruns", [])
    rerun_fracs = set(arms_config["seeds"].get("rerun_fracs", []))

    for init_name, meta in arms_config["arms"].items():
        for frac in fracs:
            seeds = list(seeds_core)
            if frac in rerun_fracs:
                seeds += list(seeds_rerun)
            for seed in seeds:
                rows.append(
                    {
                        "exp_id": f"{meta['short']}-f{int(round(frac * 100))}-s{seed}",
                        "init": init_name,
                        "track": meta["track"],
                        "role": meta["role"],
                        "label_frac": frac,
                        "seed": seed,
                    }
                )
    return rows


def collect_slice_metrics(arms_config: dict[str, Any], results_root: Path) -> list[dict[str, Any]]:
    """Collect per-run Phase-7 metrics from exported final_metrics.json files."""

    rows: list[dict[str, Any]] = []
    for row in _manifest_rows(arms_config):
        final_path = results_root / row["exp_id"] / "final_metrics.json"
        if not final_path.exists():
            continue
        payload = _read_json(final_path)
        dev = payload.get("last_dev") or {}

        dev_f1 = payload.get("best_dev_f1")
        test_f1 = payload.get("test_f1")
        oracle_f1 = _first_present(
            payload, "test_oracle_f1", "test_optimal_f1", "test_best_f1"
        )
        oracle_threshold = _first_present(
            payload,
            "test_oracle_threshold",
            "test_optimal_threshold",
            "test_best_threshold",
        )
        dark_support = payload.get("test_dark_support")
        dark_recall = (
            payload.get("test_dark_recall")
            if dark_support is not None and dark_support > 0
            else None
        )

        rows.append(
            {
                **row,
                "epochs_run": payload.get("epochs_run"),
                "dev_f1": dev_f1,
                "dev_precision": dev.get("precision"),
                "dev_recall": dev.get("recall"),
                "dev_threshold": dev.get("threshold"),
                "test_f1": test_f1,
                "test_precision": payload.get("test_precision"),
                "test_recall": payload.get("test_recall"),
                "test_threshold_applied": payload.get("test_threshold_applied"),
                "dev_to_test_f1_gap": (
                    None if dev_f1 is None or test_f1 is None else dev_f1 - test_f1
                ),
                "test_oracle_f1": oracle_f1,
                "test_oracle_threshold": oracle_threshold,
                "oracle_f1_gap": (
                    None if oracle_f1 is None or test_f1 is None else oracle_f1 - test_f1
                ),
                "test_dark_recall": dark_recall,
                "test_dark_support": dark_support,
                "test_near_shore_f1": payload.get("test_near_shore_f1"),
                "test_tp": payload.get("test_tp"),
                "test_fp": payload.get("test_fp"),
                "test_fn": payload.get("test_fn"),
            }
        )

    for row in rows:
        row["dark_slice_available"] = (row.get("test_dark_support") or 0) > 0
        row["threshold_transfer_flag"] = (row.get("dev_to_test_f1_gap") or 0.0) >= 0.20
    return sorted(rows, key=lambda r: (r["track"], r["label_frac"], r["seed"], r["role"]))


def compute_role_deltas(slice_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute track-local role deltas without mixing ViT and CNN arms."""

    rows: list[dict[str, Any]] = []
    key_cols = ["track", "label_frac", "seed"]
    metric_cols = ["test_f1", "dev_f1", "test_near_shore_f1", "test_dark_recall"]

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in slice_table:
        groups.setdefault(tuple(row[col] for col in key_cols), []).append(row)

    for key, group in groups.items():
        track, label_frac, seed = key
        by_role = {row["role"]: row for row in group}
        floor = by_role.get("floor")
        if floor is None:
            continue

        for role, row in by_role.items():
            out = {
                "track": track,
                "label_frac": label_frac,
                "seed": seed,
                "role": role,
                "init": row["init"],
                "exp_id": row["exp_id"],
            }
            for metric in metric_cols:
                value = row.get(metric)
                floor_value = floor.get(metric)
                out[metric] = value
                out[f"{metric}_vs_floor"] = (
                    None
                    if _missing(value) or _missing(floor_value)
                    else value - floor_value
                )
            rows.append(out)

        sar = by_role.get("sar")
        optical = by_role.get("optical")
        imagenet = by_role.get("imagenet")
        if sar is not None and optical is not None:
            rows.append(_contrast_row(track, label_frac, seed, "sar_minus_optical", sar, optical))
        if imagenet is not None and sar is not None:
            rows.append(
                _contrast_row(track, label_frac, seed, "imagenet_minus_sar", imagenet, sar)
            )

    return sorted(rows, key=lambda r: (r["track"], r["label_frac"], r["seed"], r["role"]))


def _contrast_row(
    track: str,
    label_frac: float,
    seed: int,
    role: str,
    left: Any,
    right: Any,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "track": track,
        "label_frac": label_frac,
        "seed": seed,
        "role": role,
        "init": "",
        "exp_id": f"{left['exp_id']}__vs__{right['exp_id']}",
    }
    for metric in ["test_f1", "dev_f1", "test_near_shore_f1", "test_dark_recall"]:
        left_value = left.get(metric)
        right_value = right.get(metric)
        out[metric] = (
            None
            if _missing(left_value) or _missing(right_value)
            else left_value - right_value
        )
        out[f"{metric}_vs_floor"] = None
    return out


def _missing(value: Any) -> bool:
    return value is None or value != value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_threshold_transfer(
    slice_table: list[dict[str, Any]],
    out_path: Path,
) -> Path | None:
    """Plot best-dev versus frozen-threshold test F1 for comparable rows."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    role_order = {"floor": 0, "optical": 1, "sar": 2, "imagenet": 3}
    rows = [
        row
        for row in slice_table
        if not _missing(row.get("dev_f1")) and not _missing(row.get("test_f1"))
    ]
    rows.sort(
        key=lambda row: (
            0 if row["track"] == "vit" else 1,
            row["label_frac"],
            row["seed"],
            role_order.get(row["role"], 99),
        )
    )
    if not rows:
        return None

    colors = {"vit": "#1f77b4", "cnn": "#ff7f0e"}
    labels = [
        f"{row['track'].upper()} · {row['role']} · "
        f"{int(round(row['label_frac'] * 100))}%"
        for row in rows
    ]
    y_values = list(reversed(range(len(rows))))

    fig_height = max(4.5, 0.48 * len(rows) + 1.8)
    fig, axis = plt.subplots(figsize=(8.5, fig_height))
    for y_value, row in zip(y_values, rows):
        color = colors[row["track"]]
        axis.plot(
            [row["test_f1"], row["dev_f1"]],
            [y_value, y_value],
            color=color,
            linewidth=2.2,
            alpha=0.8,
        )
        axis.scatter(
            row["dev_f1"],
            y_value,
            s=58,
            facecolor="white",
            edgecolor=color,
            linewidth=2,
            zorder=3,
        )
        axis.scatter(
            row["test_f1"],
            y_value,
            s=48,
            marker="s",
            color=color,
            zorder=3,
        )
        if row["threshold_transfer_flag"]:
            axis.annotate(
                f"gap {row['dev_to_test_f1_gap']:.3f}",
                (row["test_f1"], y_value),
                xytext=(7, 7),
                textcoords="offset points",
                color="#b22222",
                fontsize=8,
                fontweight="bold",
            )

    axis.set_yticks(y_values, labels)
    axis.set_xlabel("F1 (frozen scorer)")
    axis.set_title("Dev-to-test threshold transfer")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="#555555",
                markeredgewidth=2,
                color="none",
                label="best dev F1",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                markerfacecolor="#555555",
                markeredgecolor="#555555",
                color="none",
                label="test F1 at frozen dev threshold",
            ),
        ],
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"threshold-transfer figure -> {out_path}")
    return out_path


def write_phase7_tables(
    arms_config: dict[str, Any],
    results_root: Path,
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slice_table = collect_slice_metrics(arms_config, results_root)
    deltas = compute_role_deltas(slice_table)

    out_dir.mkdir(parents=True, exist_ok=True)
    slice_path = out_dir / "slice_metrics.csv"
    deltas_path = out_dir / "role_deltas.csv"
    _write_csv(slice_path, slice_table)
    _write_csv(deltas_path, deltas)
    plot_threshold_transfer(slice_table, out_dir / "threshold_transfer.png")

    print(f"{len(slice_table)} slice rows -> {slice_path}")
    print(f"{len(deltas)} delta rows -> {deltas_path}")
    if slice_table and any(row["threshold_transfer_flag"] for row in slice_table):
        flagged = ", ".join(row["exp_id"] for row in slice_table if row["threshold_transfer_flag"])
        print(f"threshold-transfer flags: {flagged}")
    if slice_table and not any(row["dark_slice_available"] for row in slice_table):
        print("dark-vessel slice has zero support in these rows; expected before final eval")
    return slice_table, deltas


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-config", default="configs/arms.yaml")
    parser.add_argument(
        "--results-root",
        default="results",
        help="root containing <exp_id>/final_metrics.json; use runs for live outputs",
    )
    parser.add_argument("--out-dir", default="results/summary")
    args = parser.parse_args(argv)

    arms_config = yaml.safe_load(Path(args.arms_config).read_text())
    write_phase7_tables(arms_config, Path(args.results_root), Path(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
