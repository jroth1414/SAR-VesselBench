"""Collect and plot the exact cohort-bound 32-cell label-efficiency grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Sequence

import yaml

from src.runtime.experiment import load_cells
from src.eval.heldout_contract import (
    COHORT_FILENAME,
    cohort_record,
    validate_complete_test_cohort,
    validate_training_cohort,
)

ROLE_COLORS = {
    "floor": "#888888",
    "optical": "#1f77b4",
    "sar": "#d62728",
    "imagenet": "#2ca02c",
}
MONOTONICITY_TOLERANCE = 0.02
REPO = Path(__file__).resolve().parents[2]
DETECTOR_PATH = REPO / "configs" / "detector.yaml"
EXPECTED_DETECTOR_SHA256 = hashlib.sha256(DETECTOR_PATH.read_bytes()).hexdigest()
EXPECTED_PRECISION = yaml.safe_load(DETECTOR_PATH.read_text())["schedule"]["precision"]
GRID_COUNT_COLUMNS = (
    "train_scene_count",
    "train_vessel_count",
    "train_dark_vessel_count",
    "train_near_shore_vessel_count",
)
GRID_COLUMNS = (
    "exp_id",
    "init",
    "track",
    "role",
    "label_frac",
    "seed",
    "precision",
    "detector_sha256",
    "git_sha",
    "dev_f1",
    "dev_threshold",
    "test_f1",
    "epochs_run",
    *GRID_COUNT_COLUMNS,
    "monotonicity_ok",
)


def _git_sha() -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={REPO}", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def metric_column(table) -> str:
    """The reportable curve always uses the complete held-out TEST cohort."""

    if len(table) == 32 and "test_f1" in table.columns and table["test_f1"].notna().all():
        return "test_f1"
    raise ValueError("label-efficiency curves require exactly 32 finite TEST results")


def _repo_path(repo: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo / path


def training_fraction_counts(
    *,
    repo: Path,
    fractions: Sequence[float],
) -> dict[float, dict[str, int]]:
    """Derive exact nested-fraction label counts from frozen TRAIN inputs.

    The same scene permutation used by ``FineTuneDataset`` is applied once,
    then every selected training row is classified through the canonical
    evaluation GT boundary. No dev, test, or verified-final label is opened.
    """

    import pandas as pd

    from src.data.datasets import nested_fraction_scenes
    from src.eval.ground_truth import classify_label, ground_truth_from_labels

    repo = repo.resolve()
    data_cfg = yaml.safe_load((repo / "configs/data.yaml").read_text())
    splits_path = _repo_path(repo, data_cfg["paths"]["splits"])
    splits = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]
    train_scenes = tuple(map(str, splits["train"]))
    if not train_scenes or len(set(train_scenes)) != len(train_scenes):
        raise ValueError("frozen TRAIN split is empty or contains duplicate scene IDs")

    raw_root = _repo_path(repo, data_cfg["paths"]["raw_xview3"])
    labels_path = raw_root / "labels/train.csv"
    labels = pd.read_csv(labels_path)
    required = {
        "scene_id",
        "confidence",
        "is_vessel",
        "source",
        "distance_from_shore_km",
        "detect_scene_column",
        "detect_scene_row",
    }
    missing_columns = sorted(required - set(labels.columns))
    if missing_columns:
        raise ValueError(
            "training labels lack count-contract columns: "
            + ", ".join(missing_columns)
        )
    labels["scene_id"] = labels["scene_id"].astype(str)
    frozen_train = set(train_scenes)
    observed_train = set(
        labels.loc[labels["scene_id"].isin(frozen_train), "scene_id"]
    )
    if observed_train != frozen_train:
        missing = sorted(frozen_train - observed_train)
        raise ValueError(
            "training labels do not cover the frozen TRAIN split"
            + (f" (first missing: {missing[:3]})" if missing else "")
        )

    per_scene = {
        scene_id: {
            "train_vessel_count": 0,
            "train_dark_vessel_count": 0,
            "train_near_shore_vessel_count": 0,
        }
        for scene_id in train_scenes
    }
    train_rows = labels[labels["scene_id"].isin(frozen_train)]
    for index, row in enumerate(train_rows.to_dict(orient="records")):
        try:
            category = classify_label(row)
            if category != "positive":
                continue
            points = ground_truth_from_labels([row])
        except ValueError as exc:
            raise ValueError(
                "invalid frozen TRAIN label while deriving grid counts "
                f"at row {index}: {exc}"
            ) from exc
        if len(points) != 1:
            raise ValueError("canonical positive TRAIN label did not produce one GT point")
        point = points[0]
        counts = per_scene[str(row["scene_id"])]
        counts["train_vessel_count"] += 1
        if point.source == "Manual":
            counts["train_dark_vessel_count"] += 1
        if (
            point.distance_from_shore_km is not None
            and point.distance_from_shore_km <= 2.0
        ):
            counts["train_near_shore_vessel_count"] += 1

    fraction_values = tuple(float(value) for value in fractions)
    if len(set(fraction_values)) != len(fraction_values):
        raise ValueError("label fractions must be unique for grid count derivation")
    result: dict[float, dict[str, int]] = {}
    for fraction in fraction_values:
        selected = nested_fraction_scenes(
            train_scenes,
            fraction,
            frac_seed=int(data_cfg["seed"]),
        )
        aggregate = {
            "train_scene_count": len(selected),
            "train_vessel_count": 0,
            "train_dark_vessel_count": 0,
            "train_near_shore_vessel_count": 0,
        }
        for scene_id in selected:
            for key, value in per_scene[scene_id].items():
                aggregate[key] += value
        result[fraction] = aggregate
    return result


def collect(arms_config: dict, runs_root: Path, out_csv: Path) -> "object":
    import pandas as pd

    runs_root = runs_root.resolve()
    cells = load_cells(REPO)
    git_sha = _git_sha()
    detector_cfg = yaml.safe_load(DETECTOR_PATH.read_text())
    cohort, cohort_sha256 = validate_training_cohort(
        path=runs_root / ".h100" / COHORT_FILENAME,
        cells=cells,
        runs_root=runs_root,
        git_sha=git_sha,
        detector_sha256=EXPECTED_DETECTOR_SHA256,
        candidate_floor=float(detector_cfg["decode"]["candidate_floor"]),
    )
    data_cfg = yaml.safe_load((REPO / "configs/data.yaml").read_text())
    splits_path = Path(str(data_cfg["paths"]["splits"]))
    if not splits_path.is_absolute():
        splits_path = REPO / splits_path
    splits = json.loads(splits_path.read_text())["splits"]
    test_scene_ids = tuple(sorted(map(str, splits["test"])))
    test_results = validate_complete_test_cohort(
        cells=cells,
        runs_root=runs_root,
        cohort=cohort,
        cohort_sha256=cohort_sha256,
        test_scene_ids=test_scene_ids,
    )
    fraction_counts = training_fraction_counts(
        repo=REPO,
        fractions=tuple(float(value) for value in arms_config["label_fracs"]),
    )

    rows = []
    expected_ids = {cell.exp_id for cell in cells}
    for init_name, meta in arms_config["arms"].items():
        for frac in arms_config["label_fracs"]:
            for seed in arms_config["seeds"]["core"]:
                exp_id = f"{meta['short']}-f{int(round(frac * 100))}-s{seed}"
                if exp_id not in expected_ids:
                    raise ValueError(f"unexpected core manifest cell: {exp_id}")
                training = cohort_record(cohort, exp_id)
                final_path = runs_root / exp_id / "final_metrics.json"
                final = json.loads(final_path.read_text())
                test = test_results[exp_id]
                rows.append(
                    {
                        "exp_id": exp_id,
                        "init": init_name,
                        "track": meta["track"],
                        "role": meta["role"],
                        "label_frac": frac,
                        "seed": seed,
                        "precision": training["recipe"]["precision"],
                        "detector_sha256": training["recipe"]["detector_sha256"],
                        "git_sha": training["recipe"]["git_sha"],
                        "dev_f1": training["best_dev"]["f1"],
                        "dev_threshold": training["best_dev"]["threshold"],
                        "test_f1": test["metrics"]["f1"],
                        "epochs_run": final["epochs_run"],
                        **fraction_counts[float(frac)],
                    }
                )

    table = pd.DataFrame(rows)
    if len(table) != 32 or set(table["exp_id"]) != expected_ids:
        raise ValueError("grid collection did not resolve the exact 32-cell cohort")
    if table.duplicated(["init", "label_frac"]).any():
        raise ValueError("duplicate arm/fraction rows violate the seed-0 design")
    table["monotonicity_ok"] = True
    for init_name, group in table.groupby("init"):
        drops = group.sort_values("label_frac")["test_f1"].diff().fillna(0.0)
        if (drops < -MONOTONICITY_TOLERANCE).any():
            table.loc[table["init"] == init_name, "monotonicity_ok"] = False
    if tuple(table.columns) != GRID_COLUMNS:
        raise ValueError(
            f"grid columns differ from the exact reportable schema: {tuple(table.columns)}"
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"{len(table)} cohort-bound TEST rows -> {out_csv}")
    if not table["monotonicity_ok"].all():
        bad = sorted(table.loc[~table["monotonicity_ok"], "init"].unique())
        print(f"WARNING: monotonicity violated for {bad} — STOP condition")
    return table


def plot(table, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.duplicated(["track", "role", "label_frac"]).any():
        raise ValueError("duplicate track/role/fraction rows cannot be plotted")
    metric = metric_column(table)
    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    for (track, role), group in table.groupby(["track", "role"]):
        points = group.sort_values("label_frac")
        axis.plot(
            points["label_frac"],
            points[metric],
            "-" if track == "vit" else "--",
            color=ROLE_COLORS[role],
            marker="o",
            label=f"{track.upper()} {role}",
        )
    axis.set_xscale("log")
    axis.set_xticks(sorted(table["label_frac"].unique()))
    axis.set_xticklabels(
        [f"{int(f * 100)}%" for f in sorted(table["label_frac"].unique())]
    )
    axis.set_xlabel("labeled xView3 training scenes (fraction, log scale)")
    axis.set_ylabel("test F1 (frozen scorer)")
    axis.set_title("Label efficiency by pretraining role — solid ViT, dashed CNN")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"figure -> {out_png}")


def main(argv: Sequence[str] | None = None) -> int:
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
        plot(table, Path(args.out_png))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
