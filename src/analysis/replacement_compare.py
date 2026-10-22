"""Compare the replacement32 cohort with the August H100 cohort.

Both cohorts pass through ``heldout_results.validate_evidence`` first, so
every development and TEST number here is one the fail-closed generator would
also accept. The replacement cohort adds the once-only 50-scene human-verified
evaluation (``final_verified_metrics.json`` per cell); this module reads it
and cross-checks each file against the cohort, its TEST result and the
completion package's ``FINAL_EVAL_COMPLETE.json`` hashes.

Outputs (``--output-dir``):
  cells.csv       one row per (cohort, cell): dev, TEST and final metrics
  claims.json     the headline checks recomputed from the rows
  comparison.md   tables behind INSPECTION.md

Usage:
  python -m src.analysis.replacement_compare \
      --replacement-evidence <staged tree> --august-evidence results/h100/evidence \
      --final-complete <package>/FINAL_EVAL_COMPLETE.json --output-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.analysis.heldout_results import (
    FRACTION_SCENES,
    FRACTIONS,
    ROLE_LABEL,
    ROLE_ORDER,
    EvidenceError,
    validate_evidence,
)

TRACKS = ("vit", "cnn")
TRACK_NAME = {"vit": "ViT-B/16", "cnn": "ConvNeXt-V2-B"}
MONOTONICITY_TOLERANCE = 0.02
FINAL_POLICY = "replacement-only-all32-final50-once-v1"
FINAL_SCENES = 50
FINAL_FIELDS = (
    "f1", "precision", "recall", "tp", "fp", "fn",
    "dark_recall", "dark_support", "near_shore_f1", "near_shore_support",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_final(
    evidence_root: Path, validated: Mapping[str, object], final_complete: Path
) -> dict[str, dict[str, float]]:
    """Read and cross-check the 32 once-only 50-scene results."""

    complete = json.loads(final_complete.read_text(encoding="utf-8"))
    if (
        complete.get("status") != "replacement-final32-complete"
        or complete.get("policy") != FINAL_POLICY
        or complete.get("cell_count") != 32
        or complete.get("scene_count") != FINAL_SCENES
    ):
        raise EvidenceError("FINAL_EVAL_COMPLETE does not describe the all-32 final evaluation")
    bound = complete["cell_result_sha256"]
    cohort_sha256 = validated["campaign"]["cohort_sha256"]  # type: ignore[index]
    out: dict[str, dict[str, float]] = {}
    for exp_id, cell in validated["cells"].items():  # type: ignore[union-attr]
        path = evidence_root / exp_id / "final_verified_metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("exp_id") != exp_id
            or payload.get("cohort_sha256") != cohort_sha256
            or payload.get("policy") != FINAL_POLICY
            or payload.get("test_result_sha256") != _sha256(evidence_root / exp_id / "test_metrics.json")
            or payload.get("checkpoint", {}).get("sha256") != cell["checkpoint_sha256"]
            or not math.isclose(payload["best_dev"]["threshold"], cell["threshold"], abs_tol=0.0)
            or len(payload.get("per_scene", {})) != FINAL_SCENES
        ):
            raise EvidenceError(f"{exp_id}: final result does not bind the cohort, test result and threshold")
        if bound.get(exp_id) != _sha256(path):
            # The completion record hashes the node-side file; a redacted copy differs only
            # if the redaction touched it, which REDACTIONS.json would list.
            redactions = json.loads((evidence_root / "REDACTIONS.json").read_text(encoding="utf-8"))
            original = redactions["files"].get(f"{exp_id}/final_verified_metrics.json", {})
            if original.get("original_sha256") != bound.get(exp_id):
                raise EvidenceError(f"{exp_id}: final result hash disagrees with FINAL_EVAL_COMPLETE")
        metrics = payload["metrics"]
        tp, fp, fn = metrics["tp"], metrics["fp"], metrics["fn"]
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if not math.isclose(f1, metrics["f1"], abs_tol=1e-9):
            raise EvidenceError(f"{exp_id}: final F1 disagrees with its counts")
        out[exp_id] = {key: float(metrics[key]) for key in FINAL_FIELDS}
    return out


def tidy_rows(cohort: str, validated: Mapping[str, object], final: Mapping[str, Mapping[str, float]] | None):
    rows = []
    tests = validated["test_results"]
    for exp_id, cell in sorted(validated["cells"].items()):  # type: ignore[union-attr]
        row = {
            "cohort": cohort,
            "exp_id": exp_id,
            "track": cell["track"],
            "role": cell["role"],
            "fraction": int(cell["label_fraction"]),
            "scenes": FRACTION_SCENES[int(cell["label_fraction"])],
            "dev_f1": cell["dev_f1"],
            "threshold": cell["threshold"],
            "best_epoch": cell["best_epoch"],
            "epochs_run": cell["epochs_run"],
            "gpu_hours": cell["gpu_hours"],
            "test_f1": tests[exp_id]["f1"],  # type: ignore[index]
            "test_precision": tests[exp_id]["precision"],  # type: ignore[index]
            "test_recall": tests[exp_id]["recall"],  # type: ignore[index]
        }
        if final is not None:
            for key, value in final[exp_id].items():
                row[f"final_{key}"] = value
        rows.append(row)
    return rows


def _grid(rows, cohort: str, metric: str) -> dict[tuple[str, str, int], float]:
    return {
        (r["track"], r["role"], r["fraction"]): float(r[metric])
        for r in rows
        if r["cohort"] == cohort and metric in r
    }


def claims(rows, cohort: str) -> dict[str, object]:
    """Recompute the report's headline checks for one cohort."""

    metrics = ["dev_f1", "test_f1"] + (["final_f1"] if any("final_f1" in r for r in rows if r["cohort"] == cohort) else [])
    out: dict[str, object] = {}
    for metric in metrics:
        grid = _grid(rows, cohort, metric)
        block: dict[str, object] = {}
        for track in TRACKS:
            block[f"{track}_sar_minus_optical"] = {
                f: round(grid[(track, "sar", f)] - grid[(track, "optical", f)], 4) for f in FRACTIONS
            }
            block[f"{track}_gain_over_random"] = {
                role: {f: round(grid[(track, role, f)] - grid[(track, "floor", f)], 4) for f in FRACTIONS}
                for role in ROLE_ORDER
                if role != "floor"
            }
            block[f"{track}_leader"] = {
                f: max(ROLE_ORDER, key=lambda role: grid[(track, role, f)]) for f in FRACTIONS
            }
            drops = {}
            for role in ROLE_ORDER:
                for low, high in zip(FRACTIONS, FRACTIONS[1:]):
                    drop = grid[(track, role, low)] - grid[(track, role, high)]
                    if drop > MONOTONICITY_TOLERANCE:
                        drops[f"{role} f{low}->f{high}"] = round(drop, 4)
            block[f"{track}_monotonicity_violations"] = drops
        out[metric] = block
    for later in [m for m in ("test_f1", "final_f1") if m in metrics]:
        gaps = [r["dev_f1"] - r[later] for r in rows if r["cohort"] == cohort]
        out[f"mean_dev_minus_{later.split('_')[0]}"] = round(sum(gaps) / len(gaps), 4)
    out["gpu_hours"] = round(sum(r["gpu_hours"] for r in rows if r["cohort"] == cohort), 1)
    return out


def _matrix_md(rows, metric: str, cohorts: Sequence[str], *, delta: bool = False) -> str:
    head = "| Track | Init | " + " | ".join(f"{f}% ({FRACTION_SCENES[f]})" for f in FRACTIONS) + " |"
    lines = [head, "|" + "---|" * (2 + len(FRACTIONS))]
    grids = {c: _grid(rows, c, metric) for c in cohorts}
    for track in TRACKS:
        for role in ROLE_ORDER:
            cells = []
            for f in FRACTIONS:
                key = (track, role, f)
                if delta:
                    a, b = grids[cohorts[0]].get(key), grids[cohorts[1]].get(key)
                    cells.append("" if a is None or b is None else f"{b - a:+.3f}")
                else:
                    cells.append(f"{grids[cohorts[0]][key]:.3f}")
            lines.append(f"| {TRACK_NAME[track]} | {ROLE_LABEL[role]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def comparison_markdown(rows, claim_sets: Mapping[str, Mapping[str, object]]) -> str:
    parts = ["# August vs replacement32: generated tables", ""]
    for metric, title in (("dev_f1", "Development-selection F1 (8 dev scenes)"), ("test_f1", "Held-out TEST F1 (16 scenes)")):
        parts += [f"## {title}", "", "### replacement32", "", _matrix_md(rows, metric, ["replacement32"]), ""]
        parts += ["### Change: replacement32 minus August", "", _matrix_md(rows, metric, ["august", "replacement32"], delta=True), ""]
    for metric, title in (
        ("final_f1", "Final 50-scene human-verified F1"),
        ("final_recall", "Final 50-scene recall"),
        ("final_dark_recall", "Final 50-scene dark-vessel recall"),
        ("final_near_shore_f1", "Final 50-scene near-shore F1"),
    ):
        parts += [f"## {title} (replacement32)", "", _matrix_md(rows, metric, ["replacement32"]), ""]
    parts += ["## Recomputed claims", "", "```json", json.dumps(claim_sets, indent=1), "```", ""]
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replacement-evidence", required=True, type=Path)
    parser.add_argument("--august-evidence", required=True, type=Path)
    parser.add_argument("--final-complete", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arms-config", default=Path("configs/arms.yaml"), type=Path)
    parser.add_argument("--detector-config", default=Path("configs/detector.yaml"), type=Path)
    parser.add_argument("--splits-config", default=Path("data/splits.json"), type=Path)
    args = parser.parse_args(argv)
    kwargs = {"arms_config": args.arms_config, "detector_config": args.detector_config, "splits_config": args.splits_config}
    replacement = validate_evidence(args.replacement_evidence, **kwargs)
    august = validate_evidence(args.august_evidence, **kwargs)
    final = load_final(args.replacement_evidence, replacement, args.final_complete)
    rows = tidy_rows("august", august, None) + tidy_rows("replacement32", replacement, final)
    claim_sets = {"august": claims(rows, "august"), "replacement32": claims(rows, "replacement32")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[-1].keys())
    with (args.output_dir / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "claims.json").write_text(json.dumps(claim_sets, indent=2) + "\n", encoding="utf-8", newline="\n")
    (args.output_dir / "comparison.md").write_text(comparison_markdown(rows, claim_sets), encoding="utf-8", newline="\n")
    print(json.dumps({
        "replacement_cohort_sha256": replacement["campaign"]["cohort_sha256"],
        "replacement_git_sha": replacement["campaign"]["git_sha"],
        "august_cohort_sha256": august["campaign"]["cohort_sha256"],
        "replacement_gpu_hours": round(replacement["campaign"]["gpu_hours"], 1),
        "hardware": replacement["campaign"]["hardware"],
        "rows": len(rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
