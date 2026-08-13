"""Fail-closed paper inputs from the committed H100 evidence tree.

``results/h100/evidence`` carries the frozen training cohort, each cell's
byte-exact completion marker, and each cell's training-curve CSV. This
generator re-derives every published number from those bytes:

* the cohort must bind the committed ``configs/detector.yaml`` and declare
  the exact 32-cell matrix from ``configs/arms.yaml``;
* every completion marker must hash to its cohort binding, agree with the
  cohort's recipe, and be internally consistent (precision/recall/F1 versus
  TP/FP/FN, marker F1 versus the training-curve maximum at the bound epoch);
* held-out TEST macros render only when all 32 immutable test results are
  present and each one revalidates against the cohort (all-or-nothing);
* the sealed 50-scene evaluation renders only from ``final_verified.csv``.

Checkpoint bytes stay outside the repository; their SHA-256 bindings are
published so an operator archive can re-verify them. Anything inconsistent
raises ``EvidenceError`` and nothing is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.analysis.h100_results import expected_cells

FRACTIONS = (10, 25, 50, 100)
FRACTION_MACRO = {10: "Ten", 25: "TwentyFive", 50: "Fifty", 100: "Hundred"}
FRACTION_SCENES = {10: 12, 25: 28, 50: 56, 100: 111}
ROLE_MACRO = {"floor": "Random", "optical": "Optical", "sar": "Sar", "imagenet": "ImageNet"}
ROLE_LABEL = {"floor": "Random", "optical": "Optical RS", "sar": "SAR", "imagenet": "ImageNet"}
TRACK_MACRO = {"vit": "ViT", "cnn": "CNN"}
TRACK_LABEL = {"vit": "ViT-B/16 track", "cnn": "ConvNeXt-V2-B track"}
ROLE_ORDER = ("floor", "optical", "sar", "imagenet")
# Okabe-Ito colorblind-safe hues, assigned to roles in fixed order; the
# floor is the neutral reference and additionally dashed (secondary encoding).
ROLE_COLOR = {
    "floor": "#5D5D5D",
    "optical": "#E69F00",
    "sar": "#0072B2",
    "imagenet": "#009E73",
}
ROLE_MARKER = {"floor": "o", "optical": "s", "sar": "^", "imagenet": "D"}
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")

EXPECTED_TEST_SCENES = 16
EXPECTED_TEST_POSITIVES = 1165
FINAL_EVAL_FRACTIONS = (10, 25, 100)


class EvidenceError(RuntimeError):
    """The committed evidence is absent, tampered, or inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EvidenceError(f"cannot hash evidence file: {path}") from exc
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"{description} root must be a JSON object")
    return payload


def _finite(value: object, description: str, *, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise EvidenceError(f"{description} is not numeric") from exc
    if not math.isfinite(number) or not low <= number <= high:
        raise EvidenceError(f"{description} is outside [{low}, {high}]")
    return number


def _consistent_prf(block: Mapping[str, object], description: str) -> None:
    tp, fp, fn = (int(block[k]) for k in ("tp", "fp", "fn"))  # type: ignore[index]
    if min(tp, fp, fn) < 0:
        raise EvidenceError(f"{description} has a negative count")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    for key, expected in (("precision", precision), ("recall", recall), ("f1", f1)):
        observed = _finite(block.get(key), f"{description}.{key}")
        if not math.isclose(observed, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise EvidenceError(f"{description}.{key} disagrees with its TP/FP/FN")


def _curve_agrees(metrics_csv: Path, best_dev: Mapping[str, object], exp_id: str) -> None:
    best_f1 = None
    best_epoch = None
    try:
        with metrics_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw = (row.get("dev_f1") or "").strip()
                if not raw:
                    continue
                value = float(raw)
                epoch = int(float(row["epoch"]))
                if best_f1 is None or value > best_f1:
                    best_f1, best_epoch = value, epoch
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise EvidenceError(f"{exp_id}: unreadable training curve") from exc
    if best_f1 is None:
        raise EvidenceError(f"{exp_id}: training curve has no development evaluation")
    marker_f1 = _finite(best_dev.get("f1"), f"{exp_id}.best_dev.f1")
    marker_epoch = int(best_dev.get("epoch"))  # type: ignore[arg-type]
    # The curve CSV logs float32-cast values; the marker stores float64.
    if not math.isclose(best_f1, marker_f1, rel_tol=0, abs_tol=1e-6) or best_epoch != marker_epoch:
        raise EvidenceError(
            f"{exp_id}: marker best_dev (F1 {marker_f1:.6f} @ {marker_epoch}) disagrees "
            f"with the training curve maximum (F1 {best_f1:.6f} @ {best_epoch})"
        )


def _validate_test_result(
    path: Path,
    *,
    exp_id: str,
    cohort_sha256: str,
    marker_sha256: str,
    detector_sha256: str,
    threshold: float,
) -> dict[str, float]:
    payload = _load_json(path, f"{exp_id} test result")
    if (
        payload.get("exp_id") != exp_id
        or payload.get("cohort_sha256") != cohort_sha256
        or payload.get("completion_marker_sha256") != marker_sha256
        or payload.get("detector_sha256") != detector_sha256
    ):
        raise EvidenceError(f"{exp_id}: test result bindings do not match the cohort")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise EvidenceError(f"{exp_id}: test metrics are absent")
    _consistent_prf(metrics, f"{exp_id}.test")
    tp, fn = int(metrics["tp"]), int(metrics["fn"])
    if tp + fn != EXPECTED_TEST_POSITIVES:
        raise EvidenceError(f"{exp_id}: test positive support must equal {EXPECTED_TEST_POSITIVES}")
    per_scene = payload.get("per_scene")
    if not isinstance(per_scene, Mapping) or len(per_scene) != EXPECTED_TEST_SCENES:
        raise EvidenceError(f"{exp_id}: test result must cover exactly {EXPECTED_TEST_SCENES} scenes")
    applied = payload.get("threshold_applied", payload.get("threshold", threshold))
    if not math.isclose(_finite(applied, f"{exp_id}.test threshold"), threshold, abs_tol=1e-12):
        raise EvidenceError(f"{exp_id}: test threshold differs from the cohort-bound dev threshold")
    return {
        "f1": _finite(metrics.get("f1"), f"{exp_id}.test.f1"),
        "precision": _finite(metrics.get("precision"), f"{exp_id}.test.precision"),
        "recall": _finite(metrics.get("recall"), f"{exp_id}.test.recall"),
        "inference_precision": str(payload.get("inference_precision", "")),
    }


def validate_evidence(
    evidence_root: str | Path,
    *,
    arms_config: str | Path,
    detector_config: str | Path,
) -> dict[str, object]:
    root = Path(evidence_root)
    detector_path = Path(detector_config)
    import yaml

    try:
        detector = yaml.safe_load(detector_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvidenceError(f"cannot read detector config: {detector_path}") from exc
    detector_sha256 = _sha256_file(detector_path)
    candidate_floor = float(detector["decode"]["candidate_floor"])

    cohort_path = root / "TRAINING_COHORT.json"
    cohort = _load_json(cohort_path, "training cohort")
    cohort_sha256 = _sha256_file(cohort_path)
    git_sha = str(cohort.get("git_sha", ""))
    if (
        cohort.get("cohort_schema") != 1
        or cohort.get("status") != "training-cohort-frozen"
        or cohort.get("policy") != "all-32-training-complete-before-any-test-access"
        or cohort.get("cell_count") != 32
        or not _HEX40.fullmatch(git_sha)
        or cohort.get("detector_sha256") != detector_sha256
        or cohort.get("candidate_floor") != candidate_floor
    ):
        raise EvidenceError("training cohort identity does not bind the committed configs")

    expected = expected_cells(arms_config)
    records = cohort.get("cells")
    if not isinstance(records, list) or {r.get("exp_id") for r in records} != set(expected):
        raise EvidenceError("training cohort is not the exact 32-cell matrix")

    cells: dict[str, dict[str, object]] = {}
    hardware: set[str] = set()
    total_active_seconds = 0.0
    test_results: dict[str, dict[str, float]] = {}
    missing_tests: list[str] = []
    for record in records:
        exp_id = str(record["exp_id"])
        metadata = expected[exp_id]
        cell_dir = root / exp_id
        marker_path = cell_dir / "final_metrics.json"
        marker_sha256 = _sha256_file(marker_path)
        binding = record.get("completion_marker")
        if not isinstance(binding, Mapping) or binding.get("sha256") != marker_sha256:
            raise EvidenceError(f"{exp_id}: committed marker does not hash to its cohort binding")
        marker = _load_json(marker_path, f"{exp_id} completion marker")
        recipe = marker.get("recipe") if isinstance(marker.get("recipe"), Mapping) else marker
        if (
            marker.get("exp_id") != exp_id
            or marker.get("git_sha", recipe.get("git_sha")) != git_sha
            or marker.get("precision", recipe.get("precision")) != "32-true"
            or recipe.get("detector_sha256", marker.get("detector_sha256")) != detector_sha256
        ):
            raise EvidenceError(f"{exp_id}: completion marker disagrees with the cohort identity")
        best_dev = marker.get("best_dev")
        if not isinstance(best_dev, Mapping):
            raise EvidenceError(f"{exp_id}: best development block is absent")
        _consistent_prf(best_dev, f"{exp_id}.best_dev")
        dev_f1 = _finite(best_dev.get("f1"), f"{exp_id}.best_dev.f1")
        if not math.isclose(
            dev_f1, _finite(marker.get("best_dev_f1"), f"{exp_id}.best_dev_f1"), abs_tol=1e-12
        ):
            raise EvidenceError(f"{exp_id}: best_dev_f1 disagrees with best_dev.f1")
        threshold = _finite(best_dev.get("threshold"), f"{exp_id}.best_dev.threshold")
        if not 0.0 < threshold < 1.0:
            raise EvidenceError(f"{exp_id}: operating threshold is degenerate")
        epochs_run = int(marker.get("epochs_run", 0))
        if epochs_run <= 0:
            raise EvidenceError(f"{exp_id}: epochs_run must be positive")
        _curve_agrees(cell_dir / "metrics.csv", best_dev, exp_id)

        runtime = _load_json(cell_dir / "runtime_provenance.json", f"{exp_id} runtime provenance")
        if runtime.get("exp_id") != exp_id or runtime.get("git_sha") != git_sha:
            raise EvidenceError(f"{exp_id}: runtime provenance identity mismatch")
        accepted = runtime.get("accepted_hardware_class")
        if isinstance(accepted, Mapping):
            hardware.add(str(accepted.get("gpu_name", "")))
        active = runtime.get("accumulated_active_seconds")
        if not isinstance(active, (int, float)) or not math.isfinite(active) or active <= 0:
            raise EvidenceError(f"{exp_id}: accumulated_active_seconds must be positive")
        total_active_seconds += float(active)

        best_checkpoint = marker.get("best_checkpoint")
        checkpoint_sha256 = (
            str(best_checkpoint.get("sha256", "")) if isinstance(best_checkpoint, Mapping) else ""
        )
        if not _HEX64.fullmatch(checkpoint_sha256):
            raise EvidenceError(f"{exp_id}: best checkpoint binding is absent")

        test_path = cell_dir / "test_metrics.json"
        if test_path.is_file():
            test_results[exp_id] = _validate_test_result(
                test_path,
                exp_id=exp_id,
                cohort_sha256=cohort_sha256,
                marker_sha256=marker_sha256,
                detector_sha256=detector_sha256,
                threshold=threshold,
            )
        else:
            missing_tests.append(exp_id)

        cells[exp_id] = {
            **metadata,
            "dev_f1": dev_f1,
            "dev_precision": _finite(best_dev.get("precision"), f"{exp_id}.best_dev.precision"),
            "dev_recall": _finite(best_dev.get("recall"), f"{exp_id}.best_dev.recall"),
            "threshold": threshold,
            "best_epoch": int(best_dev.get("epoch")),  # type: ignore[arg-type]
            "epochs_run": epochs_run,
            "checkpoint_sha256": checkpoint_sha256,
            "marker_sha256": marker_sha256,
        }

    if test_results and missing_tests:
        raise EvidenceError(
            "test results are all-or-nothing; missing: " + ", ".join(sorted(missing_tests))
        )
    if len(hardware) != 1:
        raise EvidenceError(f"evidence spans mixed hardware classes: {sorted(hardware)}")

    final_eval_path = root / "final_verified.csv"
    final_eval_rows: list[dict[str, str]] = []
    if final_eval_path.is_file():
        with final_eval_path.open(newline="", encoding="utf-8") as handle:
            final_eval_rows = list(csv.DictReader(handle))
        expected_final = {
            exp_id
            for exp_id, meta in expected.items()
            if meta["label_fraction"] in FINAL_EVAL_FRACTIONS
        }
        if {row.get("exp_id") for row in final_eval_rows} != expected_final:
            raise EvidenceError("final_verified.csv does not cover the exact 24-cell selection")

    return {
        "cells": cells,
        "test_results": test_results,
        "final_eval_rows": final_eval_rows,
        "campaign": {
            "git_sha": git_sha,
            "cohort_sha256": cohort_sha256,
            "detector_sha256": detector_sha256,
            "hardware": next(iter(hardware)),
            "gpu_hours": total_active_seconds / 3600.0,
            "created_utc": str(cohort.get("created_utc", "")),
        },
    }


def _macro(track: str, role: str, fraction: int) -> str:
    return f"{TRACK_MACRO[track]}{ROLE_MACRO[role]}{FRACTION_MACRO[fraction]}"


def render_tex(validated: Mapping[str, object]) -> str:
    cells: Mapping[str, Mapping[str, object]] = validated["cells"]  # type: ignore[assignment]
    tests: Mapping[str, Mapping[str, float]] = validated["test_results"]  # type: ignore[assignment]
    campaign: Mapping[str, object] = validated["campaign"]  # type: ignore[assignment]
    by_key = {
        (meta["track"], meta["role"], meta["label_fraction"]): (exp_id, meta)
        for exp_id, meta in cells.items()
    }
    lines = [
        "% GENERATED by src.analysis.heldout_results -- do not edit.",
        "\\newif\\ifHevDevComplete",
        "\\HevDevCompletetrue",
        "\\newif\\ifHevTestComplete",
        "\\HevTestComplete" + ("true" if tests else "false"),
        "\\newif\\ifHevFinalEval",
        "\\HevFinalEval" + ("true" if validated["final_eval_rows"] else "false"),
        f"\\def\\HevCodeSHAShort{{{str(campaign['git_sha'])[:8]}}}",
        f"\\def\\HevCohortSHAShort{{{str(campaign['cohort_sha256'])[:8]}}}",
        f"\\def\\HevHardware{{{campaign['hardware']}}}",
        f"\\def\\HevGPUHours{{{float(campaign['gpu_hours']):.1f}}}",
        f"\\def\\HevCohortCreatedUTC{{{campaign['created_utc']}}}",
    ]
    for track in ("vit", "cnn"):
        floor_f10 = float(cells[by_key[(track, "floor", 10)][0]]["dev_f1"])
        for role in ("optical", "sar", "imagenet"):
            arm_f10 = float(cells[by_key[(track, role, 10)][0]]["dev_f1"])
            delta = arm_f10 - floor_f10
            lines.append(
                f"\\def\\HevDeltaFTen{TRACK_MACRO[track]}{ROLE_MACRO[role]}"
                f"{{{'+' if delta >= 0 else '-'}{abs(delta):.3f}}}"
            )
    gaps: list[float] = []
    for track in ("vit", "cnn"):
        for role in ROLE_ORDER:
            for fraction in FRACTIONS:
                exp_id, _meta = by_key[(track, role, fraction)]
                cell = cells[exp_id]
                name = _macro(track, role, fraction)
                lines.append(f"\\def\\HevDevF{name}{{{cell['dev_f1']:.4f}}}")
                lines.append(f"\\def\\HevThr{name}{{{cell['threshold']:.3f}}}")
                lines.append(f"\\def\\HevEpoch{name}{{{cell['best_epoch']}}}")
                if tests:
                    test = tests[exp_id]
                    lines.append(f"\\def\\HevTestF{name}{{{test['f1']:.4f}}}")
                    gaps.append(float(cell["dev_f1"]) - float(test["f1"]))
                else:
                    lines.append(f"\\def\\HevTestF{name}{{\\textemdash}}")
    if gaps:
        lines.append(f"\\def\\HevTestMeanGap{{{sum(gaps) / len(gaps):.3f}}}")
        precisions = {t["inference_precision"] for t in tests.values()}
        lines.append(f"\\def\\HevTestInferencePrecision{{{'/'.join(sorted(precisions))}}}")
    else:
        lines.append("\\def\\HevTestMeanGap{\\textemdash}")
        lines.append("\\def\\HevTestInferencePrecision{\\textemdash}")
    return "\n".join(lines) + "\n"


def render_figure(validated: Mapping[str, object], out_pdf: Path) -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells: Mapping[str, Mapping[str, object]] = validated["cells"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for axis, track in zip(axes, ("vit", "cnn")):
        for role in ROLE_ORDER:
            xs, ys = [], []
            for fraction in FRACTIONS:
                for cell in cells.values():
                    if cell["track"] == track and cell["role"] == role and cell["label_fraction"] == fraction:
                        xs.append(fraction)
                        ys.append(cell["dev_f1"])
            axis.plot(
                xs,
                ys,
                color=ROLE_COLOR[role],
                marker=ROLE_MARKER[role],
                markersize=4,
                linewidth=1.6,
                linestyle="--" if role == "floor" else "-",
                label=ROLE_LABEL[role],
            )
        axis.set_title(TRACK_LABEL[track], fontsize=9)
        axis.set_xlabel("Label fraction (% of 111 scenes)", fontsize=8)
        axis.set_xticks(FRACTIONS)
        axis.set_xticklabels(
            [f"{f}\n({FRACTION_SCENES[f]})" for f in FRACTIONS], fontsize=7
        )
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(True, linewidth=0.4, alpha=0.35)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("Development F1", fontsize=8)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", metadata={"CreationDate": None})
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=Path("results/h100/evidence"))
    parser.add_argument("--arms-config", type=Path, default=Path("configs/arms.yaml"))
    parser.add_argument("--detector-config", type=Path, default=Path("configs/detector.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    validated = validate_evidence(
        args.evidence_root,
        arms_config=args.arms_config,
        detector_config=args.detector_config,
    )
    summary = {
        "cells": len(validated["cells"]),  # type: ignore[arg-type]
        "test_results": len(validated["test_results"]),  # type: ignore[arg-type]
        "final_eval_rows": len(validated["final_eval_rows"]),  # type: ignore[arg-type]
        "gpu_hours": round(float(validated["campaign"]["gpu_hours"]), 1),  # type: ignore[index]
    }
    print(json.dumps(summary, sort_keys=True))
    if args.validate_only:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = args.output_dir / "heldout_results.tex"
    tex_path.write_text(render_tex(validated), encoding="utf-8", newline="\n")
    figure_path = args.output_dir / "heldout_label_efficiency.pdf"
    render_figure(validated, figure_path)
    manifest = {
        "generator": "src.analysis.heldout_results",
        "inputs": {
            "cohort_sha256": validated["campaign"]["cohort_sha256"],  # type: ignore[index]
            "detector_sha256": validated["campaign"]["detector_sha256"],  # type: ignore[index]
            "arms_config_sha256": _sha256_file(Path(args.arms_config)),
        },
        "outputs": {
            "heldout_results.tex": _sha256_file(tex_path),
            "heldout_label_efficiency.pdf": _sha256_file(figure_path),
        },
        "summary": summary,
    }
    (args.output_dir / "heldout_generated_manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
