"""Corrected-dev diagnostics for explicitly selected legacy core checkpoints.

This command is intentionally non-reportable and read-only with respect to
canonical run directories.  It evaluates ``best.ckpt`` and/or ``last.ckpt``
independently on the same sorted first-eight DEV scenes used during training,
selecting a fresh DEV threshold for each exact checkpoint.  The only output is
an atomically published JSON artifact beneath ``<runs-root>/diagnostics``.

The command reads ``train.csv`` only.  It never opens ``validation.csv``, TEST
rasters, or verified-final rasters.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from scripts.h100.contracts import atomic_write_json
from src.eval.scorer import DatasetScoreResult, GroundTruthPoint, Metrics

DIAGNOSTIC_SCHEMA = 1
CHECKPOINT_KINDS = ("best", "last")
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
EXPECTED_DEV_COUNTS = {
    "positive": 517,
    "background": 107,
    "ignore": 118,
}


@dataclass(frozen=True)
class DevContract:
    scene_ids: tuple[str, ...]
    ground_truth_by_scene: Mapping[str, Sequence[GroundTruthPoint]]
    counts: Mapping[str, int]
    counts_by_scene: Mapping[str, Mapping[str, int]]


def sha256_file(path: Path, chunk_bytes: int = 8 * 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _label_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    from src.eval.ground_truth import classify_label

    counts = {"positive": 0, "background": 0, "ignore": 0}
    for row in rows:
        counts[classify_label(row)] += 1
    return counts


def _assert_split_disjoint(splits: Mapping[str, Sequence[str]]) -> None:
    names = ("train", "dev", "test", "eval_final")
    missing = [name for name in names if name not in splits]
    if missing:
        raise RuntimeError(f"split file is missing required partitions: {missing}")
    normalized = {name: set(map(str, splits[name])) for name in names}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(normalized[left] & normalized[right])
            if overlap:
                raise RuntimeError(
                    f"split overlap refuses diagnostics: {left}/{right}: "
                    f"{overlap[:8]}"
                )


def load_dev_contract(*, splits_path: Path, train_labels_csv: Path) -> DevContract:
    """Load and validate the only label subset authorized for diagnostics."""

    import pandas as pd

    from src.eval.ground_truth import ground_truth_from_labels

    if train_labels_csv.name != "train.csv":
        raise RuntimeError("diagnostics require the xView3 train.csv label source")
    if train_labels_csv.is_symlink():
        raise RuntimeError("diagnostic train.csv must not be a symlink")

    split_payload = json.loads(splits_path.read_text())
    splits = split_payload.get("splits")
    if not isinstance(splits, Mapping):
        raise RuntimeError("split file lacks a splits mapping")
    _assert_split_disjoint(splits)

    scene_ids = tuple(sorted(map(str, splits["dev"]))[:8])
    if len(scene_ids) != 8:
        raise RuntimeError(f"diagnostics require exactly eight DEV scenes, got {len(scene_ids)}")
    forbidden = set(map(str, splits["test"])) | set(map(str, splits["eval_final"]))
    if set(scene_ids) & forbidden:
        raise RuntimeError("diagnostic DEV scenes overlap TEST or eval_final")

    labels = pd.read_csv(train_labels_csv)
    if "scene_id" not in labels.columns:
        raise RuntimeError("train.csv lacks scene_id")
    labels = labels.copy()
    labels["scene_id"] = labels["scene_id"].astype(str)
    available = set(labels["scene_id"])
    missing = sorted(set(scene_ids) - available)
    if missing:
        raise RuntimeError(f"train.csv lacks diagnostic DEV scenes: {missing}")

    ground_truth_by_scene: dict[str, Sequence[GroundTruthPoint]] = {}
    counts_by_scene: dict[str, Mapping[str, int]] = {}
    total = {"positive": 0, "background": 0, "ignore": 0}
    for scene_id in scene_ids:
        rows = labels[labels["scene_id"] == scene_id].to_dict(orient="records")
        counts = _label_counts(rows)
        converted = ground_truth_from_labels(rows)
        if len(converted) != counts["positive"] + counts["ignore"]:
            raise RuntimeError(
                f"{scene_id}: shared GT converter/count classifier disagree"
            )
        counts_by_scene[scene_id] = counts
        ground_truth_by_scene[scene_id] = converted
        for category in total:
            total[category] += counts[category]

    if total != EXPECTED_DEV_COUNTS:
        raise RuntimeError(
            "corrected first-eight DEV counts mismatch: "
            f"{total} != {EXPECTED_DEV_COUNTS}"
        )
    return DevContract(scene_ids, ground_truth_by_scene, total, counts_by_scene)


def current_core_run_ids(arms_path: Path) -> set[str]:
    import yaml

    config = yaml.safe_load(arms_path.read_text())
    return {
        f"{meta['short']}-f{int(round(float(fraction) * 100))}-s{int(seed)}"
        for meta in config["arms"].values()
        for fraction in config["label_fracs"]
        for seed in config["seeds"]["core"]
    }


def validate_run_ids(run_ids: Sequence[str], allowed: set[str]) -> tuple[str, ...]:
    if not run_ids:
        raise RuntimeError("at least one explicit --run-id is required")
    if len(set(run_ids)) != len(run_ids):
        raise RuntimeError("duplicate diagnostic run IDs are not allowed")
    for run_id in run_ids:
        if RUN_ID_RE.fullmatch(run_id) is None or run_id not in allowed:
            raise RuntimeError(f"unsafe or non-core diagnostic run ID: {run_id!r}")
    return tuple(run_ids)


def _reject_symlink_components(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"path escapes runs root: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"diagnostic input path contains a symlink: {current}")


def checkpoint_path(runs_root: Path, run_id: str, kind: str) -> Path:
    if kind not in CHECKPOINT_KINDS:
        raise RuntimeError(f"unsupported checkpoint kind: {kind!r}")
    path = runs_root / run_id / "checkpoints" / f"{kind}.ckpt"
    _reject_symlink_components(runs_root, path)
    if not path.is_file():
        raise RuntimeError(f"diagnostic checkpoint is absent: {path}")
    return path


def checkpoint_epoch(path: Path) -> int:
    """Read Lightning's zero-based epoch from an exact checkpoint."""

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    epoch = payload.get("epoch") if isinstance(payload, Mapping) else None
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError(f"checkpoint has no valid zero-based epoch: {path}")
    return epoch


def load_checkpoint_model(path: Path, device: str):
    from src.train.lit_modules import HeatmapLitModule

    return (
        HeatmapLitModule.load_from_checkpoint(str(path), map_location="cpu")
        .to(device)
        .eval()
    )


def _metrics_payload(metrics: Metrics) -> dict[str, int | float]:
    return {
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "ignored_predictions": metrics.ignored_predictions,
    }


def _score_payload(result: DatasetScoreResult) -> dict[str, object]:
    return {
        "aggregate": _metrics_payload(result.aggregate),
        "slices": {
            name: _metrics_payload(metrics)
            for name, metrics in sorted(result.slices.items())
        },
        "per_scene": {
            scene_id: {
                "aggregate": _metrics_payload(scene.aggregate),
                "slices": {
                    name: _metrics_payload(metrics)
                    for name, metrics in sorted(scene.slices.items())
                },
            }
            for scene_id, scene in sorted(result.scene_results.items())
        },
    }


def git_sha(repo: Path) -> str:
    value = subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise RuntimeError("source checkout did not resolve to a full Git SHA")
    return value


def publish_diagnostic(
    payload: Mapping[str, object], *, runs_root: Path, timestamp: str
) -> Path:
    """Atomically publish a new directory under the ignored namespace."""

    diagnostics_root = runs_root / "diagnostics"
    if diagnostics_root.is_symlink():
        raise RuntimeError("diagnostics output root must not be a symlink")
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", timestamp):
        raise RuntimeError("diagnostic timestamp must be UTC YYYYMMDDTHHMMSSZ")
    destination = diagnostics_root / f"eval-contract-{timestamp}"
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"diagnostic destination already exists: {destination}")

    temporary = Path(
        tempfile.mkdtemp(prefix=".eval-contract-", dir=diagnostics_root)
    )
    try:
        atomic_write_json(temporary / "diagnostic.json", payload)
        os.replace(temporary, destination)
        directory_fd = os.open(diagnostics_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            artifact = temporary / "diagnostic.json"
            if artifact.exists():
                artifact.unlink()
            temporary.rmdir()
    return destination


def run_diagnostic(
    *,
    repo: Path,
    runs_root: Path,
    data_cfg: Mapping[str, object],
    det_cfg: Mapping[str, object],
    run_ids: Sequence[str],
    checkpoint_kinds: Sequence[str],
    timestamp: str | None = None,
    device: str = "cuda",
    model_loader: Callable[[Path, str], object] | None = None,
    epoch_reader: Callable[[Path], int] | None = None,
    infer_fn: Callable[..., object] | None = None,
) -> Path:
    """Evaluate exact legacy checkpoints and publish one diagnostic artifact."""

    from src.eval.infer_scene import infer_scene
    from src.eval.scorer import score_dataset
    from src.eval.threshold import apply_threshold, select_f1_threshold

    repo = repo.resolve()
    runs_root = runs_root.resolve()
    run_ids = validate_run_ids(
        run_ids, current_core_run_ids(repo / "configs" / "arms.yaml")
    )
    kinds = tuple(checkpoint_kinds)
    if not kinds or len(kinds) != len(set(kinds)) or any(
        kind not in CHECKPOINT_KINDS for kind in kinds
    ):
        raise RuntimeError("checkpoint kinds must be a unique nonempty best/last subset")

    paths = data_cfg["paths"]
    if not isinstance(paths, Mapping):
        raise RuntimeError("data config paths are malformed")
    raw_xview3 = repo / str(paths["raw_xview3"])
    contract = load_dev_contract(
        splits_path=repo / str(paths["splits"]),
        train_labels_csv=raw_xview3 / "labels" / "train.csv",
    )
    stats_path = repo / str(paths["stats"])
    stats = json.loads(stats_path.read_text())
    scene_root = raw_xview3 / "GRD"

    model_loader = model_loader or load_checkpoint_model
    epoch_reader = epoch_reader or checkpoint_epoch
    infer_fn = infer_fn or infer_scene
    records = []
    for run_id in run_ids:
        for kind in kinds:
            checkpoint = checkpoint_path(runs_root, run_id, kind)
            checkpoint_sha256 = sha256_file(checkpoint)
            epoch = epoch_reader(checkpoint)
            model = model_loader(checkpoint, device)
            predictions = {}
            for scene_id in contract.scene_ids:
                predictions[scene_id] = infer_fn(
                    model,
                    scene_root / scene_id,
                    stats=stats,
                    tau=det_cfg["decode"]["candidate_floor"],
                    d_nms_m=det_cfg["decode"]["d_nms_m"],
                    tile_px=det_cfg["eval"]["tile_px"],
                    tile_stride_px=det_cfg["eval"]["tile_stride_px"],
                    batch_size=det_cfg["eval"]["infer_batch"],
                    device=device,
                    precision=det_cfg["schedule"]["precision"],
                )
            threshold = select_f1_threshold(
                contract.ground_truth_by_scene, predictions
            )
            if not math.isfinite(threshold):
                raise RuntimeError(f"{run_id}/{kind}: selected threshold is non-finite")
            result = score_dataset(
                contract.ground_truth_by_scene,
                apply_threshold(predictions, threshold),
            )
            records.append(
                {
                    "run_id": run_id,
                    "checkpoint": {
                        "kind": kind,
                        "relative_path": checkpoint.relative_to(runs_root).as_posix(),
                        "sha256": checkpoint_sha256,
                        "epoch_zero_based": epoch,
                    },
                    "threshold": threshold,
                    "threshold_selection": "independent-dev-f1-high-threshold-tie",
                    "n_candidates": sum(len(items) for items in predictions.values()),
                    "metrics": _score_payload(result),
                }
            )
            del model
            if str(device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()

    created_at = dt.datetime.now(dt.timezone.utc)
    timestamp = timestamp or created_at.strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "diagnostic_schema": DIAGNOSTIC_SCHEMA,
        "reportable": False,
        "purpose": "superseded-v100-eval-contract-diagnostic",
        "created_at": created_at.isoformat(timespec="seconds"),
        "source_git_sha": git_sha(repo),
        "scene_policy": "sorted-first-8-frozen-dev",
        "scene_ids": list(contract.scene_ids),
        "label_source": "train.csv",
        "ground_truth_contract": {
            "positive": "is_vessel=true and confidence in {HIGH,MEDIUM}",
            "background": "is_vessel=false and confidence in {HIGH,MEDIUM}",
            "ignore": "confidence=LOW",
            "counts": dict(contract.counts),
            "counts_by_scene": contract.counts_by_scene,
        },
        "inference_precision": det_cfg["schedule"]["precision"],
        "inputs": {
            "configs/detector.yaml": sha256_file(repo / "configs/detector.yaml"),
            str(paths["splits"]): sha256_file(repo / str(paths["splits"])),
            str(paths["stats"]): sha256_file(stats_path),
            "src/eval/scorer.py": sha256_file(repo / "src/eval/scorer.py"),
        },
        "checkpoints": records,
    }
    return publish_diagnostic(payload, runs_root=runs_root, timestamp=timestamp)


def main(argv: Sequence[str] | None = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument(
        "--checkpoints", nargs="+", choices=CHECKPOINT_KINDS, default=list(CHECKPOINT_KINDS)
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    repo = Path(args.repo).resolve()
    data_cfg = yaml.safe_load((repo / args.data_config).read_text())
    det_cfg = yaml.safe_load((repo / args.detector_config).read_text())
    output = run_diagnostic(
        repo=repo,
        runs_root=repo / args.runs_root,
        data_cfg=data_cfg,
        det_cfg=det_cfg,
        run_ids=args.run_id,
        checkpoint_kinds=args.checkpoints,
        device=args.device,
    )
    print(json.dumps({"diagnostic": str(output), "status": "complete"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
