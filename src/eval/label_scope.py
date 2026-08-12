"""Validate the label view used for cohort-bound development/test scoring."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

from src.eval.ground_truth_audit import scene_ids_sha256


SCORE_VIEW_CONTRACT = "frozen-cohort-test16-v1"
SCORE_LABELS_EXPOSED_PATH = "data/raw/xview3/labels/train.csv"
EXPECTED_TRAIN_SCENES = 111
EXPECTED_DEV_SCENES = 23
EXPECTED_DEV_TRAINING_SCENES = 8
EXPECTED_TEST_SCENES = 16
EXPECTED_SCORE_LABEL_ROWS = 15_079


def _load_json(path: Path, description: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{description} must be a regular non-symlink: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} must contain a JSON object")
    return payload


def split_scope(
    splits_path: Path, *, production: bool = True
) -> dict[str, tuple[str, ...]]:
    """Return exact training, fixed development, test, and final scene IDs."""

    payload = _load_json(splits_path, "frozen split")
    try:
        splits = payload["splits"]
        train = tuple(map(str, splits["train"]))
        dev = tuple(map(str, splits["dev"]))
        test = tuple(map(str, splits["test"]))
        eval_final = tuple(map(str, splits["eval_final"]))
    except (KeyError, TypeError) as exc:
        raise RuntimeError("frozen split structure is invalid") from exc
    groups = {"train": train, "dev": dev, "test": test, "eval_final": eval_final}
    if any(not scene_id for values in groups.values() for scene_id in values):
        raise RuntimeError("frozen split contains an empty scene ID")
    if any(len(values) != len(set(values)) for values in groups.values()):
        raise RuntimeError("frozen split contains duplicate scene IDs")
    owners: dict[str, str] = {}
    for name, values in groups.items():
        for scene_id in values:
            previous = owners.setdefault(scene_id, name)
            if previous != name:
                raise RuntimeError(
                    f"scene {scene_id!r} occurs in both {previous} and {name}"
                )
    if production and (
        len(train) != EXPECTED_TRAIN_SCENES
        or len(dev) != EXPECTED_DEV_SCENES
        or len(test) != EXPECTED_TEST_SCENES
    ):
        raise RuntimeError("frozen train/dev/test scene counts are not 111/23/16")
    if len(dev) < EXPECTED_DEV_TRAINING_SCENES:
        raise RuntimeError("frozen development split has fewer than eight scenes")
    return {
        "train": tuple(sorted(train)),
        "dev8": tuple(sorted(dev)[:EXPECTED_DEV_TRAINING_SCENES]),
        "test": tuple(sorted(test)),
        "eval_final": tuple(sorted(eval_final)),
    }


def score_labels_summary(
    path: Path,
    *,
    splits_path: Path,
    production: bool = True,
) -> dict[str, object]:
    """Require exact training-plus-test rows in a post-cohort scoring view."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError("score-view labels must be a regular non-symlink")
    scope = split_scope(splits_path, production=production)
    allowed = set(scope["train"]) | set(scope["test"])
    forbidden = set(scope["dev8"]) | set(scope["eval_final"])
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if not reader.fieldnames or "scene_id" not in reader.fieldnames:
        raise RuntimeError("score-view labels lack scene_id")
    rows = 0
    observed: set[str] = set()
    for row in reader:
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id or scene_id not in allowed:
            raise RuntimeError(f"score-view labels contain forbidden scene {scene_id!r}")
        observed.add(scene_id)
        rows += 1
    if observed != allowed or observed & forbidden:
        raise RuntimeError("score-view labels do not cover exact training/test scope")
    if production and rows != EXPECTED_SCORE_LABEL_ROWS:
        raise RuntimeError("score-view labels do not contain exactly 15,079 rows")
    scene_ids = sorted(observed)
    return {
        "exposed_path": SCORE_LABELS_EXPOSED_PATH,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "row_count": rows,
        "scene_count": len(scene_ids),
        "scene_ids_sha256": scene_ids_sha256(scene_ids),
        "train_scene_ids_sha256": scene_ids_sha256(scope["train"]),
        "test_scene_ids_sha256": scene_ids_sha256(scope["test"]),
        "contract": SCORE_VIEW_CONTRACT,
    }
