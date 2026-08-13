"""Guard test: no scene_id in more than one split.

Runs against the frozen ``data/splits.json`` once it exists; skips cleanly
before then (CI is green-with-skips early in the project — see ci.yml).
Deliberately stdlib-only so the guard runs in the minimal CI environment.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pytest

SPLITS_PATH = Path(__file__).resolve().parents[1] / "data" / "splits.json"
REQUIRED_SPLITS = ("train", "dev", "test", "eval_final")


@pytest.mark.skipif(
    not SPLITS_PATH.exists(),
    reason="data/splits.json not built yet (sprint-1 artifact)",
)
def test_split_disjoint():
    payload = json.loads(SPLITS_PATH.read_text())
    splits = payload["splits"]

    for name in REQUIRED_SPLITS:
        assert name in splits, f"missing split: {name}"

    for name, scene_ids in splits.items():
        assert len(scene_ids) == len(set(scene_ids)), f"duplicate scene in {name}"

    for first, second in combinations(REQUIRED_SPLITS, 2):
        overlap = set(splits[first]) & set(splits[second])
        assert not overlap, f"scenes in both {first} and {second}: {sorted(overlap)}"

    assert splits["train"], "train split must not be empty"
    for scene_id in splits["eval_final"]:
        assert scene_id.endswith("v"), (
            f"eval_final must hold only human-verified …v scenes, got {scene_id}"
        )
