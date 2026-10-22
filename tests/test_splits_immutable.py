"""Freeze guard for data/splits.json (DEVPLAN do-not-touch manifest).

Pinned 2026-07-05 when the real study split was built and committed:
150 stratified train-pool scenes (seed 0, label-centroid region bins x
shoreline, selected from the 554 local train scenes per the human scene-count
decision) split 111 train / 23 dev / 16 test, plus all 50 human-verified
validation scenes as eval_final. Scene membership never changes mid-study;
re-pinning requires an explicit human STOP (like the scorer pin).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PINNED_SPLITS_SHA256 = (
    "0d201003999b6e634bdb3f9cccaedbd391b9d0cdcdb7f9370470d56a5e391db6"
)


def test_splits_immutable():
    splits_path = Path(__file__).resolve().parents[1] / "data" / "splits.json"
    digest = hashlib.sha256(splits_path.read_bytes()).hexdigest()

    assert digest == PINNED_SPLITS_SHA256
