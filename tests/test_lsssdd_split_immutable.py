"""Freeze guard for data/lsssdd_split.json (DEVPLAN do-not-touch manifest).

Pinned 2026-07-05 when LS-SSDD-v1.0 landed (9,000 sub-images verified):
seeded (seed 0) 90/10 internal train/val partition — 8,100 / 900 — consumed
identically by the Arm-4 (ViT) and Arm-8 (CNN) supervised pretrainings.
Deliberately spans ALL 9,000 sub-images rather than only the dataset's
official 6,000-image benchmark-train side: LS-SSDD is a pretraining source
here, never an evaluated benchmark, and the val side exists only for early
stopping (documented deviation, see DEVPLAN cold-start runbook).
Re-pinning requires an explicit human STOP.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PINNED_LSSSDD_SPLIT_SHA256 = (
    "ae6d0343d021341d58c2d037e20eb9a9f4e67ace3b7c677439973838fd1473a3"
)


def test_lsssdd_split_immutable():
    split_path = (
        Path(__file__).resolve().parents[1] / "data" / "lsssdd_split.json"
    )
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()

    assert digest == PINNED_LSSSDD_SPLIT_SHA256
