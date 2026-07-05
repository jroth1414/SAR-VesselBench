"""Freeze guard for configs/detector.yaml (DEVPLAN do-not-touch manifest).

Pinned 2026-07-05 at the sprint-3-detector merge (phase-3-done): the shared
head / optimizer / schedule / augmentation / decode contract — the fairness
guarantee every arm trains under. Never edited per study-arm; changing it
after this pin requires an explicit human STOP and re-pin (ground rule 2).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PINNED_DETECTOR_SHA256 = (
    "4fd1bfe88861cc676dd67b2092e379fbcf401dd9c1d42fb09e81a84b9cdbe2f8"
)


def test_detector_immutable():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "detector.yaml"
    )
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()

    assert digest == PINNED_DETECTOR_SHA256
