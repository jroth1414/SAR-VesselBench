"""Freeze guard for configs/detector.yaml (DEVPLAN do-not-touch manifest).

Originally pinned at the sprint-3-detector merge. Re-pinned 2026-07-26 on
``sprint-7c-fp32-grid`` after the owner approved one shared ``32-true``
amendment and a from-scratch rerun of all 32 core cells. The head, optimizer,
schedule, augmentation, and decode contract remain identical for every arm;
future changes still require an explicit human STOP and re-pin.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

PINNED_DETECTOR_SHA256 = (
    "c42ae65bf9045cc93f0d73ae1437b2f6a1300670cb49d8e93f83a39d58a62a12"
)


def test_detector_immutable():
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "detector.yaml"
    )
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()

    assert digest == PINNED_DETECTOR_SHA256

    config = yaml.safe_load(config_path.read_text())
    assert config["schedule"]["precision"] == "32-true"
