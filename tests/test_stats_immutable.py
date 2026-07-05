"""Freeze guard for data/stats.json (DEVPLAN do-not-touch manifest).

Pinned 2026-07-05 after the study-set chipping run completed (150/150 scenes,
zero failures): global per-polarization mean/std computed ONCE over the
105,408 chips of the 111 frozen train-split scenes (6.28e10 valid pixels per
channel; VH -26.448 dB sigma 5.951, VV -16.599 dB sigma 6.062), reused
unchanged for every label fraction, both tracks, and all seeds (P1.5).
Re-pinning requires an explicit human STOP.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PINNED_STATS_SHA256 = (
    "8e7691002e540f7274f9e1d67812fe8ae2b6c45775a6e9072a5177b7fadb4490"
)


def test_stats_immutable():
    stats_path = Path(__file__).resolve().parents[1] / "data" / "stats.json"
    digest = hashlib.sha256(stats_path.read_bytes()).hexdigest()

    assert digest == PINNED_STATS_SHA256
