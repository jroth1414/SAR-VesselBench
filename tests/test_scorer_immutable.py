from __future__ import annotations

import hashlib
from pathlib import Path


# Freeze baseline: src/eval/scorer.py as introduced in 5e8faf3 (sha256 below).
# Corrected 2026-07-01: the original pin (570accd4...) was mis-recorded at birth
# and never matched the committed scorer (verified: scorer.py is byte-identical
# to its 5e8faf3 blob, so this is a pin fix, not scorer drift). A future
# legitimate scorer change (e.g. DEVPLAN BLOCKER-3, the near-shore slice fix)
# requires re-recording this pin under human sign-off (guard change = STOP).
PINNED_SCORER_SHA256 = (
    "8606bb5e1fb67ff7605d40170a57626a3fa59624667c2f010f8ee09756e926ee"
)


def test_scorer_immutable():
    scorer_path = Path("src/eval/scorer.py")
    digest = hashlib.sha256(scorer_path.read_bytes()).hexdigest()

    assert digest == PINNED_SCORER_SHA256
