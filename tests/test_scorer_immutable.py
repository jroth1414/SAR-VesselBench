from __future__ import annotations

import hashlib
from pathlib import Path


# Freeze baseline: src/eval/scorer.py after sprint-2b-eval-hardening.
# Re-pinned 2026-07-01 under human sign-off for the intentional BLOCKER-3 fix:
# near_shore slice metrics now count unmatched near-shore false positives, and
# score_dataset prevents cross-scene matching. Future scorer changes require
# re-recording this pin under human sign-off (guard change = STOP).
PINNED_SCORER_SHA256 = (
    "85dec7ab083b531547fe81fea5aa02e4d828457ff157619afae29981cf49cd32"
)


def test_scorer_immutable():
    scorer_path = Path(__file__).resolve().parents[1] / "src" / "eval" / "scorer.py"
    digest = hashlib.sha256(scorer_path.read_bytes()).hexdigest()

    assert digest == PINNED_SCORER_SHA256
