from __future__ import annotations

import hashlib
from pathlib import Path


PINNED_SCORER_SHA256 = (
    "570accd42d11fe3c21714441a00346b2b5a4b02becb7c61ca07a11cc8eef3f90"
)


def test_scorer_immutable():
    scorer_path = Path("src/eval/scorer.py")
    digest = hashlib.sha256(scorer_path.read_bytes()).hexdigest()

    assert digest == PINNED_SCORER_SHA256
