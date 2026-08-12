"""Exhaustive per-cell fail-closed checks for the public H100 snapshot.

Each of the exact 32 experiment identities is independently challenged by
eight corruption classes.  These are contract tests, not duplicated smoke
assertions: each case targets a different route by which an incomplete,
unverified, mixed-hardware, or malformed cell could enter a public result.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.analysis.h100_results import SnapshotError, expected_cells, load_snapshot


REPO = Path(__file__).resolve().parents[1]
ARMS = REPO / "configs/arms.yaml"
SNAPSHOT = REPO / "results/h100/h100_campaign_snapshot.json"
CELL_IDS = tuple(expected_cells(ARMS))
INVARIANTS = (
    "missing_cell",
    "duplicate_cell",
    "identity_drift",
    "unverified_metric",
    "pending_evidence",
    "wrong_hardware",
    "nonfinite_metric",
    "incomplete_complete_profile",
)
HEX = "a" * 64


def _verified(cell: dict, hardware_sha256: str) -> None:
    cell.update(
        {
            "status": "DONE",
            "latest_epoch": 9,
            "duration_seconds": 100,
            "train_loss": 0.1,
            "development": {
                "f1": 0.8,
                "precision": 0.8,
                "recall": 0.8,
                "threshold": 0.5,
            },
            "evidence": {
                "verification": "contract-verified",
                "completion_marker_sha256": HEX,
                "runtime_provenance_sha256": HEX,
                "best_checkpoint_sha256": HEX,
                "last_checkpoint_sha256": HEX,
                "hardware_class_sha256": hardware_sha256,
            },
        }
    )


def _corrupt(payload: dict, exp_id: str, invariant: str) -> str:
    cells = payload["cells"]
    index = next(i for i, cell in enumerate(cells) if cell["exp_id"] == exp_id)
    target = cells[index]
    hardware_sha = payload["campaign"]["hardware"]["class_sha256"]

    if invariant == "missing_cell":
        cells.pop(index)
        return "exactly the 32 unique"
    if invariant == "duplicate_cell":
        cells[index] = copy.deepcopy(cells[(index + 1) % len(cells)])
        return "exactly the 32 unique"
    if invariant == "identity_drift":
        target["init"] = "tampered_initialization"
        return "metadata mismatch for init"
    if invariant == "unverified_metric":
        target.update(
            {
                "status": "DONE",
                "latest_epoch": 1,
                "duration_seconds": 1,
                "train_loss": None,
                "development": {
                    "f1": 0.8,
                    "precision": 0.8,
                    "recall": 0.8,
                    "threshold": 0.5,
                },
                "evidence": {"verification": "status-only"},
            }
        )
        return "lacks contract-verified completion evidence"
    if invariant == "pending_evidence":
        target.update(
            {
                "status": "PENDING",
                "latest_epoch": 0,
                "duration_seconds": None,
                "train_loss": None,
                "development": None,
                "evidence": {"verification": "status-only"},
            }
        )
        return "pending cell contains result evidence"
    if invariant == "wrong_hardware":
        _verified(target, "b" * 64)
        return "different hardware class"
    if invariant == "nonfinite_metric":
        _verified(target, hardware_sha)
        target["development"]["f1"] = float("nan")
        return "must be finite"
    if invariant == "incomplete_complete_profile":
        payload["profile"] = "complete"
        payload["source"] = {
            "kind": "verified-reverse-result-package",
            "description": "synthetic invariant fixture",
            "sha256": "c" * 64,
        }
        for cell in cells:
            if cell["exp_id"] != exp_id:
                _verified(cell, hardware_sha)
        target.update(
            {
                "status": "DONE",
                "latest_epoch": 1,
                "duration_seconds": 1,
                "train_loss": None,
                "development": None,
                "evidence": {"verification": "status-only"},
            }
        )
        return "32 contract-verified DONE cells"
    raise AssertionError(f"unknown invariant: {invariant}")


@pytest.mark.parametrize("exp_id", CELL_IDS)
@pytest.mark.parametrize("invariant", INVARIANTS)
def test_every_cell_fails_closed_for_every_invariant(
    tmp_path: Path, exp_id: str, invariant: str
) -> None:
    payload = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    message = _corrupt(payload, exp_id, invariant)
    candidate = tmp_path / "snapshot.json"
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SnapshotError, match=message):
        load_snapshot(candidate, ARMS)
