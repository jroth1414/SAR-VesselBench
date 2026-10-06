"""Staging a delivered run tree keeps bindings exact and redacts cluster paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import stage_evidence
from scripts.stage_evidence import StagingError, stage

PREFIX = "/projects/geofam"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_tree(root: Path, *, marker_text: str = '{"exp_id": "a-f10-s0"}\n') -> Path:
    control = root / ".h100"
    control.mkdir(parents=True)
    cell = root / "a-f10-s0"
    (cell / "metrics").mkdir(parents=True)
    (cell / "final_metrics.json").write_bytes(marker_text.encode("utf-8"))
    (cell / "metrics" / "metrics.csv").write_text("epoch,dev_f1\n0,0.5\n", encoding="utf-8")
    (cell / "runtime_provenance.json").write_text(
        json.dumps({"root": f"{PREFIX}/jroth/runs/a-f10-s0"}), encoding="utf-8"
    )
    (cell / "test_metrics.json").write_text('{"metrics": {"f1": 0.8}}', encoding="utf-8")
    (cell / "final_verified_metrics.json").write_text('{"metrics": {"f1": 0.6}}', encoding="utf-8")
    cohort = {
        "cell_count": 1,
        "cells": [
            {
                "exp_id": "a-f10-s0",
                "completion_marker": {
                    "relative_path": "a-f10-s0/final_metrics.json",
                    "sha256": _sha(marker_text.encode("utf-8")),
                },
            }
        ],
    }
    (control / "TRAINING_COHORT.json").write_text(json.dumps(cohort), encoding="utf-8")
    (control / "EVAL_GROUND_TRUTH_VALIDATED.json").write_text('{"audit_schema": 1}', encoding="utf-8")
    return root


def test_stage_copies_bound_files_exactly_and_redacts_provenance(tmp_path: Path) -> None:
    run_root = _run_tree(tmp_path / "runs")
    out = tmp_path / "evidence"

    summary = stage(run_root, out)

    original = (run_root / "a-f10-s0" / "runtime_provenance.json").read_bytes()
    staged = (out / "a-f10-s0" / "runtime_provenance.json").read_text(encoding="utf-8")
    assert PREFIX not in staged and "/cluster-site-redacted/jroth/runs" in staged
    redactions = json.loads((out / "REDACTIONS.json").read_text(encoding="utf-8"))
    assert redactions["files"] == {
        "a-f10-s0/runtime_provenance.json": {"original_sha256": _sha(original), "replaced": PREFIX}
    }
    assert (out / "a-f10-s0" / "metrics.csv").read_bytes() == (
        run_root / "a-f10-s0" / "metrics" / "metrics.csv"
    ).read_bytes()
    assert (out / "TRAINING_COHORT.json").read_bytes() == (
        run_root / ".h100" / "TRAINING_COHORT.json"
    ).read_bytes()
    assert (out / "a-f10-s0" / "final_verified_metrics.json").is_file()
    assert summary["cells"] == 1 and summary["redacted_files"] == 1


def test_stage_carries_a_terminal_recovery_record_in_place_of_a_curve(tmp_path: Path) -> None:
    run_root = _run_tree(tmp_path / "runs")
    (run_root / "a-f10-s0" / "metrics" / "metrics.csv").unlink()
    (run_root / "a-f10-s0" / "terminal_recovery.json").write_bytes(b'{"status": "terminal-dev-recovered"}\n')
    out = tmp_path / "evidence"

    stage(run_root, out)

    assert not (out / "a-f10-s0" / "metrics.csv").exists()
    assert (out / "a-f10-s0" / "terminal_recovery.json").read_bytes() == (
        run_root / "a-f10-s0" / "terminal_recovery.json"
    ).read_bytes()


def test_stage_refuses_a_marker_that_breaks_its_cohort_binding(tmp_path: Path) -> None:
    run_root = _run_tree(tmp_path / "runs")
    (run_root / "a-f10-s0" / "final_metrics.json").write_bytes(b'{"tampered": 1}\n')
    out = tmp_path / "evidence"

    with pytest.raises(StagingError, match="cohort binding"):
        stage(run_root, out)
    assert not out.exists() and not (tmp_path / "evidence.staging").exists()


def test_stage_refuses_a_byte_exact_file_carrying_the_private_prefix(tmp_path: Path) -> None:
    run_root = _run_tree(tmp_path / "runs", marker_text=f'{{"path": "{PREFIX}/x"}}\n')

    with pytest.raises(StagingError, match="private prefix"):
        stage(run_root, tmp_path / "evidence")


def test_stage_never_overwrites_an_existing_output(tmp_path: Path) -> None:
    run_root = _run_tree(tmp_path / "runs")
    out = tmp_path / "evidence"
    out.mkdir()

    assert stage_evidence.main(["--run-root", str(run_root), "--out", str(out)]) == 1
