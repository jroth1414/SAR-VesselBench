"""Collector integration keeps recipe identity separate from verified execution."""

import csv
import json
import pytest
import yaml

from scripts.h100 import resume_recovery, test_recovery
from scripts.handoff.terminal_contract import POLICY
from src.analysis import curves
from src.eval.heldout_contract import HeldoutContractError, create_training_cohort
from src.eval.result_contract import sha256_file
from src.train.terminal_recovery import write_once
from test_heldout_contract import SCENES, _test_payload
from test_terminal_cohort import cohort_args, mixed_matrix


RECIPE = POLICY["campaign_git_sha"]
EXECUTION = "84133e1ea6a3e52bc20dbed19a1a1c5706f41518"


@pytest.fixture
def grid_fixture(tmp_path, monkeypatch):
    repo, runs = tmp_path / "repo", tmp_path / "runs"
    (repo / "configs").mkdir(parents=True)
    arms = yaml.safe_load((curves.REPO / "configs/arms.yaml").read_text())
    detector = repo / "configs/detector.yaml"
    detector.write_bytes(curves.DETECTOR_PATH.read_bytes())
    splits = repo / "splits.json"
    train = [f"train-{i:02d}" for i in range(10)]
    splits.write_text(json.dumps({"splits": {"train": train, "test": SCENES}}))
    labels = repo / "raw/labels/train.csv"
    labels.parent.mkdir(parents=True)
    fields = ("scene_id", "confidence", "is_vessel", "source",
              "distance_from_shore_km", "detect_scene_column", "detect_scene_row")
    with labels.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scene in train:
            writer.writerow(dict(zip(fields, (scene, "HIGH", "true", "Manual", 1, 1, 1))))
    (repo / "configs/data.yaml").write_text(yaml.safe_dump({"seed": 0, "paths": {
        "splits": str(splits), "raw_xview3": str(labels.parents[1])}}))
    cells = mixed_matrix(runs, monkeypatch)
    cohort_path = runs / ".h100/TRAINING_COHORT.json"
    cohort = create_training_cohort(output=cohort_path, **cohort_args(runs, cells))
    for cell, record in zip(cells, cohort["cells"]):
        payload = _test_payload(cell, record)
        payload["cohort_sha256"] = sha256_file(cohort_path)
        write_once(runs / cell.exp_id / "test_metrics.json", payload)
    monkeypatch.setattr(curves, "REPO", repo)
    monkeypatch.setattr(curves, "DETECTOR_PATH", detector)
    monkeypatch.setattr(curves, "EXPECTED_DETECTOR_SHA256", "f" * 64)
    monkeypatch.setattr(curves, "load_cells", lambda root: cells)
    monkeypatch.setattr(curves, "_git_sha", lambda: EXECUTION)
    return repo, runs, arms, runs / "summary/grid.csv"


@pytest.mark.parametrize("recovery", [True, False])
def test_real_collector_validates_mixed_cohort_without_mutating_scores(grid_fixture, monkeypatch, recovery):
    repo, runs, arms, output = grid_fixture
    calls = []

    def load(root, namespace, campaign_git_sha=None, *, require_test_activation=True):
        calls.append((root, namespace, campaign_git_sha, require_test_activation))
        return {"campaign_git_sha": RECIPE, "git_sha": "3" * 40} if recovery else None

    monkeypatch.setattr(resume_recovery, "load_execution", load)
    if not recovery:
        monkeypatch.setattr(curves, "_git_sha", lambda: RECIPE)
    before = {p: (p.read_bytes(), p.stat().st_mode) for p in runs.rglob("*") if p.is_file()}
    table = curves.collect(arms, runs, output)
    assert calls == [(repo, runs, None, True)]
    assert len(table) == 32 and set(table.git_sha) == {RECIPE}
    assert table.monotonicity_ok.all() and table.epochs_run.eq(50).all()
    assert table.test_f1.nunique() == 1
    assert {p: (p.read_bytes(), p.stat().st_mode) for p in before} == before
    assert sum(json.loads(p.read_text())["result_schema"] == 3
               for p in runs.glob("*/final_metrics.json")) == 2


@pytest.mark.parametrize("change", ["wrong-recipe", "missing-test", "checkpoint-drift"])
def test_recovery_collector_retains_full_cohort_gates(grid_fixture, monkeypatch, change):
    _repo, runs, arms, output = grid_fixture
    recipe = "b" * 40 if change == "wrong-recipe" else RECIPE
    monkeypatch.setattr(resume_recovery, "load_execution", lambda *a, **k: {"campaign_git_sha": recipe})
    if change == "missing-test":
        next(runs.glob("*/test_metrics.json")).unlink()
    elif change == "checkpoint-drift":
        checkpoint = next(runs.glob("*/checkpoints/best.ckpt"))
        checkpoint.chmod(0o644)
        checkpoint.write_bytes(checkpoint.read_bytes() + b"drift")
        checkpoint.chmod(0o444)
    with pytest.raises(HeldoutContractError):
        curves.collect(arms, runs, output)
    assert not output.exists()


def test_missing_execution_receipt_rejects_before_cohort_reads(grid_fixture, monkeypatch):
    _repo, runs, arms, output = grid_fixture
    for name in ("H100_TEST_RECEIPT", "H100_TEST_RECEIPT_SHA256"):
        monkeypatch.delenv(name, raising=False)
    (runs / ".h100/test-recovery").mkdir()
    write_once(runs / ".h100/test-recovery" / test_recovery.SOURCE, {})
    def forbidden(**kwargs):
        pytest.fail("unvalidated execution reached the frozen cohort")
    monkeypatch.setattr(curves, "validate_training_cohort", forbidden)
    with pytest.raises(RuntimeError, match="receipt is required"):
        curves.collect(arms, runs, output)
    assert not output.exists()


def test_execution_validation_errors_cannot_fall_back_to_head(grid_fixture, monkeypatch):
    _repo, runs, arms, output = grid_fixture
    def rejected(*args, **kwargs):
        raise RuntimeError("actual source differs")
    monkeypatch.setattr(resume_recovery, "load_execution", rejected)
    with pytest.raises(RuntimeError, match="actual source differs"):
        curves.collect(arms, runs, output)
    assert not output.exists()
