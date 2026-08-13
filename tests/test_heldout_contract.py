"""All-32 held-out cohort barriers and immutable result regressions."""

from __future__ import annotations

import inspect
import json
import copy
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
import torch

from scripts import score_test_cohort
from src.analysis import curves
from src.eval import final_eval, heldout_contract
from src.eval.heldout_contract import (
    COHORT_FILENAME,
    HeldoutContractError,
    build_test_result,
    create_training_cohort,
    validate_complete_test_cohort,
    validate_training_cohort,
    validate_training_cohort_cell,
    write_test_result,
)
from src.eval.result_contract import atomic_write_json, sha256_file
from src.eval.scorer import GroundTruthPoint


SCENES = tuple(f"test-{index:02d}" for index in range(16))
COHORT_SHA256 = "c" * 64


def point_metric(tp: int, fp: int, fn: int, ignored: int = 0) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": ignored,
    }


def cells_and_cohort():
    cells = [SimpleNamespace(exp_id=f"cell-{index:02d}") for index in range(32)]
    records = []
    for cell in cells:
        records.append(
            {
                "exp_id": cell.exp_id,
                "completion_marker": {
                    "relative_path": f"{cell.exp_id}/final_metrics.json",
                    "sha256": "d" * 64,
                },
                "best_checkpoint": {
                    "relative_path": f"{cell.exp_id}/checkpoints/best.ckpt",
                    "sha256": "e" * 64,
                    "epoch": 3,
                },
                "best_dev": {"threshold": 0.6, "epoch": 3, "f1": 0.5},
                "recipe": {
                    "exp_id": cell.exp_id,
                    "git_sha": "a" * 40,
                    "detector_sha256": "b" * 64,
                    "precision": "32-true",
                    "micro_batch": 16,
                    "gradient_accumulation": 1,
                    "effective_batch": 16,
                },
            }
        )
    return cells, {"cells": records}


def _test_payload(cell, record):
    aggregate = point_metric(1000, 100, 165, 325)
    zero = point_metric(0, 0, 0)
    near = point_metric(1, 1, 1)
    per_scene = {
        scene: {
            "aggregate": aggregate if index == 0 else zero,
            "slices": {
                "dark": zero,
                "near_shore": near if index == 0 else zero,
            },
        }
        for index, scene in enumerate(SCENES)
    }
    return build_test_result(
        exp_id=cell.exp_id,
        cohort_sha256=COHORT_SHA256,
        cohort_cell=record,
        inference_precision="32-true",
        metrics={
            **aggregate,
            "dark_recall": 0.0,
            "dark_support": 0,
            "near_shore_f1": near["f1"],
            "near_shore_support": 2,
        },
        per_scene=per_scene,
        test_scene_ids=SCENES,
    )


def _write_training_matrix(runs):
    cells = [SimpleNamespace(exp_id=f"cell-{index:02d}") for index in range(32)]
    precision = 3 / 4
    recall = 3 / 5
    f1 = 2 * precision * recall / (precision + recall)
    best_dev = {
        "epoch": 3,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": 3,
        "fp": 1,
        "fn": 2,
        "ignored_predictions": 1,
        "threshold": 0.4,
        "n_candidates": 8,
    }
    for cell in cells:
        run = runs / cell.exp_id
        checkpoint = run / "checkpoints/best.ckpt"
        checkpoint.parent.mkdir(parents=True)
        torch.save({"epoch": 3, "callbacks": {}}, checkpoint)
        payload = {
            "result_schema": 2,
            "exp_id": cell.exp_id,
            "git_sha": "a" * 40,
            "detector_sha256": "b" * 64,
            "precision": "32-true",
            "micro_batch": 16,
            "gradient_accumulation": 1,
            "effective_batch": 16,
            "epochs_run": 4,
            "best_dev_f1": f1,
            "best_dev": best_dev,
            "best_checkpoint": {
                "relative_path": "checkpoints/best.ckpt",
                "sha256": sha256_file(checkpoint),
                "epoch": 3,
            },
            "last_dev": dict(best_dev),
            "train_loss": 1.25,
        }
        marker = run / "final_metrics.json"
        atomic_write_json(marker, payload)
        marker.chmod(0o444)
    cohort_path = runs / ".h100" / COHORT_FILENAME
    create_training_cohort(
        cells=cells,
        runs_root=runs,
        output=cohort_path,
        git_sha="a" * 40,
        detector_sha256="b" * 64,
        candidate_floor=0.05,
    )
    return cells, cohort_path, sha256_file(cohort_path)


def test_training_cohort_seals_and_requires_best_checkpoint_read_only(tmp_path):
    runs = tmp_path / "runs"
    cells, cohort_path, _cohort_sha256 = _write_training_matrix(runs)
    checkpoint = runs / cells[0].exp_id / "checkpoints/best.ckpt"
    assert checkpoint.stat().st_mode & 0o222 == 0

    checkpoint.chmod(0o644)
    with pytest.raises(HeldoutContractError, match="must be non-writable"):
        validate_training_cohort(
            path=cohort_path,
            cells=cells,
            runs_root=runs,
            git_sha="a" * 40,
            detector_sha256="b" * 64,
            candidate_floor=0.05,
        )


def test_final_eval_checkpoint_revalidation_checks_hash_and_epoch(tmp_path):
    runs = tmp_path / "runs"
    cells, cohort_path, _cohort_sha256 = _write_training_matrix(runs)
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    cell = cells[0]
    record = heldout_contract.cohort_record(cohort, cell.exp_id)
    checkpoint = runs / cell.exp_id / "checkpoints/best.ckpt"
    assert final_eval._validated_final_checkpoint(
        record=record,
        runs_root=runs,
        exp_id=cell.exp_id,
    ) == checkpoint

    checkpoint.chmod(0o644)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    checkpoint.chmod(0o444)
    with pytest.raises(HeldoutContractError, match="SHA-256 drifted"):
        final_eval._validated_final_checkpoint(
            record=record,
            runs_root=runs,
            exp_id=cell.exp_id,
        )

    checkpoint.chmod(0o644)
    torch.save({"epoch": 4, "callbacks": {}}, checkpoint)
    checkpoint.chmod(0o444)
    rebound = json.loads(json.dumps(record))
    rebound["best_checkpoint"]["sha256"] = sha256_file(checkpoint)
    with pytest.raises(HeldoutContractError, match="epoch differs"):
        final_eval._validated_final_checkpoint(
            record=rebound,
            runs_root=runs,
            exp_id=cell.exp_id,
        )


def test_final_eval_revalidates_checkpoints_before_lock_and_model_load():
    controller = inspect.getsource(final_eval.main)
    worker = inspect.getsource(final_eval.score_final_cell)
    preflight = controller.index("selected_records: dict")
    lock = controller.index("_write_once_lock(")
    label_read = controller.index("labels = pd.read_csv(")
    immediate = worker.index("checkpoint = _validated_final_checkpoint(")
    model_load = worker.index("module = HeatmapLitModule.load_from_checkpoint(")
    assert preflight < lock < label_read
    assert immediate < model_load
    assert "pd.read_csv" not in worker


def test_per_cell_cohort_validation_hashes_only_requested_checkpoint(
    tmp_path,
    monkeypatch,
):
    runs = tmp_path / "runs"
    cells, cohort_path, cohort_sha256 = _write_training_matrix(runs)
    requested = cells[7]
    calls = []
    original = heldout_contract.load_completion_marker

    def counted(marker, **kwargs):
        calls.append(marker)
        return original(marker, **kwargs)

    monkeypatch.setattr(heldout_contract, "load_completion_marker", counted)
    cohort, actual_sha256, record, training = validate_training_cohort_cell(
        path=cohort_path,
        expected_sha256=cohort_sha256,
        cells=cells,
        runs_root=runs,
        git_sha="a" * 40,
        detector_sha256="b" * 64,
        candidate_floor=0.05,
        exp_id=requested.exp_id,
    )
    assert actual_sha256 == cohort_sha256
    assert record["exp_id"] == training["exp_id"] == requested.exp_id
    assert cohort["cell_count"] == 32
    assert calls == [runs / requested.exp_id / "final_metrics.json"]

    unrelated = runs / cells[8].exp_id / "checkpoints/best.ckpt"
    unrelated.chmod(0o644)
    unrelated.write_bytes(b"tampered-unrelated-checkpoint")
    validate_training_cohort_cell(
        path=cohort_path,
        expected_sha256=cohort_sha256,
        cells=cells,
        runs_root=runs,
        git_sha="a" * 40,
        detector_sha256="b" * 64,
        candidate_floor=0.05,
        exp_id=requested.exp_id,
    )
    with pytest.raises(HeldoutContractError, match="no longer validates"):
        validate_training_cohort(
            path=cohort_path,
            cells=cells,
            runs_root=runs,
            git_sha="a" * 40,
            detector_sha256="b" * 64,
            candidate_floor=0.05,
        )

    selected = runs / requested.exp_id / "checkpoints/best.ckpt"
    selected.chmod(0o644)
    selected.write_bytes(b"tampered-selected-checkpoint")
    with pytest.raises(HeldoutContractError, match="no longer validates"):
        validate_training_cohort_cell(
            path=cohort_path,
            expected_sha256=cohort_sha256,
            cells=cells,
            runs_root=runs,
            git_sha="a" * 40,
            detector_sha256="b" * 64,
            candidate_floor=0.05,
            exp_id=requested.exp_id,
        )


def test_complete_test_cohort_requires_all_32_and_exact_support(tmp_path):
    cells, cohort = cells_and_cohort()
    for cell, record in zip(cells, cohort["cells"], strict=True):
        path = tmp_path / cell.exp_id / "test_metrics.json"
        write_test_result(path, _test_payload(cell, record))

    results = validate_complete_test_cohort(
        cells=cells,
        runs_root=tmp_path,
        cohort=cohort,
        cohort_sha256=COHORT_SHA256,
        test_scene_ids=SCENES,
    )
    assert set(results) == {cell.exp_id for cell in cells}
    assert results[cells[0].exp_id]["threshold_source"] == {
        "kind": "best-dev-checkpoint-bound",
        "threshold": 0.6,
        "dev_epoch": 3,
        "checkpoint_relative_path": "cell-00/checkpoints/best.ckpt",
        "checkpoint_sha256": "e" * 64,
        "checkpoint_epoch": 3,
    }

    writable = tmp_path / cells[0].exp_id / "test_metrics.json"
    writable.chmod(0o644)
    with pytest.raises(HeldoutContractError, match="not immutable"):
        validate_complete_test_cohort(
            cells=cells,
            runs_root=tmp_path,
            cohort=cohort,
            cohort_sha256=COHORT_SHA256,
            test_scene_ids=SCENES,
        )
    writable.chmod(0o444)

    (tmp_path / cells[-1].exp_id / "test_metrics.json").unlink()
    with pytest.raises(HeldoutContractError, match="complete test cohort"):
        validate_complete_test_cohort(
            cells=cells,
            runs_root=tmp_path,
            cohort=cohort,
            cohort_sha256=COHORT_SHA256,
            test_scene_ids=SCENES,
        )


def test_result_refuses_support_or_scene_drift():
    cells, cohort = cells_and_cohort()
    payload = _test_payload(cells[0], cohort["cells"][0])
    metrics = dict(payload["metrics"])
    metrics["near_shore_support"] = 3
    with pytest.raises(HeldoutContractError, match="near-shore support"):
        build_test_result(
            exp_id=cells[0].exp_id,
            cohort_sha256=COHORT_SHA256,
            cohort_cell=cohort["cells"][0],
            inference_precision="32-true",
            metrics=metrics,
            per_scene=payload["per_scene"],
            test_scene_ids=SCENES,
        )
    with pytest.raises(HeldoutContractError, match="16 sorted scene IDs"):
        build_test_result(
            exp_id=cells[0].exp_id,
            cohort_sha256=COHORT_SHA256,
            cohort_cell=cohort["cells"][0],
            inference_precision="32-true",
            metrics=payload["metrics"],
            per_scene=payload["per_scene"],
            test_scene_ids=SCENES[:-1],
        )


def _write_configs(repo, *, include_splits: bool = False):
    (repo / "configs").mkdir(parents=True)
    (repo / "configs/data.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "splits": "data/splits.json",
                    "raw_xview3": "data/raw/xview3",
                    "stats": "data/stats.json",
                }
            }
        )
    )
    (repo / "configs/detector.yaml").write_text(
        yaml.safe_dump(
            {
                "decode": {"candidate_floor": 0.05},
                "schedule": {"precision": "32-true"},
            }
        )
    )
    (repo / "configs/arms.yaml").write_text(
        yaml.safe_dump(
            {
                "arms": {
                    f"arm-{index}": {
                        "short": f"arm{index}",
                        "track": "vit" if index < 4 else "cnn",
                        "role": ("floor", "optical", "sar", "imagenet")[index % 4],
                    }
                    for index in range(8)
                }
            }
        )
    )
    if include_splits:
        (repo / "data").mkdir()
        (repo / "data/splits.json").write_text(
            json.dumps(
                {
                    "splits": {
                        "test": list(SCENES),
                        "eval_final": [f"final-{i:02d}" for i in range(50)],
                    }
                }
            )
        )


def test_test_scorer_does_not_open_labels_before_training_barrier(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    runs = tmp_path / "runs"
    _write_configs(repo)
    cohort_path = runs / ".h100" / COHORT_FILENAME
    cohort_path.parent.mkdir(parents=True)
    cohort_path.write_text("{}")
    cells, _cohort = cells_and_cohort()
    monkeypatch.setattr(score_test_cohort, "load_cells", lambda _repo: cells)
    monkeypatch.setattr(score_test_cohort, "_git_sha", lambda _repo: "a" * 40)
    monkeypatch.setattr(
        score_test_cohort,
        "validate_training_cohort",
        lambda **kwargs: (_ for _ in ()).throw(
            HeldoutContractError("unrelated cell drift before barrier")
        ),
    )
    monkeypatch.setattr(
        score_test_cohort,
        "validate_training_cohort_cell",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("narrow validation ran before the full barrier")
        ),
    )
    label_reads = []
    monkeypatch.setattr(pd, "read_csv", lambda path: label_reads.append(path))
    with pytest.raises(HeldoutContractError, match="unrelated cell drift"):
        score_test_cohort.main(
            [
                "--repo",
                str(repo),
                "--runs-root",
                str(runs),
                "--cohort-sha256",
                COHORT_SHA256,
                "--device",
                "cpu",
                "--only",
                cells[0].exp_id,
            ]
        )
    assert label_reads == []


def test_final_eval_does_not_lock_or_open_validation_before_all_test_results(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    runs = tmp_path / "runs"
    _write_configs(repo, include_splits=True)
    cells, cohort = cells_and_cohort()
    monkeypatch.setattr(final_eval, "load_cells", lambda _repo: cells)
    monkeypatch.setattr(final_eval, "_git_sha", lambda _repo: "a" * 40)
    monkeypatch.setattr(
        final_eval,
        "validate_training_cohort",
        lambda **kwargs: (cohort, COHORT_SHA256),
    )
    monkeypatch.setattr(
        final_eval,
        "validate_complete_test_cohort",
        lambda **kwargs: (_ for _ in ()).throw(HeldoutContractError("test barrier")),
    )
    label_reads = []
    monkeypatch.setattr(pd, "read_csv", lambda path: label_reads.append(path))
    with pytest.raises(HeldoutContractError, match="test barrier"):
        final_eval.main(
            [
                "--i-am-sure",
                "--repo",
                str(repo),
                "--runs-root",
                str(runs),
                "--device",
                "cpu",
            ]
        )
    assert label_reads == []
    assert not (runs / "final_eval.lock").exists()


def _grid_fixture(runs):
    arms = {
        f"arm-{index}": {
            "short": f"arm{index}",
            "track": "vit" if index < 4 else "cnn",
            "role": ("floor", "optical", "sar", "imagenet")[index % 4],
        }
        for index in range(8)
    }
    cells = []
    rows = []
    test_results = {}
    cohort_records = []
    for index, (init_name, arm) in enumerate(arms.items()):
        for fraction_index, fraction in enumerate((0.1, 0.25, 0.5, 1.0)):
            cell = SimpleNamespace(
                init=init_name,
                track=arm["track"],
                fraction=fraction,
                seed=0,
                exp_id=f"{arm['short']}-f{int(fraction * 100)}-s0",
            )
            test_f1 = 0.4 + fraction_index * 0.01
            cells.append(cell)
            test_results[cell.exp_id] = {"metrics": {"f1": test_f1}}
            rows.append(
                {
                    "exp_id": cell.exp_id,
                    "init": cell.init,
                    "track": cell.track,
                    "role": arm["role"],
                    "label_frac": fraction,
                    "seed": 0,
                    "precision": "32-true",
                    "detector_sha256": "b" * 64,
                    "git_sha": "a" * 40,
                    "dev_f1": 0.5,
                    "dev_threshold": 0.4,
                    "test_f1": test_f1,
                    "epochs_run": 4,
                    "train_scene_count": 10 + fraction_index,
                    "train_vessel_count": 100 + fraction_index,
                    "train_dark_vessel_count": 20 + fraction_index,
                    "train_near_shore_vessel_count": 5 + fraction_index,
                    "monotonicity_ok": True,
                }
            )
            run = runs / cell.exp_id
            run.mkdir(parents=True)
            best_dev = {"f1": 0.5, "threshold": 0.4}
            marker = run / "final_metrics.json"
            marker.write_text(
                json.dumps(
                    {
                        "exp_id": cell.exp_id,
                        "epochs_run": 4,
                        "best_dev": best_dev,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            marker.chmod(0o444)
            cohort_records.append(
                {
                    "exp_id": cell.exp_id,
                    "completion_marker": {
                        "relative_path": f"{cell.exp_id}/final_metrics.json",
                        "sha256": sha256_file(marker),
                    },
                    "best_dev": best_dev,
                }
            )
    return arms, cells, rows, test_results, {"cells": cohort_records}


def _grid_fraction_counts(rows):
    return {
        float(row["label_frac"]): {
            name: int(row[name]) for name in final_eval.GRID_COUNT_COLUMNS
        }
        for row in rows[:4]
    }


def test_final_eval_grid_gate_recomputes_monotonicity(tmp_path):
    runs = tmp_path / "runs"
    arms, cells, rows, test_results, cohort = _grid_fixture(runs)
    grid = tmp_path / "grid.csv"
    pd.DataFrame(rows, columns=final_eval.GRID_COLUMNS).to_csv(grid, index=False)
    assert final_eval.validate_grid_gate(
        grid,
        cells=cells,
        test_results=test_results,
        arms=arms,
        cohort=cohort,
        runs_root=runs,
        fraction_counts=_grid_fraction_counts(rows),
        git_sha="a" * 40,
        detector_sha256="b" * 64,
        precision="32-true",
    ) == sha256_file(grid)

    affected = next(
        row for row in rows if row["init"] == "arm-0" and row["label_frac"] == 0.25
    )
    affected["test_f1"] = 0.35
    test_results[affected["exp_id"]]["metrics"]["f1"] = 0.35
    pd.DataFrame(rows, columns=final_eval.GRID_COLUMNS).to_csv(grid, index=False)
    with pytest.raises(HeldoutContractError, match="monotonicity STOP"):
        final_eval.validate_grid_gate(
            grid,
            cells=cells,
            test_results=test_results,
            arms=arms,
            fraction_counts=_grid_fraction_counts(rows),
            cohort=cohort,
            runs_root=runs,
            git_sha="a" * 40,
            detector_sha256="b" * 64,
            precision="32-true",
        )


@pytest.mark.parametrize(
    ("field", "tampered"),
    (("dev_f1", 0.51), ("dev_threshold", 0.41), ("epochs_run", 5)),
)
def test_final_eval_grid_gate_rejects_unbound_training_fields(
    tmp_path, field, tampered
):
    runs = tmp_path / "runs"
    arms, cells, rows, test_results, cohort = _grid_fixture(runs)
    rows[0][field] = tampered
    grid = tmp_path / "grid.csv"
    pd.DataFrame(rows, columns=final_eval.GRID_COLUMNS).to_csv(grid, index=False)

    with pytest.raises(HeldoutContractError, match="disagrees with the frozen cohort"):
        final_eval.validate_grid_gate(
            grid,
            cells=cells,
            test_results=test_results,
            cohort=cohort,
            runs_root=runs,
            arms=arms,
            fraction_counts=_grid_fraction_counts(rows),
            git_sha="a" * 40,
            detector_sha256="b" * 64,
            precision="32-true",
        )


def test_final_eval_grid_gate_rehashes_cohort_bound_completion_marker(tmp_path):
    runs = tmp_path / "runs"
    arms, cells, rows, test_results, cohort = _grid_fixture(runs)
    marker = runs / cells[0].exp_id / "final_metrics.json"
    marker.chmod(0o644)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["epochs_run"] = 5
    marker.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker.chmod(0o444)
    grid = tmp_path / "grid.csv"
    pd.DataFrame(rows, columns=final_eval.GRID_COLUMNS).to_csv(grid, index=False)

    with pytest.raises(HeldoutContractError, match="completion marker drifted"):
        final_eval.validate_grid_gate(
            grid,
            cells=cells,
            test_results=test_results,
            cohort=cohort,
            runs_root=runs,
            arms=arms,
            fraction_counts=_grid_fraction_counts(rows),
            git_sha="a" * 40,
            detector_sha256="b" * 64,
            precision="32-true",
        )


def test_final_eval_monotonicity_stop_precedes_final_resource_access(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    runs = tmp_path / "runs"
    _write_configs(repo, include_splits=True)
    arms, cells, rows, test_results, cohort = _grid_fixture(runs)
    (repo / "configs/arms.yaml").write_text(yaml.safe_dump({"arms": arms}))
    detector_sha256 = sha256_file(repo / "configs/detector.yaml")
    for row in rows:
        row["detector_sha256"] = detector_sha256
    affected = next(
        row for row in rows if row["init"] == "arm-0" and row["label_frac"] == 0.25
    )
    affected["test_f1"] = 0.35
    test_results[affected["exp_id"]]["metrics"]["f1"] = 0.35
    grid = runs / "summary/grid.csv"
    grid.parent.mkdir(parents=True)
    pd.DataFrame(rows, columns=final_eval.GRID_COLUMNS).to_csv(grid, index=False)

    monkeypatch.setattr(final_eval, "load_cells", lambda _repo: cells)
    monkeypatch.setattr(final_eval, "_git_sha", lambda _repo: "a" * 40)
    monkeypatch.setattr(
        final_eval,
        "validate_training_cohort",
        lambda **kwargs: (cohort, COHORT_SHA256),
    )
    monkeypatch.setattr(
        final_eval,
        "validate_complete_test_cohort",
        lambda **kwargs: test_results,
    )
    monkeypatch.setattr(
        final_eval,
        "training_fraction_counts",
        lambda **kwargs: _grid_fraction_counts(rows),
    )
    label_reads = []
    monkeypatch.setattr(pd, "read_csv", lambda path: label_reads.append(path))
    with pytest.raises(HeldoutContractError, match="monotonicity STOP"):
        final_eval.main(
            [
                "--i-am-sure",
                "--repo",
                str(repo),
                "--runs-root",
                str(runs),
                "--device",
                "cpu",
            ]
        )
    assert label_reads == []
    assert not (runs / "final_eval.lock").exists()


def test_curves_and_legacy_entrypoint_have_no_partial_or_mutating_fallback():
    frame = pd.DataFrame({"test_f1": [0.5] * 31, "dev_f1": [0.6] * 31})
    with pytest.raises(ValueError, match="exactly 32"):
        curves.metric_column(frame)
    source = open("scripts/score_test_split.py", encoding="utf-8").read()
    assert 'payload.get("last_dev")' not in source
    assert "atomic_write_json" not in source
    assert "score_test_cohort" in source


def _final_cell_payload_fixture():
    cell = SimpleNamespace(
        exp_id="beS1-f50-s0", init="bigearthnet_s1", fraction=0.5, seed=0
    )
    checkpoint = {
        "relative_path": "beS1-f50-s0/checkpoints/best.ckpt",
        "sha256": "1" * 64,
        "epoch": 7,
    }
    record = {
        "best_checkpoint": checkpoint,
        "best_dev": {"threshold": 0.6, "epoch": 7, "f1": 0.8},
    }
    aggregate = point_metric(1, 0, 0)
    zero = point_metric(0, 0, 0)
    scene = "eval-scene"
    payload = {
        "final_result_schema": 1,
        "created_utc": "2026-08-13T06:00:00+00:00",
        "exp_id": cell.exp_id,
        "init": cell.init,
        "label_frac": cell.fraction,
        "seed": 0,
        "study_design": final_eval.FINAL_STUDY_DESIGN,
        "interpretation": final_eval.AMENDED_FINAL_INTERPRETATION,
        "threshold": 0.6,
        "threshold_source": "best-dev-checkpoint-bound",
        "dev_epoch": 7,
        "checkpoint": dict(checkpoint),
        "campaign_git_sha": "a" * 40,
        "evaluator_git_sha": "b" * 40,
        "detector_sha256": "c" * 64,
        "cohort_sha256": "d" * 64,
        "test_result_sha256": "e" * 64,
        "owner_authorization_sha256": "f" * 64,
        "final_access": {
            "lock_sha256": "2" * 64,
            "consumption_sha256": "3" * 64,
            "data_view_sha256": "4" * 64,
        },
        "test_monotonicity": {
            "ok": False,
            "tolerance": 0.02,
            "violations": [
                {
                    "init": "beS1",
                    "from_fraction": 0.5,
                    "to_fraction": 1.0,
                    "from_test_f1": 0.8521,
                    "to_test_f1": 0.8221,
                    "drop": 0.03,
                }
            ],
        },
        "inference_precision": "32-true",
        "strict_fp32": dict(final_eval._STRICT_FP32_BACKEND),
        "eval_scene_ids": [scene],
        "metrics": {
            **aggregate,
            "dark_recall": 1.0,
            "dark_support": 1,
            "near_shore_f1": 0.0,
            "near_shore_support": 0,
        },
        "per_scene": {
            scene: {
                "aggregate": aggregate,
                "slices": {"dark": aggregate, "near_shore": zero},
                "matches": [
                    {
                        "prediction_index": 0,
                        "ground_truth_index": 0,
                        "distance_m": 0.0,
                        "outcome": "tp",
                    }
                ],
            }
        },
        "thresholded_predictions": {
            scene: [
                {
                    "x_m": 10.0,
                    "y_m": 20.0,
                    "score": 0.7,
                    "distance_from_shore_km": 5.0,
                }
            ]
        },
    }
    kwargs = {
        "cell": cell,
        "record": record,
        "eval_scene_ids": (scene,),
        "campaign_git_sha": "a" * 40,
        "evaluator_git_sha": "b" * 40,
        "detector_sha256": "c" * 64,
        "cohort_sha256": "d" * 64,
        "test_result_sha256": "e" * 64,
        "grid_audit": {
            "monotonicity_ok": False,
            "monotonicity_tolerance": 0.02,
            "violations": payload["test_monotonicity"]["violations"],
        },
        "authorization_sha256": "f" * 64,
        "final_access": dict(payload["final_access"]),
        "ground_truth_by_scene": {
            scene: [
                GroundTruthPoint(
                    x_m=10.0,
                    y_m=20.0,
                    confidence="HIGH",
                    source="Manual",
                    distance_from_shore_km=5.0,
                )
            ]
        },
    }
    return payload, kwargs


def test_final_cell_payload_validator_accepts_exact_bound_scene_evidence():
    payload, kwargs = _final_cell_payload_fixture()
    assert final_eval._validate_final_cell_payload(payload, **kwargs) == payload


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["checkpoint"].update({"sha256": "9" * 64}),
        lambda value: value["strict_fp32"].update(
            {"cuda_matmul_fp32_precision": "tf32"}
        ),
        lambda value: value["final_access"].update({"lock_sha256": "9" * 64}),
        lambda value: value["per_scene"]["eval-scene"]["aggregate"].update(
            {"tp": 2}
        ),
        lambda value: value["per_scene"]["eval-scene"]["matches"].clear(),
        lambda value: value["thresholded_predictions"]["eval-scene"][0].update(
            {"score": 0.59}
        ),
    ),
)
def test_final_cell_payload_validator_rejects_tampered_evidence(mutation):
    payload, kwargs = _final_cell_payload_fixture()
    kwargs = copy.deepcopy(kwargs)
    mutation(payload)
    with pytest.raises(HeldoutContractError):
        final_eval._validate_final_cell_payload(payload, **kwargs)


def test_final_cell_payload_rejects_zero_fn_for_consumed_positive_gt():
    payload, kwargs = _final_cell_payload_fixture()
    zero = point_metric(0, 0, 0)
    payload["thresholded_predictions"]["eval-scene"] = []
    payload["per_scene"]["eval-scene"] = {
        "aggregate": dict(zero),
        "slices": {"dark": dict(zero), "near_shore": dict(zero)},
        "matches": [],
    }
    payload["metrics"] = {
        **zero,
        "dark_recall": 0.0,
        "dark_support": 0,
        "near_shore_f1": 0.0,
        "near_shore_support": 0,
    }
    with pytest.raises(HeldoutContractError, match="frozen-scorer pass"):
        final_eval._validate_final_cell_payload(payload, **kwargs)


def test_final_cell_payload_rejects_duplicate_tp_ground_truth_index():
    payload, kwargs = _final_cell_payload_fixture()
    payload["thresholded_predictions"]["eval-scene"].append(
        {
            "x_m": 10.0,
            "y_m": 20.0,
            "score": 0.65,
            "distance_from_shore_km": 5.0,
        }
    )
    payload["per_scene"]["eval-scene"]["matches"].append(
        {
            "prediction_index": 1,
            "ground_truth_index": 0,
            "distance_m": 0.0,
            "outcome": "tp",
        }
    )
    duplicate = point_metric(2, 0, 0)
    payload["per_scene"]["eval-scene"]["aggregate"] = dict(duplicate)
    payload["metrics"].update(duplicate)
    with pytest.raises(HeldoutContractError, match="match evidence is invalid"):
        final_eval._validate_final_cell_payload(payload, **kwargs)


def test_final_cell_payload_accepts_multiple_ignores_for_one_low_gt():
    payload, kwargs = _final_cell_payload_fixture()
    scene = "eval-scene"
    kwargs["ground_truth_by_scene"] = {
        scene: [
            GroundTruthPoint(
                x_m=10.0,
                y_m=20.0,
                confidence="LOW",
                source="Manual",
                distance_from_shore_km=5.0,
            )
        ]
    }
    payload["thresholded_predictions"][scene] = [
        {
            "x_m": 10.0,
            "y_m": 20.0,
            "score": 0.7,
            "distance_from_shore_km": 5.0,
        },
        {
            "x_m": 11.0,
            "y_m": 20.0,
            "score": 0.65,
            "distance_from_shore_km": 5.0,
        },
    ]
    ignored = point_metric(0, 0, 0, ignored=2)
    zero = point_metric(0, 0, 0)
    payload["per_scene"][scene] = {
        "aggregate": ignored,
        "slices": {"dark": zero, "near_shore": zero},
        "matches": [
            {
                "prediction_index": 0,
                "ground_truth_index": 0,
                "distance_m": 0.0,
                "outcome": "ignored",
            },
            {
                "prediction_index": 1,
                "ground_truth_index": 0,
                "distance_m": 1.0,
                "outcome": "ignored",
            },
        ],
    }
    payload["metrics"] = {
        **ignored,
        "dark_recall": 0.0,
        "dark_support": 0,
        "near_shore_f1": 0.0,
        "near_shore_support": 0,
    }
    assert final_eval._validate_final_cell_payload(payload, **kwargs) == payload
