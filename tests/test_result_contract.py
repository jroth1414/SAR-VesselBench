"""Schema-2 checkpoint/threshold binding and callback-state regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _symlinks_available() -> bool:
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "target"
        target.write_text("probe", encoding="utf-8")
        try:
            (Path(scratch) / "link").symlink_to(target)
        except OSError:
            return False
    return True


requires_symlinks = pytest.mark.skipif(
    not _symlinks_available(),
    reason="creating symlinks requires privilege on this platform",
)

from src.eval.result_contract import (
    RESULT_SCHEMA,
    ResultContractError,
    atomic_write_json,
    create_best_checkpoint_binding,
    load_completion_marker,
    sha256_file,
    validate_completion_payload,
    validate_dev_result,
)
from src.train.finetune import DevSceneEval


CANDIDATE_FLOOR = 0.05
GIT_SHA = "a" * 40
DETECTOR_SHA256 = "b" * 64


def dev_result(
    *,
    epoch: int = 3,
    tp: int = 3,
    fp: int = 1,
    fn: int = 2,
    ignored_predictions: int = 1,
    threshold: float = 0.4,
    n_candidates: int = 8,
) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "epoch": epoch,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_predictions": ignored_predictions,
        "threshold": threshold,
        "n_candidates": n_candidates,
    }


def recipe(exp_id: str = "vitin1k-f10-s0") -> dict:
    return {
        "exp_id": exp_id,
        "git_sha": GIT_SHA,
        "detector_sha256": DETECTOR_SHA256,
        "precision": "32-true",
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
    }


def valid_marker(tmp_path):
    run_dir = tmp_path / "vitin1k-f10-s0"
    checkpoint = run_dir / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 3, "callbacks": {}}, checkpoint)
    best_dev = dev_result()
    payload = {
        "result_schema": RESULT_SCHEMA,
        **recipe(run_dir.name),
        "epochs_run": 4,
        "best_dev_f1": best_dev["f1"],
        "best_dev": best_dev,
        "best_checkpoint": create_best_checkpoint_binding(
            run_dir=run_dir,
            checkpoint_path=checkpoint,
            best_dev=best_dev,
            candidate_floor=CANDIDATE_FLOOR,
        ),
        "last_dev": dict(best_dev),
        "train_loss": 1.25,
    }
    return run_dir, checkpoint, payload


def test_schema2_marker_round_trip_binds_real_checkpoint_bytes(tmp_path):
    run_dir, checkpoint, payload = valid_marker(tmp_path)
    marker = run_dir / "final_metrics.json"
    atomic_write_json(marker, payload)

    loaded, selected = load_completion_marker(
        marker,
        candidate_floor=CANDIDATE_FLOOR,
        expected_recipe=recipe(),
    )

    assert loaded == payload
    assert selected == checkpoint
    assert loaded["best_checkpoint"] == {
        "relative_path": "checkpoints/best.ckpt",
        "sha256": sha256_file(checkpoint),
        "epoch": 3,
    }
    assert list(run_dir.glob(".final_metrics.json.*.tmp")) == []


@pytest.mark.parametrize("schema", [None, 1, 3, True])
def test_completion_requires_exact_schema2(tmp_path, schema):
    run_dir, _checkpoint, payload = valid_marker(tmp_path)
    payload["result_schema"] = schema
    with pytest.raises(ResultContractError, match="result_schema == 2"):
        validate_completion_payload(
            payload,
            run_dir=run_dir,
            candidate_floor=CANDIDATE_FLOOR,
            expected_recipe=recipe(),
        )


def test_completion_allows_h100_runtime_contract_extension(tmp_path):
    run_dir, checkpoint, payload = valid_marker(tmp_path)
    payload["h100_runtime_contract"] = {"runtime_schema": 1}

    selected = validate_completion_payload(
        payload,
        run_dir=run_dir,
        candidate_floor=CANDIDATE_FLOOR,
    )

    assert selected == checkpoint


@pytest.mark.parametrize(
    "extra_field",
    ["test", "heldout", "legacy", "arbitrary_extra"],
)
def test_completion_rejects_noncontract_top_level_fields(tmp_path, extra_field):
    run_dir, _checkpoint, payload = valid_marker(tmp_path)
    payload[extra_field] = {}
    with pytest.raises(
        ResultContractError, match=rf"unexpected:.*{extra_field}"
    ):
        validate_completion_payload(
            payload,
            run_dir=run_dir,
            candidate_floor=CANDIDATE_FLOOR,
        )


def test_completion_requires_last_dev_and_exact_best_scalar(tmp_path):
    run_dir, _checkpoint, payload = valid_marker(tmp_path)
    payload.pop("last_dev")
    with pytest.raises(ResultContractError, match="missing: last_dev"):
        validate_completion_payload(
            payload, run_dir=run_dir, candidate_floor=CANDIDATE_FLOOR
        )

    _run_dir, _checkpoint, payload = valid_marker(tmp_path / "other")
    payload["best_dev_f1"] += 0.01
    with pytest.raises(ResultContractError, match="does not equal"):
        validate_completion_payload(
            payload,
            run_dir=tmp_path / "other" / "vitin1k-f10-s0",
            candidate_floor=CANDIDATE_FLOOR,
        )


@pytest.mark.parametrize("field", ["best_dev", "last_dev", "best_checkpoint"])
def test_completion_rejects_extra_nested_contract_fields(tmp_path, field):
    run_dir, _checkpoint, payload = valid_marker(tmp_path)
    payload[field]["unexpected_metric"] = 0
    with pytest.raises(
        ResultContractError, match=rf"{field} keys.*unexpected"
    ):
        validate_completion_payload(
            payload,
            run_dir=run_dir,
            candidate_floor=CANDIDATE_FLOOR,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("threshold", 0.049, "candidate_floor"),
        ("threshold", 1.01, "candidate_floor"),
        ("f1", float("nan"), "finite"),
        ("tp", -1, "non-negative integer"),
        ("fp", 1.0, "non-negative integer"),
        ("n_candidates", 1, "decoded candidate count"),
    ],
)
def test_dev_result_rejects_bad_threshold_nonfinite_and_counts(field, value, message):
    result = dev_result()
    result[field] = value
    with pytest.raises(ResultContractError, match=message):
        validate_dev_result(result, candidate_floor=CANDIDATE_FLOOR)


def test_dev_result_rejects_metrics_inconsistent_with_counts():
    result = dev_result()
    result["precision"] = 0.123
    with pytest.raises(ResultContractError, match="inconsistent"):
        validate_dev_result(result, candidate_floor=CANDIDATE_FLOOR)


def test_marker_rejects_hash_epoch_path_and_recipe_drift(tmp_path):
    run_dir, checkpoint, payload = valid_marker(tmp_path)

    bad = copy.deepcopy(payload)
    bad["best_checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ResultContractError, match="SHA-256"):
        validate_completion_payload(
            bad, run_dir=run_dir, candidate_floor=CANDIDATE_FLOOR
        )

    bad = copy.deepcopy(payload)
    bad["best_checkpoint"]["epoch"] = 2
    with pytest.raises(ResultContractError, match="epoch"):
        validate_completion_payload(
            bad, run_dir=run_dir, candidate_floor=CANDIDATE_FLOOR
        )

    bad = copy.deepcopy(payload)
    bad["best_checkpoint"]["relative_path"] = "../outside.ckpt"
    with pytest.raises(ResultContractError, match="unsafe|inside"):
        validate_completion_payload(
            bad, run_dir=run_dir, candidate_floor=CANDIDATE_FLOOR
        )

    bad = copy.deepcopy(payload)
    expected = recipe()
    expected["precision"] = "16-mixed"
    with pytest.raises(ResultContractError, match="recipe-matched"):
        validate_completion_payload(
            bad,
            run_dir=run_dir,
            candidate_floor=CANDIDATE_FLOOR,
            expected_recipe=expected,
        )

    torch.save({"epoch": 4}, checkpoint)
    payload["best_checkpoint"]["sha256"] = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    with pytest.raises(ResultContractError, match="Lightning checkpoint epoch"):
        validate_completion_payload(
            payload, run_dir=run_dir, candidate_floor=CANDIDATE_FLOOR
        )


@requires_symlinks
def test_marker_rejects_symlinked_checkpoint(tmp_path):
    run_dir, checkpoint, payload = valid_marker(tmp_path)
    target = checkpoint.with_name("target.ckpt")
    checkpoint.replace(target)
    checkpoint.symlink_to(target.name)
    payload["best_checkpoint"]["sha256"] = sha256_file(target)
    with pytest.raises(ResultContractError, match="symlink"):
        validate_completion_payload(
            payload, run_dir=run_dir, candidate_floor=CANDIDATE_FLOOR
        )


@requires_symlinks
def test_symlinked_runs_root_accepts_canonical_checkpoint_and_rejects_escapes(
    tmp_path,
):
    persistent_runs = tmp_path / "persistent-runs"
    persistent_runs.mkdir()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "runs").symlink_to(persistent_runs, target_is_directory=True)
    run_dir = checkout / "runs" / "vitin1k-f10-s0"
    checkpoint = persistent_runs / run_dir.name / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 3, "callbacks": {}}, checkpoint)
    best_dev = dev_result()

    binding = create_best_checkpoint_binding(
        run_dir=run_dir,
        checkpoint_path=checkpoint,
        best_dev=best_dev,
        candidate_floor=CANDIDATE_FLOOR,
    )

    assert binding == {
        "relative_path": "checkpoints/best.ckpt",
        "sha256": sha256_file(checkpoint),
        "epoch": 3,
    }

    outside = tmp_path / "outside"
    outside.mkdir()
    outside_checkpoint = outside / "best.ckpt"
    torch.save({"epoch": 3, "callbacks": {}}, outside_checkpoint)
    with pytest.raises(ResultContractError, match="inside its run directory"):
        create_best_checkpoint_binding(
            run_dir=run_dir,
            checkpoint_path=outside_checkpoint,
            best_dev=best_dev,
            candidate_floor=CANDIDATE_FLOOR,
        )

    (run_dir / "linked-checkpoints").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(ResultContractError, match="contains a symlink"):
        create_best_checkpoint_binding(
            run_dir=run_dir,
            checkpoint_path=run_dir / "linked-checkpoints" / "best.ckpt",
            best_dev=best_dev,
            candidate_floor=CANDIDATE_FLOOR,
        )


class _Module:
    device = "cpu"

    def __init__(self):
        self.logged = []
        self.train_calls = 0

    def log(self, name, value, **kwargs):
        self.logged.append((name, value, kwargs))

    def train(self):
        self.train_calls += 1


def _callback():
    callback = DevSceneEval(
        data_cfg={
            "paths": {
                "raw_xview3": "unused",
                "stats": "unused",
                "splits": "unused",
            }
        },
        det_cfg={
            "decode": {"candidate_floor": CANDIDATE_FLOOR, "d_nms_m": 120.0},
            "eval": {"tile_px": 512, "tile_stride_px": 384, "infer_batch": 8},
            "schedule": {"precision": "32-true"},
        },
        every_n_epochs=1,
        n_scenes=1,
        final_epoch=3,
    )
    callback.scene_ids = ["scene"]
    return callback


def test_dev_callback_strict_ties_and_state_round_trip(monkeypatch):
    results = iter(
        [
            dev_result(epoch=99, tp=1, fp=1, fn=1, threshold=0.3),
            dev_result(epoch=99, tp=1, fp=1, fn=1, threshold=0.8),
            dev_result(epoch=99, tp=2, fp=0, fn=0, threshold=0.9),
        ]
    )
    monkeypatch.setattr("src.eval.infer_scene.dev_f1", lambda *args, **kwargs: next(results))
    callback = _callback()
    module = _Module()

    callback.on_train_epoch_end(SimpleNamespace(current_epoch=0), module)
    first_best = dict(callback.best_result)
    callback.on_train_epoch_end(SimpleNamespace(current_epoch=1), module)
    assert callback.best_result == first_best
    assert callback.last_result["threshold"] == 0.8
    assert callback.last_result["epoch"] == 1

    restored = _callback()
    restored.load_state_dict(json.loads(json.dumps(callback.state_dict())))
    assert restored.state_dict() == callback.state_dict()

    restored.on_train_epoch_end(SimpleNamespace(current_epoch=2), module)
    assert restored.best == 1.0
    assert restored.best_result["epoch"] == 2
    assert restored.last_result == restored.best_result


def test_dev_callback_refuses_legacy_incomplete_resume_state():
    with pytest.raises(ResultContractError, match="best_dev is missing"):
        _callback().load_state_dict({"best": 0.5})
