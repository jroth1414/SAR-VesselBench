"""Training-path guards for isolated BigEarthNet-S2 diagnostics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

from src.eval.result_contract import ResultContractError  # noqa: E402
from src.train.finetune import (  # noqa: E402
    DIAGNOSTIC_PROBE_EPOCHS,
    DIAGNOSTIC_SCHEDULE_EPOCHS,
    DevSceneEval,
    _diagnostic_layout,
)


class _Parser:
    def error(self, message):
        raise ValueError(message)


def _args(root, **updates):
    values = {
        "diagnostic_variant": "current_replay",
        "diagnostic_stage": "probe",
        "diagnostic_root": root,
        "init": "bigearthnet_s2",
        "label_frac": 1.0,
        "seed": 0,
        "epochs": None,
        "smoke": False,
        "exp_suffix": None,
        "batch_size": None,
        "micro_batch": None,
        "samples_per_epoch": None,
        "dev_every": None,
        "n_dev_scenes": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _detector():
    return {
        "schedule": {
            "epochs": 50,
            "batch_size": 16,
            "precision": "32-true",
        },
        "optimizer": {"layer_decay": 0.65},
        "eval": {"dev_every_epochs": 5, "n_dev_scenes": 8},
    }


def test_probe_uses_five_epochs_with_original_fifty_epoch_horizon(tmp_path):
    root = tmp_path / "diagnostics"
    root.mkdir()
    layout = _diagnostic_layout(_args(root), _detector(), _Parser())

    assert layout["run_id"] == "bes2-current_replay-probe-f100-s0"
    assert layout["run_dir"] == root / layout["run_id"]
    assert layout["trainer_epochs"] == DIAGNOSTIC_PROBE_EPOCHS == 5
    assert layout["schedule_epochs"] == DIAGNOSTIC_SCHEDULE_EPOCHS == 50


def test_full_uses_approved_recipe_and_distinct_run_directory(tmp_path):
    root = tmp_path / "diagnostics"
    root.mkdir()
    layout = _diagnostic_layout(
        _args(
            root,
            diagnostic_variant="first_conv_reset",
            diagnostic_stage="full",
        ),
        _detector(),
        _Parser(),
    )

    assert layout["run_id"] == "bes2-first_conv_reset-full-f100-s0"
    assert layout["trainer_epochs"] == layout["schedule_epochs"] == 50


@pytest.mark.parametrize(
    "update,match",
    [
        ({"init": "cnn_random"}, "require --init bigearthnet_s2"),
        ({"label_frac": 0.5}, "require --init bigearthnet_s2"),
        ({"seed": 1}, "require --init bigearthnet_s2"),
        ({"batch_size": 8}, "refuse recipe overrides"),
        ({"samples_per_epoch": 10}, "refuse recipe overrides"),
    ],
)
def test_diagnostic_layout_rejects_recipe_drift(tmp_path, update, match):
    root = tmp_path / "diagnostics"
    root.mkdir()
    with pytest.raises(ValueError, match=match):
        _diagnostic_layout(_args(root, **update), _detector(), _Parser())


def test_diagnostic_layout_rejects_core_runs_namespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "runs" / "diagnostics"
    root.mkdir(parents=True)
    with pytest.raises(ValueError, match="outside the core runs namespace"):
        _diagnostic_layout(_args(root), _detector(), _Parser())


def _callback(*, diagnostic):
    callback = DevSceneEval(
        data_cfg={
            "paths": {
                "raw_xview3": "unused",
                "stats": "unused",
                "splits": "unused",
            }
        },
        det_cfg={
            "decode": {"candidate_floor": 0.05, "d_nms_m": 120.0},
            "eval": {"tile_px": 512, "tile_stride_px": 384, "infer_batch": 8},
            "schedule": {"precision": "32-true"},
        },
        every_n_epochs=5,
        n_scenes=8,
        final_epoch=5,
        diagnostic=diagnostic,
    )
    callback.scene_ids = ["scene"]
    return callback


class _Module:
    device = "cpu"

    def log(self, *args, **kwargs):
        pass

    def train(self):
        pass


def _dev_result():
    value = 2.0 / 3.0
    return {
        "epoch": 99,
        "f1": value,
        "precision": value,
        "recall": value,
        "tp": 2,
        "fp": 1,
        "fn": 1,
        "ignored_predictions": 0,
        "threshold": 0.2,
        "n_candidates": 3,
    }


def test_diagnostic_callback_persists_complete_history_only_in_diagnostics(
    monkeypatch,
):
    monkeypatch.setattr(
        "src.eval.infer_scene.dev_f1",
        lambda *args, **kwargs: _dev_result(),
    )
    callback = _callback(diagnostic=True)
    callback._runtime_seconds = 12.5
    callback.on_train_epoch_end(
        SimpleNamespace(current_epoch=4),
        _Module(),
    )
    state = callback.state_dict()

    assert [item["epoch"] for item in state["history"]] == [4]
    assert state["elapsed_seconds"] == 12.5

    restored = _callback(diagnostic=True)
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    core_state = _callback(diagnostic=False).state_dict()
    assert set(core_state) == {"best", "best_result", "last_result"}


def test_diagnostic_callback_rejects_history_best_mismatch(monkeypatch):
    monkeypatch.setattr(
        "src.eval.infer_scene.dev_f1",
        lambda *args, **kwargs: _dev_result(),
    )
    callback = _callback(diagnostic=True)
    callback.on_train_epoch_end(
        SimpleNamespace(current_epoch=4),
        _Module(),
    )
    state = callback.state_dict()
    state["history"] = []

    with pytest.raises(ResultContractError, match="history"):
        _callback(diagnostic=True).load_state_dict(state)
