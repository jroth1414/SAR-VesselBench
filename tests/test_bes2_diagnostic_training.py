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
    DiagnosticOptimizationTrace,
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
        "label_frac": 0.5,
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

    assert layout["run_id"] == "bes2-current_replay-probe-f50-s0"
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
            label_frac=0.1,
        ),
        _detector(),
        _Parser(),
    )

    assert layout["run_id"] == "bes2-first_conv_reset-full-f10-s0"
    assert layout["trainer_epochs"] == layout["schedule_epochs"] == 50


@pytest.mark.parametrize(
    "update,match",
    [
        ({"init": "cnn_random"}, "require --init bigearthnet_s2"),
        ({"label_frac": 1.0}, "authorize label fractions"),
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


class _TraceCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, kernel_size=1),
        )
        self.stages = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Conv2d(2, 2, kernel_size=1),
                )
                for _ in range(4)
            ]
        )


class _TraceBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _TraceCore()


class _TraceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _TraceBackbone()
        self.head = torch.nn.Conv2d(2, 1, kernel_size=1)


def test_optimization_trace_observes_exact_milestones_and_actual_updates():
    model = _TraceModule()
    optimizer = torch.optim.SGD(
        [
            {
                "params": list(model.parameters()),
                "lr": 0.01,
                "lr_scale": 0.65,
            }
        ]
    )
    callback = DiagnosticOptimizationTrace()

    for step in (1, 10, 100, 500):
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        trainer = SimpleNamespace(global_step=step - 1)
        callback.on_before_optimizer_step(trainer, model, optimizer)
        optimizer.step()
        callback.on_train_batch_end(trainer, model, None, None, 0)

    payload = callback.payload()
    assert [record["optimizer_step"] for record in payload["records"]] == [
        1,
        10,
        100,
        500,
    ]
    first = payload["records"][0]["first_convolution"]
    assert first["name"] == "backbone.model.stem.0.weight"
    assert first["lr_scale"] == pytest.approx(0.65)
    assert first["gradient_norm"] > 0.0
    assert first["update_norm"] > 0.0
    assert {
        "stem",
        "stage_0",
        "stage_1",
        "stage_2",
        "stage_3",
        "detector_head",
    } <= set(payload["records"][0]["groups"])

    restored = DiagnosticOptimizationTrace()
    restored.load_state_dict(callback.state_dict())
    assert restored.payload() == payload


def test_optimization_trace_checkpoint_finalizes_completed_pending_step():
    model = _TraceModule()
    optimizer = torch.optim.SGD(
        [
            {
                "params": list(model.parameters()),
                "lr": 0.01,
                "lr_scale": 0.65,
            }
        ]
    )
    callback = DiagnosticOptimizationTrace()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer = SimpleNamespace(global_step=0)

    callback.on_before_optimizer_step(trainer, model, optimizer)
    optimizer.step()
    trainer.global_step = 1
    state = callback.state_dict()

    assert [record["optimizer_step"] for record in state["records"]] == [1]
    assert state["records"][0]["first_convolution"]["update_norm"] > 0.0
    restored = DiagnosticOptimizationTrace()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
