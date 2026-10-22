"""CPU-only regression for TEST restore; no training, downloads or real scenes."""
import hashlib
import json
import pickle
from pathlib import Path

import lightning as L
import pandas as pd
import pytest
import torch

from scripts import score_test_cohort
from src.models.init_loaders import INIT_NAMES
from src.train import lit_modules


class TinyDetector(L.LightningModule):
    def __init__(self, init_name, weights_root, load_weights=True):
        super().__init__()
        self.save_hyperparameters()
        if load_weights:
            raise AssertionError("TEST restore attempted a pretrained artifact lookup")
        self.layer = torch.nn.Linear(2, 1)
        self.register_buffer("calibration", torch.zeros(1))

    def forward(self, value):
        return self.layer(value) + self.calibration


class UnexpectedMetadata:
    pass


class ReachedInference(Exception):
    pass


def fixture(tmp_path, monkeypatch, init_name="cnn_random", change=None):
    module = TinyDetector(init_name, Path("/absent/previous-allocation/weights"), False)
    with torch.no_grad():
        module.layer.weight.fill_(3.25)
        module.layer.bias.fill_(-1.5)
        module.calibration.fill_(0.75)
    saved = {
        "pytorch-lightning_version": L.__version__, "epoch": 39,
        "state_dict": module.state_dict(),
        "hyper_parameters": {"init_name": init_name, "load_weights": True,
                             "weights_root": Path("/absent/previous-allocation/weights")},
    }
    if change == "missing":
        del saved["state_dict"]["calibration"]
    elif change == "unexpected":
        saved["state_dict"]["unknown.weight"] = torch.ones(1)
    elif change == "global":
        saved["hyper_parameters"]["extra_metadata"] = UnexpectedMetadata()
    run = tmp_path / "synthetic-f100-s0"
    checkpoint = run / "checkpoints/best.ckpt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(saved, checkpoint)
    checkpoint.chmod(0o444)
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({}))
    binding = {"relative_path": "synthetic-f100-s0/checkpoints/best.ckpt",
               "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "epoch": 39}
    arguments = {
        "cohort": {"cells": [{"exp_id": run.name, "best_dev": {"threshold": 0.5},
                                "best_checkpoint": binding}]},
        "cohort_sha256": "a" * 64,
        "data_cfg": {"paths": {"raw_xview3": str(tmp_path / "absent-rasters"), "stats": str(stats)}},
        "det_cfg": {"decode": {"candidate_floor": 0.05, "d_nms_m": 100},
                    "eval": {"tile_px": 512, "tile_stride_px": 256, "infer_batch": 16},
                    "schedule": {"precision": "32-true"}},
        "labels": pd.DataFrame(columns=["scene_id"]),
        "test_scene_ids": ["synthetic-scene"], "device": "cpu",
    }
    monkeypatch.setattr(lit_modules, "HeatmapLitModule", TinyDetector)
    return run, checkpoint, module, arguments


@pytest.mark.parametrize("init_name", INIT_NAMES)
def test_test_restore_reproduces_path_failure_then_restores_exact_full_state(
        tmp_path, monkeypatch, init_name):
    run, checkpoint, original, arguments = fixture(tmp_path, monkeypatch, init_name)
    before = checkpoint.read_bytes()
    globals_before = set(torch.serialization.get_safe_globals())
    with pytest.raises(pickle.UnpicklingError, match="pathlib.PosixPath"):
        TinyDetector.load_from_checkpoint(str(checkpoint), map_location="cpu")
    observed = []
    from src.eval import infer_scene
    def verify(module, *args, **kwargs):
        assert not module.training and module.hparams.load_weights is False
        assert module.hparams.weights_root == Path("/absent/previous-allocation/weights")
        assert module.state_dict().keys() == original.state_dict().keys()
        for name, tensor in original.state_dict().items():
            assert torch.equal(module.state_dict()[name], tensor)
        inputs = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
        assert torch.equal(module(inputs), original(inputs))
        assert set(torch.serialization.get_safe_globals()) == globals_before
        observed.append(module.hparams.init_name)
        raise ReachedInference()
    monkeypatch.setattr(infer_scene, "infer_scene", verify)
    with pytest.raises(ReachedInference):
        score_test_cohort.score_run(run, **arguments)
    assert observed == [init_name]
    assert checkpoint.read_bytes() == before
    assert set(torch.serialization.get_safe_globals()) == globals_before
    assert not (run / "test_metrics.json").exists()


@pytest.mark.parametrize("change,exception", [("missing", RuntimeError), ("unexpected", RuntimeError),
                                               ("global", pickle.UnpicklingError)])
def test_test_restore_rejects_incomplete_state_and_additional_globals(
        tmp_path, monkeypatch, change, exception):
    run, checkpoint, _original, arguments = fixture(tmp_path, monkeypatch, change=change)
    before = checkpoint.read_bytes()
    globals_before = set(torch.serialization.get_safe_globals())
    from src.eval import infer_scene
    monkeypatch.setattr(infer_scene, "infer_scene", lambda *args, **kwargs: pytest.fail("Inference ran after invalid restore"))
    with pytest.raises(exception):
        score_test_cohort.score_run(run, **arguments)
    assert checkpoint.read_bytes() == before
    assert set(torch.serialization.get_safe_globals()) == globals_before
    assert not (run / "test_metrics.json").exists()


def test_test_restore_keeps_checkpoint_binding_gate_before_model_construction(tmp_path, monkeypatch):
    run, checkpoint, _original, arguments = fixture(tmp_path, monkeypatch)
    arguments["cohort"]["cells"][0]["best_checkpoint"]["sha256"] = "0" * 64
    monkeypatch.setattr(lit_modules, "HeatmapLitModule", lambda *args, **kwargs: pytest.fail("Constructed model before checkpoint gate"))
    with pytest.raises(score_test_cohort.HeldoutContractError, match="drifted"):
        score_test_cohort.score_run(run, **arguments)
    assert not (run / "test_metrics.json").exists()
