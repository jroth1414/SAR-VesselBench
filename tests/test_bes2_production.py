"""Production S2 fallback: exact transfer, seeded head, and historical reset parity."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from safetensors.torch import save_file

from src.models import bes2_diagnostic as diagnostic
from src.models import bes2_transfer as transfer
from src.models import init_loaders as loaders
from src.train.lit_modules import HeatmapLitModule


class TinyBackbone(torch.nn.Module):
    out_channels = 128
    out_stride = 8

    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "stem": torch.nn.Sequential(
                torch.nn.Conv2d(3, 128, 4, stride=4),
                torch.nn.GroupNorm(32, 128),
            ),
            "stages": torch.nn.Sequential(torch.nn.Conv2d(128, 128, 1, stride=2)),
        })


@pytest.fixture
def s2_fixture(tmp_path, monkeypatch):
    with torch.random.fork_rng(devices=[]):
        target = TinyBackbone().model.state_dict()
    source = {
        f"model.vision_encoder.{key}": torch.full_like(value, (index + 1) / 100)
        for index, (key, value) in enumerate(target.items())
    }
    source[transfer.SOURCE_STEM_KEY] = torch.full((128, 10, 4, 4), 0.1)
    source["model.vision_encoder.head.fc.weight"] = torch.ones(19, 128)
    source["model.vision_encoder.head.fc.bias"] = torch.ones(19)
    directory = tmp_path / "bigearthnet_s2"
    directory.mkdir()
    checkpoint = directory / "model.safetensors"
    save_file(source, checkpoint)
    (directory / "LICENSE.note").write_text("CPU fixture\n")
    monkeypatch.setattr(transfer, "SOURCE_CHECKPOINT_SHA256", transfer.sha256_file(checkpoint))
    monkeypatch.setattr(loaders, "build_backbone", lambda family: TinyBackbone())
    # The production S2 path must never call the historical conversion helper.
    monkeypatch.setattr(loaders, "repeat_with_rescaling", lambda *args: pytest.fail("S2 converted a band"))
    return tmp_path, source


def test_production_retains_seed_zero_stem_and_head_and_loads_every_other_tensor(s2_fixture):
    root, source = s2_fixture
    torch.manual_seed(0)
    fresh = HeatmapLitModule(init_name="cnn_random", load_weights=False, head_channels=32)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(0)
    loaded = HeatmapLitModule(init_name="bigearthnet_s2", weights_root=root, head_channels=32)
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert transfer.state_tensor_hashes(fresh.head) == transfer.state_tensor_hashes(loaded.head)
    before, after = fresh.backbone.model.state_dict(), loaded.backbone.model.state_dict()
    assert transfer.tensor_sha256(after[transfer.STEM_KEY]) == transfer.tensor_sha256(before[transfer.STEM_KEY])
    for key in set(after) - {transfer.STEM_KEY}:
        assert torch.equal(after[key], source[f"model.vision_encoder.{key}"])
        assert not torch.equal(after[key], before[key])
    assert sum(p.numel() for p in fresh.parameters()) == sum(p.numel() for p in loaded.parameters())
    receipt = loaded.backbone.initialization_provenance
    assert receipt["retained_seeded_target_keys"] == ["backbone.model.stem.0.weight"]
    assert set(receipt["loaded_target_keys"]) == set(after) - {transfer.STEM_KEY}
    assert receipt["source_checkpoint_sha256"] == transfer.SOURCE_CHECKPOINT_SHA256
    assert len(loaders.INIT_NAMES) == 8


def test_production_and_diagnostic_share_reset_but_replay_keeps_historical_weight(s2_fixture):
    root, source = s2_fixture
    def build(variant):
        torch.manual_seed(0)
        return HeatmapLitModule(init_name="bigearthnet_s2", weights_root=root,
                                diagnostic_variant=variant, head_channels=32)
    production, reset, replay = build(None), build("first_conv_reset"), build("current_replay")
    assert transfer.state_tensor_hashes(production) == transfer.state_tensor_hashes(reset)
    assert transfer.state_tensor_hashes(production.head) == transfer.state_tensor_hashes(replay.head)
    prod_state, replay_state = production.state_dict(), replay.state_dict()
    assert [key for key in prod_state if not torch.equal(prod_state[key], replay_state[key])] == ["backbone.model.stem.0.weight"]
    expected = diagnostic.repeat_with_rescaling(source[transfer.SOURCE_STEM_KEY], 3)
    assert torch.equal(replay_state["backbone.model.stem.0.weight"], expected)


@pytest.mark.parametrize("defect", ["missing_bias", "unexpected", "duplicate", "wrong_shape", "wrong_dtype"])
def test_transfer_rejects_any_other_partial_or_incompatible_load(s2_fixture, defect):
    _, original = s2_fixture
    source = dict(original)
    mapping = loaders.map_bigearthnet_keys(source)
    key = "model.vision_encoder.stem.0.bias"
    if defect == "missing_bias":
        mapping.pop(key)
    elif defect == "unexpected":
        mapping[key] = "unexpected.weight"
    elif defect == "duplicate":
        mapping[key] = transfer.STEM_KEY
    elif defect == "wrong_shape":
        source[key] = source[key][:1]
    else:
        source[key] = source[key].double()
    with pytest.raises(transfer.BES2TransferError):
        transfer.load_post_stem(TinyBackbone().model, source, mapping)


def test_production_rejects_unpinned_s2_before_deserialization(s2_fixture):
    root, _ = s2_fixture
    (root / "bigearthnet_s2/model.safetensors").write_bytes(b"substituted")
    with pytest.raises(transfer.BES2TransferError, match="SHA-256 mismatch"):
        loaders.build_init("bigearthnet_s2", weights_root=root)


@pytest.mark.skipif(not os.environ.get("XVIEW3_BES2_WEIGHTS_ROOT"), reason="explicit S2-only local weight acceptance")
def test_official_s2_complete_value_sensitive_transfer():
    root = Path(os.environ["XVIEW3_BES2_WEIGHTS_ROOT"])
    torch.manual_seed(0)
    fresh = HeatmapLitModule(init_name="cnn_random", load_weights=False)
    stem_hash = transfer.tensor_sha256(fresh.backbone.model.state_dict()[transfer.STEM_KEY])
    head_hashes = transfer.state_tensor_hashes(fresh.head)
    rng = torch.random.get_rng_state().clone()
    fresh_probe = fresh.backbone.model.state_dict()["stages.0.blocks.0.conv_dw.weight"].clone()
    count = sum(p.numel() for p in fresh.parameters())
    del fresh
    torch.manual_seed(0)
    loaded = HeatmapLitModule(init_name="bigearthnet_s2", weights_root=root)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert transfer.tensor_sha256(loaded.backbone.model.state_dict()[transfer.STEM_KEY]) == stem_hash
    assert transfer.state_tensor_hashes(loaded.head) == head_hashes
    assert sum(p.numel() for p in loaded.parameters()) == count
    _, source = transfer.source_state(root)
    target = loaded.backbone.model.state_dict()
    for source_key, target_key in loaders.map_bigearthnet_keys(source).items():
        if target_key != transfer.STEM_KEY:
            assert torch.equal(target[target_key], source[source_key])
    assert (target["stages.0.blocks.0.conv_dw.weight"] - fresh_probe).norm() > 1


@pytest.mark.skipif(os.environ.get("XVIEW3_BES2_H100_ACCEPTANCE") != "1", reason="explicit Judy H100 allocation only")
def test_h100_s2_batch16_lightning_forward_backward():
    import json
    import time
    import lightning as L
    from scripts.h100.lightning_contract import assert_pre_trainer_contract, assert_trainer_contract
    from scripts.h100.precision import assert_sitecustomize_active

    started = time.monotonic()
    backend = assert_sitecustomize_active(torch)
    assert torch.__version__ == "2.11.0+cu126"
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability(0) == (9, 0)
    assert "H100" in torch.cuda.get_device_name(0)
    L.seed_everything(0, workers=True)
    model = HeatmapLitModule(init_name="bigearthnet_s2", weights_root=Path(os.environ["XVIEW3_BES2_WEIGHTS_ROOT"]))
    recipe = dict(precision="32-true", devices=1, micro_batch=16, gradient_accumulation=1)
    pre = assert_pre_trainer_contract(model, **recipe)
    observed = {}

    class CheckGradients(L.Callback):
        def on_before_optimizer_step(self, trainer, module, optimizer):
            for label, parameter in (
                ("stem", module.backbone.model.stem[0].weight),
                ("post_stem", module.backbone.model.stages[0].blocks[0].conv_dw.weight),
                ("head", next(module.head.parameters())),
            ):
                assert parameter.grad is not None
                assert parameter.grad.dtype == torch.float32
                assert torch.isfinite(parameter.grad).all()
                observed[label] = float(parameter.grad.norm())
                assert observed[label] > 0

    trainer = L.Trainer(accelerator="gpu", devices=1, precision="32-true",
                        accumulate_grad_batches=1, max_epochs=50, max_steps=1,
                        logger=False, enable_checkpointing=False, enable_progress_bar=False,
                        enable_model_summary=False, num_sanity_val_steps=0,
                        callbacks=[CheckGradients()])
    assert_trainer_contract(trainer, model, pre_trainer=pre, **recipe)
    sample = {"image": torch.randn(3, 512, 512), "heatmap": torch.zeros(128, 128),
              "mask": torch.ones(128, 128)}
    sample["heatmap"][64, 64] = 1
    loader = torch.utils.data.DataLoader([sample] * 16, batch_size=16, num_workers=0)
    stem_before = model.backbone.model.stem[0].weight.detach().cpu().clone()
    trainer.fit(model, train_dataloaders=loader)
    assert trainer.global_step == 1
    assert set(observed) == {"stem", "post_stem", "head"}
    assert not torch.equal(model.backbone.model.stem[0].weight.detach().cpu(), stem_before)
    model = model.eval().cuda()
    assert_trainer_contract(trainer, model, pre_trainer=pre, **recipe)
    with torch.no_grad():
        logits = model(torch.randn(16, 3, 512, 512, device="cuda", dtype=torch.float32))
    assert logits.shape == (16, 1, 128, 128)
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    torch.cuda.synchronize()
    print(json.dumps({"status": "passed", "init": "bigearthnet_s2", "seed": 0,
                      "strict_fp32": backend, "gradient_norms": observed,
                      "source_checkpoint_sha256": transfer.SOURCE_CHECKPOINT_SHA256,
                      "gpu_hours": (time.monotonic() - started) / 3600}, sort_keys=True))


@pytest.mark.parametrize("promote_schema", [False, True])
def test_diagnostic_artifact_cannot_freeze_core_cohort(tmp_path, promote_schema):
    import json
    from scripts.h100.contracts import load_cells
    from src.eval.heldout_contract import HeldoutContractError, create_training_cohort

    cells = load_cells(Path(__file__).resolve().parents[1])
    payload = {"diagnostic_run_schema": 1, "purpose": "bes2-root-cause-training",
               "variant": "first_conv_reset", "training_result": {"result_schema": 2}}
    if promote_schema:
        payload["result_schema"] = 2
    # Even renaming a diagnostic artifact into a core cell must fail closed.
    marker = tmp_path / cells[0].exp_id / "final_metrics.json"
    marker.parent.mkdir()
    marker.write_text(json.dumps(payload))
    marker.chmod(0o444)
    output = tmp_path / ".h100/TRAINING_COHORT.json"
    with pytest.raises(HeldoutContractError, match="invalid training marker"):
        create_training_cohort(cells=cells, runs_root=tmp_path, output=output,
                               git_sha="a" * 40, detector_sha256="b" * 64,
                               candidate_floor=0.05)
    assert not output.exists()
