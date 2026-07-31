"""Fail-closed guards for the resolved H100 Lightning execution contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.h100 import lightning_contract


BACKEND = {
    "cuda_matmul_fp32_precision": "ieee",
    "cudnn_conv_fp32_precision": "ieee",
    "cudnn_rnn_fp32_precision": "ieee",
}


def _instance(class_name: str, module: str) -> object:
    cls = type(class_name, (), {})
    cls.__module__ = module
    return cls()


def _trainer() -> SimpleNamespace:
    plugin = _instance(
        "Precision", "lightning.pytorch.plugins.precision.precision"
    )
    plugin.precision = "32-true"
    strategy = _instance(
        "SingleDeviceStrategy",
        "lightning.pytorch.strategies.single_device",
    )
    strategy.root_device = SimpleNamespace(type="cuda", index=0)
    accelerator = _instance(
        "CUDAAccelerator",
        "lightning.pytorch.accelerators.cuda",
    )
    return SimpleNamespace(
        accelerator=accelerator,
        precision_plugin=plugin,
        strategy=strategy,
        num_devices=1,
        world_size=1,
        device_ids=[0],
        accumulate_grad_batches=1,
    )


@pytest.fixture(autouse=True)
def _strict_process(monkeypatch):
    monkeypatch.setenv("XVIEW3_STRICT_FP32_ACTIVE", "1")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setattr(
        lightning_contract,
        "assert_sitecustomize_active",
        lambda _torch: dict(BACKEND),
    )


def _verified_evidence(model: torch.nn.Module | None = None) -> dict:
    model = model or torch.nn.Linear(4, 2)
    pre = lightning_contract.assert_pre_trainer_contract(
        model,
        precision="32-true",
        devices=1,
        micro_batch=16,
        gradient_accumulation=1,
        torch_module=torch,
    )
    return lightning_contract.assert_trainer_contract(
        _trainer(),
        model,
        precision="32-true",
        devices=1,
        micro_batch=16,
        gradient_accumulation=1,
        pre_trainer=pre,
        torch_module=torch,
    )


def test_exact_fp32_single_device_contract_is_recordable_and_valid():
    evidence = _verified_evidence()

    assert evidence["resolved_trainer"]["gradient_scaler"] is None
    assert evidence["resolved_trainer"]["world_size"] == 1
    assert (
        lightning_contract.validate_trainer_contract_evidence(evidence)
        == evidence
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("precision", "16-mixed"),
        ("devices", 8),
        ("micro_batch", 8),
        ("gradient_accumulation", 2),
    ],
)
def test_pre_trainer_rejects_recipe_drift(field, value):
    kwargs = {
        "precision": "32-true",
        "devices": 1,
        "micro_batch": 16,
        "gradient_accumulation": 1,
    }
    kwargs[field] = value
    with pytest.raises(RuntimeError, match="pre-Trainer recipe mismatch"):
        lightning_contract.assert_pre_trainer_contract(
            torch.nn.Linear(4, 2),
            torch_module=torch,
            **kwargs,
        )


def test_pre_trainer_rejects_non_fp32_parameters():
    with pytest.raises(RuntimeError, match="non-FP32 floating parameters"):
        lightning_contract.assert_pre_trainer_contract(
            torch.nn.Linear(4, 2).half(),
            precision="32-true",
            devices=1,
            micro_batch=16,
            gradient_accumulation=1,
            torch_module=torch,
        )


def test_pre_trainer_rejects_active_autocast():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="autocast enabled"):
            lightning_contract.assert_pre_trainer_contract(
                torch.nn.Linear(4, 2),
                precision="32-true",
                devices=1,
                micro_batch=16,
                gradient_accumulation=1,
                torch_module=torch,
            )


def test_process_contract_rejects_multi_process_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(RuntimeError, match="requires one process"):
        lightning_contract.assert_launch_process_contract(torch_module=torch)


@pytest.mark.parametrize("failure", ["plugin", "scaler", "strategy", "accelerator", "root_device", "world_size"])
def test_resolved_trainer_rejects_amp_ddp_or_multiprocess(failure):
    model = torch.nn.Linear(4, 2)
    pre = lightning_contract.assert_pre_trainer_contract(
        model,
        precision="32-true",
        devices=1,
        micro_batch=16,
        gradient_accumulation=1,
        torch_module=torch,
    )
    trainer = _trainer()
    if failure == "plugin":
        trainer.precision_plugin = _instance(
            "MixedPrecision", "lightning.pytorch.plugins.precision.amp"
        )
        trainer.precision_plugin.precision = "16-mixed"
    elif failure == "scaler":
        trainer.precision_plugin.scaler = object()
    elif failure == "strategy":
        trainer.strategy = _instance(
            "DDPStrategy", "lightning.pytorch.strategies.ddp"
        )
    elif failure == "accelerator":
        trainer.accelerator = _instance(
            "CPUAccelerator", "lightning.pytorch.accelerators.cpu"
        )
    elif failure == "root_device":
        trainer.strategy.root_device = SimpleNamespace(type="cuda", index=1)
    else:
        trainer.world_size = 8
    with pytest.raises(RuntimeError, match="resolved Lightning contract mismatch"):
        lightning_contract.assert_trainer_contract(
            trainer,
            model,
            precision="32-true",
            devices=1,
            micro_batch=16,
            gradient_accumulation=1,
            pre_trainer=pre,
            torch_module=torch,
        )


def test_recorded_evidence_rejects_tampering():
    evidence = _verified_evidence()
    evidence["pre_trainer"]["autocast"]["cuda"] = True
    with pytest.raises(RuntimeError, match="process/model FP32 evidence"):
        lightning_contract.validate_trainer_contract_evidence(evidence)

def test_recorded_evidence_rejects_malformed_backend_without_type_error():
    evidence = _verified_evidence()
    evidence["pre_trainer"]["strict_fp32"]["cuda_matmul_fp32_precision"] = []
    with pytest.raises(RuntimeError, match="process/model FP32 evidence"):
        lightning_contract.validate_trainer_contract_evidence(evidence)




def test_h100_integration_calls_and_marker_binding_are_source_gated():
    repo = Path(__file__).resolve().parents[1]
    finetune = (repo / "src/train/finetune.py").read_text()
    pre = finetune.index("h100_pre_trainer = assert_pre_trainer_contract(")
    trainer = finetune.index("trainer = L.Trainer(")
    post = finetune.index("h100_runtime_contract = assert_trainer_contract(")
    assert pre < trainer < post
    assert 'if h100_runtime_contract is not None:' in finetune
    assert '"h100_runtime_contract": h100_runtime_contract,' not in finetune
    assert 'final["h100_runtime_contract"] = h100_runtime_contract' in finetune

    cell = (repo / "scripts/h100/cell.py").read_text()
    campaign = (repo / "scripts/h100/campaign.py").read_text()
    probe = (repo / "scripts/h100/strict_fp32_probe.py").read_text()
    assert "assert_launch_process_contract()" in cell
    assert "validate_trainer_contract_evidence" in cell
    assert "validate_trainer_contract_evidence" in campaign
    assert "assert_trainer_contract(" in probe
    assert "validate_trainer_contract_evidence" in probe
