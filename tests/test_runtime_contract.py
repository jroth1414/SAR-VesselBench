"""Hardware-neutral guards for the public strict-FP32 runtime contract."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.runtime import lightning_contract
from src.runtime.experiment import load_cells
from src.runtime.io import atomic_write_json, atomic_write_text, sha256_file


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
    plugin = _instance("Precision", "lightning.pytorch.plugins.precision.precision")
    plugin.precision = "32-true"
    strategy = _instance(
        "SingleDeviceStrategy", "lightning.pytorch.strategies.single_device"
    )
    strategy.root_device = SimpleNamespace(type="cuda", index=0)
    accelerator = _instance(
        "CUDAAccelerator", "lightning.pytorch.accelerators.cuda"
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
        "assert_strict_fp32_active",
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
    assert lightning_contract.validate_trainer_contract_evidence(evidence) == evidence


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
            torch.nn.Linear(4, 2), torch_module=torch, **kwargs
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


def test_process_contract_rejects_multi_process_environment(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(RuntimeError, match="requires one process"):
        lightning_contract.assert_launch_process_contract(torch_module=torch)


def test_recorded_evidence_rejects_tampering():
    evidence = _verified_evidence()
    evidence["pre_trainer"]["autocast"]["cuda"] = True
    with pytest.raises(RuntimeError, match="process/model FP32 evidence"):
        lightning_contract.validate_trainer_contract_evidence(evidence)


def test_active_manifest_resolves_exact_core_matrix():
    repo = Path(__file__).resolve().parents[1]
    cells = load_cells(repo)
    assert len(cells) == 32
    assert len({cell.exp_id for cell in cells}) == 32
    assert {cell.seed for cell in cells} == {0}


def test_durable_io_helpers_round_trip(tmp_path):
    json_path = tmp_path / "receipt.json"
    text_path = tmp_path / "receipt.txt"
    atomic_write_json(json_path, {"status": "verified", "finite": 1.0})
    atomic_write_text(text_path, "verified\n")
    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert text_path.read_text(encoding="utf-8") == "verified\n"
    assert len(sha256_file(json_path)) == 64
