"""Focused guards for parent-to-child H100 identity binding."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.h100.lightning_contract import (
    CUDA_ACCELERATOR,
    PRECISION_PLUGIN,
    SINGLE_DEVICE_STRATEGY,
)
from scripts.h100.strict_fp32_probe import (
    _visible_device_tokens,
    bind_child_probes,
    validate_diagnostic_gpu_inventory,
)


BACKEND = {
    "cuda_matmul_fp32_precision": "ieee",
    "cudnn_conv_fp32_precision": "ieee",
    "cudnn_rnn_fp32_precision": "ieee",
}


def _runtime_contract() -> dict:
    return {
        "schema": 1,
        "status": "verified",
        "pre_trainer": {
            "schema": 1,
            "status": "verified",
            "stage": "pre-trainer",
            "precision": "32-true",
            "devices": 1,
            "micro_batch": 16,
            "gradient_accumulation": 1,
            "effective_batch": 16,
            "strict_fp32": dict(BACKEND),
            "autocast": {"global": False, "cuda": False, "cpu": False},
            "process": {
                "WORLD_SIZE": "unset",
                "SLURM_NTASKS": "unset",
                "effective_world_size": 1,
            },
            "model": {
                "floating_parameter_count": 2,
                "floating_parameter_dtypes": ["torch.float32"],
            },
        },
        "resolved_trainer": {
            "accelerator": CUDA_ACCELERATOR,
            "precision_plugin": PRECISION_PLUGIN,
            "precision": "32-true",
            "gradient_scaler": None,
            "strategy": SINGLE_DEVICE_STRATEGY,
            "root_device_type": "cuda",
            "root_device_index": 0,
            "num_devices": 1,
            "world_size": 1,
            "device_ids": [0],
            "gradient_accumulation": 1,
        },
    }


def _inventory() -> list[dict]:
    return [
        {
            "index": index,
            "name": "NVIDIA H100 80GB HBM3",
            "uuid": f"GPU-H100-{index}",
            "compute_capability": [9, 0],
            "total_memory_bytes": 85_000_000_000,
        }
        for index in range(8)
    ]


def _children(inventory: list[dict]) -> list[dict]:
    return [
        {
            "device": {**device, "index": 0},
            "backend": dict(BACKEND),
            "finite": True,
            "runtime_contract": _runtime_contract(),
        }
        for device in inventory
    ]


def test_child_probes_bind_each_visible_device_to_requested_parent_gpu():
    inventory = _inventory()
    bound = bind_child_probes(
        inventory,
        _children(inventory),
        expected_backend=BACKEND,
    )

    assert [item["requested_parent_index"] for item in bound] == list(range(8))
    assert [item["expected_parent_uuid"] for item in bound] == [
        device["uuid"] for device in inventory
    ]
    assert [item["device"]["uuid"] for item in bound] == [
        device["uuid"] for device in inventory
    ]


@pytest.mark.parametrize(
    "failure",
    ["wrong_uuid", "reused_uuid", "wrong_backend", "runtime_contract"],
)
def test_child_probe_mapping_rejects_mismatched_or_reused_devices(failure):
    inventory = _inventory()
    children = _children(inventory)
    if failure == "wrong_uuid":
        children[3]["device"]["uuid"] = "GPU-NOT-THE-REQUESTED-CARD"
    elif failure == "reused_uuid":
        children[3]["device"] = copy.deepcopy(children[2]["device"])
    elif failure == "wrong_backend":
        children[3]["backend"]["cuda_matmul_fp32_precision"] = "tf32"
    else:
        children[3]["runtime_contract"]["resolved_trainer"]["world_size"] = 8

    with pytest.raises(RuntimeError, match="child probe"):
        bind_child_probes(
            inventory,
            children,
            expected_backend=BACKEND,
        )


@pytest.mark.parametrize("count", [1, 2])
def test_bes2_diagnostic_inventory_accepts_only_one_or_two_h100s(count):
    inventory = _inventory()[:count]
    assert validate_diagnostic_gpu_inventory(inventory, count) == inventory


@pytest.mark.parametrize("count", [0, 3, 8])
def test_bes2_diagnostic_inventory_rejects_other_allocation_sizes(count):
    with pytest.raises(RuntimeError, match="one or two"):
        validate_diagnostic_gpu_inventory(_inventory()[:count], count)


def test_bes2_diagnostic_inventory_rejects_duplicate_gpu_identity():
    inventory = _inventory()[:2]
    inventory[1]["uuid"] = inventory[0]["uuid"]
    with pytest.raises(RuntimeError, match="unique"):
        validate_diagnostic_gpu_inventory(inventory, 2)


def test_strict_probe_preserves_slurm_cuda_tokens(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-first,GPU-second")
    assert _visible_device_tokens(2) == ["GPU-first", "GPU-second"]
    with pytest.raises(RuntimeError, match="differs"):
        _visible_device_tokens(1)


def test_diagnostic_strict_mode_is_explicit_and_default_remains_eight_gpu():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/h100/strict_fp32_probe.py"
    ).read_text(encoding="utf-8")
    assert "default=EXPECTED_GPU_COUNT" in source
    assert "if expected_gpus != EXPECTED_GPU_COUNT:" in source
    assert 'parser.add_argument(\n        "--diagnostic"' in source
    assert "CUDA_VISIBLE_DEVICES=device_tokens[gpu]" in source
