"""Focused guards for parent-to-child H100 identity binding."""

from __future__ import annotations

import copy

import pytest

from scripts.h100.lightning_contract import (
    CUDA_ACCELERATOR,
    PRECISION_PLUGIN,
    SINGLE_DEVICE_STRATEGY,
)
from scripts.h100.strict_fp32_probe import bind_child_probes
from src.eval.final_eval import _validated_h100_hardware_class
from src.eval.heldout_contract import HeldoutContractError


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


def test_final_eval_hardware_gate_requires_exact_bound_strict_h100_receipt():
    inventory = _inventory()
    hardware = {
        "torch": "2.11.0+cu126",
        "cuda_build": "12.6",
        "driver_version": "590.1",
        "backend": dict(BACKEND),
        "devices": inventory,
        "child_probes": bind_child_probes(
            inventory, _children(inventory), expected_backend=BACKEND
        ),
    }
    assert _validated_h100_hardware_class(hardware) == {
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "total_memory_bytes": 85_000_000_000,
        "compute_capability": [9, 0],
        "driver_version": "590.1",
        "torch": "2.11.0+cu126",
        "cuda_build": "12.6",
    }

    tampered = copy.deepcopy(hardware)
    tampered["child_probes"][3]["expected_parent_uuid"] = "GPU-WRONG"
    with pytest.raises(HeldoutContractError, match="strict-FP32 H100"):
        _validated_h100_hardware_class(tampered)


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
