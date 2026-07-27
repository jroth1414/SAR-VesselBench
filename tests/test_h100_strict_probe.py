"""Focused guards for parent-to-child H100 identity binding."""

from __future__ import annotations

import copy

import pytest

from scripts.h100.strict_fp32_probe import bind_child_probes


BACKEND = {
    "cuda_matmul_fp32_precision": "ieee",
    "cudnn_conv_fp32_precision": "ieee",
    "cudnn_rnn_fp32_precision": "ieee",
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


@pytest.mark.parametrize("failure", ["wrong_uuid", "reused_uuid", "wrong_backend"])
def test_child_probe_mapping_rejects_mismatched_or_reused_devices(failure):
    inventory = _inventory()
    children = _children(inventory)
    if failure == "wrong_uuid":
        children[3]["device"]["uuid"] = "GPU-NOT-THE-REQUESTED-CARD"
    elif failure == "reused_uuid":
        children[3]["device"] = copy.deepcopy(children[2]["device"])
    else:
        children[3]["backend"]["cuda_matmul_fp32_precision"] = "tf32"

    with pytest.raises(RuntimeError, match="child probe"):
        bind_child_probes(
            inventory,
            children,
            expected_backend=BACKEND,
        )
