"""Assert eight H100s and exercise IEEE FP32 matmul/cuDNN in child processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping

from scripts.h100.contracts import (
    EXPECTED_COMPUTE_CAPABILITY,
    EXPECTED_GPU_COUNT,
    validate_gpu_inventory,
)
from scripts.h100.precision import assert_sitecustomize_active


def _device_record(torch, index: int) -> dict:
    props = torch.cuda.get_device_properties(index)
    raw_uuid = getattr(props, "uuid", None)
    if raw_uuid is None:
        raw_uuid = getattr(props, "_uuid", None)
    if raw_uuid is None:
        raw_uuid = getattr(props, "_CUuuid", None)
    uuid = str(raw_uuid or "").strip()
    if not uuid:
        raise RuntimeError(f"CUDA device {index} did not expose a hardware UUID")
    return {
        "index": index,
        "name": props.name,
        "uuid": uuid,
        "compute_capability": [props.major, props.minor],
        "total_memory_bytes": props.total_memory,
    }


def child_probe() -> dict:
    import torch

    backend = assert_sitecustomize_active(torch)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("per-GPU child must see exactly one CUDA device")
    device = torch.device("cuda:0")
    record = _device_record(torch, 0)
    if "H100" not in str(record["name"]).upper():
        raise RuntimeError(f"child device is not an H100: {record['name']!r}")
    if tuple(record["compute_capability"]) != EXPECTED_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"child device must be CC 9.0, got {record['compute_capability']}"
        )

    a = torch.randn(256, 256, dtype=torch.float32, device=device)
    b = torch.randn(256, 256, dtype=torch.float32, device=device)
    matmul = a @ b
    conv = torch.nn.Conv2d(8, 8, kernel_size=3, padding=1, device=device)
    image = torch.randn(2, 8, 64, 64, dtype=torch.float32, device=device)
    convolved = conv(image)
    torch.cuda.synchronize()
    if matmul.dtype != torch.float32 or not torch.isfinite(matmul).all():
        raise RuntimeError("strict-FP32 matmul probe failed")
    if convolved.dtype != torch.float32 or not torch.isfinite(convolved).all():
        raise RuntimeError("strict-FP32 cuDNN convolution probe failed")
    return {"device": record, "backend": backend, "finite": True}


def bind_child_probes(
    inventory: list[dict],
    children: list[object],
    *,
    expected_backend: Mapping[str, str],
) -> list[dict]:
    """Bind each one-device child to the requested parent-visible H100."""

    if len(children) != len(inventory):
        raise RuntimeError(
            "strict-FP32 child probe count does not match the parent GPU inventory"
        )
    bound: list[dict] = []
    observed_uuids: set[str] = set()
    identity_keys = (
        "name",
        "uuid",
        "compute_capability",
        "total_memory_bytes",
    )
    for parent_index, (expected, raw_child) in enumerate(
        zip(inventory, children, strict=True)
    ):
        if not isinstance(raw_child, Mapping):
            raise RuntimeError(f"child probe {parent_index} is not a JSON object")
        device = raw_child.get("device")
        if not isinstance(device, Mapping):
            raise RuntimeError(f"child probe {parent_index} lacks device identity")
        mismatches = {
            key: (expected.get(key), device.get(key))
            for key in identity_keys
            if device.get(key) != expected.get(key)
        }
        if device.get("index") != 0:
            mismatches["visible_index"] = (0, device.get("index"))
        if raw_child.get("backend") != dict(expected_backend):
            mismatches["backend"] = (dict(expected_backend), raw_child.get("backend"))
        if raw_child.get("finite") is not True:
            mismatches["finite"] = (True, raw_child.get("finite"))
        if mismatches:
            raise RuntimeError(
                f"child probe for parent GPU {parent_index} is mismatched: {mismatches}"
            )
        uuid = str(device["uuid"])
        if uuid in observed_uuids:
            raise RuntimeError(f"child probes reused GPU identity: {uuid}")
        observed_uuids.add(uuid)
        bound.append(
            {
                "requested_parent_index": parent_index,
                "expected_parent_uuid": expected["uuid"],
                "device": dict(device),
                "backend": dict(raw_child["backend"]),
                "finite": True,
            }
        )
    return bound


def parent_probe(expected_gpus: int) -> dict:
    import torch

    backend = assert_sitecustomize_active(torch)
    if torch.__version__ != "2.11.0+cu126":
        raise RuntimeError(f"expected torch 2.11.0+cu126, found {torch.__version__}")
    if "sm_90" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"torch build lacks sm_90 kernels: {torch.cuda.get_arch_list()}")
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    inventory = [_device_record(torch, index) for index in range(torch.cuda.device_count())]
    if expected_gpus != EXPECTED_GPU_COUNT:
        raise RuntimeError("the reportable H100 lane requires exactly eight GPUs")
    validate_gpu_inventory(inventory)
    driver_lines = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    drivers = {line.strip() for line in driver_lines if line.strip()}
    if len(driver_lines) != expected_gpus or len(drivers) != 1:
        raise RuntimeError("H100 allocation does not expose one exact driver version")

    children: list[object] = []
    for gpu in range(expected_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        completed = subprocess.run(
            [sys.executable, "-m", "scripts.h100.strict_fp32_probe", "--child"],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        children.append(json.loads(completed.stdout))
    bound_children = bind_child_probes(
        inventory,
        children,
        expected_backend=backend,
    )
    return {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "driver_version": drivers.pop(),
        "backend": backend,
        "devices": inventory,
        "child_probes": bound_children,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpus", type=int, default=EXPECTED_GPU_COUNT)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    payload = child_probe() if args.child else parent_probe(args.expected_gpus)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
