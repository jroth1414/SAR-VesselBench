"""Fail-closed Lightning execution contract for the strict-FP32 H100 lane."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from scripts.h100.contracts import (
    EFFECTIVE_BATCH,
    EXPECTED_PRECISION,
    GRADIENT_ACCUMULATION,
    MICRO_BATCH,
)
from scripts.h100.precision import STRICT_SENTINEL, assert_sitecustomize_active

CONTRACT_SCHEMA = 1
CUDA_ACCELERATOR = "lightning.pytorch.accelerators.cuda.CUDAAccelerator"
PRECISION_PLUGIN = "lightning.pytorch.plugins.precision.precision.Precision"
SINGLE_DEVICE_STRATEGY = (
    "lightning.pytorch.strategies.single_device.SingleDeviceStrategy"
)
STRICT_BACKEND_KEYS = {
    "cuda_matmul_fp32_precision",
    "cudnn_conv_fp32_precision",
    "cudnn_rnn_fp32_precision",
}


def h100_runtime_active() -> bool:
    """Whether this process is inside the sitecustomize-guarded H100 runtime."""

    return os.environ.get(STRICT_SENTINEL) == "1"


def _qualified_type(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _autocast_state(torch_module: Any) -> dict[str, bool]:
    states = {"global": bool(torch_module.is_autocast_enabled())}
    for device_type in ("cuda", "cpu"):
        try:
            states[device_type] = bool(
                torch_module.is_autocast_enabled(device_type)
            )
        except TypeError as exc:
            raise RuntimeError(
                "pinned torch lacks device-specific autocast inspection"
            ) from exc
    if any(states.values()):
        raise RuntimeError(f"H100 runtime entered with autocast enabled: {states}")
    return states


def _single_process_state() -> dict[str, object]:
    observed: dict[str, object] = {}
    for name in ("WORLD_SIZE", "SLURM_NTASKS"):
        raw = os.environ.get(name)
        observed[name] = raw if raw not in (None, "") else "unset"
        if raw not in (None, ""):
            try:
                value = int(raw)
            except ValueError as exc:
                raise RuntimeError(f"{name} is not an integer: {raw!r}") from exc
            if value != 1:
                raise RuntimeError(
                    f"H100 core runtime requires one process, but {name}={value}"
                )
    observed["effective_world_size"] = 1
    return observed


def _model_parameter_state(
    model: Any, torch_module: Any
) -> dict[str, object]:
    floating = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point()
    ]
    if not floating:
        raise RuntimeError("H100 model exposes no floating parameters")
    mismatched = [
        f"{name}:{parameter.dtype}"
        for name, parameter in floating
        if parameter.dtype != torch_module.float32
    ]
    if mismatched:
        raise RuntimeError(
            "H100 model has non-FP32 floating parameters: "
            + ", ".join(mismatched[:8])
        )
    return {
        "floating_parameter_count": len(floating),
        "floating_parameter_dtypes": ["torch.float32"],
    }


def assert_launch_process_contract(
    *, torch_module: Any | None = None
) -> dict[str, object]:
    """Assert process-level state before an H100 model or Trainer can run."""

    if torch_module is None:
        import torch as torch_module

    backend = assert_sitecustomize_active(torch_module)
    return {
        "strict_fp32": backend,
        "autocast": _autocast_state(torch_module),
        "process": _single_process_state(),
    }


def assert_pre_trainer_contract(
    model: Any,
    *,
    precision: str,
    devices: int,
    micro_batch: int,
    gradient_accumulation: int,
    torch_module: Any | None = None,
) -> dict[str, object]:
    """Assert the exact H100 core recipe immediately before Trainer creation."""

    if torch_module is None:
        import torch as torch_module

    effective_batch = micro_batch * gradient_accumulation
    expected = {
        "precision": EXPECTED_PRECISION,
        "devices": 1,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch": EFFECTIVE_BATCH,
    }
    observed = {
        "precision": precision,
        "devices": devices,
        "micro_batch": micro_batch,
        "gradient_accumulation": gradient_accumulation,
        "effective_batch": effective_batch,
    }
    if observed != expected:
        raise RuntimeError(
            f"H100 pre-Trainer recipe mismatch: {observed} != {expected}"
        )
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "verified",
        "stage": "pre-trainer",
        **observed,
        **assert_launch_process_contract(torch_module=torch_module),
        "model": _model_parameter_state(model, torch_module),
    }


def assert_trainer_contract(
    trainer: Any,
    model: Any,
    *,
    precision: str,
    devices: int,
    micro_batch: int,
    gradient_accumulation: int,
    pre_trainer: Mapping[str, object],
    torch_module: Any | None = None,
) -> dict[str, object]:
    """Assert Lightning resolved no AMP/scaler/DDP immediately after creation."""

    current_pre = assert_pre_trainer_contract(
        model,
        precision=precision,
        devices=devices,
        micro_batch=micro_batch,
        gradient_accumulation=gradient_accumulation,
        torch_module=torch_module,
    )
    if dict(pre_trainer) != current_pre:
        raise RuntimeError("H100 pre-Trainer evidence changed during construction")

    plugin = trainer.precision_plugin
    strategy = trainer.strategy
    root_device = getattr(strategy, "root_device", None)
    device_ids = list(trainer.device_ids)
    resolved = {
        "accelerator": _qualified_type(trainer.accelerator),
        "precision_plugin": _qualified_type(plugin),
        "precision": str(getattr(plugin, "precision", "")),
        "gradient_scaler": getattr(plugin, "scaler", None),
        "strategy": _qualified_type(strategy),
        "root_device_type": str(getattr(root_device, "type", "")),
        "root_device_index": getattr(root_device, "index", None),
        "num_devices": trainer.num_devices,
        "world_size": trainer.world_size,
        "device_ids": device_ids,
        "gradient_accumulation": trainer.accumulate_grad_batches,
    }
    expected = {
        "accelerator": CUDA_ACCELERATOR,
        "precision_plugin": PRECISION_PLUGIN,
        "precision": EXPECTED_PRECISION,
        "gradient_scaler": None,
        "strategy": SINGLE_DEVICE_STRATEGY,
        "root_device_type": "cuda",
        "root_device_index": 0,
        "num_devices": 1,
        "world_size": 1,
        "device_ids": [0],
        "gradient_accumulation": GRADIENT_ACCUMULATION,
    }
    if resolved != expected:
        raise RuntimeError(
            f"H100 resolved Lightning contract mismatch: {resolved} != {expected}"
        )
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "verified",
        "pre_trainer": current_pre,
        "resolved_trainer": resolved,
    }


def validate_trainer_contract_evidence(
    evidence: object,
) -> dict[str, object]:
    """Validate recorded evidence without trusting an unverified completion marker."""

    if not isinstance(evidence, Mapping):
        raise RuntimeError("H100 completion lacks Lightning runtime-contract evidence")
    if set(evidence) != {
        "schema",
        "status",
        "pre_trainer",
        "resolved_trainer",
    }:
        raise RuntimeError("H100 Lightning runtime-contract evidence keys are invalid")
    pre = evidence.get("pre_trainer")
    resolved = evidence.get("resolved_trainer")
    if (
        evidence.get("schema") != CONTRACT_SCHEMA
        or evidence.get("status") != "verified"
        or not isinstance(pre, Mapping)
        or not isinstance(resolved, Mapping)
    ):
        raise RuntimeError("H100 Lightning runtime-contract evidence is invalid")
    expected_pre = {
        "precision": EXPECTED_PRECISION,
        "devices": 1,
        "micro_batch": MICRO_BATCH,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "effective_batch": EFFECTIVE_BATCH,
    }
    expected_pre_keys = set(expected_pre) | {
        "schema",
        "status",
        "stage",
        "strict_fp32",
        "autocast",
        "process",
        "model",
    }
    if (
        set(pre) != expected_pre_keys
        or pre.get("schema") != CONTRACT_SCHEMA
        or pre.get("status") != "verified"
        or pre.get("stage") != "pre-trainer"
        or any(
            pre.get(name) != value for name, value in expected_pre.items()
        )
    ):
        raise RuntimeError("H100 recorded pre-Trainer recipe is invalid")
    autocast = pre.get("autocast")
    model = pre.get("model")
    strict_fp32 = pre.get("strict_fp32")
    process = pre.get("process")
    if (
        not isinstance(autocast, Mapping)
        or set(autocast) != {"global", "cuda", "cpu"}
        or any(value is not False for value in autocast.values())
        or not isinstance(model, Mapping)
        or set(model)
        != {"floating_parameter_count", "floating_parameter_dtypes"}
        or type(model.get("floating_parameter_count")) is not int
        or model.get("floating_parameter_count", 0) <= 0
        or model.get("floating_parameter_dtypes") != ["torch.float32"]
        or not isinstance(strict_fp32, Mapping)
        or set(strict_fp32) != STRICT_BACKEND_KEYS
        or any(
            strict_fp32.get(name) != "ieee"
            for name in STRICT_BACKEND_KEYS
        )
        or not isinstance(process, Mapping)
        or set(process) != {"WORLD_SIZE", "SLURM_NTASKS", "effective_world_size"}
        or process.get("effective_world_size") != 1
        or any(
            process.get(name) not in ("unset", "1")
            for name in ("WORLD_SIZE", "SLURM_NTASKS")
        )
    ):
        raise RuntimeError("H100 recorded process/model FP32 evidence is invalid")
    expected_resolved = {
        "accelerator": CUDA_ACCELERATOR,
        "precision_plugin": PRECISION_PLUGIN,
        "precision": EXPECTED_PRECISION,
        "gradient_scaler": None,
        "strategy": SINGLE_DEVICE_STRATEGY,
        "root_device_type": "cuda",
        "root_device_index": 0,
        "num_devices": 1,
        "world_size": 1,
        "device_ids": [0],
        "gradient_accumulation": GRADIENT_ACCUMULATION,
    }
    if dict(resolved) != expected_resolved:
        raise RuntimeError("H100 recorded resolved Trainer evidence is invalid")
    return dict(evidence)
