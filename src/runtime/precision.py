"""Strict IEEE-FP32 backend policy shared by training and inference."""

from __future__ import annotations

import os
from typing import Any


STRICT_SENTINEL = "XVIEW3_STRICT_FP32_ACTIVE"


def assert_strict_fp32_environment() -> None:
    """Fail before importing Torch when the strict startup marker is absent."""

    if os.environ.get(STRICT_SENTINEL) != "1":
        raise RuntimeError("the strict-FP32 startup hook was not loaded")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE changed after process start")


def apply_strict_fp32(torch_module: Any | None = None) -> dict[str, str]:
    """Disable reduced-mantissa TF32 and return the asserted backend state."""

    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError(
            "NVIDIA_TF32_OVERRIDE=0 must be exported before Python/CUDA starts"
        )
    if torch_module is None:
        import torch as torch_module

    torch_module.backends.cuda.matmul.fp32_precision = "ieee"
    torch_module.backends.cudnn.conv.fp32_precision = "ieee"
    torch_module.backends.cudnn.rnn.fp32_precision = "ieee"
    state = backend_state(torch_module)
    if set(state.values()) != {"ieee"}:
        raise RuntimeError(f"strict FP32 backend assertion failed: {state}")
    os.environ[STRICT_SENTINEL] = "1"
    return state


def backend_state(torch_module: Any | None = None) -> dict[str, str]:
    """Return the three PyTorch FP32 backend settings used by the study."""

    if torch_module is None:
        import torch as torch_module
    return {
        "cuda_matmul_fp32_precision": str(
            torch_module.backends.cuda.matmul.fp32_precision
        ),
        "cudnn_conv_fp32_precision": str(
            torch_module.backends.cudnn.conv.fp32_precision
        ),
        "cudnn_rnn_fp32_precision": str(
            torch_module.backends.cudnn.rnn.fp32_precision
        ),
    }


def assert_strict_fp32_active(torch_module: Any | None = None) -> dict[str, str]:
    """Fail if process startup did not install the strict IEEE-FP32 policy."""

    assert_strict_fp32_environment()
    state = backend_state(torch_module)
    if set(state.values()) != {"ieee"}:
        raise RuntimeError(f"non-IEEE backend state in child process: {state}")
    return state
