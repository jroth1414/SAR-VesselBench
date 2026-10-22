"""Strict IEEE-FP32 policy for the H100 lane.

PyTorch 2.11 deprecated the legacy ``allow_tf32`` switches in favor of
per-backend ``fp32_precision`` settings.  The H100 lane uses only the new
API, and ``sitecustomize.py`` applies it before any training/inference module
can initialize CUDA.
"""

from __future__ import annotations

import os
from typing import Any

STRICT_SENTINEL = "XVIEW3_STRICT_FP32_ACTIVE"


def apply_strict_fp32(torch_module: Any | None = None) -> dict[str, str]:
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


def assert_sitecustomize_active(torch_module: Any | None = None) -> dict[str, str]:
    if os.environ.get(STRICT_SENTINEL) != "1":
        raise RuntimeError(
            "H100 sitecustomize was not loaded in this Python subprocess"
        )
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("NVIDIA_TF32_OVERRIDE changed after process start")
    state = backend_state(torch_module)
    if set(state.values()) != {"ieee"}:
        raise RuntimeError(f"non-IEEE backend state in child process: {state}")
    return state
