"""Fixture helpers shared by the terminal-recovery tests.

The cluster runtime tests that originally defined these helpers exercise the
scheduler tooling, which this repository does not include; only the payload
builders the remaining tests import are kept here.
"""

from __future__ import annotations

from scripts.h100 import lightning_contract


def strict_backend():
    return {
        "cuda_matmul_fp32_precision": "ieee",
        "cudnn_conv_fp32_precision": "ieee",
        "cudnn_rnn_fp32_precision": "ieee",
    }


def h100_runtime_contract():
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
            "strict_fp32": strict_backend(),
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
            "accelerator": lightning_contract.CUDA_ACCELERATOR,
            "precision_plugin": lightning_contract.PRECISION_PLUGIN,
            "precision": "32-true",
            "gradient_scaler": None,
            "strategy": lightning_contract.SINGLE_DEVICE_STRATEGY,
            "root_device_type": "cuda",
            "root_device_index": 0,
            "num_devices": 1,
            "world_size": 1,
            "device_ids": [0],
            "gradient_accumulation": 1,
        },
    }
