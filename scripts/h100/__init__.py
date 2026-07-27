"""Strict-FP32 H100/Slurm execution lane.

This package is orchestration only.  It deliberately delegates every core
cell to the unchanged :mod:`src.train.finetune` entrypoint.
"""
