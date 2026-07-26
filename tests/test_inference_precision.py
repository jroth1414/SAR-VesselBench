"""Regression tests for the shared core-inference precision recipe."""

import pytest

from src.eval.infer_scene import _autocast_enabled


def test_fp32_recipe_disables_model_forward_autocast():
    assert not _autocast_enabled("32-true", "cuda")
    assert not _autocast_enabled("32-true", "cpu")
    assert not _autocast_enabled("16-mixed", "cpu")


def test_inference_precision_rejects_unknown_recipe():
    with pytest.raises(ValueError, match="unsupported shared inference precision"):
        _autocast_enabled("bf16-mixed", "cuda")
