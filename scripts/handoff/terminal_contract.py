"""Exact owner-authorized scope for the two interrupted terminal DEV callbacks."""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping

from .package import PackageError, _canonical_json
from .recovery_contract import CAMPAIGN_SOURCE, PROTECTED_OBJECTS as _PREVIOUS_PROTECTED

TERMINAL_ANCESTOR = "3808f4ab5f8aea8e6b552f7f2b100fd47adc919f"
CELLS = ("beS2-f10-s0", "vitin1k-f10-s0")
CHECKPOINT_SHA256 = {
    "beS2-f10-s0": "fce3eb4460eca2380b60c2528820ab46526c878f474d37649fd6098c4216a7ee",
    "vitin1k-f10-s0": "d9d1e9af69d2d2a631be53f997d15619bc72231da6bac0b46f8147d76132b1df",
}
# Sealed to the reviewed implementation before any production package is built.
PROTECTED_OBJECTS = {
    **_PREVIOUS_PROTECTED,
    "src/train": "8bb9d845735a07025e13352527c1a019c7450aca",
    "src/eval": "69e5b30b6b5b45f56bd7fb08de561b2e3e486f85",
}
POLICY = {
    "campaign_git_sha": CAMPAIGN_SOURCE,
    "failed_execution_git_sha": TERMINAL_ANCESTOR,
    "h100_ready_sha256": "2b9f6c8435dc0a536fbcd3a45b2144fd72f30d8b04ed6c31771bda059191349d",
    "failed_manifest_sha256": "1b91413111d36fcc977f1cec295aa9a52c8bfa5d2a9db1f61f6eee2259daba8e",
    "previous_source_receipt_sha256": "62b1b35dca4dedb1d2dda2c575bbdcbfded3f6dbb0899e08ac735a297ed11b07",
    "previous_activation_sha256": "aa13767cd7e2dc4e0cc2cbce240a602538831fd4d176e2ad0bd2be19a70d7280",
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "completed_cells": sorted(
        f"{arm}-f{fraction}-s0"
        for arm in ("vitrand", "satdino", "sarmae", "vitin1k", "cnnrand", "beS2", "beS1", "cnnin1k")
        for fraction in (10, 25, 50, 100)
        if f"{arm}-f{fraction}-s0" not in CELLS
    ),
    "cell_order": list(CELLS),
    "node": "dgx18",
    "gpus": 1,
    "workers": 1,
    "approval_date": "2026-10-01",
    "scope": "terminal-epoch49-DEV-only-zero-added-training",
    "epoch": 49,
    "global_step": 35050,
    "last_completed_dev_epoch": 44,
    "added_training_batches": 0,
    "added_optimizer_steps": 0,
    "added_scheduler_steps": 0,
    "completion_schema": 3,
    "train_loss": None,
    "missing_train_loss_reason": "terminal-resume-did-not-persist-train-loss",
    "missing_metrics_csv": "confirmed-absent-before-terminal-recovery",
    "test": "only-after-immutable-all32-training-cohort",
    "final": "forbidden-new-two-cohort-64-cell-authorization-required",
}


def runtime_contract(maximum: int) -> dict:
    if type(maximum) is not int or maximum <= 0:
        raise PackageError("terminal recovery physical file limit must be a positive integer")
    return {
        "schema": 1, "payload": "one-git-bundle-plus-control-json",
        "data_artifacts": 0, "checkpoint_artifacts": 0, "credentials": 0,
        "maximum_physical_file_bytes": maximum,
    }


def validate_terminal_control(control: Mapping[str, object]) -> dict:
    if (not isinstance(control, Mapping)
            or set(control) != {"schema", "policy", "source", "protected_source", "contract"}
            or type(control.get("schema")) is not int or control["schema"] != 1):
        raise PackageError("terminal recovery control schema is invalid")
    if _canonical_json(control["policy"]) != _canonical_json(POLICY):
        raise PackageError("terminal recovery policy differs from owner approval")
    contract = control["contract"]
    if (not isinstance(contract, Mapping)
            or contract != runtime_contract(contract.get("maximum_physical_file_bytes"))
            or any(type(contract[key]) is not int for key in (
                "schema", "data_artifacts", "checkpoint_artifacts", "credentials"))):
        raise PackageError("terminal recovery package contract is invalid")
    if (any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in PROTECTED_OBJECTS.values())
            or control["protected_source"] != {
                "required_ancestor": TERMINAL_ANCESTOR, "git_objects": PROTECTED_OBJECTS}):
        raise PackageError("terminal recovery source is not sealed to reviewed objects")
    source = control["source"]
    if (not isinstance(source, Mapping)
            or set(source) != {"branch", "git_commit", "required_ancestor", "git_bundle_sha256"}
            or re.fullmatch(r"sprint-10[a-z]+-replacement-[a-z-]+", str(source.get("branch"))) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit"))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(source.get("git_bundle_sha256"))) is None
            or source.get("required_ancestor") != TERMINAL_ANCESTOR
            or source.get("git_commit") in (CAMPAIGN_SOURCE, TERMINAL_ANCESTOR)):
        raise PackageError("terminal recovery execution source identity is invalid")
    return copy.deepcopy(dict(control))


def validate_execution_identity(value: Mapping[str, object]) -> dict:
    """Validate pathless actual-source identity before archive/runtime binding."""
    keys = {"campaign_git_sha", "git_sha", "terminal_package_id", "terminal_manifest_sha256", "source_receipt_sha256"}
    if (not isinstance(value, Mapping) or set(value) != keys
            or value.get("campaign_git_sha") != CAMPAIGN_SOURCE
            or re.fullmatch(r"[0-9a-f]{40}", str(value.get("git_sha"))) is None
            or value.get("git_sha") in (CAMPAIGN_SOURCE, TERMINAL_ANCESTOR)
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("terminal_manifest_sha256"))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("source_receipt_sha256"))) is None
            or re.fullmatch(r"xview3-h100-runtime-" + re.escape(str(value.get("git_sha"))) + r"-[0-9a-f]{64}",
                            str(value.get("terminal_package_id"))) is None):
        raise PackageError("terminal recovery execution identity is invalid")
    return copy.deepcopy(dict(value))
