"""Exact owner-approved TEST checkpoint-loading recovery for failed job 659990."""
from __future__ import annotations
import copy
import re
from .package import PackageError, _canonical_json
from .terminal_contract import CAMPAIGN_SOURCE, PROTECTED_OBJECTS as TERMINAL_PROTECTED, runtime_contract

TEST_ANCESTOR = "716f9267af594765e08211d00681414ee9fd8574"
TERMINAL_SOURCE = "1ec7a68fb3b25f90c55ef74950e7b7045d509d80"
CELL = "cnnrand-f100-s0"
CELLS = sorted(f"{arm}-f{fraction}-s0" for arm in
    ("vitrand", "satdino", "sarmae", "vitin1k", "cnnrand", "beS2", "beS1", "cnnin1k")
    for fraction in (10, 25, 50, 100))
PROTECTED_OBJECTS = {**TERMINAL_PROTECTED,
    "scripts/score_test_cohort.py": "4fd536f94979c6f67e68ecad794dfccf1f97c2cc"}
POLICY = {
    "campaign_git_sha": CAMPAIGN_SOURCE, "failed_execution_git_sha": TERMINAL_SOURCE,
    "h100_ready_sha256": "2b9f6c8435dc0a536fbcd3a45b2144fd72f30d8b04ed6c31771bda059191349d",
    "failed_manifest_sha256": "20055c899c3f260c42315afc799e3bc90afea1a9b39e7f8d03f19518c9fcbc41",
    "failed_provenance_sha256": "e5c142fc8e9a4e909d3c6274aa1a35a6d66ee10cc4a6a0cc4a7837b465af6085",
    "cohort_sha256": "a8996efd01a95e78bfb4ae46bc93120d842407d109ee74d111d310e8aa5cbef5",
    "prior_test_launch_sha256": "5875f6f21859446393d012c8696853ecf1b9169f0378b25826d70636cd9f5d21",
    "terminal_source_receipt_sha256": "acc25ad1c94b2b67dd6815a359b44f6d228153a058629e6eca2afe3f060092cb",
    "selected_checkpoint_sha256": "9b6bfab109409e5d93ac948448930568188b9c25bce4da503be8266dc8295d47",
    "selected_checkpoint_epoch": 39, "failed_cell": CELL, "failed_attempt": 4,
    "failed_job": "659990", "failure_utc": "2026-10-01T19:50:21.596817+00:00",
    "node": "dgx18", "gpus": 1, "workers": 1, "cpus": 6, "memory": "64G",
    "time_limit": "1-12:30:00", "approval_date": "2026-10-01",
    "scope": "TEST-only-known-PosixPath-restricted-strict-trained-state-restore",
    "retry_first": CELL, "cells": CELLS, "test_scenes": 16,
    "added_training_batches": 0, "added_optimizer_steps": 0,
    "added_scheduler_steps": 0, "final": "forbidden", "later_failure_reset": "forbidden",
}


def validate_test_recovery_control(control):
    if (not isinstance(control, dict) or set(control) !=
            {"schema", "policy", "source", "protected_source", "contract"}
            or type(control.get("schema")) is not int or control["schema"] != 1
            or _canonical_json(control["policy"]) != _canonical_json(POLICY)):
        raise PackageError("TEST recovery control differs from exact owner approval")
    contract = control["contract"]
    if contract != runtime_contract(contract.get("maximum_physical_file_bytes")):
        raise PackageError("TEST recovery package contract differs")
    if control["protected_source"] != {"required_ancestor": TEST_ANCESTOR, "git_objects": PROTECTED_OBJECTS}:
        raise PackageError("TEST recovery protected source differs")
    source = control["source"]
    if (set(source) != {"branch", "git_commit", "required_ancestor", "git_bundle_sha256"}
            or re.fullmatch(r"sprint-10[a-z]+-replacement-[a-z-]+", str(source.get("branch"))) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit"))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(source.get("git_bundle_sha256"))) is None
            or source.get("required_ancestor") != TEST_ANCESTOR
            or source.get("git_commit") in (CAMPAIGN_SOURCE, TERMINAL_SOURCE, TEST_ANCESTOR)):
        raise PackageError("TEST recovery execution source differs")
    return copy.deepcopy(control)
