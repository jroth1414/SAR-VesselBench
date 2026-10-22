"""Exact approved CPU-only finalization of the completed job-660080 TEST grid."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

from .package import PackageError, _canonical_json
from .test_recovery_contract import PROTECTED_OBJECTS as TEST_PROTECTED
from .terminal_contract import runtime_contract

GRID_ANCESTOR = "b36fffa0fed0af9868b6043b62a6fc9b25d63250"
INVENTORY_SHA256 = "eafca17583c6704283b3d7e3615345371eadafe3148202178704e68429bce8bb"
_raw_inventory = Path(__file__).with_name("grid_inventory.json").read_bytes()
if hashlib.sha256(_raw_inventory).hexdigest() != INVENTORY_SHA256:
    raise PackageError("approved grid-finalization inventory bytes differ")
INVENTORY = json.loads(_raw_inventory)
PROTECTED_OBJECTS = {
    **TEST_PROTECTED,
    "src/analysis/curves.py": "3b0cec5646089ae7967bdaf324534f63b41fd175",
    "scripts/h100/campaign.py": "99eac0a777a950daa67b94e1a681af48c16572af",
    "scripts/h100/cell.py": "493d6017fa4d2047d0c957f7c0cc6734237a1482",
    "scripts/h100/resume_recovery.py": "a055118f1ca3669d773f392c7a4883edf6eb85bd",
    "scripts/h100/test_recovery.py": "77de537802ad7c191b4de393143a99b1a2db94d2",
    "scripts/h100/terminal_recovery.py": "8a7c2fca2622a0977a73c9aa0650035bbd7a26e3",
}
POLICY = {
    "approval_date": "2026-10-02", "scope": "completed-all32-TEST-grid-only",
    "approved_proposal_sha256": "173efef340da661325d109b14353b6c83a3399b78029a2c47a0295a4327cf635",
    "approved_inventory_sha256": INVENTORY_SHA256, "inventory": INVENTORY,
    "node": "dgx18", "gpus": 0, "workers": 0, "cpus": 6,
    "memory": "64G", "time_limit": "04:00:00", "requeue": False,
    "added_training_batches": 0, "added_optimizer_steps": 0,
    "added_scheduler_steps": 0, "added_model_forwards": 0,
    "test_inputs": "forbidden", "final_inputs": "forbidden",
    "later_failure_reset": "forbidden", "reporting_gates": "unchanged",
}


def validate_grid_control(control):
    if (not isinstance(control, dict) or set(control) !=
            {"schema", "policy", "source", "protected_source", "contract"}
            or type(control.get("schema")) is not int or control["schema"] != 1
            or _canonical_json(control["policy"]) != _canonical_json(POLICY)):
        raise PackageError("grid finalization control differs from exact owner approval")
    contract = control["contract"]
    if not isinstance(contract, dict) or contract != runtime_contract(contract.get("maximum_physical_file_bytes")):
        raise PackageError("grid finalization package contract differs")
    if control["protected_source"] != {"required_ancestor": GRID_ANCESTOR, "git_objects": PROTECTED_OBJECTS}:
        raise PackageError("grid finalization protected source differs")
    source = control["source"]
    if (not isinstance(source, dict)
            or set(source) != {"branch", "git_commit", "required_ancestor", "git_bundle_sha256"}
            or re.fullmatch(r"sprint-10[a-z]+-replacement-[a-z-]+", str(source.get("branch"))) is None
            or re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit"))) is None
            or re.fullmatch(r"[0-9a-f]{64}", str(source.get("git_bundle_sha256"))) is None
            or source.get("required_ancestor") != GRID_ANCESTOR
            or source.get("git_commit") in (GRID_ANCESTOR, INVENTORY["recipe_git_sha"], INVENTORY["test_execution"]["git_sha"])):
        raise PackageError("grid finalization source differs")
    return copy.deepcopy(control)
