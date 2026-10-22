"""Pathless, owner-approved identities for the fresh 32-cell H100 cohort."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

from .package import PackageError, _canonical_json, _hash_file, _load_json
from .runtime_amendment import _expected_base_identity

APPROVED_SOURCE = "93d479419c6b3a1d314b0753b4edd84495ed9cac"
ACCEPTED_READY_SHA256 = "6dc4a77ff57e2a21a8e3aa39eda5a6e12e500ef759e4ff34a1ee2fa6df1c8f4e"
ACCEPTED_BINDINGS_SHA256 = "2ab650673fa4554272ffab55389982e69c15334469d7e62bc6409041ad8ad85a"
PROTECTED_OBJECTS = {
    "configs": "58e1d496056a5485e001a9f279f7cab440311ea6",
    "data": "f8218453bc0ba623b0043d860d8b3d25ffe90ede",
    "locks": "0df8b4b5f502f59e41e3206dbd83a102ab9b7f9b",
    "src/data": "ad81aa9720877978233fb52e02f5ac444e32c052",
    "src/eval": "06feb6992463f15bf5eb8d38f6318acac4a7f739",
    "src/models": "1fd2f60db4d3b95e5c51c77438d2b3727398d394",
    "src/train": "383beffd2f9fb406863aa5f0262ff7d4507acf8f",
    "tests/test_backbone_parity.py": "a0db2742f6924a37803c2587ca8b038373b36e78",
    "tests/test_fm_checkpoints_load.py": "f26d483b367f9c47a78f7e8cc92d87654ca60ef0",
    "tests/test_scorer_immutable.py": "faa061b409245bbf3585957efa508d15cf7dc3c5",
    "tests/test_split_disjoint.py": "93d4f6f1df9b016ecdb0820e2e5c97fffc6a7ab4",
}
BINDING_KEYS = {
    "accepted_h100", "accepted_h100_ready_sha256", "accepted_data_view",
    "accepted_data_runtime", "evaluation_ground_truth",
}


OWNER_LAUNCH_POLICY = {
    "schema": 1,
    "status": "waived-by-owner",
    "approved_utc_date": "2026-09-14",
    "standalone_smoke": "waived-not-passed",
    "node": "dgx18",
    "failed_smoke_job_id": "643872",
    "failed_smoke_source": "c7ea82c378ac134321405182e58aba372409a6fb",
    "rationale": "owner-reports-recent-successful-dgx18-diagnostics",
    "sealed_runtime": "unchanged-exact-verification-required",
    "acceptance": "fresh-required-before-automatic-campaign",
    "scientific_and_data_gates": "unchanged",
}


def accepted_bindings(path: Path) -> dict:
    """Read accepted JSON controls only; never dereference their archived paths."""
    if path.is_symlink() or _hash_file(path) != ACCEPTED_READY_SHA256:
        raise PackageError("replacement requires the exact accepted H100_READY bytes")
    ready = _load_json(path)
    venv = ready["venv"]
    accepted = {
        "schema": 1,
        "ready_sha256": ACCEPTED_READY_SHA256,
        "source": ready["source"],
        "strict_fp32": ready["strict_fp32"],
        "base_payload": ready["base_payload"],
        "runtime_amendment": ready["runtime_amendment"],
        "venv": {
            "sha256": venv["sha256"],
            "venv_build_sha256": venv["venv_build_sha256"],
            "base_python_sha256": venv["base_python"]["executable_sha256"],
            "base_python_runtime_sha256": venv["base_python"]["runtime"]["sha256"],
            "wheelhouse_sha256": venv["wheelhouse"]["identity"]["sha256"],
            "base_extraction_receipt_sha256": venv["wheelhouse"]["base_extraction"]["sha256"],
        },
    }
    return {
        "accepted_h100": accepted,
        "accepted_h100_ready_sha256": ACCEPTED_READY_SHA256,
        "accepted_data_view": {
            key: venv["staged_data_view"][key] for key in ("sha256", "receipt")
        },
        "accepted_data_runtime": ready["runtime_amendment"],
        "evaluation_ground_truth": {
            key: ready["evaluation_ground_truth"][key] for key in ("sha256", "receipt")
        },
    }


def runtime_contract(maximum: int) -> dict:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise PackageError("replacement physical file limit must be a positive integer")
    return {
        "schema": 2, "payload": "one-git-bundle-plus-control-json",
        "data_artifacts": 0, "checkpoint_artifacts": 0, "credentials": 0,
        "maximum_physical_file_bytes": maximum,
        "core_cells": 32, "arms": list(range(1, 9)),
        "fractions": [0.1, 0.25, 0.5, 1.0], "seed": 0,
        "initialization": "fresh-all-32-no-prior-or-diagnostic-checkpoints",
        "arm6": "BigEarthNet-S2 post-stem transfer with a seeded three-channel first convolution",
        "retained_tensor": "backbone.model.stem.0.weight",
        "s2_checkpoint_sha256": "b09d0e41cc683878243a9128a6f4724d6a71d562318beeae716f0dce9cbbf454",
        "other_arms": "unchanged", "new_s1_checkpoint_tests": False,
        "precision": "32-true", "tf32": False,
        "micro_batch": 16, "gradient_accumulation": 1, "effective_batch": 16,
        "processes_per_gpu": 1, "ddp": False, "layer_decay": 0.65,
        "max_epochs": 50, "early_stopping": "unchanged-frozen-recipe",
        "pre_cohort_view": "train111-fixed-dev8-no-test-v1",
        "test": "only-after-new-immutable-all32-training-cohort",
        "final": "forbidden-new-two-cohort-64-cell-authorization-required",
        "original_campaign": "preserved-immutable", "v100": "untouched-diagnostic",
    }


def validate_replacement_control(control: Mapping[str, object]) -> dict:
    """Pure validation of exact accepted bindings and the owner-approved recipe."""
    import re
    expected_keys = BINDING_KEYS | {"schema", "source", "contract", "protected_source"}
    if isinstance(control, Mapping) and "owner_launch_policy" in control:
        expected_keys.add("owner_launch_policy")
        if control["owner_launch_policy"] != OWNER_LAUNCH_POLICY:
            raise PackageError("replacement owner launch policy differs from approval")
    if not isinstance(control, Mapping) or set(control) != expected_keys or control.get("schema") != 2:
        raise PackageError("replacement control schema is invalid")
    bindings = {key: control[key] for key in BINDING_KEYS}
    if hashlib.sha256(_canonical_json(bindings)).hexdigest() != ACCEPTED_BINDINGS_SHA256:
        raise PackageError("replacement accepted data/runtime/readiness binding mismatch")
    contract = control["contract"]
    if not isinstance(contract, Mapping) or contract != runtime_contract(contract.get("maximum_physical_file_bytes")):
        raise PackageError("replacement runtime contract differs from owner approval")
    if control["protected_source"] != {"git_commit": APPROVED_SOURCE, "git_objects": PROTECTED_OBJECTS}:
        raise PackageError("replacement protected source binding mismatch")
    source = control["source"]
    if (
        not isinstance(source, Mapping)
        or set(source) != {"branch", "git_commit", "required_ancestor", "git_bundle_sha256"}
        or re.fullmatch(r"sprint-10[j-z]-replacement-[a-z-]+", str(source.get("branch"))) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(source.get("git_bundle_sha256"))) is None
        or source.get("required_ancestor") != APPROVED_SOURCE
    ):
        raise PackageError("replacement source identity is invalid")
    return dict(control)


def base_payload_identity() -> dict:
    return _expected_base_identity()
