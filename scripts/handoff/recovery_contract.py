"""Exact owner-approved identities for the checkpoint-load recovery only."""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping

from .package import PackageError
from .replacement_contract import PROTECTED_OBJECTS as _ORIGINAL_PROTECTED_OBJECTS

CAMPAIGN_SOURCE = "06b2e61dbb85dc5fa5a5d5853f9ba15f6a87dca9"
RECOVERY_ANCESTOR = "f856c1acd6b9f646f66ef15bc31476b5f4ebab02"
PREVIOUS_RECOVERY_SOURCE = "d471e4ca175bbd65c1cb09559673f35dbdfe9111"
PREVIOUS_RECOVERY_RECEIPT_SHA256 = "bb42e5c78b75fa18b84aa9935ac5eef42980d03dd73c5a5ccda3e8a1b4df62d5"
PROTECTED_OBJECTS = {
    **_ORIGINAL_PROTECTED_OBJECTS,
    "src/train": "db1ac3ff149473c80fb4718295afaed1e3913ba1",
}
POLICY = {
    "campaign_git_sha": CAMPAIGN_SOURCE,
    "h100_ready_sha256": "2b9f6c8435dc0a536fbcd3a45b2144fd72f30d8b04ed6c31771bda059191349d",
    "failed_manifest_sha256": "49e7ac65cd83b59add7616d7a33ec5d5ac75bc4f75a1649f8bc721313a2ab643",
    "completed_cells": [
        "beS1-f100-s0", "sarmae-f100-s0", "satdino-f100-s0",
        "vitin1k-f100-s0", "vitrand-f100-s0",
    ],
    "checkpoint_sha256": {
        "beS1-f50-s0": "860903b566d08dac95799680f99c1aadc4a19f69a3c5fd9e2a61001d7b8f5051",
        "beS2-f100-s0": "dd1043a0118c29f58471bc52b8572313627ffd6fa410d3cb02d083c91fb03d4d",
        "beS2-f50-s0": "19b6e7e946eff2cdf77312536953388b548db755abe3c9e8873b7b49a597e1db",
        "cnnin1k-f100-s0": "780e64f24d14d9a78eb06312615a5af0475ede59472328f41419b451a3fe4902",
        "cnnin1k-f50-s0": "8d1f9f1ff3e72d48091792a41e0bbb1222bded69950369178d64ffd425026e17",
        "cnnrand-f100-s0": "1596b80c47eb6deaa87f1df77c62e4c9099174706e7e24800fd38b4dd576a0e1",
        "cnnrand-f50-s0": "1575af54ec9bff0921e5dba1e6be715e47d3da6cfcd72b557a0b6667c64a9674",
        "vitrand-f50-s0": "66190a2819ad289ca2768fe22cf58edaef93c0abd4c8858e50d8a72827faa5ad",
    },
    "node": "dgx18",
    "approval_date": "2026-09-24",
    "scope": "restricted-PosixPath-resume",
}


def runtime_contract(maximum: int) -> dict:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise PackageError("recovery physical file limit must be a positive integer")
    return {
        "schema": 1,
        "payload": "one-git-bundle-plus-control-json",
        "data_artifacts": 0,
        "checkpoint_artifacts": 0,
        "credentials": 0,
        "maximum_physical_file_bytes": maximum,
    }


def validate_recovery_control(control: Mapping[str, object]) -> dict:
    """Validate execution lineage without opening campaign data or checkpoints."""
    if (
        not isinstance(control, Mapping)
        or set(control) != {"schema", "policy", "source", "protected_source", "contract"}
        or type(control.get("schema")) is not int
        or control["schema"] != 1
    ):
        raise PackageError("recovery control schema is invalid")
    if control["policy"] != POLICY:
        raise PackageError("recovery policy differs from owner approval")
    contract = control["contract"]
    if (
        not isinstance(contract, Mapping)
        or contract != runtime_contract(contract.get("maximum_physical_file_bytes"))
        or any(type(contract[key]) is not int for key in (
            "schema", "data_artifacts", "checkpoint_artifacts", "credentials",
        ))
    ):
        raise PackageError("recovery package contract is invalid")
    if control["protected_source"] != {
        "git_commit": RECOVERY_ANCESTOR, "git_objects": PROTECTED_OBJECTS,
    }:
        raise PackageError("recovery protected source differs from reviewed loader fix")
    source = control["source"]
    if (
        not isinstance(source, Mapping)
        or set(source) != {"branch", "git_commit", "required_ancestor", "git_bundle_sha256"}
        or re.fullmatch(r"sprint-10[s-z]-replacement-[a-z-]+", str(source.get("branch"))) is None
        or re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit"))) is None
        or re.fullmatch(r"[0-9a-f]{64}", str(source.get("git_bundle_sha256"))) is None
        or source.get("required_ancestor") != RECOVERY_ANCESTOR
        or source.get("git_commit") == CAMPAIGN_SOURCE
    ):
        raise PackageError("recovery execution source identity is invalid")
    return copy.deepcopy(dict(control))
