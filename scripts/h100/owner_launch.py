"""Source-bound owner smoke waiver and acceptance-to-campaign transition."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path

from scripts.h100.contracts import atomic_write_json, sha256_file
from scripts.handoff.package import _canonical_json, _require_no_symlink_components
from scripts.handoff.replacement_contract import OWNER_LAUNCH_POLICY, validate_replacement_control

OWNER_POLICY_FILENAME = "OWNER_LAUNCH_POLICY.json"
AUTOMATIC_READY_FILENAME = "AUTOMATIC_CAMPAIGN_READY.json"


def owner_policy(control: dict | None) -> dict | None:
    if control is None or "owner_launch_policy" not in control:
        return None
    validate_replacement_control(control)
    return dict(OWNER_LAUNCH_POLICY)


def require_owner_node(control: dict, node: str | None = None) -> None:
    policy = owner_policy(control)
    actual = socket.gethostname() if node is None else node
    if policy and (not isinstance(actual, str) or actual.split(".")[0] != policy["node"]):
        raise RuntimeError("owner launch policy permits only dgx18")


def owner_receipt(control: dict, bindings: dict) -> dict:
    policy = owner_policy(control)
    if policy is None:
        raise RuntimeError("standalone smoke waiver requires explicit replacement owner policy")
    if bindings["source"]["git_sha"] != control["source"]["git_commit"]:
        raise RuntimeError("owner waiver source binding mismatch")
    return {"schema": 1, "kind": "owner-standalone-smoke-waiver",
            "status": "waived-by-owner", "owner_launch_policy": policy, "bindings": bindings}


def validate_owner_smoke_binding(binding: dict, *, control: dict, expected_bindings: dict) -> dict:
    expected = owner_receipt(control, expected_bindings)
    if (
        not isinstance(binding, dict) or set(binding) != {"status", "path", "sha256", "receipt"}
        or binding.get("status") != "waived-by-owner" or binding.get("receipt") != expected
        or binding.get("sha256") != hashlib.sha256(_canonical_json(expected)).hexdigest()
        or not Path(str(binding.get("path", ""))).is_absolute()
        or Path(binding["path"]).name != OWNER_POLICY_FILENAME
    ):
        raise RuntimeError("owner smoke waiver evidence binding mismatch")
    return expected


def smoke_prerequisite(*, control: dict | None, bindings: dict, measured_path: Path,
                       runs_root: Path, create: bool = False) -> dict:
    if owner_policy(control) is None:
        from scripts.h100.slurm_smoke import validate_smoke_receipt
        receipt = validate_smoke_receipt(measured_path, expected_bindings=bindings)
        return {"path": str(measured_path.absolute()), "sha256": sha256_file(measured_path), "receipt": receipt}
    path = runs_root.absolute() / ".h100" / OWNER_POLICY_FILENAME
    receipt = owner_receipt(control, bindings)
    if create:
        require_owner_node(control)
        _require_no_symlink_components(path.parent, leaf="directory")
        if os.path.lexists(path):
            raise RuntimeError("owner launch policy evidence already exists; refusing overwrite")
        atomic_write_json(path, receipt)
        path.chmod(0o444)
    path = _require_no_symlink_components(path, leaf="file")
    if path.stat().st_mode & 0o222 or json.loads(path.read_text()) != receipt:
        raise RuntimeError("owner launch policy evidence is not immutable or exact")
    binding = {"status": "waived-by-owner", "path": str(path), "sha256": sha256_file(path), "receipt": receipt}
    validate_owner_smoke_binding(binding, control=control, expected_bindings=bindings)
    return binding


def readiness_smoke_bindings(payload: dict) -> dict:
    from scripts.h100.slurm_smoke import make_bindings
    venv = payload["venv"]
    return make_bindings(
        git_sha=payload["source"]["git_sha"],
        detector_sha256=payload["source"]["frozen_sha256"]["configs/detector.yaml"],
        venv_sha256=venv["sha256"], venv_build_sha256=venv["venv_build_sha256"],
        base_python_sha256=venv["base_python"]["executable_sha256"],
        base_python_runtime_sha256=venv["base_python"]["runtime"]["sha256"],
        wheelhouse_sha256=venv["wheelhouse"]["identity"]["sha256"],
        base_extraction_receipt_sha256=venv["wheelhouse"]["base_extraction"]["sha256"],
        base_payload=payload["base_payload"], runtime_amendment=payload["runtime_amendment"],
    )


def full_mode(*, runs_root: Path, control: dict, authorize: bool = False) -> tuple[str, str]:
    """Pin one accepted readiness digest before the automatic campaign requeue."""
    if owner_policy(control) is None:
        raise RuntimeError("full launch requires the explicit owner launch policy")
    require_owner_node(control)
    ready = runs_root / ".h100/H100_READY.json"
    marker = runs_root / ".h100" / AUTOMATIC_READY_FILENAME
    if not os.path.lexists(ready):
        if authorize or os.path.lexists(marker):
            raise RuntimeError("automatic campaign has no successful fresh acceptance")
        return "acceptance", "absent"
    ready = _require_no_symlink_components(ready, leaf="file")
    payload = json.loads(ready.read_text())
    if (
        ready.stat().st_mode & 0o222 or payload.get("schema") != 2 or payload.get("status") != "ready"
        or payload.get("replacement", {}).get("control") != control
    ):
        raise RuntimeError("automatic campaign readiness is not canonical")
    from scripts.h100.replacement import validate_readiness_replacement
    validate_readiness_replacement(payload, ready)
    expected = {"schema": 1, "status": "accepted-for-automatic-campaign", "node": "dgx18",
                "git_sha": control["source"]["git_commit"], "runs_root": str(runs_root),
                "readiness_sha256": sha256_file(ready)}
    if authorize:
        if os.path.lexists(marker):
            raise RuntimeError("automatic campaign already authorized; refusing overwrite")
        atomic_write_json(marker, expected)
        marker.chmod(0o444)
    marker = _require_no_symlink_components(marker, leaf="file")
    if marker.stat().st_mode & 0o222 or json.loads(marker.read_text()) != expected:
        raise RuntimeError("automatic campaign readiness digest changed")
    return "campaign", expected["readiness_sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("node", "full-mode", "authorize-full"))
    parser.add_argument("--runtime-package-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--runs-root", type=Path)
    args = parser.parse_args()
    from scripts.h100.replacement import control_from_manifest
    control = control_from_manifest(args.runtime_package_root / "manifest.json", args.expected_manifest_sha256, args.expected_git_sha)
    if args.command == "node":
        policy = owner_policy(control)
        if policy and os.environ.get("SLURM_JOB_ID"):
            require_owner_node(control)
        print(policy["node"] if policy else "unrestricted")
    else:
        if args.runs_root is None:
            parser.error("--runs-root is required")
        print(*full_mode(runs_root=args.runs_root, control=control, authorize=args.command == "authorize-full"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
