"""Replacement namespace preflight and additive readiness bindings; no final access."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from scripts.h100.contracts import atomic_write_json, sha256_file
from scripts.h100.data_staging import replacement_data_binding, validate_replacement_namespaces
from scripts.handoff.package import _canonical_json, _require_no_symlink_components
from scripts.handoff.replacement_contract import validate_replacement_control
from src.analysis.bes2_contract import assert_no_final_consumption

NAMESPACE_FILE = "REPLACEMENT_NAMESPACE.json"
LOAD_TESTS = [
    f"tests/test_fm_checkpoints_load.py::test_value_sensitive_load[{name}]"
    for name in ("satdino_b", "sarmae_b", "bigearthnet_s2", "vit_imagenet", "cnn_imagenet")
]
S2_TESTS = [
    "tests/test_bes2_production.py::test_official_s2_complete_value_sensitive_transfer",
    "tests/test_bes2_production.py::test_h100_s2_batch16_lightning_forward_backward",
]


def control_from_manifest(path: Path, expected_sha256: str, expected_git_sha: str) -> dict | None:
    path = _require_no_symlink_components(path, leaf="file")
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("replacement runtime manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("package_type") != "h100-replacement-amendment":
        return None
    control = validate_replacement_control(manifest.get("amendment_identity"))
    source = control["source"]
    identity_sha = hashlib.sha256(_canonical_json(control)).hexdigest()
    if (
        source["git_commit"] != expected_git_sha or manifest.get("source") != source
        or manifest.get("amendment_identity_sha256") != identity_sha
        or manifest.get("package_id") != f"xview3-h100-runtime-{expected_git_sha}-{identity_sha}"
    ):
        raise RuntimeError("replacement runtime source/content binding mismatch")
    return control


def classify_runtime(path: Path, expected_sha256: str, expected_git_sha: str) -> str:
    control = control_from_manifest(path, expected_sha256, expected_git_sha)
    if control is not None:
        from scripts.handoff.replacement_amendment import verify_replacement_amendment
        verify_replacement_amendment(path.parent)
        return "replacement"
    manifest = json.loads(path.read_text())
    if (
        manifest.get("package_type") != "h100-runtime-amendment"
        or manifest.get("format_version") != 2
        or manifest.get("source", {}).get("git_commit") != expected_git_sha
    ):
        raise RuntimeError("unsupported or unbound H100 runtime package")
    return "legacy"


def assert_final_unconsumed(root: Path) -> None:
    assert_no_final_consumption(root)
    summary = root / "summary/final_verified.csv"
    if summary.exists() or summary.is_symlink():
        raise RuntimeError("verified-final summary exists; owner STOP required")


def preflight(*, repo: Path, runtime_root: Path, manifest_sha256: str,
              git_sha: str, runs_root: Path, original_runs_root: Path,
              data_runtime_root: Path, initialize: bool = False) -> dict:
    from scripts.h100.source_validation import git_output
    from scripts.handoff.replacement_amendment import verify_protected_source
    if git_output(repo, "rev-parse", "HEAD") != git_sha or git_output(repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("replacement preflight requires the exact clean received source")
    verify_protected_source(repo)
    control = control_from_manifest(runtime_root / "manifest.json", manifest_sha256, git_sha)
    if control is None:
        raise RuntimeError("replacement preflight requires a replacement amendment")
    new, original = validate_replacement_namespaces(runs_root=runs_root, original_runs_root=original_runs_root)
    old_ready = _require_no_symlink_components(original / ".h100/H100_READY.json", leaf="file")
    if sha256_file(old_ready) != control["accepted_h100_ready_sha256"]:
        raise RuntimeError("protected original namespace does not contain accepted H100_READY")
    original_ready = json.loads(old_ready.read_text())
    if original_ready.get("evaluation_ground_truth", {}).get("path") != str(original / ".h100/EVAL_GROUND_TRUTH_VALIDATED.json"):
        raise RuntimeError("accepted H100_READY belongs to a different original namespace")
    # Control/path existence checks only. No label, raster, TEST metric or old
    # checkpoint is opened, and the old authorization remains paused.
    assert_final_unconsumed(original)
    assert_final_unconsumed(new)
    data_root = _require_no_symlink_components(data_runtime_root, leaf="directory")
    for name, key in (("manifest.json", "manifest_sha256"), ("READY.json", "ready_sha256"), ("SHA256SUMS", "sha256sums_sha256")):
        path = _require_no_symlink_components(data_root / name, leaf="file")
        if sha256_file(path) != control["accepted_data_runtime"][key]:
            raise RuntimeError("replacement accepted data-runtime control mismatch")
    expected = {
        "schema": 1, "status": "initialized-empty", "git_sha": git_sha,
        "runtime_manifest_sha256": manifest_sha256,
        "runs_root": str(new), "original_runs_root": str(original),
        "accepted_h100_ready_sha256": control["accepted_h100_ready_sha256"],
        "data_binding": replacement_data_binding(control),
    }
    marker = new / ".h100" / NAMESPACE_FILE
    if marker.exists() or marker.is_symlink():
        marker = _require_no_symlink_components(marker, leaf="file")
        if marker.stat().st_mode & 0o222 or json.loads(marker.read_text()) != expected:
            raise RuntimeError("replacement namespace belongs to a different source or cohort")
    else:
        if not initialize:
            raise RuntimeError("replacement namespace was not initialized by submission preflight")
        if new.exists() and any(new.iterdir()):
            raise RuntimeError("replacement namespace must start completely empty")
        new.mkdir(parents=True, exist_ok=True)
        (new / ".h100").mkdir()
        atomic_write_json(marker, expected)
        marker.chmod(0o444)
    return expected


def validate_readiness_replacement(payload: dict, ready_path: Path) -> dict | None:
    """Validate the new control/test extension while preserving legacy receipts."""
    extension = payload.get("replacement")
    if extension is None:
        return None
    if not isinstance(extension, dict) or set(extension) != {"control", "data_binding", "checkpoint_acceptance"}:
        raise RuntimeError("replacement readiness extension schema is invalid")
    control = validate_replacement_control(extension["control"])
    source = control["source"]
    identity_sha = hashlib.sha256(_canonical_json(control)).hexdigest()
    runtime = payload["runtime_amendment"]
    if (
        source["git_commit"] != payload["source"]["git_sha"]
        or runtime["git_sha"] != source["git_commit"]
        or runtime["runtime_bundle_sha256"] != source["git_bundle_sha256"]
        or runtime["package_id"] != f"xview3-h100-runtime-{source['git_commit']}-{identity_sha}"
        or payload["base_payload"] != control["accepted_h100"]["base_payload"]
    ):
        raise RuntimeError("replacement readiness source identity mismatch")
    venv = payload["venv"]
    current = {
        "sha256": venv["sha256"], "venv_build_sha256": venv["venv_build_sha256"],
        "base_python_sha256": venv["base_python"]["executable_sha256"],
        "base_python_runtime_sha256": venv["base_python"]["runtime"]["sha256"],
        "wheelhouse_sha256": venv["wheelhouse"]["identity"]["sha256"],
        "base_extraction_receipt_sha256": venv["wheelhouse"]["base_extraction"]["sha256"],
    }
    if current != control["accepted_h100"]["venv"]:
        raise RuntimeError("replacement venv differs from the accepted sealed runtime")
    from scripts.h100.owner_launch import (
        owner_policy, require_owner_node, readiness_smoke_bindings, smoke_prerequisite,
    )
    if owner_policy(control):
        require_owner_node(control, payload.get("host", "missing"))
        expected_smoke = smoke_prerequisite(
            control=control, bindings=readiness_smoke_bindings(payload),
            measured_path=ready_path.parent / "slurm-smoke/SLURM_SMOKE_READY.json",
            runs_root=ready_path.parent.parent,
        )
        if payload.get("slurm_smoke") != expected_smoke:
            raise RuntimeError("replacement readiness owner waiver mismatch")
    elif payload.get("slurm_smoke", {}).get("status") == "waived-by-owner":
        raise RuntimeError("replacement readiness waiver lacks owner policy")
    binding = replacement_data_binding(control)
    if extension["data_binding"] != binding:
        raise RuntimeError("replacement readiness data binding mismatch")
    gate = extension["checkpoint_acceptance"]
    expected_command = ["-m", "pytest", "-q", "-s", *LOAD_TESTS, *S2_TESTS]
    if (
        not isinstance(gate, dict)
        or set(gate) != {"schema", "status", "command", "duration_seconds", "log", "offline_suite_weights_absent", "inherited_s1_ready_sha256"}
        or gate["schema"] != 1 or gate["status"] != "passed"
        or gate["command"] != expected_command or gate["offline_suite_weights_absent"] is not True
        or gate["inherited_s1_ready_sha256"] != control["accepted_h100_ready_sha256"]
        or isinstance(gate["duration_seconds"], bool)
        or not math.isfinite(float(gate["duration_seconds"])) or gate["duration_seconds"] <= 0
    ):
        raise RuntimeError("replacement checkpoint acceptance contract mismatch")
    log = _require_no_symlink_components(ready_path.parent / "acceptance-logs/replacement-checkpoints.log", leaf="file")
    if gate["log"] != {"path": str(log), "sha256": sha256_file(log)}:
        raise RuntimeError("replacement checkpoint acceptance log binding mismatch")
    return binding


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "classify":
        parser = argparse.ArgumentParser(description="Classify the hash-bound runtime before any namespace write")
        parser.add_argument("--manifest", type=Path, required=True)
        parser.add_argument("--expected-manifest-sha256", required=True)
        parser.add_argument("--expected-git-sha", required=True)
        args = parser.parse_args(sys.argv[2:])
        print(classify_runtime(args.manifest, args.expected_manifest_sha256, args.expected_git_sha))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runtime-package-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--original-runs-root", type=Path, required=True)
    parser.add_argument("--data-runtime-package-root", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    result = preflight(
        repo=args.repo, runtime_root=args.runtime_package_root,
        manifest_sha256=args.expected_manifest_sha256, git_sha=args.expected_git_sha,
        runs_root=args.runs_root, original_runs_root=args.original_runs_root,
        data_runtime_root=args.data_runtime_package_root, initialize=args.initialize,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
