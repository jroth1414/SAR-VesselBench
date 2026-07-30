"""Host-only validation of the detached checkout used by the H100 venv."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Mapping

import yaml

from scripts.h100.contracts import (
    EXPECTED_PRECISION,
    FROZEN_PATHS,
    atomic_write_json,
    sha256_file,
    verify_expected_hashes,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_RECEIPT_NAME = "SOURCE_VALIDATED.json"
BASE_PAYLOAD_GIT_SHA = "2726199efcebbebc89156e708b89df2a3415468a"
BASE_PAYLOAD_PACKAGE_ID = f"xview3-h100-fp32-{BASE_PAYLOAD_GIT_SHA}"
BASE_PAYLOAD_KEYS = {
    "package_id",
    "git_sha",
    "manifest_sha256",
    "ready_sha256",
    "sha256sums_sha256",
    "repo_bundle_sha256",
}
RUNTIME_AMENDMENT_KEYS = {
    "package_id",
    "git_sha",
    "manifest_sha256",
    "ready_sha256",
    "sha256sums_sha256",
    "runtime_bundle_sha256",
}
SOURCE_KEYS = {
    "schema",
    "status",
    "scope",
    "git_sha",
    "clean_worktree",
    "frozen_sha256",
    "base_payload",
    "runtime_amendment",
}


def git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo}", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def expected_receipt(
    *,
    git_sha: str,
    frozen_sha256: Mapping[str, str],
    base_payload: Mapping[str, str],
    runtime_amendment: Mapping[str, str],
) -> dict:
    if not HEX40.fullmatch(git_sha):
        raise RuntimeError("source validation requires a full 40-hex git SHA")
    if set(frozen_sha256) != set(FROZEN_PATHS):
        raise RuntimeError("source validation requires every frozen path")
    if any(not HEX64.fullmatch(str(value)) for value in frozen_sha256.values()):
        raise RuntimeError("source validation frozen bindings must be SHA-256")
    if set(base_payload) != BASE_PAYLOAD_KEYS:
        raise RuntimeError("source validation base-payload keys are invalid")
    if (
        base_payload.get("package_id") != BASE_PAYLOAD_PACKAGE_ID
        or base_payload.get("git_sha") != BASE_PAYLOAD_GIT_SHA
        or any(
            not HEX64.fullmatch(str(base_payload[key]))
            for key in BASE_PAYLOAD_KEYS - {"package_id", "git_sha"}
        )
    ):
        raise RuntimeError("source validation base-payload bindings are invalid")
    if set(runtime_amendment) != RUNTIME_AMENDMENT_KEYS:
        raise RuntimeError("source validation runtime-amendment keys are invalid")
    runtime_id = str(runtime_amendment.get("package_id", ""))
    if (
        runtime_amendment.get("git_sha") != git_sha
        or not re.fullmatch(rf"xview3-h100-runtime-{git_sha}-[0-9a-f]{{64}}", runtime_id)
        or any(
            not HEX64.fullmatch(str(runtime_amendment[key]))
            for key in RUNTIME_AMENDMENT_KEYS - {"package_id", "git_sha"}
        )
    ):
        raise RuntimeError("source validation runtime-amendment bindings are invalid")
    return {
        "schema": 2,
        "status": "source-validated",
        "scope": "detached-clean-runtime-amendment-checkout-before-native-venv-links",
        "git_sha": git_sha,
        "clean_worktree": True,
        "frozen_sha256": dict(frozen_sha256),
        "base_payload": dict(base_payload),
        "runtime_amendment": dict(runtime_amendment),
    }


def validate_checkout(
    *,
    repo: Path,
    expected_git_sha: str,
    expected_hashes: Mapping[str, str],
    expected_base_payload: Mapping[str, str],
    expected_runtime_amendment: Mapping[str, str],
) -> dict:
    repo = repo.resolve()
    git_sha = git_output(repo, "rev-parse", "HEAD")
    if git_sha != expected_git_sha:
        raise RuntimeError(f"git SHA mismatch: expected {expected_git_sha}, got {git_sha}")
    dirty = git_output(repo, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RuntimeError(f"target checkout is dirty (including untracked files):\n{dirty}")
    actual_hashes = verify_expected_hashes(repo, expected_hashes)
    detector = yaml.safe_load((repo / "configs/detector.yaml").read_text())
    if detector["schedule"]["precision"] != EXPECTED_PRECISION:
        raise RuntimeError("detector precision is not shared 32-true")
    return expected_receipt(
        git_sha=git_sha,
        frozen_sha256=actual_hashes,
        base_payload=expected_base_payload,
        runtime_amendment=expected_runtime_amendment,
    )


def write_once(path: Path, payload: Mapping[str, object]) -> None:
    path = path.absolute()
    if path.name != SOURCE_RECEIPT_NAME:
        raise RuntimeError(f"source receipt must be named {SOURCE_RECEIPT_NAME}")
    if path.is_symlink():
        raise RuntimeError("source validation receipt cannot be a symlink")
    if path.exists():
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("existing source validation receipt is invalid") from exc
        if prior != dict(payload):
            raise RuntimeError("existing source validation receipt binding mismatch")
    else:
        atomic_write_json(path, payload)
    path.chmod(0o444)


def validate_source_receipt(
    path: Path,
    *,
    expected_sha256: str,
    expected_git_sha: str,
    expected_hashes: Mapping[str, str],
    expected_base_payload: Mapping[str, str],
    expected_runtime_amendment: Mapping[str, str],
) -> dict:
    path = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("source validation receipt must be a regular non-symlink")
    if not HEX64.fullmatch(expected_sha256) or sha256_file(path) != expected_sha256:
        raise RuntimeError("source validation receipt SHA-256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("source validation receipt is invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != SOURCE_KEYS:
        raise RuntimeError("source validation receipt keys do not match the contract")
    expected = expected_receipt(
        git_sha=expected_git_sha,
        frozen_sha256=expected_hashes,
        base_payload=expected_base_payload,
        runtime_amendment=expected_runtime_amendment,
    )
    if payload != expected:
        raise RuntimeError("source validation receipt binding mismatch")
    return payload


def verify_transfer_bindings(
    *,
    base_payload_root: Path,
    runtime_amendment_root: Path,
    expected_base_payload: Mapping[str, str],
    expected_runtime_amendment: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Verify both actual packages and compare every supplied identity field."""

    # Keep the training/runtime import graph independent of Box tooling.  This
    # host-only entrypoint runs in the transfer environment before the native
    # venv boundary.  The prepared verifier reads and verifies the 294 GB base
    # exactly once, then verifies the small amendment against that identity.
    from scripts.handoff.runtime_amendment import prepare_runtime_verifier

    verifier = prepare_runtime_verifier(base_payload_root)
    manifest = verifier(runtime_amendment_root)
    base = manifest.get("base_payload")
    source = manifest.get("source")
    if not isinstance(base, Mapping) or not isinstance(source, Mapping):
        raise RuntimeError("verified runtime amendment lacks transfer identities")
    observed_base = {
        "package_id": str(base.get("package_id", "")),
        "git_sha": str(base.get("source_git_commit", "")),
        "manifest_sha256": str(base.get("manifest_sha256", "")),
        "ready_sha256": str(base.get("ready_sha256", "")),
        "sha256sums_sha256": str(base.get("sha256sums_sha256", "")),
        "repo_bundle_sha256": str(base.get("repo_bundle_sha256", "")),
    }
    observed_runtime = {
        "package_id": str(manifest.get("package_id", "")),
        "git_sha": str(source.get("git_commit", "")),
        "manifest_sha256": sha256_file(runtime_amendment_root / "manifest.json"),
        "ready_sha256": sha256_file(runtime_amendment_root / "READY.json"),
        "sha256sums_sha256": sha256_file(runtime_amendment_root / "SHA256SUMS"),
        "runtime_bundle_sha256": str(source.get("git_bundle_sha256", "")),
    }
    if observed_base != dict(expected_base_payload):
        raise RuntimeError(
            "verified base-payload identity differs from the supplied bindings"
        )
    if observed_runtime != dict(expected_runtime_amendment):
        raise RuntimeError(
            "verified runtime-amendment identity differs from the supplied bindings"
        )
    return {
        "base_payload": observed_base,
        "runtime_amendment": observed_runtime,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-payload-root", type=Path, required=True)
    parser.add_argument("--runtime-amendment-root", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--frozen-sha256", action="append", required=True)
    parser.add_argument("--base-payload-package-id", required=True)
    parser.add_argument("--base-payload-git-sha", required=True)
    parser.add_argument("--base-payload-manifest-sha256", required=True)
    parser.add_argument("--base-payload-ready-sha256", required=True)
    parser.add_argument("--base-payload-sha256sums-sha256", required=True)
    parser.add_argument("--base-payload-repo-bundle-sha256", required=True)
    parser.add_argument("--runtime-amendment-package-id", required=True)
    parser.add_argument("--runtime-amendment-git-sha", required=True)
    parser.add_argument("--runtime-amendment-manifest-sha256", required=True)
    parser.add_argument("--runtime-amendment-ready-sha256", required=True)
    parser.add_argument("--runtime-amendment-sha256sums-sha256", required=True)
    parser.add_argument("--runtime-amendment-bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.frozen_sha256) != len(FROZEN_PATHS):
        parser.error(f"--frozen-sha256 must be repeated {len(FROZEN_PATHS)} times")
    frozen = dict(zip(FROZEN_PATHS, args.frozen_sha256, strict=True))
    base_payload = {
        "package_id": args.base_payload_package_id,
        "git_sha": args.base_payload_git_sha,
        "manifest_sha256": args.base_payload_manifest_sha256,
        "ready_sha256": args.base_payload_ready_sha256,
        "sha256sums_sha256": args.base_payload_sha256sums_sha256,
        "repo_bundle_sha256": args.base_payload_repo_bundle_sha256,
    }
    runtime_amendment = {
        "package_id": args.runtime_amendment_package_id,
        "git_sha": args.runtime_amendment_git_sha,
        "manifest_sha256": args.runtime_amendment_manifest_sha256,
        "ready_sha256": args.runtime_amendment_ready_sha256,
        "sha256sums_sha256": args.runtime_amendment_sha256sums_sha256,
        "runtime_bundle_sha256": args.runtime_amendment_bundle_sha256,
    }
    verify_transfer_bindings(
        base_payload_root=args.base_payload_root,
        runtime_amendment_root=args.runtime_amendment_root,
        expected_base_payload=base_payload,
        expected_runtime_amendment=runtime_amendment,
    )
    payload = validate_checkout(
        repo=args.repo,
        expected_git_sha=args.expected_git_sha,
        expected_hashes=frozen,
        expected_base_payload=base_payload,
        expected_runtime_amendment=runtime_amendment,
    )
    write_once(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
