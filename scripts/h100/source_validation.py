"""Host-only validation of the detached checkout used by the slim H100 SIF."""

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
PACKAGE_KEYS = {
    "manifest_sha256",
    "ready_sha256",
    "sha256sums_sha256",
    "repo_bundle_sha256",
}
SOURCE_KEYS = {
    "schema",
    "status",
    "scope",
    "git_sha",
    "clean_worktree",
    "frozen_sha256",
    "package",
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
    package: Mapping[str, str],
) -> dict:
    if not HEX40.fullmatch(git_sha):
        raise RuntimeError("source validation requires a full 40-hex git SHA")
    if set(frozen_sha256) != set(FROZEN_PATHS):
        raise RuntimeError("source validation requires every frozen path")
    if any(not HEX64.fullmatch(str(value)) for value in frozen_sha256.values()):
        raise RuntimeError("source validation frozen bindings must be SHA-256")
    if set(package) != PACKAGE_KEYS or any(
        not HEX64.fullmatch(str(value)) for value in package.values()
    ):
        raise RuntimeError("source validation package bindings are invalid")
    return {
        "schema": 1,
        "status": "source-validated",
        "scope": "detached-clean-checkout-before-runtime-links",
        "git_sha": git_sha,
        "clean_worktree": True,
        "frozen_sha256": dict(frozen_sha256),
        "package": dict(package),
    }


def validate_checkout(
    *,
    repo: Path,
    expected_git_sha: str,
    expected_hashes: Mapping[str, str],
    expected_package: Mapping[str, str],
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
        package=expected_package,
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
    expected_package: Mapping[str, str],
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
        package=expected_package,
    )
    if payload != expected:
        raise RuntimeError("source validation receipt binding mismatch")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--frozen-sha256", action="append", required=True)
    parser.add_argument("--package-manifest-sha256", required=True)
    parser.add_argument("--package-ready-sha256", required=True)
    parser.add_argument("--package-sha256sums-sha256", required=True)
    parser.add_argument("--package-repo-bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.frozen_sha256) != len(FROZEN_PATHS):
        parser.error(f"--frozen-sha256 must be repeated {len(FROZEN_PATHS)} times")
    frozen = dict(zip(FROZEN_PATHS, args.frozen_sha256, strict=True))
    package = {
        "manifest_sha256": args.package_manifest_sha256,
        "ready_sha256": args.package_ready_sha256,
        "sha256sums_sha256": args.package_sha256sums_sha256,
        "repo_bundle_sha256": args.package_repo_bundle_sha256,
    }
    payload = validate_checkout(
        repo=args.repo,
        expected_git_sha=args.expected_git_sha,
        expected_hashes=frozen,
        expected_package=package,
    )
    write_once(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
