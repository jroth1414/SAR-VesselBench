"""Run the git/zstd-dependent handoff tests with the host transfer Python."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from scripts.h100.contracts import atomic_write_json, sha256_file
from scripts.h100.source_validation import HEX64

HOST_TESTS = [
    "tests/test_h100_handoff.py",
    "tests/test_h100_submit_isolation.py",
    "tests/test_experiment_manifest.py",
]
HOST_COMMAND = ["-m", "pytest", "-q", *HOST_TESTS]
RECEIPT_KEYS = {
    "schema",
    "status",
    "slice",
    "command",
    "source_validation_sha256",
    "duration_seconds",
    "log",
}


def run_host_gate(
    *,
    repo: Path,
    log_path: Path,
    receipt_path: Path,
    source_validation_sha256: str,
) -> dict:
    repo = repo.resolve()
    log_path = log_path.absolute()
    receipt_path = receipt_path.absolute()
    if not HEX64.fullmatch(source_validation_sha256):
        raise RuntimeError("host test gate source receipt SHA-256 is invalid")
    for path, label in ((log_path, "host test log"), (receipt_path, "host test receipt")):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"{label} must not pre-exist: {path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8") as output:
        subprocess.run(
            [sys.executable, *HOST_COMMAND],
            cwd=repo,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    duration = time.monotonic() - started
    payload = {
        "schema": 1,
        "status": "passed",
        "slice": "host-handoff",
        "command": HOST_COMMAND,
        "source_validation_sha256": source_validation_sha256,
        "duration_seconds": duration,
        "log": {"path": str(log_path), "sha256": sha256_file(log_path)},
    }
    atomic_write_json(receipt_path, payload)
    receipt_path.chmod(0o444)
    return payload


def validate_host_gate(
    path: Path,
    *,
    expected_sha256: str,
    expected_source_validation_sha256: str,
) -> dict:
    path = path.absolute()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("host test receipt must be a regular non-symlink")
    if not HEX64.fullmatch(expected_sha256) or sha256_file(path) != expected_sha256:
        raise RuntimeError("host test receipt SHA-256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("host test receipt is invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != RECEIPT_KEYS:
        raise RuntimeError("host test receipt keys do not match the contract")
    if (
        payload.get("schema") != 1
        or payload.get("status") != "passed"
        or payload.get("slice") != "host-handoff"
        or payload.get("command") != HOST_COMMAND
        or payload.get("source_validation_sha256")
        != expected_source_validation_sha256
    ):
        raise RuntimeError("host test receipt binding mismatch")
    try:
        duration = float(payload.get("duration_seconds"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("host test duration is invalid") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("host test duration must be finite and positive")
    log = payload.get("log")
    if not isinstance(log, dict) or set(log) != {"path", "sha256"}:
        raise RuntimeError("host test log binding is invalid")
    log_path = Path(str(log["path"]))
    if (
        not log_path.is_absolute()
        or log_path.is_symlink()
        or not log_path.is_file()
        or not HEX64.fullmatch(str(log["sha256"]))
        or sha256_file(log_path) != log["sha256"]
    ):
        raise RuntimeError("host test log hash/path binding mismatch")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-validation-sha256", required=True)
    args = parser.parse_args()
    payload = run_host_gate(
        repo=args.repo,
        log_path=args.log,
        receipt_path=args.receipt,
        source_validation_sha256=args.source_validation_sha256,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
