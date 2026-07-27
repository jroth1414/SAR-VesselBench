"""Build the digest-pinned H100 SIF from a verified offline wheelhouse."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from packaging.utils import canonicalize_name

from scripts.h100.contracts import atomic_write_json, atomic_write_text, sha256_file

PINNED_BASE = (
    "python:3.11.15-slim-bookworm@sha256:"
    "28255a3ace7eb4c48bc1b57b90af29e1bc82b4fd6c60614a8e3dce61b87ff941"
)
OCI_DIGEST_RE = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
EXPECTED_PYTHON_VERSION = "3.11.15"


def definition_base(definition: str | Path) -> str:
    for line in Path(definition).read_text().splitlines():
        if line.startswith("From:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("Apptainer definition has no From header")


def wheelhouse_manifest(wheelhouse: str | Path) -> dict[str, dict[str, object]]:
    root = Path(wheelhouse)
    if not root.is_dir():
        raise RuntimeError(f"offline wheelhouse does not exist: {root}")
    entries = sorted(root.iterdir())
    invalid = [
        path.name
        for path in entries
        if path.suffix != ".whl"
        or path.is_symlink()
        or not stat.S_ISREG(path.lstat().st_mode)
    ]
    if invalid:
        raise RuntimeError(
            "offline wheelhouse may contain only regular .whl files: "
            + ", ".join(invalid)
        )
    artifacts = {
        path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in entries
    }
    if not artifacts:
        raise RuntimeError("offline wheelhouse contains no files")
    return artifacts


def assert_wheelhouse_unchanged(
    wheelhouse: str | Path,
    expected: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    actual = wheelhouse_manifest(wheelhouse)
    if actual != expected:
        raise RuntimeError("offline wheelhouse changed while the SIF was building")
    return actual


def normalized_pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise RuntimeError(f"environment contains a non-exact pin: {line}")
        name, version = line.split("==", 1)
        pins[canonicalize_name(name)] = version
    return pins


def assert_freeze_matches_lock(lock_text: str, freeze_text: str) -> dict[str, str]:
    locked = normalized_pins(lock_text)
    installed = normalized_pins(freeze_text)
    mismatches = {
        name: (version, installed.get(name))
        for name, version in locked.items()
        if installed.get(name) != version
    }
    if mismatches:
        first = dict(list(mismatches.items())[:8])
        raise RuntimeError(f"SIF freeze does not match exact environment lock: {first}")
    # These are the only packages supplied by the digest-pinned Python base
    # rather than env-v100node.txt.  Any other extra means the SIF is not the
    # normalized lock environment we reviewed.
    bootstrap_only = {"pip", "setuptools", "wheel"}
    extras = set(installed) - set(locked) - bootstrap_only
    if extras:
        raise RuntimeError(f"SIF freeze has unexpected packages: {sorted(extras)}")
    return {name: installed[name] for name in sorted(locked)}


def build(
    *,
    repo: Path,
    wheelhouse: Path,
    output: Path,
    apptainer: str = "apptainer",
) -> dict:
    definition = repo / "containers/h100-strict-fp32.def"
    env_lock = repo / "locks/env-v100node.txt"
    base = definition_base(definition)
    if not OCI_DIGEST_RE.fullmatch(base) or base != PINNED_BASE:
        raise RuntimeError(f"container base is not the approved OCI digest: {base}")
    wheels = wheelhouse_manifest(wheelhouse)
    lock_text = env_lock.read_text()

    # Resolve the complete lock before the expensive build, explicitly with
    # networking disabled.  A missing or substituted wheel fails here.
    with tempfile.NamedTemporaryFile(suffix=".json") as report:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--ignore-installed",
                "--no-index",
                f"--find-links={wheelhouse}",
                "--requirement",
                str(env_lock),
                "--report",
                report.name,
            ],
            env={
                **os.environ,
                "PIP_NO_INDEX": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            },
            check=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing SIF: {output}")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    temporary_sif = staging / output.name
    try:
        with tempfile.TemporaryDirectory(
            prefix="xview3-h100-apptainer-"
        ) as temporary:
            context = Path(temporary)
            shutil.copy2(definition, context / definition.name)
            shutil.copy2(env_lock, context / "env-v100node.txt")
            # Apptainer follows host-origin symlinks in %files.  Keeping the
            # context link avoids a second wheelhouse copy; the complete map
            # is re-hashed after build to close concurrent-mutation risk.
            os.symlink(
                wheelhouse.resolve(),
                context / "wheelhouse",
                target_is_directory=True,
            )
            version = subprocess.run(
                [apptainer, "--version"],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            # OCI availability is a distinct, digest-pinned precheck.  Package
            # dependency resolution above and inside the SIF stays offline.
            subprocess.run(
                [apptainer, "inspect", f"docker://{base}"],
                text=True,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                [
                    apptainer,
                    "build",
                    "--fakeroot",
                    str(temporary_sif.resolve()),
                    definition.name,
                ],
                cwd=context,
                check=True,
            )

        assert_wheelhouse_unchanged(wheelhouse, wheels)
        sif_sha256 = sha256_file(temporary_sif)
        freeze = subprocess.run(
            [
                apptainer,
                "exec",
                str(temporary_sif),
                "python",
                "-m",
                "pip",
                "freeze",
                "--all",
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        matched_freeze = assert_freeze_matches_lock(lock_text, freeze)
        python_version = subprocess.run(
            [
                apptainer,
                "exec",
                str(temporary_sif),
                "python",
                "-c",
                "import platform; print(platform.python_version())",
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if python_version != EXPECTED_PYTHON_VERSION:
            raise RuntimeError(
                "SIF Python version mismatch: expected "
                f"{EXPECTED_PYTHON_VERSION}, got {python_version}"
            )
        sif_bytes = temporary_sif.stat().st_size
        os.replace(temporary_sif, output)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    atomic_write_text(
        output.with_suffix(output.suffix + ".sha256"),
        f"{sif_sha256}  {output.name}\n"
    )
    payload = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_oci": base,
        "definition_sha256": sha256_file(definition),
        "environment_lock_sha256": sha256_file(env_lock),
        "wheelhouse": wheels,
        "wheelhouse_reverified_after_build": True,
        "python_version": python_version,
        "normalized_environment": matched_freeze,
        "sif_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "apptainer_version": version,
        "sif": {
            "path": str(output.resolve()),
            "bytes": sif_bytes,
            "sha256": sif_sha256,
        },
        "network_policy": "Python packages installed with --no-index from wheelhouse",
        "build_mode": "apptainer build --fakeroot",
    }
    atomic_write_json(output.with_suffix(output.suffix + ".build.json"), payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apptainer", default="apptainer")
    args = parser.parse_args()
    payload = build(
        repo=args.repo.resolve(),
        wheelhouse=args.wheelhouse.resolve(),
        output=args.output.resolve(),
        apptainer=args.apptainer,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
