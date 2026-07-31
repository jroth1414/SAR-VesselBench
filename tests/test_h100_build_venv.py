"""Focused CPU fixture tests for the native Judy H100 venv seal."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.h100 import build_venv
from scripts.h100.contracts import sha256_file


def _write_executable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755)


def _metadata(path: Path, *, prefix: str, base_prefix: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    root = Path(base_prefix)
    metadata: dict[str, object] = {
        "version": "3.11.15",
        "implementation": "cpython",
        "implementation_version": [3, 11, 15, "final", 0],
        "soabi": "cpython-311-x86_64-linux-gnu",
        "platform": "Linux-fixture-x86_64",
        "libc": ["glibc", "2.36"],
        "prefix": prefix,
        "base_prefix": base_prefix,
        "stdlib": f"{base_prefix}/lib/python3.11",
        "platstdlib": f"{base_prefix}/lib/python3.11",
        "executable": str(resolved),
        "config_vars": {
            "LIBDIR": f"{base_prefix}/lib",
            "LIBPL": f"{base_prefix}/lib/python3.11/config",
            "LDLIBRARY": "libpython3.11.so",
            "LIBRARY": None,
            "INSTSONAME": None,
        },
        "mapped_runtime_files": [str(root / "lib/libsystem-runtime.so")],
        "probed_modules": {
            name: "loaded" for name in build_venv.BASE_RUNTIME_PROBE_MODULES
        },
        "requested_path": str(path.absolute()),
        "resolved_path": str(resolved),
        "executable_sha256": sha256_file(resolved),
    }
    metadata["runtime"] = build_venv.base_python_runtime_manifest(metadata)
    return metadata


def _fixture_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    repo = tmp_path / "repo"
    lock = repo / "locks/env-v100node.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text("# exact fixture\nDemo_Pkg==1.0\n")

    extracted = tmp_path / "base-extracted"
    wheelhouse = extracted / "environment/wheelhouse"
    wheelhouse.mkdir(parents=True)
    (wheelhouse / "demo_pkg-1.0-py3-none-any.whl").write_bytes(b"fixture wheel")
    base_package_id = "xview3-h100-fp32-" + "1" * 40
    base_manifest_sha256 = "2" * 64
    extraction_receipt = extracted / "HANDOFF_EXTRACTED.json"
    extraction_receipt.write_text(
        __import__("json").dumps(
            {
                "format_version": 1,
                "package_id": base_package_id,
                "manifest_sha256": base_manifest_sha256,
                "wheelhouse": build_venv.wheelhouse_identity(wheelhouse),
            },
            sort_keys=True,
        )
    )

    base_python = tmp_path / "python311/bin/python3.11"
    _write_executable(base_python, b"#!/bin/sh\nexit 0\n")
    output = tmp_path / "persistent/xview3-h100-venv"
    base_prefix = str(base_python.parent.parent)
    stdlib = Path(base_prefix) / "lib/python3.11"
    stdlib.mkdir(parents=True)
    (stdlib / "site.py").write_text("BASE_SITE = 1\n")
    (stdlib / "site-packages").mkdir()
    (stdlib / "site-packages/ignored.py").write_text("MUTABLE = 1\n")
    (Path(base_prefix) / "lib/libpython3.11.so").write_bytes(b"fixture libpython")
    (Path(base_prefix) / "lib/libsystem-runtime.so").write_bytes(b"fixture system")
    state = {"freeze": "demo-pkg==1.0\npip==24.0\n"}

    def fake_inspect(path: str | Path) -> dict[str, object]:
        candidate = Path(path)
        if candidate.resolve(strict=True) == base_python.resolve(strict=True):
            return _metadata(
                candidate,
                prefix=base_prefix,
                base_prefix=base_prefix,
            )
        return _metadata(
            candidate,
            prefix=str(output),
            base_prefix=base_prefix,
        )

    def fake_run(
        command: list[str],
        *,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output
        if "venv" in command:
            target = Path(command[-1])
            _write_executable(target / "bin/python", b"fixture copied python")
            (target / "lib").mkdir()
            (target / "lib/site.py").write_text("VALUE = 1\n")
            cache = target / "lib/__pycache__"
            cache.mkdir()
            (cache / "site.cpython-311.pyc").write_bytes(b"bytecode")
            os.symlink("lib", target / "lib64")
        stdout = state["freeze"] if "freeze" in command else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(build_venv, "inspect_python", fake_inspect)
    monkeypatch.setattr(build_venv, "_run", fake_run)
    return {
        "repo": repo,
        "wheelhouse": wheelhouse,
        "output": output,
        "base_python": base_python,
        "base_extraction_receipt": extraction_receipt,
        "base_package_id": base_package_id,
        "base_manifest_sha256": base_manifest_sha256,
        "state": state,
    }


def _build(runtime: dict) -> dict[str, object]:
    return build_venv.build(
        repo=runtime["repo"],
        wheelhouse=runtime["wheelhouse"],
        output=runtime["output"],
        base_python=runtime["base_python"],
        base_extraction_receipt=runtime["base_extraction_receipt"],
        expected_base_payload_package_id=runtime["base_package_id"],
        expected_base_payload_manifest_sha256=runtime["base_manifest_sha256"],
    )


def _verify(runtime: dict, payload: dict[str, object]) -> dict[str, object]:
    root = runtime["output"]
    return build_venv.verify(
        repo=runtime["repo"],
        venv_root=root,
        base_python=runtime["base_python"],
        wheelhouse=runtime["wheelhouse"],
        base_extraction_receipt=runtime["base_extraction_receipt"],
        expected_venv_sha256=payload["venv"]["tree"]["sha256"],
        expected_receipt_sha256=sha256_file(build_venv.receipt_path(root)),
        expected_base_python_sha256=payload["base_python"]["executable_sha256"],
        expected_base_python_runtime_sha256=payload["base_python"]["runtime"]["sha256"],
        expected_wheelhouse_sha256=payload["wheelhouse"]["identity"]["sha256"],
        expected_base_extraction_receipt_sha256=payload["wheelhouse"]["base_extraction"]["sha256"],
        expected_base_payload_package_id=runtime["base_package_id"],
        expected_base_payload_manifest_sha256=runtime["base_manifest_sha256"],
    )


def test_build_seals_and_verifies_native_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    payload = _build(runtime)
    root = runtime["output"]

    assert payload["kind"] == build_venv.RECEIPT_KIND
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    assert stat.S_IMODE((root / "bin/python").stat().st_mode) == 0o555
    assert stat.S_IMODE((root / "lib/site.py").stat().st_mode) == 0o444
    assert not (root / "lib/__pycache__").exists()
    assert os.readlink(root / "lib64") == "lib"
    assert stat.S_IMODE(build_venv.checksum_path(root).stat().st_mode) == 0o444
    assert stat.S_IMODE(build_venv.receipt_path(root).stat().st_mode) == 0o444
    assert _verify(runtime, payload) == payload
    assert build_venv.verify(
        environment_lock=runtime["repo"] / "locks/env-v100node.txt",
        build_receipt=build_venv.receipt_path(root),
        venv_root=root,
        base_python=runtime["base_python"],
        wheelhouse=runtime["wheelhouse"],
        base_extraction_receipt=runtime["base_extraction_receipt"],
        expected_venv_sha256=payload["venv"]["tree"]["sha256"],
        expected_receipt_sha256=sha256_file(build_venv.receipt_path(root)),
        expected_base_python_sha256=payload["base_python"]["executable_sha256"],
        expected_base_python_runtime_sha256=payload["base_python"]["runtime"]["sha256"],
        expected_wheelhouse_sha256=payload["wheelhouse"]["identity"]["sha256"],
        expected_base_extraction_receipt_sha256=payload["wheelhouse"]["base_extraction"]["sha256"],
        expected_base_payload_package_id=runtime["base_package_id"],
        expected_base_payload_manifest_sha256=runtime["base_manifest_sha256"],
    ) == payload


def test_build_rejects_base_extraction_format_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    receipt = runtime["base_extraction_receipt"]
    receipt_payload = __import__("json").loads(receipt.read_text())
    receipt_payload["format_version"] = 2
    receipt.write_text(__import__("json").dumps(receipt_payload, sort_keys=True))

    with pytest.raises(RuntimeError, match="format version mismatch"):
        _build(runtime)
    assert not runtime["output"].exists()


def test_build_failure_at_receipt_last_cleans_only_new_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    events: list[str] = []
    real_write_text = build_venv.atomic_write_text

    def tracked_write_text(path: Path, text: str) -> None:
        events.append("checksum")
        real_write_text(path, text)

    def fail_receipt(_path: Path, _payload: object) -> None:
        events.append("receipt")
        raise RuntimeError("simulated receipt publication failure")

    monkeypatch.setattr(build_venv, "atomic_write_text", tracked_write_text)
    monkeypatch.setattr(build_venv, "atomic_write_json", fail_receipt)
    with pytest.raises(RuntimeError, match="publication failure"):
        _build(runtime)

    root = runtime["output"]
    assert events == ["checksum", "receipt"]
    assert not root.exists()
    assert not build_venv.checksum_path(root).exists()
    assert not build_venv.receipt_path(root).exists()
    assert runtime["base_python"].is_file()
    assert runtime["wheelhouse"].is_dir()


def test_verify_rejects_venv_tree_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    payload = _build(runtime)
    root = runtime["output"]
    root.chmod(0o755)
    target = root / "lib/site.py"
    target.chmod(0o644)
    target.write_text("VALUE = 2\n")

    with pytest.raises(RuntimeError, match="root must be mode 0555|tree digest mismatch"):
        _verify(runtime, payload)


def test_verify_rejects_base_python_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    payload = _build(runtime)
    runtime["base_python"].write_bytes(b"changed base interpreter")
    runtime["base_python"].chmod(0o755)

    with pytest.raises(RuntimeError, match="base Python provenance mismatch"):
        _verify(runtime, payload)


def test_verify_rejects_base_runtime_stdlib_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    payload = _build(runtime)
    stdlib_file = runtime["base_python"].parent.parent / "lib/python3.11/site.py"
    stdlib_file.write_text("BASE_SITE = 2\n")

    with pytest.raises(RuntimeError, match="base Python provenance mismatch"):
        _verify(runtime, payload)


def test_verify_rejects_libpython_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    payload = _build(runtime)
    libpython = runtime["base_python"].parent.parent / "lib/libpython3.11.so"
    libpython.write_bytes(b"changed libpython")

    with pytest.raises(RuntimeError, match="base Python provenance mismatch"):
        _verify(runtime, payload)


def test_verify_rejects_mapped_system_library_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    payload = _build(runtime)
    system_library = runtime["base_python"].parent.parent / "lib/libsystem-runtime.so"
    system_library.write_bytes(b"changed system runtime")

    with pytest.raises(RuntimeError, match="base Python provenance mismatch"):
        _verify(runtime, payload)


def test_verify_rejects_wheelhouse_or_extraction_receipt_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    payload = _build(runtime)
    wheel = runtime["wheelhouse"] / "demo_pkg-1.0-py3-none-any.whl"
    wheel.write_bytes(b"changed wheel")
    with pytest.raises(RuntimeError, match="wheelhouse|extraction"):
        _verify(runtime, payload)


def test_verify_rejects_freeze_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _fixture_runtime(tmp_path, monkeypatch)
    payload = _build(runtime)
    runtime["state"]["freeze"] = "demo-pkg==2.0\npip==24.0\n"

    with pytest.raises(RuntimeError, match="does not match exact environment lock"):
        _verify(runtime, payload)


@pytest.mark.parametrize("target", ["../outside", "/absolute/outside"])
def test_tree_manifest_rejects_unsafe_symlink(tmp_path: Path, target: str) -> None:
    root = tmp_path / "venv"
    root.mkdir()
    os.symlink(target, root / "unsafe")

    with pytest.raises(RuntimeError, match="must be safe and relative"):
        build_venv.tree_manifest(root)


def test_freeze_requires_exact_lock_and_only_bootstrap_extras() -> None:
    lock = "Demo_Pkg==1.0\nsecond==2\n"
    assert build_venv.assert_freeze_matches_lock(
        lock,
        "demo-pkg==1.0\nSECOND==2\npip==24.0\n",
    ) == {"demo-pkg": "1.0", "second": "2"}

    with pytest.raises(RuntimeError, match="unexpected packages"):
        build_venv.assert_freeze_matches_lock(lock, "demo-pkg==1.0\nsecond==2\nrogue==3\n")
