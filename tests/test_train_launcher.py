"""Process-boundary guards for the public strict-FP32 training launcher."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.runtime import lightning_contract, precision, train, train_child


class _ExecCalled(RuntimeError):
    pass


def test_launcher_exports_driver_policy_and_replaces_stale_sentinel(monkeypatch):
    observed: dict[str, object] = {}

    def fake_execvpe(executable, command, environment):
        observed.update(
            executable=executable,
            command=command,
            environment=environment,
        )
        raise _ExecCalled

    monkeypatch.setenv(precision.STRICT_SENTINEL, "1")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "unexpected")
    monkeypatch.setattr(train.os, "execvpe", fake_execvpe)

    with pytest.raises(_ExecCalled):
        train.main(["--init", "sarmae_b"])

    assert observed["executable"] == sys.executable
    assert observed["command"] == [
        sys.executable,
        "-m",
        "src.runtime.train_child",
        "--init",
        "sarmae_b",
    ]
    environment = observed["environment"]
    assert isinstance(environment, dict)
    assert environment["NVIDIA_TF32_OVERRIDE"] == "0"
    assert precision.STRICT_SENTINEL not in environment


def test_child_applies_then_asserts_policy_before_imported_trainer(monkeypatch):
    events: list[str] = []
    fake_finetune = ModuleType("src.train.finetune")

    def fake_finetune_main(argv):
        events.append("finetune")
        assert argv == ["--help"]
        return 19

    fake_finetune.main = fake_finetune_main
    monkeypatch.setitem(sys.modules, "src.train.finetune", fake_finetune)
    monkeypatch.setattr(
        precision,
        "apply_strict_fp32",
        lambda: events.append("apply"),
    )
    monkeypatch.setattr(
        lightning_contract,
        "assert_launch_process_contract",
        lambda: events.append("assert"),
    )

    assert train_child.main(["--help"]) == 19
    assert events == ["apply", "assert", "finetune"]


def test_direct_finetune_module_rejects_missing_startup_marker():
    repo = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop(precision.STRICT_SENTINEL, None)
    environment.pop("NVIDIA_TF32_OVERRIDE", None)
    completed = subprocess.run(
        [sys.executable, "-m", "src.train.finetune", "--help"],
        cwd=repo,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "strict-FP32 startup hook was not loaded" in completed.stderr


def test_public_entrypoints_use_strict_launcher():
    repo = Path(__file__).resolve().parents[1]
    pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
    readme = (repo / "README.md").read_text(encoding="utf-8")
    child = (repo / "src/runtime/train_child.py").read_text(encoding="utf-8")
    assert 'xview3-train = "src.runtime.train:main"' in pyproject
    assert "python -m src.runtime.train" in readme
    assert "python -m src.train.finetune \\" not in readme
    assert child.index("apply_strict_fp32()") < child.index(
        "assert_launch_process_contract()"
    )
    assert child.index("assert_launch_process_contract()") < child.index(
        "from src.train.finetune"
    )
