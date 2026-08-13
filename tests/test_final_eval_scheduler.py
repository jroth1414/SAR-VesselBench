from __future__ import annotations

import json
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest

from src.eval import final_eval
from src.eval.heldout_contract import HeldoutContractError


@dataclass(frozen=True)
class _Cell:
    exp_id: str
    init: str = "fixture-init"
    fraction: float = 0.1
    seed: int = 0


class _InputPipe:
    def __init__(self, expected: bytes, *, fail_write: bool = False) -> None:
        self.expected = expected
        self.fail_write = fail_write
        self.closed = False
        self.writes: list[bytes] = []

    def write(self, value: bytes) -> int:
        self.writes.append(value)
        if self.fail_write:
            raise BrokenPipeError("fixture pipe failure")
        assert value == self.expected
        return len(value)

    def close(self) -> None:
        self.closed = True


class _Process:
    def __init__(
        self,
        *,
        exp_id: str,
        returns: list[int | None],
        runs_root: Path,
        normalized_gt: bytes,
        on_terminal: Callable[[str], None],
    ) -> None:
        self.exp_id = exp_id
        self.returns = list(returns)
        self.runs_root = runs_root
        self.stdin = _InputPipe(normalized_gt)
        self.on_terminal = on_terminal
        self.terminal = False
        self.poll_calls = 0
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls: list[int | None] = []

    def _publish_success(self) -> None:
        result = (
            self.runs_root
            / self.exp_id
            / final_eval.FINAL_CELL_RESULT_FILENAME
        )
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(
            json.dumps({"exp_id": self.exp_id}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result.chmod(0o444)

    def poll(self) -> int | None:
        self.poll_calls += 1
        value = self.returns.pop(0) if self.returns else 0
        if value is not None and not self.terminal:
            self.terminal = True
            self.on_terminal(self.exp_id)
            if value == 0:
                self._publish_success()
        return value

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, timeout: int | None = None) -> int:
        self.wait_calls.append(timeout)
        return 0


class _BrokenPipeProcess(_Process):
    def __init__(
        self,
        *,
        exp_id: str,
        runs_root: Path,
        normalized_gt: bytes,
        on_terminal: Callable[[str], None],
    ) -> None:
        super().__init__(
            exp_id=exp_id,
            returns=[None],
            runs_root=runs_root,
            normalized_gt=normalized_gt,
            on_terminal=on_terminal,
        )
        self.stdin = _InputPipe(normalized_gt, fail_write=True)
        self._timed_out = False

    def poll(self) -> None:
        self.poll_calls += 1
        return None

    def wait(self, timeout: int | None = None) -> int:
        self.wait_calls.append(timeout)
        if timeout == 30 and not self._timed_out:
            self._timed_out = True
            raise subprocess.TimeoutExpired("fixture-worker", timeout)
        return 0


@dataclass
class _Harness:
    repo: Path
    runs_root: Path
    selected: list[_Cell]
    normalized_gt: bytes
    lock_payload: dict[str, object]
    validated: list[str]


def _harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    repo = tmp_path / "repo"
    runs_root = tmp_path / "runs"
    (repo / "data").mkdir(parents=True)
    (runs_root / ".h100").mkdir(parents=True)
    scenes = [f"final-{index:02d}v" for index in range(50)]
    (repo / "data/splits.json").write_text(
        json.dumps({"splits": {"eval_final": scenes}}),
        encoding="utf-8",
    )
    selected = [_Cell(f"cell-{index:02d}") for index in range(32)]
    normalized_gt = json.dumps(
        {scene: [] for scene in scenes}, sort_keys=True
    ).encode("utf-8")
    lock_payload: dict[str, object] = {
        "lock_schema": 3,
        "owner_amendment": {"sha256": "a" * 64},
        "grid": {
            "monotonicity_ok": False,
            "monotonicity_tolerance": 0.02,
            "violations": [{"fixture": True}],
        },
        "final_data_view": {"sha256": "b" * 64},
        "test_results": [
            {"exp_id": cell.exp_id, "sha256": f"{index:064x}"}
            for index, cell in enumerate(selected)
        ],
        "campaign_git_sha": "c" * 40,
        "evaluator_git_sha": "d" * 40,
        "detector_sha256": "e" * 64,
        "cohort_sha256": "f" * 64,
    }

    def regular_json(path: Path, _description: str) -> dict[str, object]:
        if path.name == "final_eval.lock":
            return lock_payload
        if path.name == final_eval.COHORT_FILENAME:
            return {"fixture": "cohort"}
        return {"exp_id": path.parent.name}

    validated: list[str] = []

    def validate_payload(
        payload: object, *, cell: object, **_arguments: object
    ) -> dict[str, object]:
        assert payload == {"exp_id": cell.exp_id}
        validated.append(str(cell.exp_id))
        return {"exp_id": str(cell.exp_id)}

    monkeypatch.setattr(final_eval, "sha256_file", lambda _path: "9" * 64)
    monkeypatch.setattr(final_eval, "_regular_json", regular_json)
    monkeypatch.setattr(
        final_eval,
        "cohort_record",
        lambda _cohort, exp_id: {"exp_id": exp_id},
    )
    monkeypatch.setattr(final_eval, "_validate_final_cell_payload", validate_payload)
    monkeypatch.setattr(final_eval.time, "sleep", lambda _seconds: None)
    return _Harness(
        repo=repo,
        runs_root=runs_root,
        selected=selected,
        normalized_gt=normalized_gt,
        lock_payload=lock_payload,
        validated=validated,
    )


def _exp_id(command: list[str]) -> str:
    assert command[:4] == [
        final_eval.sys.executable,
        "-B",
        "-m",
        "src.eval.final_worker",
    ]
    return command[command.index("--exp-id") + 1]


def test_success_runs_eight_at_a_time_and_returns_all_cells_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    launched: list[str] = []
    gpu_assignments: list[int] = []
    active: set[str] = set()
    maximum_active = 0
    processes: dict[str, _Process] = {}

    def popen(command: list[str], **arguments: object) -> _Process:
        nonlocal maximum_active
        exp_id = _exp_id(command)
        assert exp_id not in processes
        assert arguments["cwd"] == harness.repo
        assert arguments["stdin"] is subprocess.PIPE
        assert arguments["stderr"] is subprocess.STDOUT
        gpu = int(arguments["env"]["CUDA_VISIBLE_DEVICES"])  # type: ignore[index]
        launched.append(exp_id)
        gpu_assignments.append(gpu)
        active.add(exp_id)
        maximum_active = max(maximum_active, len(active))
        process = _Process(
            exp_id=exp_id,
            returns=[0],
            runs_root=harness.runs_root,
            normalized_gt=harness.normalized_gt,
            on_terminal=active.remove,
        )
        processes[exp_id] = process
        return process

    monkeypatch.setattr(final_eval.subprocess, "Popen", popen)
    results = final_eval._run_parallel_final_workers(
        repo=harness.repo,
        runs_root=harness.runs_root,
        selected=harness.selected,
        normalized_gt_bytes=harness.normalized_gt,
        worker_count=8,
    )

    expected = [cell.exp_id for cell in harness.selected]
    assert launched[:8] == expected[:8]
    assert launched == expected
    assert len(launched) == len(set(launched)) == 32
    assert gpu_assignments == list(range(8)) * 4
    assert maximum_active == 8
    assert harness.validated == expected
    assert results == [{"exp_id": exp_id} for exp_id in expected]
    assert all(process.stdin.closed for process in processes.values())
    assert all(process.stdin.writes == [harness.normalized_gt] for process in processes.values())
    assert all(process.poll_calls == 1 for process in processes.values())
    assert all(
        stat.S_IMODE(
            (harness.runs_root / "logs/h100-final" / f"{exp_id}.log").stat().st_mode
        )
        == 0o444
        for exp_id in expected
    )


@pytest.mark.parametrize("failure_kind", ["nonzero", "invalid-result"])
def test_worker_failure_stops_pending_but_finishes_all_running_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    launched: list[str] = []
    terminal: list[str] = []
    processes: dict[str, _Process] = {}

    def popen(command: list[str], **_arguments: object) -> _Process:
        exp_id = _exp_id(command)
        launched.append(exp_id)
        index = int(exp_id.rsplit("-", 1)[1])
        returns: list[int | None]
        if failure_kind == "nonzero" and index == 0:
            returns = [7]
        elif failure_kind == "nonzero":
            returns = [None, 0]
        else:
            returns = [0]
        process = _Process(
            exp_id=exp_id,
            returns=returns,
            runs_root=harness.runs_root,
            normalized_gt=harness.normalized_gt,
            on_terminal=terminal.append,
        )
        processes[exp_id] = process
        return process

    if failure_kind == "invalid-result":
        original = final_eval._validate_final_cell_payload

        def reject_first(
            payload: object, *, cell: object, **arguments: object
        ) -> dict[str, object]:
            if cell.exp_id == "cell-00":
                raise HeldoutContractError("fixture invalid result")
            return original(payload, cell=cell, **arguments)

        monkeypatch.setattr(final_eval, "_validate_final_cell_payload", reject_first)
    monkeypatch.setattr(final_eval.subprocess, "Popen", popen)

    with pytest.raises(HeldoutContractError, match="permanently incomplete"):
        final_eval._run_parallel_final_workers(
            repo=harness.repo,
            runs_root=harness.runs_root,
            selected=harness.selected,
            normalized_gt_bytes=harness.normalized_gt,
            worker_count=8,
        )

    assert launched == [cell.exp_id for cell in harness.selected[:8]]
    assert terminal == [cell.exp_id for cell in harness.selected[:8]]
    assert all(process.stdin.closed for process in processes.values())
    if failure_kind == "nonzero":
        assert harness.validated == [cell.exp_id for cell in harness.selected[1:8]]
        assert processes["cell-00"].poll_calls == 1
        assert all(processes[f"cell-{index:02d}"].poll_calls == 2 for index in range(1, 8))
    else:
        assert harness.validated == [cell.exp_id for cell in harness.selected[1:8]]
        assert all(process.poll_calls == 1 for process in processes.values())
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444
        for path in (harness.runs_root / "logs/h100-final").glob("*.log")
    )


def test_pipe_failure_after_popen_terminates_kills_and_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    created: list[_BrokenPipeProcess] = []

    def popen(command: list[str], **_arguments: object) -> _BrokenPipeProcess:
        process = _BrokenPipeProcess(
            exp_id=_exp_id(command),
            runs_root=harness.runs_root,
            normalized_gt=harness.normalized_gt,
            on_terminal=lambda _exp_id: None,
        )
        created.append(process)
        return process

    monkeypatch.setattr(final_eval.subprocess, "Popen", popen)
    with pytest.raises(HeldoutContractError, match="permanently incomplete"):
        final_eval._run_parallel_final_workers(
            repo=harness.repo,
            runs_root=harness.runs_root,
            selected=harness.selected,
            normalized_gt_bytes=harness.normalized_gt,
            worker_count=8,
        )

    assert len(created) == 1
    process = created[0]
    assert process.stdin.closed
    assert process.stdin.writes == [harness.normalized_gt]
    assert process.poll_calls == 1
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls == [30, None]
    log = harness.runs_root / "logs/h100-final/cell-00.log"
    assert stat.S_IMODE(log.stat().st_mode) == 0o444


@pytest.mark.parametrize("occupied", ["log", "result"])
def test_preexisting_log_or_result_refuses_process_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied: str,
) -> None:
    harness = _harness(tmp_path, monkeypatch)
    first = harness.selected[0]
    if occupied == "log":
        path = harness.runs_root / "logs/h100-final" / f"{first.exp_id}.log"
    else:
        path = (
            harness.runs_root
            / first.exp_id
            / final_eval.FINAL_CELL_RESULT_FILENAME
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("occupied\n", encoding="utf-8")
    path.chmod(0o444)
    launches: list[str] = []

    def popen(command: list[str], **_arguments: object) -> _Process:
        launches.append(_exp_id(command))
        raise AssertionError("occupied cell must be rejected before Popen")

    monkeypatch.setattr(final_eval.subprocess, "Popen", popen)
    with pytest.raises(HeldoutContractError, match="permanently incomplete"):
        final_eval._run_parallel_final_workers(
            repo=harness.repo,
            runs_root=harness.runs_root,
            selected=harness.selected,
            normalized_gt_bytes=harness.normalized_gt,
            worker_count=8,
        )
    assert launches == []
