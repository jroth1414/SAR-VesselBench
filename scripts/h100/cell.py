"""Run one unchanged fine-tune followed by its frozen test-split scoring."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

from scripts.h100.contracts import EXPECTED_PRECISION, atomic_write_json, atomic_write_text
from scripts.h100.lightning_contract import (
    assert_launch_process_contract,
    validate_trainer_contract_evidence,
)
from scripts.h100.precision import assert_sitecustomize_active

PREEMPTED_EXIT_CODE = 75


def _request_requeue() -> None:
    request_dir = Path(os.environ["H100_REQUEUE_REQUEST_DIR"])
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    atomic_write_text(request_dir / f"gpu-{gpu}.request", os.environ["SLURM_JOB_ID"] + "\n")


def _validate_scored(run_dir: Path, exp_id: str) -> dict:
    payload = json.loads((run_dir / "final_metrics.json").read_text())
    expected = {
        "exp_id": exp_id,
        "precision": EXPECTED_PRECISION,
        "test_inference_precision": EXPECTED_PRECISION,
    }
    mismatches = {
        key: (value, payload.get(key))
        for key, value in expected.items()
        if payload.get(key) != value
    }
    for key in ("test_f1", "test_precision", "test_recall", "test_near_shore_f1"):
        try:
            finite = math.isfinite(float(payload[key]))
        except (KeyError, TypeError, ValueError):
            finite = False
        if not finite:
            mismatches[key] = ("finite", payload.get(key))
    for name in ("best.ckpt", "last.ckpt"):
        checkpoint = run_dir / "checkpoints" / name
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            mismatches[name] = ("nonempty checkpoint", None)
    try:
        validate_trainer_contract_evidence(payload.get("h100_runtime_contract"))
    except RuntimeError as exc:
        mismatches["h100_runtime_contract"] = ("valid", str(exc))
    if mismatches:
        raise RuntimeError(f"{exp_id}: incomplete train+test cell: {mismatches}")
    return payload


def run_cell(args: argparse.Namespace) -> int:
    assert_sitecustomize_active()
    assert_launch_process_contract()
    repo = args.repo.resolve()
    runs_root = args.runs_root.resolve()
    run_dir = runs_root / args.exp_id
    state_path = run_dir / "cell_wrapper.json"
    active: dict[str, object] = {"process": None, "phase": None, "preempted": False}

    def request_if_completed_during_signal_race() -> None:
        request = Path(os.environ["H100_REQUEUE_REQUEST_DIR"]) / (
            f"gpu-{os.environ['CUDA_VISIBLE_DEVICES']}.request"
        )
        if request.is_file() and not request.is_symlink():
            return
        best = run_dir / "checkpoints/best.ckpt"
        last = run_dir / "checkpoints/last.ckpt"
        if all(path.is_file() and path.stat().st_size > 0 for path in (best, last)):
            active["phase"] = "score-test"
            atomic_write_json(
                state_path, {"phase": "score-test", "exp_id": args.exp_id}
            )
            _request_requeue()

    def on_usr1(_signum, _frame) -> None:
        active["preempted"] = True
        process = active["process"]
        if (
            active["phase"] == "train"
            and process is not None
            and process.poll() is None
        ):
            try:
                os.kill(process.pid, signal.SIGUSR1)
                return
            except ProcessLookupError:
                pass
        # Scoring is restartable from the already-durable best/last pair; it
        # has no Lightning loop that could issue the deferred request itself.
        best = run_dir / "checkpoints/best.ckpt"
        last = run_dir / "checkpoints/last.ckpt"
        if (
            not best.is_file()
            or best.stat().st_size <= 0
            or not last.is_file()
            or last.stat().st_size <= 0
        ):
            return
        active["phase"] = "score-test"
        atomic_write_json(state_path, {"phase": "score-test", "exp_id": args.exp_id})
        _request_requeue()
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    def on_stop(signum, _frame) -> None:
        process = active["process"]
        if process is not None and process.poll() is None:
            process.send_signal(signum)

    prior_handler = signal.signal(signal.SIGUSR1, on_usr1)
    prior_int = signal.signal(signal.SIGINT, on_stop)
    prior_term = signal.signal(signal.SIGTERM, on_stop)
    try:
        final_path = run_dir / "final_metrics.json"
        if not final_path.exists():
            train = [
                sys.executable,
                "-m",
                "src.train.finetune",
                "--init",
                args.init,
                "--label_frac",
                str(args.label_frac),
                "--seed",
                "0",
                "--git-sha",
                args.git_sha,
                "--workers",
                str(args.workers),
                "--micro-batch",
                "16",
            ]
            atomic_write_json(state_path, {"phase": "train", "exp_id": args.exp_id})
            active["phase"] = "train"
            active["process"] = subprocess.Popen(train, cwd=repo)
            code = active["process"].wait()
            if active["preempted"]:
                if code == 0:
                    request_if_completed_during_signal_race()
                return PREEMPTED_EXIT_CODE
            if code != 0:
                return code

        atomic_write_json(state_path, {"phase": "score-test", "exp_id": args.exp_id})
        active["phase"] = "score-test"
        score = [
            sys.executable,
            str(repo / "scripts/score_test_split.py"),
            "--runs-root",
            str(runs_root),
            "--only",
            args.exp_id,
            "--device",
            "cuda",
        ]
        active["process"] = subprocess.Popen(score, cwd=repo)
        code = active["process"].wait()
        if active["preempted"]:
            if code == 0:
                request_if_completed_during_signal_race()
            return PREEMPTED_EXIT_CODE
        if code != 0:
            return code
        _validate_scored(run_dir, args.exp_id)
        atomic_write_json(state_path, {"phase": "complete", "exp_id": args.exp_id})
        return 0
    finally:
        signal.signal(signal.SIGUSR1, prior_handler)
        signal.signal(signal.SIGINT, prior_int)
        signal.signal(signal.SIGTERM, prior_term)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--label-frac", type=float, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--workers", type=int, required=True)
    return run_cell(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
