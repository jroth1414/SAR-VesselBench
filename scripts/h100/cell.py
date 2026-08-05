"""Run one explicit H100 cell phase behind the all-training cohort barrier."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

from scripts.h100.contracts import (
    EXPECTED_PRECISION,
    atomic_write_json,
    atomic_write_text,
    load_cells,
)
from scripts.h100.lightning_contract import (
    assert_launch_process_contract,
    validate_trainer_contract_evidence,
)
from scripts.h100.precision import assert_sitecustomize_active
from src.eval.heldout_contract import (
    TEST_RESULT_FILENAME,
    validate_test_result,
    validate_training_cohort_cell,
)
from src.eval.result_contract import ResultContractError, load_completion_marker

PREEMPTED_EXIT_CODE = 75


def _request_requeue() -> None:
    request_dir = Path(os.environ["H100_REQUEUE_REQUEST_DIR"])
    gpu = os.environ["CUDA_VISIBLE_DEVICES"]
    atomic_write_text(request_dir / f"gpu-{gpu}.request", os.environ["SLURM_JOB_ID"] + "\n")


def _expected_recipe(args: argparse.Namespace) -> dict[str, object]:
    return {
        "exp_id": args.exp_id,
        "git_sha": args.git_sha,
        "detector_sha256": args.detector_sha256,
        "precision": EXPECTED_PRECISION,
        "micro_batch": 16,
        "gradient_accumulation": 1,
        "effective_batch": 16,
    }


def _validate_training(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    seal: bool,
) -> dict[str, object]:
    marker = run_dir / "final_metrics.json"
    try:
        payload, _checkpoint = load_completion_marker(
            marker,
            candidate_floor=args.candidate_floor,
            expected_recipe=_expected_recipe(args),
        )
    except ResultContractError as exc:
        raise RuntimeError(
            f"{args.exp_id}: invalid schema-2 training marker: {exc}"
        ) from exc
    try:
        validate_trainer_contract_evidence(payload.get("h100_runtime_contract"))
    except RuntimeError as exc:
        raise RuntimeError(
            f"{args.exp_id}: invalid H100 trainer evidence: {exc}"
        ) from exc
    last = run_dir / "checkpoints/last.ckpt"
    if last.is_symlink() or not last.is_file() or last.stat().st_size <= 0:
        raise RuntimeError(f"{args.exp_id}: training marker lacks durable last.ckpt")
    if seal:
        marker.chmod(0o444)
    if stat.S_IMODE(marker.stat().st_mode) & 0o222:
        raise RuntimeError(f"{args.exp_id}: training marker is not immutable")
    return payload


def _validated_cohort(
    args: argparse.Namespace,
    *,
    repo: Path,
    runs_root: Path,
) -> tuple[dict[str, object], str]:
    cohort_path = args.training_cohort
    if cohort_path is None:
        raise RuntimeError("score-test phase requires the canonical training cohort")
    cohort, cohort_sha256, _record, _training = validate_training_cohort_cell(
        path=cohort_path,
        expected_sha256=args.training_cohort_sha256,
        cells=load_cells(repo),
        runs_root=runs_root,
        git_sha=args.git_sha,
        detector_sha256=args.detector_sha256,
        candidate_floor=args.candidate_floor,
        exp_id=args.exp_id,
    )
    if cohort_sha256 != args.training_cohort_sha256:
        raise RuntimeError(
            "score-test cohort SHA-256 differs from the controller binding"
        )
    return cohort, cohort_sha256


def _validate_scored(
    run_dir: Path,
    args: argparse.Namespace,
    *,
    repo: Path,
    cohort: dict[str, object],
    cohort_sha256: str,
) -> dict[str, object]:
    try:
        splits = json.loads((repo / "data/splits.json").read_text(encoding="utf-8"))
        test_scene_ids = tuple(sorted(map(str, splits["splits"]["test"])))
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("frozen TEST scene IDs are absent or invalid") from exc
    return validate_test_result(
        path=run_dir / TEST_RESULT_FILENAME,
        exp_id=args.exp_id,
        cohort=cohort,
        cohort_sha256=cohort_sha256,
        test_scene_ids=test_scene_ids,
    )


def run_cell(args: argparse.Namespace) -> int:
    assert_sitecustomize_active()
    assert_launch_process_contract()
    if args.phase not in {"train", "score-test"}:
        raise RuntimeError(f"unsupported H100 cell phase: {args.phase!r}")
    repo = args.repo.resolve()
    runs_root = args.runs_root.resolve()
    run_dir = runs_root / args.exp_id
    state_path = run_dir / "cell_wrapper.json"
    test_path = run_dir / TEST_RESULT_FILENAME
    active: dict[str, object] = {
        "process": None,
        "phase": args.phase,
        "preempted": False,
    }

    cohort: dict[str, object] | None = None
    cohort_sha256: str | None = None
    if args.phase == "train":
        if test_path.exists() or test_path.is_symlink():
            raise RuntimeError(
                f"{args.exp_id}: test result exists before the training cohort"
            )
    else:
        cohort, cohort_sha256 = _validated_cohort(
            args,
            repo=repo,
            runs_root=runs_root,
        )

    def request_if_completed_during_signal_race() -> None:
        request = Path(os.environ["H100_REQUEUE_REQUEST_DIR"]) / (
            f"gpu-{os.environ['CUDA_VISIBLE_DEVICES']}.request"
        )
        if request.is_file() and not request.is_symlink():
            return
        if args.phase == "train":
            _validate_training(run_dir, args, seal=True)
            completed_phase = "train-complete"
        else:
            assert cohort is not None and cohort_sha256 is not None
            _validate_scored(
                run_dir,
                args,
                repo=repo,
                cohort=cohort,
                cohort_sha256=cohort_sha256,
            )
            completed_phase = "score-test-complete"
        active["phase"] = completed_phase
        state = {"phase": completed_phase, "exp_id": args.exp_id}
        if cohort_sha256 is not None:
            state["training_cohort_sha256"] = cohort_sha256
        atomic_write_json(state_path, state)
        _request_requeue()

    def on_usr1(_signum, _frame) -> None:
        active["preempted"] = True
        process = active["process"]
        if args.phase == "train" and process is not None and process.poll() is None:
            try:
                os.kill(process.pid, signal.SIGUSR1)
                return
            except ProcessLookupError:
                pass
        if args.phase == "score-test":
            atomic_write_json(
                state_path,
                {
                    "phase": "score-test",
                    "exp_id": args.exp_id,
                    "training_cohort_sha256": cohort_sha256,
                },
            )
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
        if args.phase == "train":
            final_path = run_dir / "final_metrics.json"
            if final_path.exists() or final_path.is_symlink():
                _validate_training(run_dir, args, seal=True)
                atomic_write_json(
                    state_path,
                    {"phase": "train-complete", "exp_id": args.exp_id},
                )
                return 0
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
            atomic_write_json(
                state_path,
                {"phase": "train", "exp_id": args.exp_id},
            )
            active["process"] = subprocess.Popen(train, cwd=repo)
            code = active["process"].wait()
            if active["preempted"]:
                if code == 0:
                    request_if_completed_during_signal_race()
                return PREEMPTED_EXIT_CODE
            if code != 0:
                return code
            _validate_training(run_dir, args, seal=True)
            atomic_write_json(
                state_path,
                {"phase": "train-complete", "exp_id": args.exp_id},
            )
            return 0

        assert cohort is not None and cohort_sha256 is not None
        _validate_training(run_dir, args, seal=False)
        if test_path.exists() or test_path.is_symlink():
            _validate_scored(
                run_dir,
                args,
                repo=repo,
                cohort=cohort,
                cohort_sha256=cohort_sha256,
            )
            atomic_write_json(
                state_path,
                {
                    "phase": "score-test-complete",
                    "exp_id": args.exp_id,
                    "training_cohort_sha256": cohort_sha256,
                },
            )
            return 0
        atomic_write_json(
            state_path,
            {
                "phase": "score-test",
                "exp_id": args.exp_id,
                "training_cohort_sha256": cohort_sha256,
            },
        )
        score = [
            sys.executable,
            str(repo / "scripts/score_test_split.py"),
            "--repo",
            str(repo),
            "--runs-root",
            str(runs_root),
            "--cohort",
            str(args.training_cohort),
            "--cohort-sha256",
            cohort_sha256,
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
        _validate_scored(
            run_dir,
            args,
            repo=repo,
            cohort=cohort,
            cohort_sha256=cohort_sha256,
        )
        atomic_write_json(
            state_path,
            {
                "phase": "score-test-complete",
                "exp_id": args.exp_id,
                "training_cohort_sha256": cohort_sha256,
            },
        )
        return 0
    finally:
        signal.signal(signal.SIGUSR1, prior_handler)
        signal.signal(signal.SIGINT, prior_int)
        signal.signal(signal.SIGTERM, prior_term)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "score-test"), required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--init", required=True)
    parser.add_argument("--label-frac", type=float, required=True)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--detector-sha256", required=True)
    parser.add_argument("--candidate-floor", type=float, required=True)
    parser.add_argument("--training-cohort", type=Path)
    parser.add_argument("--training-cohort-sha256")
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.phase == "score-test" and (
        args.training_cohort is None or args.training_cohort_sha256 is None
    ):
        parser.error("score-test requires --training-cohort and its SHA-256")
    if args.phase == "train" and (
        args.training_cohort is not None or args.training_cohort_sha256 is not None
    ):
        parser.error("train phase must not receive a training cohort")
    return run_cell(args)


if __name__ == "__main__":
    raise SystemExit(main())
