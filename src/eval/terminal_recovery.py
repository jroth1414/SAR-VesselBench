"""Owner-approved, checkpoint-bound exception for two terminal DEV recoveries."""
from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping
from pathlib import Path, PosixPath

from src.eval.result_contract import (
    ResultContractError, checkpoint_path_from_marker, sha256_file,
    read_lightning_checkpoint_epoch, validate_dev_result,
)

RECEIPT_NAME = "terminal_recovery.json"
RECEIPT_FIELDS = frozenset({
    "schema", "status", "exp_id", "campaign_git_sha", "input_checkpoint",
    "input_global_step", "output_global_step", "added_training_batches",
    "added_optimizer_steps", "added_scheduler_steps", "last_dev_before",
    "prior_best_dev", "prior_best_checkpoint", "terminal_dev", "selected_dev",
    "selected_checkpoint", "terminal_execution", "activation_sha256",
    "loss_unavailable_reason", "metrics_csv_absent", "terminal_model_unchanged",
    "runtime_seconds",
})
LOSS_REASON = "terminal-resume-did-not-persist-train-loss"


def require(condition, message):
    if not condition:
        raise ResultContractError("terminal recovery: " + message)


def checkpoint_hashes():
    from scripts.handoff.terminal_contract import CHECKPOINT_SHA256
    return CHECKPOINT_SHA256


def select_terminal(prior, terminal):
    """Match DEV's strict comparison and Lightning's FP32 monitor; reject drift."""
    def fp32(value):
        return struct.unpack("f", struct.pack("f", value))[0]
    improved = terminal["f1"] > prior["f1"]
    require(improved == (fp32(terminal["f1"]) > fp32(prior["f1"])),
            "DEV and checkpoint monitor disagree after FP32 rounding")
    return improved


def validate_saved_state(saved, *, candidate_floor):
    """Prove the exact interrupted epoch boundary without changing loop state."""
    from scripts.handoff.terminal_contract import POLICY
    epoch, step = POLICY["epoch"], POLICY["global_step"]
    require(isinstance(saved, Mapping), "checkpoint root is not a mapping")
    require(type(saved.get("epoch")) is int and saved["epoch"] == epoch
            and type(saved.get("global_step")) is int and saved["global_step"] == step,
            "checkpoint is not the approved terminal step")
    fit = saved.get("loops", {}).get("fit_loop", {})
    for scope in ("current", "total"):
        require(fit.get("epoch_progress", {}).get(scope) == {
            "ready": epoch + 1, "started": epoch + 1, "processed": epoch + 1, "completed": epoch},
            "epoch progress differs from the approved interruption")
        count = step // (epoch + 1) if scope == "current" else step
        require(fit.get("epoch_loop.batch_progress", {}).get(scope) == {
            key: count for key in ("ready", "started", "processed", "completed")},
            "training batches remain unfinished")
    require(fit.get("epoch_loop.batch_progress", {}).get("is_last_batch") is True,
            "checkpoint is not at the final batch")
    callbacks = saved.get("callbacks", {})
    dev = callbacks.get("DevSceneEval", {})
    last = validate_dev_result(dev.get("last_result"), candidate_floor=candidate_floor)
    best = validate_dev_result(dev.get("best_result"), candidate_floor=candidate_floor)
    require(last["epoch"] == POLICY["last_completed_dev_epoch"] and best["epoch"] <= last["epoch"]
            and best["f1"] >= last["f1"] and dev.get("best") == best["f1"],
            "saved DEV selection state differs from the interruption")
    selection = [v for k, v in callbacks.items() if str(k).startswith("ModelCheckpoint{")]
    require(len(selection) == 1, "expected exactly one checkpoint-selection callback")
    score = selection[0].get("best_model_score")
    expected_score = struct.unpack("f", struct.pack("f", best["f1"]))[0]
    require(score is not None and float(score) == expected_score,
            "checkpoint monitor does not match saved best DEV")
    require(bool(saved.get("state_dict")) and bool(saved.get("optimizer_states"))
            and bool(saved.get("lr_schedulers")), "saved training state is incomplete")
    return last, best, selection[0]


def load_restricted(path):
    import torch
    with torch.serialization.safe_globals([PosixPath]):
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def validate_receipt_payload(receipt, marker, *, candidate_floor):
    """Pure schema/selection check, also used by archive-bound receivers."""
    from scripts.handoff.terminal_contract import POLICY
    require(isinstance(receipt, Mapping) and set(receipt) == RECEIPT_FIELDS,
            "receipt fields are not exact")
    require(type(receipt["schema"]) is int and receipt["schema"] == 1
            and receipt["status"] == "terminal-dev-recovered", "receipt schema/status is invalid")
    from scripts.h100.lightning_contract import validate_trainer_contract_evidence
    try:
        validate_trainer_contract_evidence(marker.get("h100_runtime_contract"))
    except RuntimeError as exc:
        raise ResultContractError("terminal recovery requires measured H100 runtime evidence") from exc
    cell = marker.get("exp_id")
    require(cell in checkpoint_hashes() and receipt["exp_id"] == cell,
            "cell is outside the owner-approved exception")
    require(receipt["campaign_git_sha"] == marker.get("git_sha") == POLICY["campaign_git_sha"],
            "campaign identity changed")
    require(marker.get("train_loss") is None and receipt["loss_unavailable_reason"] == LOSS_REASON
            and receipt["metrics_csv_absent"] is True and receipt["terminal_model_unchanged"] is True,
            "missing-loss exception lacks its evidence")
    expected_input = {"relative_path": "checkpoints/last.ckpt",
                      "sha256": checkpoint_hashes()[cell], "epoch": 49}
    require(receipt["input_checkpoint"] == expected_input, "terminal input checkpoint changed")
    for field in ("input_global_step", "output_global_step"):
        require(type(receipt[field]) is int and receipt[field] == 35050, "global step changed")
    for field in ("added_training_batches", "added_optimizer_steps", "added_scheduler_steps"):
        require(type(receipt[field]) is int and receipt[field] == 0, "additional training is forbidden")
    require(type(marker.get("epochs_run")) is int and marker["epochs_run"] == 50,
            "epoch count differs from the approved limit")
    results = {name: validate_dev_result(receipt[name], candidate_floor=candidate_floor)
               for name in ("last_dev_before", "prior_best_dev", "terminal_dev", "selected_dev")}
    prior, terminal = results["prior_best_dev"], results["terminal_dev"]
    require(results["last_dev_before"]["epoch"] == 44 and prior["epoch"] <= 44
            and prior["f1"] >= results["last_dev_before"]["f1"] and terminal["epoch"] == 49,
            "DEV epoch sequence is invalid")
    old = receipt["prior_best_checkpoint"]
    require(isinstance(old, Mapping) and set(old) == {"relative_path", "sha256", "epoch"}
            and old["relative_path"] == "checkpoints/best.ckpt" and old["epoch"] == prior["epoch"],
            "prior best checkpoint binding is invalid")
    improved = select_terminal(prior, terminal)
    expected_selected = ({**expected_input, "relative_path": "checkpoints/terminal-best.ckpt"}
                         if improved else dict(old))
    require(receipt["selected_checkpoint"] == marker.get("best_checkpoint") == expected_selected
            and receipt["selected_dev"] == marker.get("best_dev") == (terminal if improved else prior)
            and marker.get("last_dev") == terminal,
            "result marker differs from the existing selection policy")
    from scripts.handoff.terminal_contract import validate_execution_identity
    validate_execution_identity(receipt["terminal_execution"])
    import re
    require(re.fullmatch(r"[0-9a-f]{64}", str(receipt["activation_sha256"])) is not None
            and re.fullmatch(r"[0-9a-f]{64}", str(old["sha256"])) is not None,
            "receipt hash binding is invalid")
    seconds = receipt["runtime_seconds"]
    require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
            "finalization runtime is invalid")
    return receipt


def validate_terminal_receipt(receipt, marker, *, run_dir, candidate_floor):
    validate_receipt_payload(receipt, marker, candidate_floor=candidate_floor)
    from scripts.h100.terminal_recovery import validate_receipt_authorization
    try:
        validate_receipt_authorization(Path(run_dir), receipt)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        raise ResultContractError("terminal recovery authorization does not match immutable controls") from exc
    root = Path(run_dir)
    for binding in (receipt["input_checkpoint"], receipt["prior_best_checkpoint"]):
        checkpoint = checkpoint_path_from_marker(root, binding["relative_path"])
        require(sha256_file(checkpoint) == binding["sha256"], "preserved checkpoint bytes changed")
        require(read_lightning_checkpoint_epoch(checkpoint, restricted=True) == binding["epoch"],
                "preserved checkpoint epoch changed")
    saved = load_restricted(root / "checkpoints/last.ckpt")
    last, best, selection = validate_saved_state(saved, candidate_floor=candidate_floor)
    require(last == receipt["last_dev_before"] and best == receipt["prior_best_dev"],
            "receipt prior selection differs from the actual saved callback")
    # Persistent absolute paths in the original checkpoint do not follow export relocation.
    require(Path(selection["best_model_path"]).name == "best.ckpt",
            "saved checkpoint selection path is unexpected")
    return receipt


def load_terminal_receipt(marker, *, run_dir, candidate_floor):
    reference = marker.get("terminal_recovery")
    require(isinstance(reference, Mapping) and set(reference) == {"relative_path", "sha256"}
            and reference["relative_path"] == RECEIPT_NAME, "receipt reference is invalid")
    path = Path(run_dir) / RECEIPT_NAME
    require(path.resolve() == path.absolute() and path.is_file()
            and not path.stat().st_mode & 0o222, "receipt must be canonical and immutable")
    require(sha256_file(path) == reference["sha256"], "receipt SHA-256 changed")
    try:
        receipt = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ResultContractError("terminal recovery receipt is invalid JSON") from exc
    return validate_terminal_receipt(receipt, marker, run_dir=run_dir, candidate_floor=candidate_floor)
