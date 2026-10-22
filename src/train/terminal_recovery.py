"""Finish two approved interrupted DEV callbacks, without entering a train loop."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

from src.eval.result_contract import (
    atomic_write_json, sha256_file, validate_completion_payload, validate_dev_result,
)
from src.eval.terminal_recovery import (
    LOSS_REASON, RECEIPT_NAME, load_restricted, require, select_terminal,
    validate_saved_state, validate_terminal_receipt,
)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_once(path, value):
    if os.path.lexists(path):
        require(not path.is_symlink() and path.is_file()
                and not path.stat().st_mode & 0o222
                and json.loads(path.read_text()) == value, "immutable evidence changed")
        return
    fd, name = tempfile.mkstemp(prefix=".terminal-json-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w") as output:
            output.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            os.fchmod(output.fileno(), 0o444)
            os.fsync(output.fileno())
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def copy_once(source, destination, expected_sha):
    if os.path.lexists(destination):
        require(not destination.is_symlink() and not destination.stat().st_mode & 0o222
                and sha256_file(destination) == expected_sha, "selected checkpoint copy changed")
        return
    fd, name = tempfile.mkstemp(prefix=".terminal-copy-", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as source_file:
            shutil.copyfileobj(source_file, output)
            output.flush()
            os.fsync(output.fileno())
        require(sha256_file(temporary) == expected_sha, "checkpoint copy is not byte-identical")
        temporary.chmod(0o444)
        os.link(temporary, destination)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def model_digest(module):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def finish_dev(module, saved, evaluator, *, candidate_floor):
    """Shared LightningModule, exact saved weights, inference only; no optimizer exists."""
    validate_saved_state(saved, candidate_floor=candidate_floor)
    import torch
    module.load_state_dict(saved["state_dict"], strict=True)
    for name, value in module.state_dict().items():
        expected = saved["state_dict"][name]
        require(value.dtype == expected.dtype and torch.equal(value.detach().cpu(), expected.cpu()),
                "restored model tensor differs from the terminal checkpoint: " + name)
    before = model_digest(module)
    result = validate_dev_result({**evaluator(module), "epoch": 49}, candidate_floor=candidate_floor)
    require(model_digest(module) == before, "DEV evaluation changed model tensors")
    return result


def _binding(path, relative, epoch):
    return {"relative_path": relative, "sha256": sha256_file(path), "epoch": epoch}


def recover(args):
    import lightning as L
    import torch
    import yaml
    from scripts.h100.lightning_contract import assert_pre_trainer_contract, assert_trainer_contract
    from scripts.h100.terminal_recovery import validate_worker_context
    from scripts.handoff.terminal_contract import CHECKPOINT_SHA256, POLICY
    from src.eval.infer_scene import dev_f1
    from src.train.lit_modules import HeatmapLitModule

    repo = Path.cwd().resolve(strict=True)
    run = args.run_dir
    require(run.is_absolute() and run.resolve(strict=True) == run, "run directory is not canonical")
    cell = run.name
    execution = validate_worker_context(
        repo=repo, runs_root=run.parent, exp_id=cell,
        source_receipt=args.execution_json, expected_sha256=args.execution_sha256,
    )
    activation = run.parent / ".h100/terminal-recovery/TERMINAL_ACTIVATION.json"
    activation_sha = sha256_file(activation)
    config = yaml.safe_load((run / "config.yaml").read_text())
    detector = yaml.safe_load((repo / "configs/detector.yaml").read_text())
    data = yaml.safe_load(args.data_config.read_text())
    require(config["detector"] == detector and config["detector_sha256"] == sha256_file(repo / "configs/detector.yaml")
            and config["git_sha"] == POLICY["campaign_git_sha"], "resolved scientific recipe changed")
    require(data == yaml.safe_load((repo / "configs/data.yaml").read_text()), "data configuration differs from reviewed source")
    expected_init = {"beS2-f10-s0": "bigearthnet_s2", "vitin1k-f10-s0": "vit_imagenet"}[cell]
    require(config["args"]["init"] == expected_init and config["args"]["seed"] == 0
            and config["args"]["label_frac"] == 0.1, "cell initialization identity changed")
    input_path = run / "checkpoints/last.ckpt"
    input_sha = CHECKPOINT_SHA256[cell]
    require(sha256_file(input_path) == input_sha, "terminal checkpoint hash changed")
    require(not os.path.lexists(run / "metrics/metrics.csv"), "missing-loss CSV evidence changed")
    saved = load_restricted(input_path)
    floor = detector["decode"]["candidate_floor"]
    previous_dev, previous_best, selection = validate_saved_state(saved, candidate_floor=floor)
    best_path = run / "checkpoints/best.ckpt"
    require(Path(selection["best_model_path"]) == best_path, "prior selected checkpoint path changed")
    prior_binding = _binding(best_path, "checkpoints/best.ckpt", previous_best["epoch"])
    from src.eval.result_contract import read_lightning_checkpoint_epoch
    require(read_lightning_checkpoint_epoch(best_path, restricted=True) == previous_best["epoch"],
            "prior checkpoint epoch differs from saved selection")

    parameters = dict(init_name=expected_init, lr=detector["optimizer"]["lr"],
                      layer_decay=detector["optimizer"]["layer_decay"],
                      weight_decay=detector["optimizer"]["weight_decay"],
                      epochs=50, warmup_epochs=detector["schedule"]["warmup_epochs"],
                      head_channels=detector["head"]["channels"])
    require(all(saved.get("hyper_parameters", {}).get(k) == v for k, v in parameters.items()),
            "checkpoint hyperparameters differ from the shared recipe")
    L.seed_everything(0, workers=True)
    module = HeatmapLitModule(**parameters, load_weights=False)
    contract_args = dict(precision="32-true", devices=1, micro_batch=16, gradient_accumulation=1)
    pre = assert_pre_trainer_contract(module, **contract_args)
    trainer = L.Trainer(max_epochs=50, accelerator="gpu", devices=1, precision="32-true",
                        accumulate_grad_batches=1, logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    runtime = assert_trainer_contract(trainer, module, pre_trainer=pre, **contract_args)
    scene_ids = sorted(json.loads(Path(data["paths"]["splits"]).read_text())["splits"]["dev"])[:8]
    require(len(scene_ids) == detector["eval"]["n_dev_scenes"] == 8, "DEV8 scene inventory changed")
    cache_path = run / "terminal-dev.json"
    cache_identity = {"schema": 1, "exp_id": cell, "input_checkpoint_sha256": input_sha,
                      "terminal_execution": execution, "activation_sha256": activation_sha}

    def evaluate(model):
        return dev_f1(model, scene_ids=scene_ids,
            raw_root=Path(data["paths"]["raw_xview3"]) / "GRD",
            labels_csv=Path(data["paths"]["raw_xview3"]) / "labels/train.csv",
            stats_path=data["paths"]["stats"], tau=floor,
            d_nms_m=detector["decode"]["d_nms_m"], tile_px=detector["eval"]["tile_px"],
            tile_stride_px=detector["eval"]["tile_stride_px"],
            batch_size=detector["eval"]["infer_batch"], device=torch.device("cuda", 0), precision="32-true")

    if os.path.lexists(cache_path):
        require(not cache_path.is_symlink() and not cache_path.stat().st_mode & 0o222,
                "terminal DEV cache is not immutable")
        cache = json.loads(cache_path.read_text())
        require(set(cache) == set(cache_identity) | {"terminal_dev", "runtime_seconds"}
                and all(cache[k] == v for k, v in cache_identity.items()), "terminal DEV cache identity changed")
        result = validate_dev_result(cache["terminal_dev"], candidate_floor=floor)
        elapsed = cache["runtime_seconds"]
    else:
        started = time.monotonic()
        result = finish_dev(module, saved, evaluate, candidate_floor=floor)
        elapsed = time.monotonic() - started
        require(sha256_file(input_path) == input_sha, "terminal input changed during DEV")
        write_once(cache_path, {**cache_identity, "terminal_dev": result, "runtime_seconds": elapsed})
    improved = select_terminal(previous_best, result)
    if improved:
        copy_once(input_path, run / "checkpoints/terminal-best.ckpt", input_sha)
    chosen_dev = result if improved else previous_best
    chosen_checkpoint = ({"relative_path": "checkpoints/terminal-best.ckpt", "sha256": input_sha, "epoch": 49}
                         if improved else prior_binding)
    receipt = {
        "schema": 1, "status": "terminal-dev-recovered", "exp_id": cell,
        "campaign_git_sha": POLICY["campaign_git_sha"],
        "input_checkpoint": {"relative_path": "checkpoints/last.ckpt", "sha256": input_sha, "epoch": 49},
        "input_global_step": 35050, "output_global_step": 35050,
        "added_training_batches": 0, "added_optimizer_steps": 0, "added_scheduler_steps": 0,
        "last_dev_before": previous_dev, "prior_best_dev": previous_best,
        "prior_best_checkpoint": prior_binding, "terminal_dev": result,
        "selected_dev": chosen_dev, "selected_checkpoint": chosen_checkpoint,
        "terminal_execution": execution, "activation_sha256": activation_sha,
        "loss_unavailable_reason": LOSS_REASON, "metrics_csv_absent": True,
        "terminal_model_unchanged": True, "runtime_seconds": elapsed,
    }
    marker = {
        "result_schema": 3, "exp_id": cell, "git_sha": POLICY["campaign_git_sha"],
        "detector_sha256": config["detector_sha256"], "precision": "32-true",
        "micro_batch": 16, "gradient_accumulation": 1, "effective_batch": 16,
        "epochs_run": 50, "best_dev_f1": chosen_dev["f1"], "best_dev": chosen_dev,
        "best_checkpoint": chosen_checkpoint, "last_dev": result, "train_loss": None,
        "h100_runtime_contract": runtime,
    }
    validate_terminal_receipt(receipt, marker, run_dir=run, candidate_floor=floor)
    write_once(run / RECEIPT_NAME, receipt)
    marker["terminal_recovery"] = {"relative_path": RECEIPT_NAME, "sha256": sha256_file(run / RECEIPT_NAME)}
    validate_completion_payload(marker, run_dir=run, candidate_floor=floor)
    write_once(run / "final_metrics.json", marker)
    print(json.dumps({"status": "terminal-recovery-complete", "exp_id": cell,
                      "added_training_steps": 0, "result_schema": 3}, sort_keys=True))
    return 0


class Preempted(Exception):
    pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--execution-json", type=Path, required=True)
    parser.add_argument("--execution-sha256", required=True)
    args = parser.parse_args(argv)
    def stop(_signum, _frame):
        raise Preempted()
    for sig in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    try:
        return recover(args)
    except Preempted:
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
