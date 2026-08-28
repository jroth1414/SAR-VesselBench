"""Shared fine-tune entrypoint for all eight arms (DEVPLAN P3.5).

``--init`` selects the loader (and implicitly the backbone family); head,
loss, sampler, augmentation, decode, schedule, and seeds are identical across
both tracks. ``--label_frac`` subsamples train scenes (scene-level, seeded,
NESTED). Every 5 epochs a callback runs tiled whole-scene inference on the
fixed dev scenes and logs dev F1 through the FROZEN scorer; early stopping
waits 4 dev evals.

Run directory contract (ground rule 8): ``runs/<exp_id>/`` with config.yaml,
metrics.csv, final_metrics.json, checkpoints/. Experiment ids follow the
Section-12 manifest: ``{init_short}-f{frac}-s{seed}``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Sequence

from lightning.pytorch.callbacks import Callback

from src.analysis.bes2_contract import (
    DIAGNOSTIC_VARIANTS,
    OPTIMIZATION_TRACE_STEPS,
    diagnostic_run_id,
    stage_fractions,
    validate_optimization_trace,
)
from src.eval.result_contract import (
    RESULT_SCHEMA,
    ResultContractError,
    atomic_write_json,
    create_best_checkpoint_binding,
    validate_completion_payload,
    validate_dev_result,
)

INIT_SHORT = {
    "vit_random": "vitrand",
    "satdino_b": "satdino",
    "sarmae_b": "sarmae",
    "vit_imagenet": "vitin1k",
    "cnn_random": "cnnrand",
    "bigearthnet_s2": "beS2",
    "bigearthnet_s1": "beS1",
    "cnn_imagenet": "cnnin1k",
}
DIAGNOSTIC_STAGES = ("probe", "full")
DIAGNOSTIC_RUN_SCHEMA = 2
DIAGNOSTIC_PROBE_EPOCHS = 5
DIAGNOSTIC_SCHEDULE_EPOCHS = 50



def exp_id(init_name: str, label_frac: float, seed: int) -> str:
    return f"{INIT_SHORT[init_name]}-f{int(round(label_frac * 100))}-s{seed}"


def _diagnostic_layout(args, det_cfg, parser):
    """Resolve a fail-closed namespace and unchanged recipe for S2 diagnostics."""

    supplied = (
        args.diagnostic_variant,
        args.diagnostic_stage,
        args.diagnostic_root,
    )
    if not any(value is not None for value in supplied):
        return None
    if not all(value is not None for value in supplied):
        parser.error(
            "--diagnostic-variant, --diagnostic-stage, and "
            "--diagnostic-root must be supplied together"
        )
    if (
        args.init != "bigearthnet_s2"
        or args.seed != 0
    ):
        parser.error(
            "BigEarthNet-S2 diagnostics require --init bigearthnet_s2, "
            "and --seed 0"
        )
    if not any(
        math.isclose(args.label_frac, allowed, rel_tol=0.0, abs_tol=1.0e-12)
        for allowed in stage_fractions(args.diagnostic_stage)
    ):
        parser.error(
            f"BigEarthNet-S2 {args.diagnostic_stage} diagnostics authorize "
            f"label fractions {stage_fractions(args.diagnostic_stage)}"
        )
    forbidden = {
        "--epochs": args.epochs,
        "--smoke": args.smoke,
        "--exp-suffix": args.exp_suffix,
        "--batch-size": args.batch_size,
        "--micro-batch": args.micro_batch,
        "--samples-per-epoch": args.samples_per_epoch,
        "--dev-every": args.dev_every,
        "--n-dev-scenes": args.n_dev_scenes,
    }
    changed = sorted(
        name
        for name, value in forbidden.items()
        if value not in (None, False)
    )
    if changed:
        parser.error(
            "diagnostic runs refuse recipe overrides: " + ", ".join(changed)
        )

    expected_recipe = {
        "schedule.epochs": DIAGNOSTIC_SCHEDULE_EPOCHS,
        "schedule.batch_size": 16,
        "schedule.precision": "32-true",
        "optimizer.layer_decay": 0.65,
        "eval.dev_every_epochs": 5,
        "eval.n_dev_scenes": 8,
    }
    observed_recipe = {
        "schedule.epochs": det_cfg["schedule"]["epochs"],
        "schedule.batch_size": det_cfg["schedule"]["batch_size"],
        "schedule.precision": det_cfg["schedule"]["precision"],
        "optimizer.layer_decay": det_cfg["optimizer"]["layer_decay"],
        "eval.dev_every_epochs": det_cfg["eval"]["dev_every_epochs"],
        "eval.n_dev_scenes": det_cfg["eval"]["n_dev_scenes"],
    }
    if observed_recipe != expected_recipe:
        parser.error(
            "diagnostic detector recipe differs from the approved contract: "
            f"{observed_recipe} != {expected_recipe}"
        )

    root = Path(args.diagnostic_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        parser.error(
            "--diagnostic-root must be an existing absolute non-symlink directory"
        )
    resolved_root = root.resolve(strict=True)
    core_root = (Path.cwd() / "runs").resolve(strict=False)
    if resolved_root == core_root or resolved_root.is_relative_to(core_root):
        parser.error("diagnostic runs must be outside the core runs namespace")

    run_id = diagnostic_run_id(
        args.diagnostic_variant, args.diagnostic_stage, args.label_frac
    )
    trainer_epochs = (
        DIAGNOSTIC_PROBE_EPOCHS
        if args.diagnostic_stage == "probe"
        else DIAGNOSTIC_SCHEDULE_EPOCHS
    )
    return {
        "run_id": run_id,
        "run_dir": resolved_root / run_id,
        "trainer_epochs": trainer_epochs,
        "schedule_epochs": DIAGNOSTIC_SCHEDULE_EPOCHS,
        "fraction": args.label_frac,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import lightning as L
    import yaml
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger

    from scripts.h100.lightning_contract import (
        assert_pre_trainer_contract,
        assert_trainer_contract,
        h100_runtime_active,
    )
    from src.train.datamodule import FineTuneDataModule
    from src.train.lit_modules import HeatmapLitModule

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, choices=sorted(INIT_SHORT))
    parser.add_argument("--label_frac", type=float, default=1.0, choices=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--git-sha",
        default=None,
        help="full host-validated source SHA (avoids requiring git in a slim SIF)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override detector.yaml (smoke only)")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--weights-root", type=Path, default=Path("data/weights"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="3-epoch dev-box sanity: small batch, capped steps, 1 dev scene",
    )
    # Check-level overrides (P3.6 early-signal runs on the dev card). The
    # frozen detector.yaml stays authoritative for the real grid — these
    # exist so short comparisons can share IDENTICAL reduced settings.
    parser.add_argument("--exp-suffix", default=None, help="append to the run id (e.g. p36)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=None,
        help="hardware adaptation: split the recipe batch into micro-batches "
        "with gradient accumulation (recipe batch stays the effective batch; "
        "needed for ConvNeXt cells on the 16 GB dev card where batch 16 "
        "overflows VRAM into shared memory)",
    )
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--dev-every", type=int, default=None)
    parser.add_argument("--n-dev-scenes", type=int, default=None)
    parser.add_argument(
        "--diagnostic-variant",
        choices=DIAGNOSTIC_VARIANTS,
        default=None,
        help="diagnostic-only S2 initialization; never a core arm name",
    )
    parser.add_argument(
        "--diagnostic-stage",
        choices=DIAGNOSTIC_STAGES,
        default=None,
    )
    parser.add_argument(
        "--diagnostic-root",
        type=Path,
        default=None,
        help="existing persistent namespace outside core runs",
    )
    args = parser.parse_args(argv)

    data_cfg = yaml.safe_load(Path(args.data_config).read_text())
    detector_path = Path(args.detector_config)
    det_cfg = yaml.safe_load(detector_path.read_text())
    detector_sha256 = hashlib.sha256(detector_path.read_bytes()).hexdigest()
    git_sha = args.git_sha or _git_sha()
    if not re.fullmatch(r"[0-9a-f]{40}", git_sha):
        parser.error("--git-sha/source checkout must resolve to a full 40-hex SHA")

    diagnostic = _diagnostic_layout(args, det_cfg, parser)
    if diagnostic is not None and not h100_runtime_active():
        parser.error(
            "BigEarthNet-S2 diagnostic training requires the accepted "
            "strict-FP32 H100 runtime"
        )

    if diagnostic is None:
        run_id = exp_id(args.init, args.label_frac, args.seed)
        if args.exp_suffix:
            run_id = f"{run_id}-{args.exp_suffix}"
        run_dir = Path("runs") / run_id
        trainer_epochs = args.epochs or det_cfg["schedule"]["epochs"]
        schedule_epochs = trainer_epochs
    else:
        run_id = diagnostic["run_id"]
        run_dir = diagnostic["run_dir"]
        trainer_epochs = diagnostic["trainer_epochs"]
        schedule_epochs = diagnostic["schedule_epochs"]
        if (run_dir / "final_metrics.json").exists():
            parser.error(
                "diagnostic namespace contains a forbidden core completion marker"
            )
        if (run_dir / "training_metrics.json").exists():
            parser.error("diagnostic training is already complete and immutable")
    run_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(args.seed, workers=True)

    batch_size = args.batch_size or det_cfg["schedule"]["batch_size"]
    accumulate = 1
    if args.micro_batch and args.micro_batch < batch_size:
        if batch_size % args.micro_batch:
            raise SystemExit("--micro-batch must divide the recipe batch size")
        accumulate = batch_size // args.micro_batch
        batch_size = args.micro_batch
    limit_train_batches = None
    if args.smoke:
        # dev-box smoke: smaller batch for the 16 GB card, full epochs — at
        # ~8 it/s the whole f0.1 dataset is ~3 min/epoch, and the capped-step
        # variant produced too little signal to decode a single detection.
        batch_size = min(batch_size, 8)

    datamodule = FineTuneDataModule(
        chips_root=data_cfg["paths"]["chips"],
        splits_path=data_cfg["paths"]["splits"],
        stats_path=data_cfg["paths"]["stats"],
        label_frac=args.label_frac,
        frac_seed=int(data_cfg["seed"]),  # FIXED data seed => fractions nest
        batch_size=batch_size,
        num_workers=args.workers,
        crop=det_cfg["input"]["crop_px"],
        fg_frac=det_cfg["sampler"]["fg_frac"],
        seed=args.seed,
        samples_per_epoch=args.samples_per_epoch,
    )
    module = HeatmapLitModule(
        init_name=args.init,
        lr=det_cfg["optimizer"]["lr"],
        layer_decay=det_cfg["optimizer"]["layer_decay"],
        weight_decay=det_cfg["optimizer"]["weight_decay"],
        epochs=schedule_epochs,
        warmup_epochs=det_cfg["schedule"]["warmup_epochs"],
        head_channels=det_cfg["head"]["channels"],
        diagnostic_variant=args.diagnostic_variant,
        weights_root=args.weights_root,
    )
    if diagnostic is not None:
        initialization = module.diagnostic_initialization
        if not isinstance(initialization, dict):
            raise RuntimeError("diagnostic initialization manifest is absent")
        initialization_path = run_dir / "initialization.json"
        if initialization_path.exists():
            if initialization_path.is_symlink() or json.loads(
                initialization_path.read_text()
            ) != initialization:
                raise RuntimeError(
                    "diagnostic initialization changed across resume"
                )
        else:
            atomic_write_json(initialization_path, initialization)

    dev_eval = DevSceneEval(
        data_cfg=data_cfg,
        det_cfg=det_cfg,
        every_n_epochs=args.dev_every
        or (1 if args.smoke else det_cfg["eval"]["dev_every_epochs"]),
        n_scenes=args.n_dev_scenes
        or (1 if args.smoke else det_cfg["eval"]["n_dev_scenes"]),
        final_epoch=trainer_epochs,
        diagnostic=diagnostic is not None,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        filename="best",
        monitor="dev_f1",
        mode="max",
        save_last=True,
        save_top_k=1,
        every_n_epochs=dev_eval.every,
        save_on_train_epoch_end=True,
        # A stable filename is convenient but is not itself the binding:
        # consumers must follow best_checkpoint.relative_path from schema 2.
        enable_version_counter=False,
    )
    optimization_trace = (
        DiagnosticOptimizationTrace() if diagnostic is not None else None
    )
    callbacks = [
        dev_eval,
        checkpoint_callback,
        EarlyStopping(
            monitor="dev_f1",
            mode="max",
            # PATIENCE IS IN EPOCHS, NOT DEV EVALS: EarlyStopping checks every
            # epoch and the logged dev_f1 PERSISTS between our every-N-epoch
            # evals, so stale-value checks count against patience. Multiply by
            # the eval cadence so "patience 4 dev evals" means 4 REAL evals —
            # the un-multiplied version stopped every run at epoch ~9 after a
            # single eval (caught 2026-07-07 when the first grid cells all
            # reported epochs_run=9).
            patience=det_cfg["eval"]["early_stop_patience"] * dev_eval.every,
            strict=False,
            check_on_train_epoch_end=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if optimization_trace is not None:
        callbacks.append(optimization_trace)

    h100_pre_trainer = None
    h100_runtime_contract = None
    if h100_runtime_active():
        h100_pre_trainer = assert_pre_trainer_contract(
            module,
            precision=det_cfg["schedule"]["precision"],
            devices=1,
            micro_batch=batch_size,
            gradient_accumulation=accumulate,
        )
    trainer = L.Trainer(
        max_epochs=trainer_epochs,
        accelerator="gpu",
        devices=1,
        precision=det_cfg["schedule"]["precision"],
        gradient_clip_val=det_cfg["optimizer"]["grad_clip"],
        accumulate_grad_batches=accumulate,
        limit_train_batches=limit_train_batches,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(run_dir), name="", version="metrics"),
        default_root_dir=str(run_dir),
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    if h100_pre_trainer is not None:
        h100_runtime_contract = assert_trainer_contract(
            trainer,
            module,
            precision=det_cfg["schedule"]["precision"],
            devices=1,
            micro_batch=batch_size,
            gradient_accumulation=accumulate,
            pre_trainer=h100_pre_trainer,
        )

    resolved_args = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in vars(args).items()
    }
    resolved = {
        "exp_id": run_id,
        "args": resolved_args,
        "detector": det_cfg,
        "data": data_cfg,
        "git_sha": git_sha,
        "detector_sha256": detector_sha256,
        "execution": {
            "micro_batch": batch_size,
            "gradient_accumulation": accumulate,
            "effective_batch": batch_size * accumulate,
        },
    }
    if diagnostic is not None:
        resolved["diagnostic"] = {
            "schema": DIAGNOSTIC_RUN_SCHEMA,
            "variant": args.diagnostic_variant,
            "stage": args.diagnostic_stage,
            "fraction": args.label_frac,
            "trainer_max_epochs": trainer_epochs,
            "schedule_horizon_epochs": schedule_epochs,
            "core_completion_marker_forbidden": True,
        }
    if h100_runtime_contract is not None:
        resolved["execution"]["h100_runtime_contract"] = h100_runtime_contract
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved), newline="\n")

    last_ckpt = run_dir / "checkpoints" / "last.ckpt"
    trainer.fit(
        module,
        datamodule=datamodule,
        # resume after interruption (reboot/sleep) instead of restarting
        ckpt_path=str(last_ckpt) if last_ckpt.exists() else None,
    )

    candidate_floor = det_cfg["decode"]["candidate_floor"]
    try:
        best_dev = validate_dev_result(
            dev_eval.best_result,
            candidate_floor=candidate_floor,
        )
        if dev_eval.best is None or float(dev_eval.best) != best_dev["f1"]:
            raise ResultContractError("callback best scalar does not equal best_dev.f1")
        best_checkpoint = create_best_checkpoint_binding(
            run_dir=run_dir,
            checkpoint_path=checkpoint_callback.best_model_path,
            best_dev=best_dev,
            candidate_floor=candidate_floor,
        )
    except ResultContractError as exc:
        raise RuntimeError(
            "training finished without a valid checkpoint-bound best-dev result"
        ) from exc

    final = {
        "result_schema": RESULT_SCHEMA,
        "exp_id": run_id,
        "git_sha": git_sha,
        "detector_sha256": detector_sha256,
        "precision": det_cfg["schedule"]["precision"],
        "micro_batch": batch_size,
        "gradient_accumulation": accumulate,
        "effective_batch": batch_size * accumulate,
        "epochs_run": trainer.current_epoch,
        "best_dev_f1": best_dev["f1"],
        "best_dev": best_dev,
        "best_checkpoint": best_checkpoint,
        "last_dev": dev_eval.last_result,
        "train_loss": float(trainer.callback_metrics.get("train_loss", float("nan"))),
    }
    if h100_runtime_contract is not None:
        final["h100_runtime_contract"] = h100_runtime_contract
    expected_recipe = {
        name: final[name]
        for name in (
            "exp_id",
            "git_sha",
            "detector_sha256",
            "precision",
            "micro_batch",
            "gradient_accumulation",
            "effective_batch",
        )
    }
    validate_completion_payload(
        final,
        run_dir=run_dir,
        candidate_floor=candidate_floor,
        expected_recipe=expected_recipe,
    )
    if diagnostic is None:
        atomic_write_json(run_dir / "final_metrics.json", final)
        output = final
    else:
        if (run_dir / "final_metrics.json").exists():
            raise RuntimeError(
                "diagnostic training must never publish final_metrics.json"
            )
        runtime_seconds = dev_eval.elapsed_seconds()
        output = {
            "diagnostic_run_schema": DIAGNOSTIC_RUN_SCHEMA,
            "purpose": "bes2-root-cause-training",
            "variant": args.diagnostic_variant,
            "stage": args.diagnostic_stage,
            "fraction": args.label_frac,
            "trainer_max_epochs": trainer_epochs,
            "schedule_horizon_epochs": schedule_epochs,
            "initialization": module.diagnostic_initialization,
            "dev_history": dev_eval.history,
            "optimization_trace": (
                optimization_trace.payload()
                if optimization_trace is not None
                else None
            ),
            "runtime": {
                "gpu_count": 1,
                "seconds": runtime_seconds,
                "gpu_hours": runtime_seconds / 3600.0,
            },
            "training_result": final,
        }
        atomic_write_json(run_dir / "training_metrics.json", output)
    print(json.dumps(output, indent=1))
    return 0


def _git_sha() -> str:
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    return subprocess.run(
        ["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class DiagnosticOptimizationTrace(Callback):
    """Observe early gradients and actual parameter updates without mutation."""

    FIRST_CONVOLUTION = "backbone.model.stem.0.weight"

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, object]] = []
        self._pending: dict[str, object] | None = None
        self._pending_trainer = None

    @staticmethod
    def _logical_group(name: str) -> str:
        if name.startswith("backbone.model.stem."):
            return "stem"
        for stage in range(4):
            if name.startswith(f"backbone.model.stages.{stage}."):
                return f"stage_{stage}"
        if name.startswith("backbone.model."):
            return "backbone_other"
        if name.startswith("head."):
            return "detector_head"
        raise RuntimeError(
            f"unclassified diagnostic trainable parameter: {name}"
        )

    @staticmethod
    def _squared_norm(tensor) -> float:
        return float(tensor.detach().float().square().sum().item())

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator > 0.0 else 0.0

    @staticmethod
    def _new_accumulator() -> dict[str, object]:
        return {
            "parameter_count": 0,
            "parameter_norm_sq": 0.0,
            "gradient_norm_sq": 0.0,
            "update_norm_sq": 0.0,
            "lrs": [],
            "lr_scales": [],
        }

    @classmethod
    def _group_payload(cls, accumulator: dict[str, object]) -> dict[str, object]:
        parameter_norm = math.sqrt(float(accumulator["parameter_norm_sq"]))
        gradient_norm = math.sqrt(float(accumulator["gradient_norm_sq"]))
        update_norm = math.sqrt(float(accumulator["update_norm_sq"]))
        lrs = [float(value) for value in accumulator["lrs"]]
        lr_scales = [
            float(value) for value in accumulator["lr_scales"]
        ]
        if not lrs or not lr_scales:
            raise RuntimeError("diagnostic optimization group has no parameters")
        return {
            "parameter_count": int(accumulator["parameter_count"]),
            "parameter_norm_before": parameter_norm,
            "gradient_norm": gradient_norm,
            "gradient_to_parameter_norm": cls._ratio(
                gradient_norm, parameter_norm
            ),
            "lr_min": min(lrs),
            "lr_max": max(lrs),
            "lr_scale_min": min(lr_scales),
            "lr_scale_max": max(lr_scales),
            "update_norm": update_norm,
            "relative_update_norm": cls._ratio(update_norm, parameter_norm),
        }

    def _snapshot(self, pl_module, optimizer, step: int) -> dict[str, object]:
        import torch

        optimizer_metadata: dict[int, tuple[float, float]] = {}
        for optimizer_group in optimizer.param_groups:
            lr = float(optimizer_group["lr"])
            lr_scale = float(optimizer_group.get("lr_scale", 1.0))
            if (
                not math.isfinite(lr)
                or not math.isfinite(lr_scale)
                or lr < 0.0
                or lr_scale < 0.0
            ):
                raise RuntimeError(
                    "diagnostic optimizer contains a non-finite learning rate"
                )
            for parameter in optimizer_group["params"]:
                identity = id(parameter)
                if identity in optimizer_metadata:
                    raise RuntimeError(
                        "diagnostic parameter appears in multiple optimizer groups"
                    )
                optimizer_metadata[identity] = (lr, lr_scale)

        groups: dict[str, dict[str, object]] = {}
        entries: list[dict[str, object]] = []
        first: dict[str, object] | None = None
        with torch.no_grad():
            for name, parameter in pl_module.named_parameters():
                if not parameter.requires_grad:
                    continue
                metadata = optimizer_metadata.get(id(parameter))
                if metadata is None:
                    raise RuntimeError(
                        f"diagnostic trainable parameter is absent from optimizer: {name}"
                    )
                lr, lr_scale = metadata
                logical = self._logical_group(name)
                accumulator = groups.setdefault(
                    logical, self._new_accumulator()
                )
                parameter_norm_sq = self._squared_norm(parameter)
                gradient_norm_sq = (
                    self._squared_norm(parameter.grad)
                    if parameter.grad is not None
                    else 0.0
                )
                accumulator["parameter_count"] = int(
                    accumulator["parameter_count"]
                ) + parameter.numel()
                accumulator["parameter_norm_sq"] = float(
                    accumulator["parameter_norm_sq"]
                ) + parameter_norm_sq
                accumulator["gradient_norm_sq"] = float(
                    accumulator["gradient_norm_sq"]
                ) + gradient_norm_sq
                accumulator["lrs"].append(lr)
                accumulator["lr_scales"].append(lr_scale)
                entry = {
                    "name": name,
                    "logical_group": logical,
                    "parameter": parameter,
                    "before": parameter.detach().clone(),
                    "lr": lr,
                    "lr_scale": lr_scale,
                    "parameter_norm_sq": parameter_norm_sq,
                    "gradient_norm_sq": gradient_norm_sq,
                }
                entries.append(entry)
                if name == self.FIRST_CONVOLUTION:
                    first = entry

        if first is None:
            raise RuntimeError(
                "diagnostic first-convolution parameter was not found"
            )
        return {
            "optimizer_step": step,
            "groups": groups,
            "entries": entries,
            "first": first,
        }

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        step = int(trainer.global_step) + 1
        completed = {
            int(record["optimizer_step"]) for record in self.records
        }
        if step not in OPTIMIZATION_TRACE_STEPS or step in completed:
            return
        if self._pending is not None:
            raise RuntimeError("diagnostic optimization snapshot is still pending")
        self._pending = self._snapshot(pl_module, optimizer, step)
        self._pending_trainer = trainer

    def _finalize_pending(self) -> None:
        if self._pending is None:
            return
        import torch

        pending = self._pending
        with torch.no_grad():
            for entry in pending["entries"]:
                update_sq = self._squared_norm(
                    entry["parameter"].detach() - entry["before"]
                )
                accumulator = pending["groups"][entry["logical_group"]]
                accumulator["update_norm_sq"] = float(
                    accumulator["update_norm_sq"]
                ) + update_sq
                entry["update_norm_sq"] = update_sq

        groups = {
            name: self._group_payload(accumulator)
            for name, accumulator in sorted(pending["groups"].items())
        }
        first = pending["first"]
        parameter_norm = math.sqrt(float(first["parameter_norm_sq"]))
        gradient_norm = math.sqrt(float(first["gradient_norm_sq"]))
        update_norm = math.sqrt(float(first["update_norm_sq"]))
        record = {
            "optimizer_step": int(pending["optimizer_step"]),
            "groups": groups,
            "first_convolution": {
                "name": self.FIRST_CONVOLUTION,
                "parameter_count": int(first["parameter"].numel()),
                "parameter_norm_before": parameter_norm,
                "gradient_norm": gradient_norm,
                "gradient_to_parameter_norm": self._ratio(
                    gradient_norm, parameter_norm
                ),
                "lr": float(first["lr"]),
                "lr_scale": float(first["lr_scale"]),
                "update_norm": update_norm,
                "relative_update_norm": self._ratio(
                    update_norm, parameter_norm
                ),
            },
        }
        self.records.append(record)
        self._pending = None
        self._pending_trainer = None
        validate_optimization_trace(self._payload(), require_complete=False)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        del trainer, pl_module, outputs, batch, batch_idx
        self._finalize_pending()

    def _payload(self) -> dict[str, object]:
        return {
            "schema": 1,
            "milestones": list(OPTIMIZATION_TRACE_STEPS),
            "records": self.records,
        }

    def payload(self) -> dict[str, object]:
        if self._pending is not None:
            raise RuntimeError("diagnostic optimization update remains pending")
        return validate_optimization_trace(self._payload())

    def state_dict(self) -> dict[str, object]:
        if self._pending is not None:
            trainer = self._pending_trainer
            if trainer is None:
                raise RuntimeError(
                    "diagnostic optimizer snapshot lacks trainer state"
                )
            if int(trainer.global_step) >= int(
                self._pending["optimizer_step"]
            ):
                self._finalize_pending()
        return validate_optimization_trace(
            self._payload(), require_complete=False
        )

    def load_state_dict(self, state_dict) -> None:
        payload = validate_optimization_trace(
            state_dict, require_complete=False
        )
        self.records = [dict(record) for record in payload["records"]]
        self._pending = None
        self._pending_trainer = None


class DevSceneEval(Callback):
    """Every N epochs: tiled inference on the fixed dev scenes -> dev F1."""

    def __init__(
        self,
        *,
        data_cfg,
        det_cfg,
        every_n_epochs,
        n_scenes,
        final_epoch,
        diagnostic: bool = False,
    ):
        super().__init__()
        self.data_cfg = data_cfg
        self.det_cfg = det_cfg
        self.every = every_n_epochs
        self.n_scenes = n_scenes
        self.final_epoch = final_epoch
        self.diagnostic = diagnostic
        self.best: float | None = None
        self.best_result: dict | None = None
        self.last_result: dict | None = None
        self.history: list[dict[str, object]] = []
        self._runtime_seconds = 0.0
        self._segment_started: float | None = None

    def on_fit_start(self, trainer, pl_module):
        if self.diagnostic and self._segment_started is None:
            self._segment_started = time.monotonic()

    def on_fit_end(self, trainer, pl_module):
        if self.diagnostic and self._segment_started is not None:
            self._runtime_seconds = self.elapsed_seconds()
            self._segment_started = None

    def elapsed_seconds(self) -> float:
        elapsed = self._runtime_seconds
        if self._segment_started is not None:
            elapsed += time.monotonic() - self._segment_started
        return float(elapsed)

    # Lightning duck-typed callback hooks -------------------------------
    def setup(self, trainer, pl_module, stage=None):
        splits = json.loads(Path(self.data_cfg["paths"]["splits"]).read_text())["splits"]
        self.scene_ids = sorted(splits["dev"])[: self.n_scenes]

    def on_train_epoch_end(self, trainer, pl_module):
        completed_epoch = trainer.current_epoch + 1
        if completed_epoch % self.every and completed_epoch != self.final_epoch:
            return
        from src.eval.infer_scene import dev_f1

        result = dev_f1(
            pl_module,
            scene_ids=self.scene_ids,
            raw_root=Path(self.data_cfg["paths"]["raw_xview3"]) / "GRD",
            labels_csv=Path(self.data_cfg["paths"]["raw_xview3"]) / "labels" / "train.csv",
            stats_path=self.data_cfg["paths"]["stats"],
            tau=self.det_cfg["decode"]["candidate_floor"],
            d_nms_m=self.det_cfg["decode"]["d_nms_m"],
            tile_px=self.det_cfg["eval"]["tile_px"],
            tile_stride_px=self.det_cfg["eval"]["tile_stride_px"],
            batch_size=self.det_cfg["eval"]["infer_batch"],
            device=pl_module.device,
            precision=self.det_cfg["schedule"]["precision"],
        )
        # Lightning checkpoint metadata uses a zero-based epoch. Persist that
        # exact value so checkpoint and operating-point binding is unambiguous.
        result = {**result, "epoch": int(trainer.current_epoch)}
        result = validate_dev_result(
            result,
            candidate_floor=self.det_cfg["decode"]["candidate_floor"],
        )
        if self.diagnostic:
            self.history.append(dict(result))
        self.last_result = dict(result)
        if self.best is None or result["f1"] > self.best:
            self.best = result["f1"]
            self.best_result = dict(result)
        pl_module.log("dev_f1", result["f1"], prog_bar=True)
        pl_module.log("dev_recall", result["recall"])
        pl_module.log("dev_precision", result["precision"])
        print(f"[dev eval, checkpoint epoch {trainer.current_epoch}] {result}")
        pl_module.train()

    def state_dict(self):
        state = {
            "best": self.best,
            "best_result": self.best_result,
            "last_result": self.last_result,
        }
        if self.diagnostic:
            state.update(
                {
                    "history": self.history,
                    "elapsed_seconds": self.elapsed_seconds(),
                }
            )
        return state

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ResultContractError("DevSceneEval checkpoint state must be a mapping")
        if self.diagnostic:
            raw_history = state.get("history")
            elapsed = state.get("elapsed_seconds")
            if not isinstance(raw_history, list):
                raise ResultContractError(
                    "diagnostic DevSceneEval history must be a list"
                )
            self.history = [
                validate_dev_result(
                    item,
                    candidate_floor=self.det_cfg["decode"]["candidate_floor"],
                    description=f"history[{index}]",
                )
                for index, item in enumerate(raw_history)
            ]
            if (
                isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float))
                or not math.isfinite(float(elapsed))
                or float(elapsed) < 0.0
            ):
                raise ResultContractError(
                    "diagnostic DevSceneEval elapsed_seconds is invalid"
                )
            self._runtime_seconds = float(elapsed)
            self._segment_started = None
        best = state.get("best")
        best_result = state.get("best_result")
        last_result = state.get("last_result")
        if best is None:
            if best_result is not None or last_result is not None:
                raise ResultContractError(
                    "DevSceneEval empty best state cannot contain dev results"
                )
            self.best = None
            self.best_result = None
            self.last_result = None
            if self.diagnostic and self.history:
                raise ResultContractError(
                    "empty diagnostic best state cannot contain DEV history"
                )
            return
        validated_best = validate_dev_result(
            best_result,
            candidate_floor=self.det_cfg["decode"]["candidate_floor"],
        )
        validated_last = validate_dev_result(
            last_result,
            candidate_floor=self.det_cfg["decode"]["candidate_floor"],
            description="last_result",
        )
        if isinstance(best, bool):
            raise ResultContractError("DevSceneEval best must be a finite number")
        try:
            normalized_best = float(best)
        except (TypeError, ValueError) as exc:
            raise ResultContractError("DevSceneEval best must be a finite number") from exc
        if not math.isfinite(normalized_best):
            raise ResultContractError("DevSceneEval best must be a finite number")
        if normalized_best != validated_best["f1"]:
            raise ResultContractError("DevSceneEval best does not equal best_result.f1")
        self.best = normalized_best
        self.best_result = validated_best
        self.last_result = validated_last
        if self.diagnostic:
            if not self.history or self.history[-1] != validated_last:
                raise ResultContractError(
                    "diagnostic DEV history does not end at last_result"
                )
            expected_best = max(
                self.history, key=lambda item: float(item["f1"])
            )
            if expected_best != validated_best:
                raise ResultContractError(
                    "diagnostic DEV history does not reproduce best_result"
                )


if __name__ == "__main__":
    raise SystemExit(main())
