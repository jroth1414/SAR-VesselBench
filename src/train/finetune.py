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
import json
from pathlib import Path
from typing import Sequence

from lightning.pytorch.callbacks import Callback

INIT_SHORT = {
    "vit_random": "vitrand",
    "satdino_b": "satdino",
    "sarmae_b": "sarmae",
    "vit_supervised": "vitsup",
    "cnn_random": "cnnrand",
    "bigearthnet_s2": "beS2",
    "bigearthnet_s1": "beS1",
    "cnn_supervised": "cnnsup",
}


def exp_id(init_name: str, label_frac: float, seed: int) -> str:
    return f"{INIT_SHORT[init_name]}-f{int(round(label_frac * 100))}-s{seed}"


def main(argv: Sequence[str] | None = None) -> int:
    import lightning as L
    import yaml
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger

    from src.train.datamodule import FineTuneDataModule
    from src.train.lit_modules import HeatmapLitModule

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True, choices=sorted(INIT_SHORT))
    parser.add_argument("--label_frac", type=float, default=1.0, choices=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="override detector.yaml (smoke only)")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--supervised-checkpoint", default=None, help="for vit_supervised / cnn_supervised")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="3-epoch dev-box sanity: small batch, capped steps, 1 dev scene",
    )
    args = parser.parse_args(argv)

    data_cfg = yaml.safe_load(Path(args.data_config).read_text())
    det_cfg = yaml.safe_load(Path(args.detector_config).read_text())

    run_id = exp_id(args.init, args.label_frac, args.seed)
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    L.seed_everything(args.seed, workers=True)

    epochs = args.epochs or det_cfg["schedule"]["epochs"]
    batch_size = det_cfg["schedule"]["batch_size"]
    limit_train_batches = None
    if args.smoke:
        batch_size = min(batch_size, 8)
        limit_train_batches = 300

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
    )
    module = HeatmapLitModule(
        init_name=args.init,
        lr=det_cfg["optimizer"]["lr"],
        layer_decay=det_cfg["optimizer"]["layer_decay"],
        weight_decay=det_cfg["optimizer"]["weight_decay"],
        epochs=epochs,
        warmup_epochs=det_cfg["schedule"]["warmup_epochs"],
        head_channels=det_cfg["head"]["channels"],
        supervised_checkpoint=args.supervised_checkpoint,
    )

    dev_eval = DevSceneEval(
        data_cfg=data_cfg,
        det_cfg=det_cfg,
        every_n_epochs=1 if args.smoke else det_cfg["eval"]["dev_every_epochs"],
        n_scenes=1 if args.smoke else det_cfg["eval"]["n_dev_scenes"],
        final_epoch=epochs,
    )
    callbacks = [
        dev_eval,
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best",
            monitor="dev_f1",
            mode="max",
            save_last=True,
            save_top_k=1,
            every_n_epochs=dev_eval.every,
            save_on_train_epoch_end=True,
        ),
        EarlyStopping(
            monitor="dev_f1",
            mode="max",
            patience=det_cfg["eval"]["early_stop_patience"],
            strict=False,
            check_on_train_epoch_end=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        precision=det_cfg["schedule"]["precision"],
        gradient_clip_val=det_cfg["optimizer"]["grad_clip"],
        limit_train_batches=limit_train_batches,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(run_dir), name="", version="metrics"),
        default_root_dir=str(run_dir),
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    resolved = {
        "exp_id": run_id,
        "args": vars(args),
        "detector": det_cfg,
        "data": data_cfg,
        "git_sha": _git_sha(),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved), newline="\n")

    trainer.fit(module, datamodule=datamodule)

    final = {
        "exp_id": run_id,
        "epochs_run": trainer.current_epoch,
        "best_dev_f1": dev_eval.best,
        "last_dev": dev_eval.last_result,
        "train_loss": float(trainer.callback_metrics.get("train_loss", float("nan"))),
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=1), newline="\n")
    print(json.dumps(final, indent=1))
    return 0


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


class DevSceneEval(Callback):
    """Every N epochs: tiled inference on the fixed dev scenes -> dev F1."""

    def __init__(self, *, data_cfg, det_cfg, every_n_epochs, n_scenes, final_epoch):
        super().__init__()
        self.data_cfg = data_cfg
        self.det_cfg = det_cfg
        self.every = every_n_epochs
        self.n_scenes = n_scenes
        self.final_epoch = final_epoch
        self.best: float | None = None
        self.last_result: dict | None = None

    # Lightning duck-typed callback hooks -------------------------------
    def setup(self, trainer, pl_module, stage=None):
        splits = json.loads(Path(self.data_cfg["paths"]["splits"]).read_text())["splits"]
        self.scene_ids = sorted(splits["dev"])[: self.n_scenes]

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.every and epoch != self.final_epoch:
            return
        from src.eval.infer_scene import dev_f1

        result = dev_f1(
            pl_module,
            scene_ids=self.scene_ids,
            raw_root=Path(self.data_cfg["paths"]["raw_xview3"]) / "GRD",
            labels_csv=Path(self.data_cfg["paths"]["raw_xview3"]) / "labels" / "train.csv",
            stats_path=self.data_cfg["paths"]["stats"],
            tau=self.det_cfg["decode"]["tau"],
            d_nms_m=self.det_cfg["decode"]["d_nms_m"],
            tile_px=self.det_cfg["eval"]["tile_px"],
            tile_stride_px=self.det_cfg["eval"]["tile_stride_px"],
            batch_size=self.det_cfg["eval"]["infer_batch"],
            device=pl_module.device,
        )
        self.last_result = result
        if self.best is None or result["f1"] > self.best:
            self.best = result["f1"]
        pl_module.log("dev_f1", result["f1"], prog_bar=True)
        pl_module.log("dev_recall", result["recall"])
        pl_module.log("dev_precision", result["precision"])
        print(f"[dev eval, epoch {epoch}] {result}")
        pl_module.train()

    def state_dict(self):
        return {"best": self.best}

    def load_state_dict(self, state):
        self.best = state.get("best")


if __name__ == "__main__":
    raise SystemExit(main())
