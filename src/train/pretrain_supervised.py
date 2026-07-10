"""LS-SSDD supervised pretraining — Arms 4 and 8's source (DEVPLAN P5.1).

Pretrains a backbone + heatmap head on LS-SSDD-v1.0 ONLY (centroids as
targets, P1.4), with the SAME loss / sampler / augmentation as the fine-tune
pipeline via the shared LightningModule. Run twice: ``--backbone vit`` ->
``runs/vitsup-lsssdd`` (Arm 4's init) and ``--backbone cnn`` ->
``runs/cnnsup-lsssdd`` (Arm 8's init). Both consume the identical frozen
``data/lsssdd_split.json``; the val part (900 sub-images) drives val_loss
early stopping (no prescribed stopping epoch — stop when validation turns;
50-epoch safety ceiling).

The resulting ``checkpoints/best.ckpt`` feeds ``init_loaders`` via
``finetune.py --supervised-checkpoint`` (keys under ``backbone.``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

EXP_IDS = {"vit": "vitsup-lsssdd", "cnn": "cnnsup-lsssdd"}
RANDOM_INIT = {"vit": "vit_random", "cnn": "cnn_random"}


def main(argv: Sequence[str] | None = None) -> int:
    import lightning as L
    import yaml
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

    from src.train.datamodule import SupervisedSARDataModule
    from src.train.lit_modules import HeatmapLitModule

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", required=True, choices=["vit", "cnn"])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--detector-config", default="configs/detector.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience-epochs", type=int, default=5)
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=None,
        help="hardware adaptation as in finetune.py: micro-batch with gradient "
        "accumulation, effective batch stays the recipe's (ConvNeXt at batch "
        "16 overflows the 16 GB dev card into shared memory)",
    )
    args = parser.parse_args(argv)

    data_cfg = yaml.safe_load(Path(args.data_config).read_text())
    det_cfg = yaml.safe_load(Path(args.detector_config).read_text())

    run_id = EXP_IDS[args.backbone]
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    L.seed_everything(args.seed, workers=True)

    batch_size = det_cfg["schedule"]["batch_size"]
    accumulate = 1
    if args.micro_batch and args.micro_batch < batch_size:
        if batch_size % args.micro_batch:
            raise SystemExit("--micro-batch must divide the recipe batch size")
        accumulate = batch_size // args.micro_batch
        batch_size = args.micro_batch

    datamodule = SupervisedSARDataModule(
        lsssdd_root=data_cfg["paths"]["raw_lsssdd"],
        split_path=data_cfg["paths"]["lsssdd_split"],
        batch_size=batch_size,
        num_workers=args.workers,
        crop=det_cfg["input"]["crop_px"],
        fg_frac=det_cfg["sampler"]["fg_frac"],
        seed=args.seed,
    )
    module = HeatmapLitModule(
        init_name=RANDOM_INIT[args.backbone],
        lr=det_cfg["optimizer"]["lr"],
        layer_decay=det_cfg["optimizer"]["layer_decay"],
        weight_decay=det_cfg["optimizer"]["weight_decay"],
        epochs=det_cfg["schedule"]["epochs"],
        warmup_epochs=det_cfg["schedule"]["warmup_epochs"],
        head_channels=det_cfg["head"]["channels"],
        load_weights=False,  # random init — LS-SSDD supervision is the point
    )

    trainer = L.Trainer(
        max_epochs=det_cfg["schedule"]["epochs"],
        accelerator="gpu",
        devices=1,
        precision=det_cfg["schedule"]["precision"],
        gradient_clip_val=det_cfg["optimizer"]["grad_clip"],
        accumulate_grad_batches=accumulate,
        callbacks=[
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                filename="best",
                monitor="val_loss",
                mode="min",
                save_last=True,
                save_top_k=1,
            ),
            EarlyStopping(
                monitor="val_loss", mode="min", patience=args.patience_epochs
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=CSVLogger(save_dir=str(run_dir), name="", version="metrics"),
        default_root_dir=str(run_dir),
        num_sanity_val_steps=2,
        log_every_n_steps=10,
    )

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"exp_id": run_id, "args": vars(args), "detector": det_cfg}),
        newline="\n",
    )
    trainer.fit(module, datamodule=datamodule)

    final = {
        "exp_id": run_id,
        "epochs_run": trainer.current_epoch,
        "best_val_loss": float(trainer.checkpoint_callback.best_model_score or float("nan")),
        "best_checkpoint": str(run_dir / "checkpoints" / "best.ckpt"),
    }
    (run_dir / "final_metrics.json").write_text(json.dumps(final, indent=1), newline="\n")
    print(json.dumps(final, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
