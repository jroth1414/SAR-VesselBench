"""The shared LightningModule every arm trains through (DEVPLAN P3.5).

One module: backbone (selected by init name) + shared heatmap head +
penalty-reduced focal loss + AdamW with layer-wise lr decay and a warmup +
cosine schedule. Within each track all four arms fine-tune end-to-end with
the identical schedule; only the loaded weights differ. The head, loss,
optimizer, and schedule are identical across BOTH tracks (ground rule 2).
"""

from __future__ import annotations

import math
from pathlib import Path

import lightning as L
import torch

from src.models.heatmap_head import HeatmapHead
from src.models.init_loaders import INIT_FAMILY, build_init
from src.train.losses import penalty_reduced_focal_loss


class HeatmapLitModule(L.LightningModule):
    def __init__(
        self,
        *,
        init_name: str,
        lr: float = 1.0e-4,
        layer_decay: float = 0.65,
        weight_decay: float = 0.05,
        epochs: int = 50,
        warmup_epochs: int = 5,
        head_channels: int = 256,
        load_weights: bool = True,
        weights_root: str | Path = "data/weights",
        supervised_checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = build_init(
            init_name,
            load_weights=load_weights,
            weights_root=weights_root,
            supervised_checkpoint=supervised_checkpoint,
        )
        self.head = HeatmapHead(
            self.backbone.out_channels,
            self.backbone.out_stride,
            head_channels=head_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Images -> heatmap logits (B, 1, H/4, W/4)."""

        return self.head(self.backbone(x))

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        logits = self(batch["image"]).squeeze(1)
        loss = penalty_reduced_focal_loss(
            logits, batch["heatmap"], batch["mask"]
        )
        self.log(f"{stage}_loss", loss, prog_bar=stage == "train", sync_dist=stage != "train")
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Used only by the LS-SSDD pretraining (P5.1); the fine-tune arms
        # evaluate via whole-scene dev inference instead.
        return self._step(batch, "val")

    def configure_optimizers(self):
        param_groups = self._layer_decay_param_groups()
        optimizer = torch.optim.AdamW(param_groups, lr=self.hparams.lr)

        warmup = self.hparams.warmup_epochs
        total = self.hparams.epochs

        def schedule(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _layer_decay_param_groups(self) -> list[dict]:
        """timm layer-wise lr decay on the backbone; head at full lr."""

        from timm.optim import param_groups_layer_decay

        backbone_groups = param_groups_layer_decay(
            self.backbone.model,
            weight_decay=self.hparams.weight_decay,
            layer_decay=self.hparams.layer_decay,
        )
        head_decay = [p for p in self.head.parameters() if p.ndim > 1]
        head_no_decay = [p for p in self.head.parameters() if p.ndim <= 1]
        return [
            *backbone_groups,
            {"params": head_decay, "weight_decay": self.hparams.weight_decay},
            {"params": head_no_decay, "weight_decay": 0.0},
        ]

    @property
    def family(self) -> str:
        return INIT_FAMILY[self.hparams.init_name]
