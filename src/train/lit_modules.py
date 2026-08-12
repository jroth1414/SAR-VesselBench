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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = build_init(
            init_name,
            load_weights=load_weights,
            weights_root=weights_root,
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
        if not torch.isfinite(loss):
            # Fail LOUDLY: a non-finite forward is permanent and silently
            # burns epochs — the P3.6 bigearthnet_s1 run trained two more
            # epochs on NaN before the dev eval refused
            # the non-finite heatmap.
            raise RuntimeError(
                f"non-finite {stage} loss at epoch {self.current_epoch} — "
                "under the shared precision recipe; stop and diagnose "
                "(DEVPLAN risk register)"
            )
        self.log(f"{stage}_loss", loss, prog_bar=stage == "train", sync_dist=stage != "train")
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def configure_optimizers(self):
        param_groups = self._layer_decay_param_groups()
        # Bake lr_scale into each group's lr: timm only APPLIES lr_scale via
        # its own schedulers, so with torch's LambdaLR the ladder would sit
        # inert in the group dicts and every layer would train at base lr
        # (caught by test_layer_decay). LambdaLR then scales each group's
        # own initial lr, preserving the ladder across the cosine schedule.
        for group in param_groups:
            group["lr"] = self.hparams.lr * group.get("lr_scale", 1.0)
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
        """timm layer-wise lr decay on the backbone; head at full lr.

        Asserts the decay actually took: timm's group matcher fails SILENTLY
        on models whose module names it cannot parse (the P3.6 sweep caught
        the CNN track training with lr_scale=1.0 everywhere because the
        features_only wrapper had flattened the names — a cross-track
        fairness violation). A degenerate grouping now raises instead.
        """

        from timm.optim import param_groups_layer_decay

        if self.family == "cnn":
            # ConvNeXt official fine-tuning convention: blocks are mapped
            # into TWELVE layer-ids to mirror ViT-B's depth (ConvNeXt repo
            # optim_factory.get_num_layer_for_convnext). timm's native
            # per-block matcher would build a ~38-rung ladder whose earliest
            # rungs decay to ~1e-7 — effectively frozen — while the ViT track
            # bottoms out at 0.65^12; equalizing ladder depth keeps the
            # shared layer_decay=0.65 meaning the same thing in both tracks.
            backbone_groups = _convnext_layer_decay_groups(
                self.backbone.model,
                weight_decay=self.hparams.weight_decay,
                layer_decay=self.hparams.layer_decay,
            )
        else:
            backbone_groups = param_groups_layer_decay(
                self.backbone.model,
                weight_decay=self.hparams.weight_decay,
                layer_decay=self.hparams.layer_decay,
            )
        lr_scales = {round(g.get("lr_scale", 1.0), 6) for g in backbone_groups}
        if len(lr_scales) < 3:
            raise RuntimeError(
                f"layer-wise lr decay degenerated to {sorted(lr_scales)} — "
                "the group matcher did not recognize the backbone's module "
                "names; refusing to train with a broken schedule (ground rule 2)"
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


def _convnext_layer_id(name: str) -> int:
    """Param name -> layer id per the ConvNeXt official fine-tune recipe.

    ConvNeXt repo ``get_num_layer_for_convnext``: stem -> 0, stage-0 blocks
    -> 1, stage-1 blocks -> 2, stage-2 block j -> 3 + j//3 (27 blocks spread
    over ids 3..11), stage-3 blocks -> 12; stage downsamples group with
    their stage; final norms / head remnants -> the last rung. Mirrors
    ViT-B's 13-rung ladder so the shared layer_decay means the same thing
    in both tracks.
    """

    parts = name.split(".")
    if name.startswith("stem"):
        return 0
    if name.startswith("stages"):
        stage = int(parts[1])
        if stage == 0:
            return 1
        if stage == 1:
            return 2
        if stage == 2:
            block = int(parts[3]) if parts[2] == "blocks" else 0
            return 3 + block // 3
        return 12
    return 12  # norm_pre / head remnants train with the last rung


def _convnext_layer_decay_groups(
    model, *, weight_decay: float, layer_decay: float
) -> list[dict]:
    """Manual layer-decay param groups (timm has no custom layer-map hook).

    Matches timm's group format (params / lr_scale / weight_decay) and its
    exclude-1d-params-from-decay convention.
    """

    max_id = 12
    groups: dict[tuple[int, bool], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid = _convnext_layer_id(name)
        decay = param.ndim > 1
        key = (lid, decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr_scale": layer_decay ** (max_id - lid),
                "weight_decay": weight_decay if decay else 0.0,
            }
        groups[key]["params"].append(param)
    return list(groups.values())
