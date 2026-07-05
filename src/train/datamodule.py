"""One LightningDataModule shared by every arm (DEVPLAN ground rule 9).

Wraps ``FineTuneDataset`` (xView3 fine-tunes, all 8 arms) or
``SupervisedSARDataset`` (the two LS-SSDD pretrainings) behind one loader
configuration, so batching, worker seeding, and sampling are identical by
construction. Dev/test evaluation does NOT flow through here — it is
whole-scene tiled inference (src/eval/infer_scene.py).
"""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from src.data.datasets import FineTuneDataset, SupervisedSARDataset
from src.train.sampler import build_foreground_balanced_sampler


class FineTuneDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        chips_root: str | Path,
        splits_path: str | Path,
        stats_path: str | Path,
        label_frac: float = 1.0,
        frac_seed: int = 0,
        batch_size: int = 16,
        num_workers: int = 4,
        crop: int = 512,
        fg_frac: float = 0.5,
        seed: int = 0,
        samples_per_epoch: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        self.train_set = FineTuneDataset(
            chips_root=hp.chips_root,
            splits_path=hp.splits_path,
            stats_path=hp.stats_path,
            split="train",
            label_frac=hp.label_frac,
            frac_seed=hp.frac_seed,
            crop=hp.crop,
            augment=True,
            seed=hp.seed,
        )

    def train_dataloader(self) -> DataLoader:
        hp = self.hparams
        sampler = build_foreground_balanced_sampler(
            self.train_set.foreground,
            fg_frac=hp.fg_frac,
            num_samples=hp.samples_per_epoch,
            seed=hp.seed,
        )
        return DataLoader(
            self.train_set,
            batch_size=hp.batch_size,
            sampler=sampler,
            num_workers=hp.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=hp.num_workers > 0,
        )


class SupervisedSARDataModule(L.LightningDataModule):
    """LS-SSDD loaders for the Arm-4/Arm-8 supervised pretrainings (P5.1)."""

    def __init__(
        self,
        *,
        lsssdd_root: str | Path,
        split_path: str | Path,
        batch_size: int = 16,
        num_workers: int = 4,
        crop: int = 512,
        fg_frac: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        self.train_set = SupervisedSARDataset(
            lsssdd_root=hp.lsssdd_root,
            split_path=hp.split_path,
            part="train",
            crop=hp.crop,
            augment=True,
            seed=hp.seed,
        )
        self.val_set = SupervisedSARDataset(
            lsssdd_root=hp.lsssdd_root,
            split_path=hp.split_path,
            part="val",
            crop=hp.crop,
            augment=False,
            seed=hp.seed,
        )

    def train_dataloader(self) -> DataLoader:
        hp = self.hparams
        sampler = build_foreground_balanced_sampler(
            self.train_set.foreground, fg_frac=hp.fg_frac, seed=hp.seed
        )
        return DataLoader(
            self.train_set,
            batch_size=hp.batch_size,
            sampler=sampler,
            num_workers=hp.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=hp.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        hp = self.hparams
        return DataLoader(
            self.val_set,
            batch_size=hp.batch_size,
            shuffle=False,
            num_workers=hp.num_workers,
            pin_memory=True,
        )
