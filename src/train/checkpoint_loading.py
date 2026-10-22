"""Restricted Lightning restore for this project's Path-valued hyperparameters."""
from pathlib import PosixPath

import torch


def fit_with_path_metadata(trainer, module, *, datamodule, ckpt_path):
    """Allow only our known path metadata type, with no permanent allowlist change.

    This does not rewrite a checkpoint or opt into unrestricted pickle loading.
    Any further unsupported type remains a hard failure.
    """
    with torch.serialization.safe_globals([PosixPath]):
        return trainer.fit(
            module, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=True,
        )
