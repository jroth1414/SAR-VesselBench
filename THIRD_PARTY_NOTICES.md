# Third-party notices

The repository's MIT License covers original project code and documentation.
It does not change the terms for datasets, pretrained weights, model code, or
vendored publication files supplied by other authors.

## Data

- **xView3-SAR:** distributed under the terms published with the xView3
  challenge and dataset. This repository contains split identifiers and
  aggregate normalization statistics, but no imagery or label CSV files.
- **LS-SSDD-v1.0:** no longer participates in the experiment. Its frozen split
  manifest remains as historical provenance; no LS-SSDD image is distributed.

Users must obtain data from its publisher and comply with its license and use
restrictions.

## Pretrained weights

| Initialization | Upstream source | Terms noted by this project |
|---|---|---|
| SatDINO ViT-B/16 | `strakajk/satdino-vit_base-16` | Apache-2.0 model repository |
| SARMAE ViT-B/16 | `Wenquandan777/SARMAE` | CC BY-NC 4.0, gated access |
| BigEarthNet S2 ConvNeXt-V2-B | `BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0` | upstream model-card terms |
| BigEarthNet S1 ConvNeXt-V2-B | `BIFOLD-BigEarthNetv2-0/convnextv2_base-s1-v0.2.0` | upstream model-card terms |
| ImageNet ViT-B/16 | `timm/vit_base_patch16_224.augreg_in1k` | upstream `timm` model-card terms |
| ImageNet ConvNeXt-V2-B | `timm/convnextv2_base.fcmae_ft_in1k` | upstream `timm` model-card terms |
| YOLO26 reference | Ultralytics | upstream software and model terms |
| LocateAnything-3B reference | NVIDIA | custom upstream terms |

No pretrained weight file is included in the repository or class submission.
Users must review the current upstream terms before downloading a checkpoint.
The SARMAE checkpoint's noncommercial restriction is narrower than this
repository's MIT License.

## Software and publication assets

The project depends on the packages named in `pyproject.toml` and the
environment locks. Each dependency retains its own license.

The supplemental manuscript vendors Springer's LNCS class and bibliography
style for reproducible typesetting. Springer retains the applicable rights to
those files. They are not relicensed under MIT.
