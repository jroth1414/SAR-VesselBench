# Superseded random-to-LS-SSDD Arms 4 and 8

These artifacts belong to the pre-2026-07-22 design in which random
ViT-B/16 and ConvNeXt-V2-Base backbones were trained on LS-SSDD before xView3
fine-tuning. The owner superseded that role after the f10/seed-0 results were
observed and replaced it with downloaded ImageNet-1K initialization in both
tracks.

Archived experiment IDs:

- `vitsup-lsssdd` and `cnnsup-lsssdd`: source-training records.
- `vitsup-f10-s0` and `cnnsup-f10-s0`: superseded xView3 f10 cells.
- `summary/`: the eight-row curve/table rendered before the amendment.

These records are preserved for design provenance only. They are excluded
from the active manifest, current curves, and the 34-experiment count. Current
replacement IDs are `vitin1k-f10-s0` and `cnnin1k-f10-s0`; do not move the
archived files back into the top-level active result namespace.
