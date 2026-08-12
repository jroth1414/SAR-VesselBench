# Development and Reproduction Plan

This document is the public source of truth for the xView3 label-efficiency
study. It records the scientific contract, the state of the final result set,
and the checks required before publication.

## Current state

- `dev` contains the complete research history through the Sprint 9b paper
  draft. Development changes still use reviewed topic branches.
- `final-submission` contains the public repository, class report, and
  supplemental AIPR manuscript.
- The canonical core campaign contains 32 strict-FP32 H100 runs: eight arms,
  four label fractions, and seed 0. The final result snapshot has not yet been
  imported into this branch.
- V100 runs are engineering diagnostics. They cannot supply values, fill gaps,
  or support claims in an H100 curve or table.
- The 50 human-verified scenes remain sealed. The class report may report
  corrected development-selection metrics, but it must identify them as such.

The repository must build an honest deadline report from any verified,
completion-closed H100 cohort. The complete report profile remains locked until
all 32 core completion records pass the result contract.

## Research question and controlled design

The study asks how much labeled xView3 data different pretrained backbones save
for dark-vessel detection, and whether the effect of pretraining source holds
for both transformer and convolutional backbones.

Two architecture-matched tracks use one point detector:

| Track | Role | Arm | Initialization |
|---|---|---|---|
| ViT-B/16 | random floor | `vitrand` | random |
| ViT-B/16 | optical | `satdino` | SatDINO fMoW RGB |
| ViT-B/16 | SAR | `sarmae` | SARMAE encoder |
| ViT-B/16 | generic | `vitin1k` | ImageNet-1K AugReg |
| ConvNeXt-V2-B | random floor | `cnnrand` | random |
| ConvNeXt-V2-B | optical | `beS2` | BigEarthNet S2 |
| ConvNeXt-V2-B | SAR | `beS1` | BigEarthNet S1 |
| ConvNeXt-V2-B | generic | `cnnin1k` | FCMAE then ImageNet-1K |

Each arm runs at 10%, 25%, 50%, and 100% of the fixed training scenes. Every
cell uses seed 0, input channels `[VH, VV, VH-VV]`, effective batch 16, the
shared CenterNet-style head and decoder, and Lightning `32-true`. Initialization
is the only within-track variable. The backbone architecture is the only
matched-role cross-track variable. The ViT and CNN ImageNet checkpoints share
their final source dataset and classification supervision, but not their full
training histories.

The two external references, YOLO26 and LocateAnything, retain their published
independent recipes and do not enter the controlled core curves.

## Data and evaluation contract

- Scene identifiers cannot cross train, development, test, or final-evaluation
  partitions.
- Training labels include vessel-positive HIGH/MEDIUM rows and LOW-confidence
  ignore rows. HIGH/MEDIUM non-vessels cannot become positive targets.
- Development threshold selection and each reported score bind to the same
  checkpoint. A `best.ckpt` score cannot use a threshold selected by
  `last.ckpt`.
- The scorer remains a pure, frozen implementation. Near-shore unmatched
  predictions count as false positives; dark-vessel recall applies only where
  manual dark-vessel labels exist.
- The 50 human-verified scenes may be opened once through the held-out tripwire.
  They are outside the current class-report result contract.

The following files are byte-frozen:

| File | SHA-256 |
|---|---|
| `configs/detector.yaml` | `c42ae65bf9045cc93f0d73ae1437b2f6a1300670cb49d8e93f83a39d58a62a12` |
| `data/splits.json` | `0d201003999b6e634bdb3f9cccaedbd391b9d0cdcdb7f9370470d56a5e391db6` |
| `data/stats.json` | `8e7691002e540f7274f9e1d67812fe8ae2b6c45775a6e9072a5177b7fadb4490` |
| `data/lsssdd_split.json` | `ae6d0343d021341d58c2d037e20eb9a9f4e67ace3b7c677439973838fd1473a3` |
| `src/eval/scorer.py` | `85dec7ab083b531547fe81fea5aa02e4d828457ff157619afae29981cf49cd32` |

`data/lsssdd_split.json` is retained as immutable historical provenance. The
active study does not train on LS-SSDD.

## Checkpoint sources

The project downloads all six pretrained core encoders. It does not pretrain
them.

- SatDINO: `strakajk/satdino-vit_base-16`, Apache-2.0.
- SARMAE: `Wenquandan777/SARMAE`, file
  `SARMAE_vitb_checkpoint-last`, CC BY-NC 4.0.
- ViT ImageNet: `timm/vit_base_patch16_224.augreg_in1k`.
- BigEarthNet S2 and S1:
  `BIFOLD-BigEarthNetv2-0/convnextv2_base-{s2,s1}-v0.2.0`.
- ConvNeXt-V2 ImageNet:
  `timm/convnextv2_base.fcmae_ft_in1k`.

The value-sensitive checkpoint tests must show that each loaded encoder differs
from a fresh random initialization. Structural key-manifest checks run in CPU
CI; value-sensitive checks require the training environment and downloaded
weights.

## Result publication gates

The machine-readable H100 snapshot is the only source for paper values. The
importer must reject a snapshot unless it satisfies all applicable checks:

1. The snapshot identifies the campaign commit, environment, H100 hardware,
   strict-FP32 state, and frozen hashes.
2. It contains exactly the declared 32 core cell identifiers with no duplicate.
3. Only `DONE` cells expose finite reportable metrics and immutable completion
   evidence.
4. A reported label fraction includes all eight arms. Incomplete fractions
   appear as status information without numeric comparisons.
5. The complete profile requires all 32 cells. No V100 value can appear in the
   snapshot or generated paper data.
6. External references remain separate and carry their own validated
   provenance.

After import, regenerate every table, figure, macro, and claim. Review the
abstract, results, discussion, and conclusion against the generated values.

## Public deliverables

The class deliverable contains:

- `final_report.pdf`, with at most five main-body pages from Introduction
  through Conclusion. References and appendices follow outside that limit.
- The supplemental AIPR manuscript.
- Source code, environment instructions, sanitized logs, and result provenance.
- A deterministic `Roth_John_final_project.zip` with checksums.

The Canvas archive excludes `data/`, imagery, annotations, weights,
checkpoints, run directories, environments, credentials, and private system
paths. The Git repository retains the frozen split/statistics metadata because
the guard tests and scientific record depend on those exact bytes.

## Required checks

Before a public release:

1. Run the full CPU test suite and every frozen guard test.
2. Run the six value-sensitive checkpoint-load tests in the training
   environment.
3. Validate the H100 snapshot and regenerate both manuscripts.
4. Confirm the class main body ends within its fifth page.
5. Build the submission ZIP from its allowlist, extract it in a clean
   directory, and run its documented checks.
6. Scan the final tree and retained Git refs for secrets, private paths, raw
   data, weights, and checkpoints.

Any proposed change to a frozen artifact, scorer semantics, split membership,
precision recipe, or result-admission rule requires explicit owner approval,
a new decision record, and updated immutable hashes where applicable.
