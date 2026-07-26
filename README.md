# xView3 — Label-Efficient Dark-Vessel Detection in SAR

**Does SAR-domain pretraining beat optical and ImageNet transfer — and does it hold across ViT and CNN?**

A controlled study of how much labeled data different classes of *pretrained backbone* save
for detecting **dark vessels** (ships not broadcasting AIS) in Sentinel-1 synthetic-aperture-radar
(SAR) imagery, under one fixed point detector.

*John Roth & Kyle Wagner · JHU EN.705.643 Deep Learning Developments with PyTorch*

> **Documentation map.** This README is the orientation. The authoritative design and phase plan
> is [`DEVPLAN.md`](DEVPLAN.md) — start with its **"Current repository state — READ FIRST"** cold-start
> runbook to see what is built vs. not. [`AGENTS.md`](AGENTS.md) is the short list of non-negotiables
> for any coding agent. The amended working proposal is available as
> [`docs/proposal.tex`](docs/proposal.tex) and [`docs/proposal.pdf`](docs/proposal.pdf). The
> [`signed course proposal`](docs/proposal_signed.pdf) remains an unchanged historical record.
> The results-neutral final-paper abstract draft is
> [`docs/final_paper_abstract.tex`](docs/final_paper_abstract.tex), with a rendered
> [`PDF`](docs/final_paper_abstract.pdf).

---

## 1. The research question

Labeled dark-vessel examples in SAR are scarce and expensive (human-verified). Pretrained backbones
promise to close that gap, but it is unclear **which kind of pretraining transfers best** to this
scarce-label regime — generic ImageNet, optical remote sensing, or SAR-specific — and **whether the
answer depends on the backbone architecture**.

We hold the detector, its head, its training schedule, and its input fixed, and vary only the
downloaded **initialization** (within a track) and the **architecture family** (across tracks):

> *Under one fixed point detector, how much labeled xView3 data does each class of pretrained
> backbone save for dark-vessel detection; does a SAR-domain checkpoint beat optical remote-sensing
> and generic ImageNet transfer in the scarce-label regime; and does that finding hold across two
> architecture families (ViT and CNN)?*

**Hypothesis:** within each architecture, SAR-domain pretraining gives the steepest label-efficiency
gains and the largest dark-vessel-recall advantage at low label budgets, and this ordering is
*architecture-general* (holds for both the ViT and the CNN).

We **pretrain no backbone** — all six non-random core initializations are downloaded. The controlled
matrix is 8 arms × 4 label fractions at one fixed run seed (`0`) = 32 core xView3 fine-tunes; R2 and R3
bring the total to **34 reported experiments**. There is no active LS-SSDD stage and no seed-rerun
matrix. This removes self-pretraining compute and novel pretraining code from the study.

---

## 2. Experimental design — two architecture-matched tracks, eight arms

The study has **two tracks**, matched by role so that comparisons isolate one variable at a time:

- **ViT track** — ViT-B/16 (~86M params), arms 1–4.
- **CNN track** — ConvNeXt-V2-Base (~89M params, deliberately size-matched), arms 5–8.

*Initialization* is the only variable **within** a track; *architecture* is the only variable **across**
matched roles. Comparing arm *k* to arm *k+4* isolates the architecture family with the pretraining role
held fixed.

| Arm | Track | Backbone | Initialization | On curve? | Role |
|----:|-------|----------|----------------|:---------:|------|
| 1 | ViT | ViT-B/16 | random init | yes | ViT floor |
| 2 | ViT | ViT-B/16 | **SatDINO** (fMoW-RGB, DINO) | yes | optical-domain FM transfer |
| 3 | ViT | ViT-B/16 | **SARMAE** (SAR-1M, MAE) | yes | SAR-domain FM transfer |
| 4 | ViT | ViT-B/16 | **ImageNet-1K AugReg** (`timm/vit_base_patch16_224.augreg_in1k`) | yes | generic ImageNet transfer |
| 5 | CNN | ConvNeXt-V2-B | random init | yes | CNN floor |
| 6 | CNN | ConvNeXt-V2-B | **BigEarthNet-S2** (optical RS) | yes | optical-RS transfer |
| 7 | CNN | ConvNeXt-V2-B | **BigEarthNet-S1** (SAR RS) | yes | SAR-RS transfer |
| 8 | CNN | ConvNeXt-V2-B | **ImageNet-1K FCMAE→supervised** (`timm/convnextv2_base.fcmae_ft_in1k`) | yes | generic ImageNet transfer |
| R2 | ref | YOLO26 | COCO-pretrained | no | detector reference |
| R3 | ref | LocateAnything-3B | zero-shot | no | VLM reference |

**Fairness by construction.** Within each track, all four arms share the same backbone, detection head,
optimizer, schedule, run seed `0`, and the fixed input. The head/optimizer/schedule are shared across *both*
tracks too — only the backbone (and a small per-family head adapter) differs. Guard tests in CI enforce
the invariants that keep the arms comparable (see §6).

**ImageNet matched-role caveat.** Arms 4 and 8 match on generic source dataset and final ImageNet-1K
classification supervision, but not on full training history: Arm 4 is supervised AugReg, whereas
Arm 8 is FCMAE-pretrained and then supervised-fine-tuned. Cross-track comparisons must state this
limitation; this pair is not a matched MAE/FCMAE experiment.

**Headline deliverable:** a **two-track label-efficiency curve** — detection F1 vs. 10 / 25 / 50 / 100%
of real labels, eight lines (solid ViT, dashed CNN; one color per pretraining role) — plus the hardest
slices: **dark-vessel recall** and **near-shore F1**, where any SAR advantage should concentrate.

---

## 3. Models

**Backbones (the two architectures every arm uses):**

- **ViT-B/16** via `timm` `vit_base_patch16_224`, `in_chans=3`.
- **ConvNeXt-V2-Base** via `timm` `convnextv2_base`, `in_chans=3`.

**Downloaded foundation / pretrained backbones (we do not pretrain these):**

| Model | Repo | Domain / method | License |
|-------|------|-----------------|---------|
| **SatDINO** ViT-B/16 (fMoW-RGB) | `strakajk/satdino-vit_base-16` | optical satellite, DINO self-distillation | Apache-2.0 |
| **SARMAE** ViT-B/16 (SAR-1M) | weights `Wenquandan777/SARMAE` (code `MiliLab/SARMAE`) | SAR, masked autoencoder w/ speckle-aware enhancement | CC BY-NC 4.0 (gated) |
| **BigEarthNet ConvNeXt-V2-B (S2)** | `BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0` | optical (Sentinel-2), supervised land-cover | see repo |
| **BigEarthNet ConvNeXt-V2-B (S1)** | `BIFOLD-BigEarthNetv2-0/convnextv2_base-s1-v0.2.0` | SAR (Sentinel-1), supervised land-cover | see repo |
| **ImageNet ViT-B/16** | `timm/vit_base_patch16_224.augreg_in1k` | ImageNet-1K, supervised AugReg | see upstream model card |
| **ImageNet ConvNeXt-V2-B** | `timm/convnextv2_base.fcmae_ft_in1k` | ImageNet-1K, FCMAE then supervised fine-tuning | see upstream model card |

**Detector.** A point-native heatmap head (CenterNet / *Objects as Points* style, with the TRANSAR SAR
precedent): each vessel is a Gaussian blob on a **stride-4** response map, decoded by peak-finding with
**distance-based non-maximum suppression**. This matches xView3's point labels and its distance-based
official metric exactly, avoiding synthetic bounding boxes the data never had. One `PyTorch Lightning`
code path is shared across all arms.

**Reference arms** (reported separately, *not* on the controlled curves): YOLO26 (COCO init, Ultralytics),
and LocateAnything-3B (zero-shot VLM). ImageNet is now a matched core role, so the old R1 is removed.

---

## 4. Datasets

- **xView3-SAR** — locally acquired DIU-format Sentinel-1 **GRD** archives, dual-pol (VH, VV),
  registered into `data/raw/xview3`. The SARFish Hugging Face mirror contains raw SAFE products,
  not the canonical xView3 GeoTIFF/label payload used here. Training scenes are split
  **75 / 15 / 10** at the **scene level** (no chip-level leakage). Exactly 50 human-verified scenes
  form `eval_final` and are **touched once**, after the grid and thresholds are frozen.
  *SLC products are never fetched* (multi-TB, unneeded).

- **LS-SSDD-v1.0 (retired)** was used by the superseded design for Arms 4/8 and is **not an active
  training source**. Its frozen `data/lsssdd_split.json` remains committed, immutable, and clearly
  retired for provenance; the signed historical proposal and Git history preserve the old design.


**Input representation (study-wide, all arms):** a fixed 3-channel tensor **[VH, VV, VH−VV]** in dB, so the
input is physically identical across arms and channel handling is never a confound. Each backbone's stem
adapts to this input via the standard `timm` `in_chans` path.

**Labels / dark-vessel proxy:** detection positives are `is_vessel` with confidence ∈ {HIGH, MEDIUM};
LOW-confidence labels become ignore regions; a vessel is treated as **"dark"** when `source == "Manual"`
(no AIS correlate). Scoring is a per-scene distance-matched greedy F1 (200 m tolerance) with
dark-vessel recall and near-shore F1 slices.

---

## 5. Repository layout & status

This is a **plan-driven** research repo: [`DEVPLAN.md`](DEVPLAN.md) is the source of truth; its
cold-start runbook holds the live status ledger. Current state:

- **DONE — Phases 0-3** (tagged `phase-1-done`, `phase-2-done`, `phase-3-done`): environment +
  lockfiles, full data pipeline with the three frozen artifacts (`data/splits.json` — 150 study
  scenes split 111/23/16 plus the 50 verified scenes as `eval_final`; `data/stats.json`;
  retired historical `data/lsssdd_split.json`), the frozen scorer/decode/threshold stack, and the shared detector
  (both backbone tracks behind one head, eight init loaders, `configs/detector.yaml` frozen).
  All five freeze guards plus the parity and checkpoint-load guards run in CI.
- **PASSED — historical P3.6 early-signal gate**: the four remote-sensing pretrained arms tested
  at an equal reduced budget beat their track floors, and SAR led optical on the eight-scene dev
  gate (ViT 0.788 floor / 0.835 SatDINO / 0.858 SARMAE; CNN 0.677 floor /
  0.726 BigEarthNet-S2 / 0.819 BigEarthNet-S1). This was a pipeline/signal check, not a final
  result; the completed mature grid and once-only verified evaluation determine paper claims.
- **AMENDED — 2026-07-22**: Arms 4/8 changed from random→LS-SSDD to the two downloaded ImageNet-1K
  checkpoints above; the old `vitsup-f10-s0` and `cnnsup-f10-s0` cells are superseded and excluded.
  Seed reruns and R1 are removed. The target is 32 seed-0 core cells plus R2/R3 = 34 experiments.
- **RESET — full core grid**: every mixed-precision core result is superseded after the V100
  `cnnin1k-f100-s0` epoch-30 divergence. The stopped campaign is preserved for diagnostics; all
  32 canonical core IDs and both reference IDs restart empty under the new campaign.
- **EXECUTION DECISION — 8x V100 full fp32 (2026-07-26)**: the owner approved shared
  Lightning `32-true` training and model-forward inference, micro-batch/effective-batch 16,
  a detector re-pin, all-32 restart, and the >2x/1,300-GPU-hour override. Likely wall clock is
  about eight days; reserve nine days, or 10–11 if references cannot overlap the core queue.
- **READY — Phase 6 tripwires** (`src/eval/final_eval.py`: `--i-am-sure` + lockfile, hard
  preconditions). The 50 raw eval-scene rasters are not in the current transfer payload, and final
  evaluation remains untouched. Phase 7 analysis is also pending; do not promote P3.6 or partial f10
  observations to final-paper findings.

```
JHU-xView3/
  DEVPLAN.md          # single source of truth: design, phases, gates, risks
  AGENTS.md           # non-negotiables for any coding agent
  docs/               # decisions, node handoff, amended proposal + final-paper abstract
  requirements-ci.txt # minimal CPU deps for CI guard tests
  configs/            # data.yaml, detector.yaml (frozen), arms.yaml (run manifest)
  locks/              # verified active V100 pin + historical 5070/P100 pins
  src/data/           # download/registration, chipper, centroids, splits, datasets
  src/models/         # backbones, heatmap head + adapters, 8 init loaders
  src/train/          # shared Lightning module/datamodule + fine-tune; retired LS tooling remains
  src/eval/           # scorer.py (frozen), decode, threshold, infer_scene, final_eval
  src/references/     # YOLO26 (R2), LocateAnything zero-shot (R3)
  src/analysis/       # curves + grid.csv collector, QA galleries
  scripts/            # gpu_sanity, chipping driver, grid queues (dev card + node), test scoring
  tests/              # unit tests + the anti-drift guard tests
  .github/workflows/  # CI: guard tests + full suite
  data/, runs/        # gitignored (frozen JSON artifacts are the committed exceptions)
```

---

## 6. Getting started

The full-fp32 amendment is on `sprint-7c-fp32-grid`, opened from `dev`:

```bash
git clone https://github.com/jroth1414/JHU-xView3
cd JHU-xView3
git fetch origin
git switch sprint-7c-fp32-grid

# 1) On the active V100 host, install the exact verified environment:
#    pip install -r locks/env-v100node.txt --extra-index-url https://download.pytorch.org/whl/cu126
#    Never use a bare moving torch install.

# 2) Install the checked-out package (the only sanctioned package install):
pip install -e .[dev]

# 3) Sanity-check the GPU + kernels, then run the tests:
python scripts/gpu_sanity.py    # or: make env-check
pytest tests/ -q                # or: make test
```

### Launch the fresh 34-run campaign on eight V100s

The active campaign starts from empty canonical namespaces. Do not restore or
copy mixed-precision completion markers into `runs/`; the archived AMP campaign
is diagnostic provenance only. Core runs use shared `32-true` training and
model-forward inference at micro-batch/effective-batch 16. R2/R3 are fresh
external-reference runs and retain their independently validated precision.

Before launch, verify the reviewed branch, payload, exact ImageNet bytes, GPU
leases, full test suite, all six value-sensitive loads, and both-family fp32
train/inference memory gates:

```bash
git status --short --branch
gpu info
nvidia-smi -L
python scripts/gpu_sanity.py
pytest tests/ -q
pytest tests/test_fm_checkpoints_load.py -q
find data/chips -mindepth 1 -maxdepth 1 -type d
find data/raw/xview3/GRD -mindepth 1 -maxdepth 1 -type d
sha256sum data/weights/imagenet_vit_augreg_in1k/model.safetensors
sha256sum data/weights/imagenet_cnn_fcmae_ft_in1k/model.safetensors
```

Expected ImageNet hashes, in the order above:

```text
678a1ce471be7da9822fe2508497a5bcf6da4c6802053151b232ba88a42c21a2
ec152f1e375edc2b3dfac7a81155a449b4c5cbb7c5cf0b9494838f6c87518d73
```

The runtime launcher under `runs/launch/` is intentionally gitignored and must
pin a clean committed SHA, the verified environment hash, the re-pinned
detector hash, and every other frozen artifact hash. One controller starts R2
on GPU 0, R3 on GPU 1, and core cells on GPUs 2–7; each released reference GPU
joins the same core queue. It rejects recipe-mismatched completion markers and
fail-stops new launches if any cell fails. The 50 `eval_final` scenes remain
untouched until the complete fp32 grid passes the 0.02 monotonicity check.

**Active V100 gate.** Eight Tesla V100-SXM2-32GB cards (Volta sm_70) are
attached as local CUDA 0–7. The verified environment is Python 3.11.15 with
torch 2.11.0+cu126. A 200-step ConvNeXt fp32 batch-16 probe completed finite at
about 0.72 step/s; a full step reserved 30.516 GiB, so the margin is tight and
the pre-launch memory gate is mandatory. Reserve GPUs through `gpu`, then use
only container-local indices from `nvidia-smi -L`, never host IDs from
`gpu info`.

**Continuous integration.** `.github/workflows/ci.yml` runs, on every push/PR to `dev`/`main`, the
anti-drift **guard tests** that encode study validity:

- `test_split_disjoint` — no scene appears in two splits (anti-leakage).
- `test_backbone_parity` — within-track param parity + matched head-adapter output geometry.
- `test_fm_checkpoints_load` — every downloaded backbone loads value-sensitively (not silently random).
- `test_scorer_immutable` — the frozen scorer's hash is unchanged.

CI runs the CPU/offline structural guard halves; value-sensitive checkpoint loading runs on the
active V100 host with pinned local weights. The real training environment is **not** CI.
`locks/env-v100node.txt` is the active verified lock; 5070/P100 locks are historical.

**Experiment budget:** 32 core seed-0 fine-tunes plus fresh R2/R3 = **34 experiments**. No
LS-SSDD backbone training, R1, seed reruns, or mixed-precision core result remains active. The
shared-fp32 projection is about 1,550 GPU-hours. With eight V100s, expect roughly eight days,
reserve nine days with safe reference overlap, and use 10–11 days as the serialized-reference
conservative bound. Results are seed-0 point estimates; do not report seed-derived uncertainty.

---

## 7. Licensing & attribution notes

- **SARMAE** weights are **CC BY-NC 4.0** (non-commercial) and gated; **BigEarthNet** weight licenses must
  be verified at download. This project's use is academic/non-commercial; any released code or checkpoints
  are scoped accordingly. See the DEVPLAN risk register.
- The two ImageNet checkpoints have pinned upstream revisions, exact SHA-256 hashes, and local source/license
  notes. Arm 4 is Apache-2.0; Arm 8 is CC BY-NC 4.0. Verify the exact bytes before transfer or
  training—a model alias without recorded provenance is not sufficient.
- Datasets and downloaded weights are pinned by revision for reproducibility; large data, checkpoints, and
  `runs/` are never committed (see `.gitignore`).

## 8. Key references

xView3-SAR (Paolo et al., NeurIPS 2022) · SatDINO (Straka & Gruber, 2025) · SARMAE (Liu et al., CVPR 2026) ·
ViT AugReg (Steiner et al., 2021) · ImageNet (Deng et al., 2009) · ConvNeXt V2 (Woo et al., CVPR 2023) ·
reBEN / BigEarthNet v2.0 (Clasen et al., IGARSS 2025) · Rethinking Pre-training and Self-training
(Zoph et al., NeurIPS 2020) ·
Ultralytics YOLO26 (2026). The amended proposal bibliography covers the active study design; exact
checkpoint variants, revisions, hashes, and licenses remain canonical in [`DEVPLAN.md`](DEVPLAN.md) §1a.
