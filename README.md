# xView3 — Label-Efficient Dark-Vessel Detection in SAR

**Does SAR-domain pretraining beat optical — and does it hold across ViT and CNN?**

A controlled study of how much labeled data different classes of *pretrained backbone* save
for detecting **dark vessels** (ships not broadcasting AIS) in Sentinel-1 synthetic-aperture-radar
(SAR) imagery, under one fixed point detector.

*John Roth & Kyle Wagner · JHU EN.705.643 Deep Learning Developments with PyTorch*

> **Documentation map.** This README is the orientation. The authoritative design and phase plan
> is [`DEVPLAN.md`](DEVPLAN.md) — start with its **"Current repository state — READ FIRST"** cold-start
> runbook to see what is built vs. not. [`AGENTS.md`](AGENTS.md) is the short list of non-negotiables
> for any coding agent. [`proposal.pdf`](proposal.pdf) is the 1.5-page course proposal.

---

## 1. The research question

Labeled dark-vessel examples in SAR are scarce and expensive (human-verified). Pretrained backbones
promise to close that gap, but it is unclear **which kind of pretraining transfers best** to this
scarce-label regime — generic optical, SAR-specific, or labeled out-of-domain SAR — and **whether the
answer depends on the backbone architecture**.

We hold the detector, its head, its training schedule, and its input fixed, and vary only the
downloaded **initialization** (within a track) and the **architecture family** (across tracks):

> *Under one fixed point detector, how much labeled xView3 data does each class of pretrained
> backbone save for dark-vessel detection; does a SAR-domain checkpoint beat an optical one in the
> scarce-label regime; and does that finding hold across two architecture families (ViT and CNN)?*

**Hypothesis:** within each architecture, SAR-domain pretraining gives the steepest label-efficiency
gains and the largest dark-vessel-recall advantage at low label budgets, and this ordering is
*architecture-general* (holds for both the ViT and the CNN).

We **pretrain no foundation model** — all foundation backbones are downloaded. The only training we run
is a cheap supervised-detection backbone per architecture on LS-SSDD. This deliberately removes the
project's two largest risks (self-pretraining compute and novel pretraining code) and keeps the focus
on a clean, metric-matched comparison.

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
| 1 | ViT | ViT-B/16 | random init | ✅ | ViT floor |
| 2 | ViT | ViT-B/16 | **SatDINO** (fMoW-RGB, DINO) | ✅ | optical-domain FM transfer |
| 3 | ViT | ViT-B/16 | **SARMAE** (SAR-1M, MAE) | ✅ | SAR-domain FM transfer |
| 4 | ViT | ViT-B/16 | LS-SSDD supervised (we train) | ✅ | labeled SAR-detection transfer |
| 5 | CNN | ConvNeXt-V2-B | random init | ✅ | CNN floor |
| 6 | CNN | ConvNeXt-V2-B | **BigEarthNet-S2** (optical RS) | ✅ | optical-RS transfer |
| 7 | CNN | ConvNeXt-V2-B | **BigEarthNet-S1** (SAR RS) | ✅ | SAR-RS transfer |
| 8 | CNN | ConvNeXt-V2-B | LS-SSDD supervised (we train) | ✅ | labeled SAR-detection transfer |
| R1 | ref | ConvNeXt-V2-B | ImageNet (contingent, if time) | ❌ | generic-transfer reference |
| R2 | ref | YOLO26 | COCO-pretrained | ❌ | detector reference |
| R3 | ref | LocateAnything-3B | zero-shot | ❌ | VLM reference |

**Fairness by construction.** Within each track, all four arms share the same backbone, detection head,
optimizer, schedule, seeds, and the fixed input. The head/optimizer/schedule are shared across *both*
tracks too — only the backbone (and a small per-family head adapter) differs. Guard tests in CI enforce
the invariants that keep the arms comparable (see §6).

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

**Detector.** A point-native heatmap head (CenterNet / *Objects as Points* style, with the TRANSAR SAR
precedent): each vessel is a Gaussian blob on a **stride-4** response map, decoded by peak-finding with
**distance-based non-maximum suppression**. This matches xView3's point labels and its distance-based
official metric exactly, avoiding synthetic bounding boxes the data never had. One `PyTorch Lightning`
code path is shared across all arms.

**Reference arms** (reported separately, *not* on the controlled curves): YOLO26 (COCO init, Ultralytics),
LocateAnything-3B (zero-shot VLM), and a contingent ImageNet-ConvNeXt (RS-vs-generic-pretraining reference,
only if the eight core arms finish early).

---

## 4. Datasets

- **xView3-SAR** (via the **SARFish** mirror, `ConnorLuckettDSTG/SARFish`) — Sentinel-1 **GRD**, dual-pol
  (VH, VV). Training scenes are split **75 / 15 / 10** at the **scene level** (no chip-level leakage). The
  ~50 human-verified validation scenes form `eval_final` and are **touched exactly once**, at final eval.
  *SLC products are never fetched* (multi-TB, unneeded).
- **LS-SSDD-v1.0** (`TianwenZhang0825/LS-SSDD-v1.0-OPEN`) — native Sentinel-1, cut into 800×800 sub-images;
  the labeled supervised-transfer source for **Arms 4 and 8** only (never mixed into the xView3 splits).

**Input representation (study-wide, all arms):** a fixed 3-channel tensor **[VH, VV, VH−VV]** in dB, so the
input is physically identical across arms and channel handling is never a confound. Each backbone's stem
adapts to this input via the standard `timm` `in_chans` path.

**Labels / dark-vessel proxy:** detection positives are `is_vessel` with confidence ∈ {HIGH, MEDIUM};
LOW-confidence labels become ignore regions; a vessel is treated as **"dark"** when `source == "Manual"`
(no AIS correlate). Scoring is a per-scene distance-matched greedy F1 (200 m tolerance) with
dark-vessel recall and near-shore F1 slices.

---

## 5. Repository layout & status

This is a **plan-driven** research repo: [`DEVPLAN.md`](DEVPLAN.md) is the source of truth and most of the
pipeline is still to be built. Current state (see the DEVPLAN cold-start runbook for the live ledger):

- ✅ **Phase 2 (scorer + decode + thresholds)** — `src/eval/scorer.py` (frozen, per-scene
  distance-matched F1 + slices), `src/eval/decode.py` (peak-finding + distance-NMS), and
  `src/eval/threshold.py` (dev-selected operating point), with tests. Tagged `phase-2-done`.
- ✅ **Phase 0 (env/scaffold)** — CI, docs, `.gitignore`, `.gitattributes` (LF-normalized so the
  frozen-artifact hash pins hold on every platform), full `pyproject.toml` (`pip install -e .`),
  `Makefile`, `configs/data.yaml`, `scripts/gpu_sanity.py`, and the two GPU lockfiles under `locks/`
  (the 5070 Ti lock is a real freeze verified on the box; the V100-node lock is a candidate pin to
  re-freeze on the node — see its header).
- 🟡 **Phase 1 (data pipeline)** — code + tests done (`src/data/`: registration/download, chipper,
  centroid conversion, split builder; exercised end-to-end on a 7-scene subset of the locally
  downloaded full xView3 set). The three frozen artifacts (`data/splits.json`, `data/stats.json`,
  `data/lsssdd_split.json`) are **not yet built for real**: the xView3 label CSVs (auth-gated at DIU)
  and the LS-SSDD imagery (radars.ac.cn) are still to be fetched — see the DEVPLAN cold-start
  runbook, BLOCKER-4/5.
- ⬜ **Phases 3–8** (shared detector, arms, grid, final eval, analysis) — not started.

```
JHU-xView3/
  DEVPLAN.md          # single source of truth: design, phases, gates, risks
  AGENTS.md           # non-negotiables for any coding agent
  proposal.tex/.pdf   # 1.5-page course proposal
  requirements-ci.txt # minimal CPU deps for CI guard tests
  src/eval/           # scorer.py (frozen), decode.py, threshold.py  ← built
  tests/              # scorer / decode / guard tests
  .github/workflows/  # CI: anti-drift guard tests
  # planned: src/{data,models,train,references,analysis}, configs/, locks/, Makefile, data/, runs/
```

---

## 6. Getting started

```bash
git clone https://github.com/jroth1414/JHU-xView3
cd JHU-xView3

# 1) Install torch/torchvision from the index that matches YOUR machine FIRST
#    (never bare `pip install torch` — see DEVPLAN Appendix C and locks/):
#      5070 Ti (Blackwell sm_120):  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
#      V100 node (Volta sm_70):     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#    or reproduce a box exactly from its lockfile in locks/.

# 2) Then the package itself (the only sanctioned install):
pip install -e .[dev]

# 3) Sanity-check the GPU + kernels, then run the tests:
python scripts/gpu_sanity.py    # or: make env-check
pytest                          # or: make test
```

**GPU sanity outputs (P0.3).** RTX 5070 Ti box, recorded 2026-07-04 (torch 2.11.0+cu128, stable —
Blackwell no longer needs a nightly):

```
torch 2.11.0+cu128 | cuda build 12.8
device: NVIDIA GeForce RTX 5070 Ti | capability sm_120 | 15.9 GiB
fp16 matmul 4096x4096: ok (mean abs 51.06)
sdpa backends enabled: {'flash': True, 'mem_efficient': True, 'math': True}
sdpa: ok | first usable backend: mem_efficient
gpu_sanity: PASS
```

V100 node: not yet run — the node lock (`locks/env-v100node.txt`) is a candidate pin; re-freeze and
paste the sanity output here on first use (its header has the exact recipe).

**Continuous integration.** `.github/workflows/ci.yml` runs, on every push/PR to `dev`/`main`, the
anti-drift **guard tests** that encode study validity:

- `test_split_disjoint` — no scene appears in two splits (anti-leakage).
- `test_backbone_parity` — within-track param parity + matched head-adapter output geometry.
- `test_fm_checkpoints_load` — every downloaded backbone loads value-sensitively (not silently random).
- `test_scorer_immutable` — the frozen scorer's hash is unchanged.

Guards run only once their target files exist, so a fresh clone is green-with-skips for the not-yet-written
ones. The real training environment is **not** CI — see `locks/` (to be created) and DEVPLAN Appendix C for
the two GPU machines (an 8×V100 node, Volta/sm_70, fp16-only; and an RTX 5070 Ti, Blackwell/sm_120).

**Compute budget:** ~293–360 GPU-hours total (foundation models are downloaded, so there is no
from-scratch pretraining) — roughly two to three nights on the 8×V100 node.

---

## 7. Licensing & attribution notes

- **SARMAE** weights are **CC BY-NC 4.0** (non-commercial) and gated; **BigEarthNet** weight licenses must
  be verified at download. This project's use is academic/non-commercial; any released code or checkpoints
  are scoped accordingly. See the DEVPLAN risk register.
- Datasets and downloaded weights are pinned by revision for reproducibility; large data, checkpoints, and
  `runs/` are never committed (see `.gitignore`).

## 8. Key references

xView3-SAR (Paolo et al., NeurIPS 2022) · SatDINO (Straka & Gruber, 2025) · SARMAE (Liu et al., CVPR 2026) ·
ConvNeXt V2 (Woo et al., CVPR 2023) · reBEN / BigEarthNet v2.0 (Clasen et al., IGARSS 2025) ·
LS-SSDD-v1.0 (Zhang et al., 2020) · Rethinking Pre-training and Self-training (Zoph et al., NeurIPS 2020) ·
Ultralytics YOLO26 (2026). Full citations in [`proposal.tex`](proposal.tex).
