# DEVPLAN — xView3 Label-Efficient Dark Vessel Detection

Development plan for a coding agent (Claude Code or similar). Execute phases in order; each phase ends with explicit acceptance criteria. Do not start a phase until the previous phase's criteria pass, except where a phase is marked parallel-safe.

**Project question (the study).** Under one fixed point detector, how much labeled xView3 data does each class of *pretrained backbone* save for dark-vessel detection — does a SAR-domain checkpoint beat an optical one in the scarce-label regime, and **does that finding hold across architecture families (ViT and CNN)?** The headline deliverable is a **label-efficiency curve** with two matched tracks (ViT and CNN); the optical-vs-SAR contrast and its architecture-generality are the central results.

**What this is and is not.** This is a *controlled label-efficiency comparison of pretrained backbones across two architecture families*. Foundation-model backbones are downloaded; the only thing we train ourselves is a supervised-detection backbone per family (on LS-SSDD), which is cheap. This scope removes the project's two largest risks (self-pretraining a foundation model — its GPU-hours and novel code) while still answering both the domain-of-pretraining question and the architecture-generality question. The novelty is the systematic comparison — optical vs SAR pretraining, ViT vs CNN, on *dark-vessel, scarce-label* detection through a point-native detector — which to our knowledge has not been done.

**Design principle: architecture-matched fairness.** The study has **two tracks — a ViT track (ViT-B/16, ~86M) and a CNN track (ConvNeXt-V2-Base, ~89M, size-matched).** Within each track, all four arms share the *same* backbone architecture, detection head, fine-tuning schedule, and seeds; only the downloaded initialization differs. Across tracks, the two backbones are deliberately size-matched (~86M vs ~89M) so ViT-vs-CNN is a fair architecture comparison. Every claim about pretraining is made *within* a track (architecture held fixed); every architecture-generality claim is made by comparing *matched roles across* tracks (pretraining role held fixed). We report parameter count and GPU-hours per arm.

**Secondary deliverable removed.** The unconstrained challenge/leaderboard arm has been **cut** to protect the Sep 1 deadline; the controlled two-track study is the entire deliverable. An optional contingent reference arm (ImageNet-ConvNeXt) may be added if the eight core arms complete early (see Section 12).

## ⚑ Current repository state — READ THIS FIRST (cold-start runbook)

> **Purpose.** This section lets any fresh agent (or human) resume the project *from any point* using only the repo. Read it **before** AGENTS.md and before choosing a phase or a branch. **Section 1's repository layout below is the TARGET tree, not the current one** — most of it does not exist yet.
>
> **Update discipline.** Update the status ledger and add a `phase-N-done` git tag at every sprint merge. The `sprint-2b-eval-hardening` merge must be tagged `phase-2-done`; later phase merges follow the same pattern. If the ledger looks stale, rebuild it with the *state-detection runbook* below — **the repo is ground truth; this table is a cache.**

### Branch model (as actually built — this overrides the "main" phrasing elsewhere)
- The integration / default branch is **`dev`** (GitHub `HEAD → dev`), **not `main`**; `main` currently lags `dev`. Everywhere this plan or AGENTS.md says "land on `main`" / "no direct commits to `main`," read **`dev`**: open each sprint branch off `dev` and PR back into `dev`.
- Branches present: `dev` (active), `main` (behind `dev`), `sprint-2-scorer` (merged into `dev` via PR #1), and repair branch `sprint-2b-eval-hardening` until merged.
- CI must trigger on `dev` (see the CI-trigger fix in §1b) or it never runs on the active branch.

### Status ledger (ground truth as of this revision)
| Phase | Sprint branch | State | Evidence | Missing to reach DONE |
|---|---|---|---|---|
| 0 Env/scaffold | `sprint-0-env` | **DONE** | full `pyproject.toml` (`pip install -e .`), `Makefile`, `locks/env-5070ti.txt` (real on-box freeze; torch 2.11.0+cu128 stable — Blackwell needs no nightly anymore; `gpu_sanity` PASS pasted in README), `locks/env-v100node.txt` (candidate cu126 pin — re-freeze on the node per its header), `configs/data.yaml`, `scripts/gpu_sanity.py`, `.gitattributes` (LF-normalized checkouts so frozen-artifact sha256 pins hold on Windows) | on-node verification of the V100 lock + its `gpu_sanity` output (needs the physical node — non-blocking human item) |
| 1 Data/splits | `sprint-1-data` + `sprint-1b-freeze-splits` | **MOSTLY DONE — splits + lsssdd_split frozen; stats pending the chip job** | all P1 code + tests green; labels acquired and profiled (BLOCKER-4); **`data/splits.json` FROZEN** (150 scenes: 111/23/16 + 50 eval_final, seed 0, stratified from labels) pinned by `test_splits_immutable`; **`data/lsssdd_split.json` FROZEN** (seeded 90/10 over the verified 9,000 sub-images, 8,100/900) pinned by `test_lsssdd_split_immutable`; label projection validated on real scenes; chipping of the 150 study scenes running on the dev box (resumable `scripts/chip_study_scenes.py`, log `runs/logs/chip_study.log`) | `data/stats.json` (the driver computes it when chipping finishes; then pin it analogously + commit); then tag `phase-1-done` |
| 2 Scorer/decode/threshold | `sprint-2-scorer` + `sprint-2b-eval-hardening` | **DONE — scorer re-frozen after eval hardening; tagged `phase-2-done`** | `scorer.py` counts near-shore FPs and exposes per-scene aggregation; `threshold.py` owns dev threshold selection; `decode.py` rejects non-finite heatmaps; Phase-2 tests pass | — |
| 3 Detector | `sprint-3-detector` | NOT STARTED | — | `src/models/*`, `src/train/*`, `configs/detector.yaml` |
| 4 FM+floor arms+refs | `sprint-4/5/6` | NOT STARTED | — | — |
| 5 Supervised+grid | `sprint-7-grid` | NOT STARTED | — | — |
| 6 Final eval | `sprint-8-final-eval` | NOT STARTED | — | — |
| 7 Analysis | `sprint-9-analysis` | NOT STARTED | — | — |
| 8 Contingent ref (opt) | `sprint-10-ref-optional` | NOT STARTED | — | — |

### Known blockers — resolve before the affected phase (do not skip)
- **BLOCKER-1 — scorer freeze pin (RESOLVED).** The original pin in `tests/test_scorer_immutable.py` was mis-recorded at birth; `sprint-2b-eval-hardening` intentionally re-pinned the scorer after the BLOCKER-3 metric fix. Future scorer changes require human sign-off and a new pin.
- **BLOCKER-2 (RESOLVED).** Dev-tuned threshold selection lives in non-frozen `src/eval/threshold.py`; `scorer.py` stays pure scoring.
- **BLOCKER-3 (RESOLVED).** Near-shore F1 now counts unmatched FP predictions with `PredictionPoint.distance_from_shore_km <= 2.0`; missing shore distance on an unmatched FP raises. Dark-vessel remains a GT-defined recall slice, not a precision/F1 slice.
- **BLOCKER-4 (RESOLVED 2026-07-05).** Label CSVs acquired by the human: `D:\train.csv` (64,113 rows, 554 scenes), `D:\validation.csv` (19,224 rows, 50 scenes). **Measured label facts every later phase must respect (extends Appendix B):** (a) `source` values are LOWERCASE — `ais` / `manual` / `ais/manual`; the frozen scorer's dark test is `source == "Manual"`, so the eval adapter (P3 `infer_scene.py` / `final_eval.py`) MUST normalize case when building `GroundTruthPoint`s — the scorer itself stays frozen. (b) The **train CSV is 100% `ais`-sourced** — dark vessels (`manual`) exist ONLY in the validation/eval_final scenes (8,022 rows, 6,420 vessels), so the dark-recall slice is structurally empty on dev/test and is measurable only at final eval. (c) Train confidence↔is_vessel is degenerate: HIGH ⇒ is_vessel=False (16,692 fixed objects), MEDIUM ⇒ is_vessel=True (36,375 vessels), LOW ⇒ NaN (11,046, ignore); the standard positives rule (`is_vessel & conf∈{HIGH,MEDIUM}`) therefore selects exactly the 36,375 MEDIUM vessels in train. (d) bbox fields exist only in validation (19,049 rows); YOLO-reference boxes for train are always synthesized from points+lengths. (e) `distance_from_shore_km` uses a `9999.99` far-from-shore sentinel.
- **BLOCKER-5 (RESOLVED 2026-07-05).** (a) Local DIU-format archives are the canonical imagery source (the SARFish HF mirror hosts raw `SAFE.zip` products, not xView3 GeoTIFFs, and no labels); `download_sarfish.py --from-local` registers them. (b) LS-SSDD-v1.0 imagery landed (`D:\LS-SSDDv1.0.zip` from radars.ac.cn — the human's first pull, `Official-SSDD-OPEN.rar`, was SSDD, the wrong dataset, and is unused): 6,000+3,000 800×800 JPGs + 9,000 VOC XMLs verified, extracted to `data/raw/lsssdd/`, provenance in SOURCE.note, license Apache-2.0. **`data/lsssdd_split.json` FROZEN**: seeded 90/10 over all 9,000 sub-images (8,100/900), pinned by `test_lsssdd_split_immutable`. Documented deviation: the dataset's official 6,000/3,000 partition is a detection *benchmark* protocol; LS-SSDD is used here purely as a pretraining source (never evaluated), and the val side exists only for early stopping, so the plan's own P1.5 spec (seeded 90/10 by `splits.py`) governs.
- **Scene-count decision (human, 2026-07-05): 150 study scenes** of the 554 available — keeps the frozen ~300 GPU-h budget and the disk envelope; selected stratified (seed 0, 6 label-centroid region bins × shoreline<5 km), split 111 train / 23 dev / 16 test; all 50 verified scenes form `eval_final`. Frozen in `data/splits.json`, pinned by `tests/test_splits_immutable.py`.
- **Expected CI color.** `test_scorer_immutable` GREEN (pin corrected); `test_split_disjoint` + `test_splits_immutable` GREEN once the frozen `data/splits.json` lands on `dev`. The two not-yet-written guards (`test_backbone_parity`, `test_fm_checkpoints_load`) are *skipped* until their files exist, so the guard job is green-with-skips — not a fresh invariant violation. Cross-check any red against this ledger before treating it as a STOP.

### State-detection runbook (rebuild the ledger from the repo)
Run from the repo root; the repo, not this table, is ground truth:
```bash
git branch -a && git tag                                   # branches + phase-N-done tags
git log --oneline --all --decorate -20                     # what merged where
ls src/data src/models src/train configs locks 2>/dev/null # which phases are scaffolded
python -m pytest --collect-only -q                         # which tests exist / import cleanly
python -c "import hashlib;print(hashlib.sha256(open('src/eval/scorer.py','rb').read()).hexdigest())"  # vs the pin
```
Map: `src/data/` + `data/splits.json` present ⇒ Phase 1 underway/done; `configs/detector.yaml` present ⇒ Phase 3 underway; a `phase-N-done` tag ⇒ that phase merged.

### Per-phase gates
Each phase header below carries an **Entry preconditions** block (artifacts/tests that must already exist) and a **Definition of Done — machine-checkable** command whose exit code *is* the gate. Subjective "eyeball" checks are listed separately as **Human review (non-blocking for an unattended agent)** so a cold agent is never forced to wait on a human that is not there.

## The arms

The experiment is a set of **downloaded (or, for the two supervised arms, cheaply-trained) backbone initializations** fed through one fixed detector, in two architecture-matched tracks. The initialization is the only variable *within* each track; the architecture is the only variable *across* matched roles.

| Arm | Track | Backbone | Initialization | On curve? | Role |
|-----|-------|----------|----------------|-----------|------|
| 1 | ViT | ViT-B/16 (~86M) | random init | yes | ViT floor |
| 2 | ViT | ViT-B/16 | **SatDINO** (fMoW-RGB, DINO, downloaded) | yes | optical-domain FM transfer |
| 3 | ViT | ViT-B/16 | **SARMAE** (SAR-1M, MAE, downloaded) | yes | SAR-domain FM transfer |
| 4 | ViT | ViT-B/16 | LS-SSDD supervised (we train) | yes | labeled SAR-detection transfer |
| 5 | CNN | ConvNeXt-V2-B (~89M) | random init | yes | CNN floor |
| 6 | CNN | ConvNeXt-V2-B | **BigEarthNet-S2** (optical RS, downloaded) | yes | optical-RS transfer |
| 7 | CNN | ConvNeXt-V2-B | **BigEarthNet-S1** (SAR RS, downloaded) | yes | SAR-RS transfer |
| 8 | CNN | ConvNeXt-V2-B | LS-SSDD supervised (we train) | yes | labeled SAR-detection transfer |
| R1 | ref | ConvNeXt-V2-B | ImageNet (contingent, if time) | **no** | generic-transfer reference |
| R2 | ref | YOLO26 | COCO-pretrained | **no** | detector reference |
| R3 | ref | LocateAnything-3B | zero-shot | **no** | VLM reference |

The two tracks are matched by *role*: floor (1,5), optical-domain pretraining (2,6), SAR-domain pretraining (3,7), labeled SAR-detection transfer (4,8). Comparing arm *k* to arm *k+4* isolates architecture family with the pretraining role held fixed.

**Foundation models (downloaded, we do not pretrain them):**
- **SatDINO** (Straka & Gruber 2025, `strakajk/satdino-vit_base-16`): ViT-B/16, DINO self-distillation on fMoW-RGB optical satellite imagery. Apache-2.0. The ViT optical anchor.
- **SARMAE** (Liu et al., CVPR 2026; HF **weights** `Wenquandan777/SARMAE`, file `SARMAE_vitb_checkpoint-last`; **code** `github.com/MiliLab/SARMAE`): ViT-B/16 masked autoencoder pretrained on SAR-1M (million-scale SAR) with speckle-aware enhancement. CC BY-NC 4.0 (gated — accept terms). The ViT SAR anchor. *(The HF org `MiliLab/SARMAE` is the code mirror and is gated; the downloadable ViT-B weights are at `Wenquandan777/SARMAE` — verified. §1a is canonical for this id.)*
- **BigEarthNet ConvNeXt-V2-B** (TU Berlin RSiM/BIFOLD, reBEN; `BIFOLD-BigEarthNetv2-0/convnextv2_base-{s1,s2}-v0.2.0`): ConvNeXt-V2-Base supervised on BigEarthNet v2.0 land-cover classification, in Sentinel-1 (SAR) and Sentinel-2 (optical) variants. The CNN optical/SAR anchors. *These are supervised-classification pretrained, not self-supervised foundation models — a documented distinction from SatDINO/SARMAE; they still provide in-domain SAR/optical feature learning.*

**Cross-family method caveats (stated openly in the writeup, not hidden):**
- The two *ViT* FM arms differ in SSL method (SatDINO=DINO, SARMAE=MAE) as well as domain — no downloadable optical ViT-B MAE existed to match SARMAE's method, so we compare the best released FM of each domain as-is.
- The two *CNN* RS arms (BigEarthNet S1/S2) are the *cleanest* domain contrast in the study: identical model, identical dataset, identical supervised task, differing *only* in input modality (SAR vs optical). The CNN track is therefore the controlled domain comparison; the ViT track is the best-available-FM comparison.
- ViT FM arms are self-supervised; CNN RS arms are supervised-classification. This SSL-vs-supervised difference is a track-level property, documented, not a within-track confound.
- The ViT hands the head a stride-16 (32×32) latent while the ConvNeXt hands it a stride-32 (16×16) latent the CNN head upsamples once more; both decode on the same stride-4 (128×128) map, but the *pre-head* feature resolution and amount of learned upsampling differ by 2×. This is an inherent ViT-vs-CNN architecture-family property, reported alongside the cross-track (arm *k* vs *k+4*) gap as a limitation — not a defect the shared head removes.

**Channel representation (study-wide, all arms):** the fixed 3-channel [VH, VV, VH−VV] in dB. Every downloaded backbone was pretrained on a 3-channel-ish input that is not dual-pol SAR (SatDINO=optical RGB; SARMAE=single-pol SAR→3ch; BigEarthNet-S1=Sentinel-1 VV/VH 2-band; BigEarthNet-S2=10-band Sentinel-2), so each backbone's stem adapts to the [VH,VV,VH−VV] input during fine-tuning via the `timm` `in_chans` path. Bounded, documented, and applied identically to every arm.

**ImageNet is intentionally NOT a core arm** — it appears only as the optional contingent reference R1 (ConvNeXt-V2-B, ImageNet), gating the "RS-specific vs generic pretraining" question for the CNN track if time permits.

## 0. Ground rules

1. **The eval scorer is sacred.** Written first (Phase 2), unit-tested, never modified after Phase 3 begins. Every reported number in the project flows through `src/eval/scorer.py`.
2. **Initialization is the only variable *within* a track; architecture is the only variable *across* matched roles.** Two backbone definitions only: ViT-B/16 (arms 1–4) and ConvNeXt-V2-Base (arms 5–8). Within each track, one head, one optimizer config, one augmentation policy, one decode config, one fine-tuning schedule, shared seeds — every arm fine-tunes end-to-end, no frozen/fine-tuned asymmetry. The head/optimizer/schedule are shared across *both* tracks too (only the backbone differs), so ViT-vs-CNN is fair. If you are tempted to tune something per-arm, stop — that breaks the study. The two supervised arms (4, 8) additionally involve a cheap LS-SSDD backbone pretraining before the sweep; that is the only training we run.
3. **Scene-level splits only.** No chip from a dev/test/eval scene may appear in any training or pretraining corpus. Membership is keyed on `scene_id`, recorded once in `data/splits.json`; every dataloader asserts membership at construction.
4. **The ~50 human-verified xView3 validation scenes are touched exactly once,** by `src/eval/final_eval.py`, after the grid is complete. Tripwire: the script refuses to run without `--i-am-sure` and writes a timestamped lockfile on first use.
5. **Two machines, distinct roles.** Dev/iteration card: one RTX 5070 Ti (16 GB, Blackwell sm_120 — needs PyTorch nightly / cu12x). Heavy lifting: one **8× V100 node** (32 GB each, Volta sm_70). Volta constraints in Appendix C are mandatory — CUDA 12.x wheels, fp16 + GradScaler with norms in fp32, no bf16, no FlashAttention, no bitsandbytes. The two boxes have *different* CUDA/torch builds; pin each in its own lockfile.
6. **Core arms vs. references are reported separately.** The eight core arms (1–8) go in the study tables and the two-track label-efficiency figure. The reference arms (R1 ImageNet-ConvNeXt if run, R2 YOLO26, R3 LocateAnything) go in a separate "references" section for context; they are not on the controlled curves. There is no challenge/leaderboard arm.
7. **Determinism.** Every run takes `--seed`; seed torch, numpy, random, and dataloader workers (Lightning's `seed_everything(workers=True)`). Log the resolved config (full YAML), git SHA, and an environment hash into the run directory.
8. **Run directories.** Every experiment writes `runs/<exp_id>/` with `config.yaml`, `metrics.csv` (per-epoch), `final_metrics.json`, `checkpoints/`, `log.txt`. Experiment IDs come from the manifest in Section 12 (Run manifest) — never invent ad-hoc names.
9. **One framework for every arm: PyTorch Lightning.** All training entrypoints (the eight arms' fine-tuning, plus the two supervised-transfer backbone trainings on LS-SSDD for arms 4 and 8) are `LightningModule`s sharing one `LightningDataModule` and one `Trainer` config, so DDP across the 8×V100 node, mixed-precision, seeding, and checkpointing are identical across arms by construction. Backbones load as plain `nn.Module`s inside the LightningModule (timm for both ViT-B and ConvNeXt-V2-B; SatDINO/SARMAE via their loaders; BigEarthNet ConvNeXt via the `configilm`/reBEN loader). One head class attaches to either backbone via a small **per-family** adapter: ViT stride-16 tokens (32×32) reshape then upsample to the stride-4 map; the ConvNeXt stride-32 stage-3 map (16×16) needs one **extra** upsample block to reach the same stride-4 map (see P3.3). Both decode on the same stride-4 (128×128) output, but pre-head feature resolution differs by 2× — an inherent ViT-vs-CNN property reported as a cross-track limitation, not something the shared head equalizes. Keep configs in YAML; avoid hydra. This consistency is itself part of the fairness argument.
10. **Fairness accounting.** Every arm logs parameter count and fine-tuning GPU-hours. Pretraining GPU-hours are 0 on our side for the six downloaded/random arms (1,2,3,5,6,7) and external/cited for the four downloaded FMs. The two supervised-transfer arms (4, 8) each run a cheap LS-SSDD backbone pretraining, measured. These populate a fairness table in the writeup. Within each track all four arms share architecture and fine-tuning compute; across tracks the two backbones are size-matched (~86M vs ~89M).
11. **Commit per task** with messages referencing the task ID (e.g. `P1.3: scene-level split builder`). Never commit data, checkpoints, or anything under `runs/`.
12. **Follow cited methods; do not invent.** For any component that names a reference method (Section 1a), implement that method's published recipe — do not substitute a "better," "simpler," or "more modern" approach reasoned up independently, however plausible it seems. Novelty in this project lives in the *experimental comparison*, not in re-deriving pretraining or channel-adaptation mechanics. If a reference is ambiguous, unavailable, or two references conflict, STOP and surface the question to a human rather than improvising — a silently-wrong method (e.g. a hand-rolled patch-embed hack or an ad-hoc reconstruction target) corrupts every downstream number without throwing an error.

## 1. Repository layout

**This is the TARGET tree, not the current one** — see the cold-start runbook at the top of this file for what actually exists today (Phase 0 is only PARTIAL). Create the missing scaffolding as part of `sprint-0-env`.

```
JHU-xView3/
  README.md
  DEVPLAN.md                  # this file
  pyproject.toml              # base deps; see two lockfiles below
  locks/
    env-5070ti.txt            # cu12x / nightly for sm_120
    env-v100node.txt          # cu12x for sm_70 (no bf16/flash)
  Makefile                    # one target per phase entrypoint
  configs/
    data.yaml                 # paths, chip size, split fractions, seed
    detector.yaml              # SHARED: head, optimizer, aug, decode (never edited per study-arm)
    arms.yaml                 # the run matrix (Section 13)
  data/                       # gitignored; symlink to large storage
    raw/
      xview3/                 # SARFish GRD GeoTIFFs + label CSVs
      lsssdd/                 # LS-SSDD-v1.0 (Arm 4 supervised source)
    chips/                    # 800x800 chips + per-chip JSON sidecars
    splits.json               # scene_id -> split (frozen after sprint-1)
    stats.json                # train-split-only per-pol mean/std (frozen after sprint-1)
    lsssdd_split.json         # LS-SSDD internal train/val split (frozen after sprint-1)
    manifests/                # parquet chip manifests per split/source
  src/
    data/
      download_sarfish.py     # selective HF GRD pulls
      download_aux.py         # LS-SSDD fetcher (+ BigEarthNet ConvNeXt weights via HF)
      chipper.py              # scene -> chips + label projection (xView3 + LS-SSDD)
      to_centroids.py         # boxes/masks -> center points (LS-SSDD)
      splits.py               # scene-level split builder
      datasets.py             # FineTuneDataset, SupervisedSARDataset
      transforms.py           # log-norm, crops, flips
    models/
      backbones.py            # ViT-B/16 + ConvNeXt-V2-B, both in_chans=3 (VH,VV,VH-VV)
      heatmap_head.py         # deconv tower + 1ch sigmoid head + per-family adapter
      init_loaders.py         # 8 loaders: vit_random/satdino_b/sarmae_b/vit_supervised + cnn_random/bigearthnet_s2/bigearthnet_s1/cnn_supervised
    train/
      lit_modules.py          # LightningModules: supervised-pretrain, finetune
      datamodule.py           # one LightningDataModule (chips, splits, channel rep)
      pretrain_supervised.py  # entrypoint: heatmap pretrain on LS-SSDD (run twice: ViT->Arm4, CNN->Arm8)
      finetune.py             # entrypoint: shared fine-tune loop (all 8 arms, both tracks)
      sampler.py              # foreground-balanced chip sampler
      losses.py               # penalty-reduced focal
    eval/
      decode.py               # peaks, distance NMS
      scorer.py               # distance-matched F1 + slices  (SACRED)
      infer_scene.py          # tiled whole-scene inference
      final_eval.py           # verified-scenes tripwire script
    references/
      yolo26_ref.py           # boxes from points+lengths, ultralytics run
      locateanything_zs.py    # zero-shot grounding VLM probe
      imagenet_cnn_ref.py     # contingent R1: ImageNet ConvNeXt-V2-B fine-tune (Phase 8, optional)
    analysis/
      curves.py               # two-track label-efficiency plots
      error_slices.py         # dark-vessel / shoreline breakdowns (both tracks)
      qualitative.py          # chip galleries with predictions
      architecture_comparison.py # ViT-vs-CNN matched-role summary
  tests/
    test_scorer.py
    test_decode.py
    test_chipper.py
    test_centroids.py
    test_splits.py
    test_split_disjoint.py      # guard: no scene_id in two splits (1b)
    test_backbone_parity.py     # guard: within-track param parity (ViT arms=ViT-B, CNN arms=ConvNeXt-V2-B) (1b)
    test_fm_checkpoints_load.py  # guard: SatDINO,SARMAE,BigEarthNet-S1,BigEarthNet-S2 load w/ expected keys (1b)
    test_scorer_immutable.py    # guard: scorer.py hash matches pinned value (1b)
  runs/                       # gitignored
```

## 1a. Reference implementations — adapt these, do not reinvent

Every component below has a canonical paper and/or repository. **The agent's job is to adapt the reference implementation, not to design a new method.** When a detail is unspecified by this plan, copy the reference's choice rather than inventing one. Pin versions; for dataset and downloaded-weight sources a **revision pin is mandatory** (see the note below the table). **This table is the single canonical source for every reference id — any other mention in this plan or in AGENTS.md must match it verbatim.**

| Component | Follow this reference | What to copy |
|---|---|---|
| ViT-B backbone (shared by all arms) | `timm` `vit_base_patch16_224` | the one architecture every arm uses; FM checkpoints load into it |
| Arm-3 weights (ViT SAR FM) | SARMAE — HF **weights** `Wenquandan777/SARMAE`, file `SARMAE_vitb_checkpoint-last` (code `github.com/MiliLab/SARMAE`; Liu et al. CVPR 2026) | ViT-B SAR-1M checkpoint load; encoder only, drop optical-branch/decoder; CC BY-NC 4.0 (gated). **Canonical id — arm list, P3.2, and AGENTS.md must match this verbatim.** |
| Arms 6,7 weights (CNN RS) | BigEarthNet ConvNeXt-V2-B, `BIFOLD-BigEarthNetv2-0/convnextv2_base-{s2,s1}-v0.2.0` (reBEN, Clasen et al. IGARSS 2025) | ConvNeXt-V2-Base; load via `configilm` `BigEarthNetv2_0_ImageClassifier.from_pretrained`, take the backbone, drop the classification head; S2=optical, S1=SAR |
| CNN backbone + stem adaptation | `timm` `convnextv2_base` + `in_chans=3` | ConvNeXt-V2-Base as the CNN arm backbone; stem takes the fixed [VH,VV,VH−VV] input via `in_chans`, same Repeat-with-rescaling policy as the ViT stem |
| CNN floor / supervised / R1 | `timm` `convnextv2_base` | Arm 5 = trunc-normal random init; Arm 8 supervised-pretrains on LS-SSDD like Arm 4; R1 (contingent) = ImageNet FCMAE weights |
| Backbone + channel adapt | `timm` (`vit_base_patch16_224`, `in_chans`) | ViT-B/16 definition; `in_chans` Repeat-with-rescaling for any channel change |
| Arm-2 weights (optical FM) | SatDINO, `strakajk/satdino-vit_base-16` (Straka & Gruber 2025, arXiv 2508.21402) | ViT-B/16 fMoW-RGB DINO checkpoint; `AutoModel.from_pretrained(..., trust_remote_code=True)`; Apache-2.0 |
| Heatmap head + decode | CenterNet (Objects as Points); SAR precedent TRANSAR | penalty-reduced focal target, peak + distance-NMS decode |
| Supervised SAR detection (Arm 4) | LS-SSDD-v1.0 repo conventions | box→centroid conversion, 800px sub-image handling |
| Arm R3 weights (VLM reference) | LocateAnything-3B — **repo id UNVERIFIED**: resolve before `sprint-6`/P4.6 | zero-shot grounding on ~200 dev chips, prompts {ship,vessel,boat}; centers through the SACRED scorer. R3 is off the controlled curves and is **DROPPABLE** — if the id cannot be verified, skip it (do not guess a checkpoint). |

**Source-revision pinning is mandatory (not "where practical").** Every dataset and downloaded-weight source must be pinned to an exact revision — HuggingFace `revision=<commit-sha>`, GitHub tag/commit — recorded in a `SOURCE.note` beside the download. The frozen `data/splits.json` assumes identical scene contents on any re-pull, so an unpinned source silently voids reproducibility (see P1.1/P1.2 and the risk register).

If a referenced repo is unavailable, a citation is ambiguous, or two references conflict, **stop and flag it for a human** — do not paper over the gap with an improvised method (see ground rule 12). This applies to **any** named reference, including the R2/R3 reference arms, not only the §1a table rows.

## 1b. Execution discipline — sprints, gates, and stop conditions

This section governs *how* the plan is executed, not what it builds. Its purpose is to limit agent drift: the failure mode where an agent, run unsupervised against a large spec, gradually optimizes for "make this file work" instead of "serve the study," and silently substitutes plausible-but-wrong choices that never throw an error. Every rule here shrinks the unsupervised interval, makes failures loud, or gates progression behind a human.

### Schedule (calendar)

The project runs **July 1 to September 1, 2026** (about nine weeks); the final project is due **Sep 1**. All dates in phase headers below are calendar dates on this window. Because foundation models are downloaded (no self-pretraining of an FM), the schedule is gated by engineering and analysis, not training compute (~293–360 GPU-hours total, roughly three to four nights on the node). High-level mapping of phases to dates:

| Dates (2026) | Focus (phases) |
|---|---|
| Jul 1–14 | Data pipeline + sacred scorer + detector (Phases 0–3, parallel lanes) |
| Jul 15–28 | Both tracks' FM + floor arms (ViT: SatDINO, SARMAE; CNN: BigEarthNet S1/S2); channel-format check; references (Phase 4) |
| Jul 29–Aug 11 | Supervised-transfer arms (ViT+CNN on LS-SSDD) + full label-fraction grid + seed reruns (Phase 5) |
| Aug 12–25 | Final verified-scene eval + ViT-vs-CNN error analysis; begin writeup (Phases 6–7) |
| Aug 26–Sep 1 | Finish writeup and submit; optional contingent ImageNet-ConvNeXt reference only if time remains (Phase 8) |

The writeup deliberately overlaps the Aug 12–25 analysis block rather than starting cold at the end; the final week is for finishing and submission, not first drafting. The optional contingent reference arm (Phase 8) is the schedule's release valve — cut it first if anything runs late, since the eight-arm study is the deliverable. (The unconstrained challenge arm was removed entirely to protect the deadline.)

### Git and repository

- **Remote:** `https://github.com/jroth1414/JHU-xView3`.
- **Authorship — strict.** All commits are authored by the repository owner (the human). Set `git config user.name` and `git config user.email` to the owner's identity. **Do NOT add `Co-Authored-By:` trailers, "Generated with …", "🤖", or any AI/agent attribution** in commit messages, commit bodies, PR titles, or PR descriptions. The agent is a tool, not a listed author. (Note for the human: some agent CLIs inject attribution by default; disable it in the agent's own settings too, and verify the first few commits land clean — the repo instruction is necessary but may not be sufficient on its own.)
- **Integration branch is `dev`, not `main`.** The GitHub default/HEAD branch is `dev`; `main` currently lags `dev` (see the cold-start runbook). Read every "`main`" reference in this plan as `dev` until the team explicitly decides otherwise.
- **Branching:** one branch per sprint, named as below, opened off `dev`. No direct commits to `dev` or `main`.
- **Merge = human gate.** Every sprint lands on `dev` via a pull request the human reviews. The PR is where drift is caught before it compounds. Keep PRs small enough to actually read line-by-line; if a sprint's diff is growing past a few hundred lines, split it.
- **Commits:** small and frequent, each referencing the task ID (e.g. `P1.3: scene-level split builder`). Small commits make the review diff legible and `git bisect` cheap.

### Sprints (one branch each; mirror the phases, no renumbering)

Each sprint branch carries a short `SPRINT.md` stating its goal, its acceptance criteria (copied from the referenced phase), and its definition of done — so the agent on that branch holds a *small* contract in context, not the whole plan. Sprints are tagged by **review tier**, which sets how much human attention the merge gets — vigilance is finite, so spend it where drift actually corrupts the study.

| Sprint (branch) | Phase | Review tier | Why this tier |
|---|---|---|---|
| `sprint-0-env` | Phase 0 | Leaf | tooling only; can't corrupt results |
| `sprint-1-data` | Phase 1 | **Foundation** | a leaky split silently poisons every run |
| `sprint-2-scorer` | Phase 2 | **Foundation** | defines correctness for every reported number |
| `sprint-3-detector` | Phase 3 | Spine | the detector *is* the fairness guarantee |
| `sprint-4-vit-arms` | Phase 4 (Arms 2,3) | Spine | ViT FM arms (SatDINO, SARMAE) + channel-format check |
| `sprint-5-cnn-arms` | Phase 4 (Arms 6,7) | Spine | CNN backbone integration + BigEarthNet S1/S2 arms |
| `sprint-6-floor-refs` | Phase 4 (rest) | Leaf | both floors (1,5) + external refs (R2,R3); off the controlled curves |
| `sprint-7-grid` | Phase 5 | Spine | two supervised arms (4,8) + full label-fraction grid |
| `sprint-8-final-eval` | Phase 6 | **Foundation** | touches the once-only verified-scene eval |
| `sprint-9-analysis` | Phase 7 | Leaf | ViT-vs-CNN figures/slices; read the output, trust the code |
| `sprint-10-ref-optional` | Phase 8 | Leaf | contingent ImageNet-ConvNeXt reference, only if core arms done |

- **Foundation tier** — review every line; slow, careful merge. These define or consume ground truth.
- **Spine tier** — review that the acceptance criteria are genuinely met; spot-check the implementation. These carry the study's validity.
- **Leaf tier** — review the output for sanity; trust the code. These cannot corrupt the controlled comparison.

**Hard branch-ordering constraint:** no model-code sprint (`sprint-3` onward) may open a PR until **`sprint-2-scorer` is merged to `dev`**. Building arms against a still-moving scorer makes every early number untrustworthy. Likewise `sprint-1-data` must merge before any sprint that trains on chips.

### Sub-agent distribution

Sub-agents are safe only on **independent** work; they drift badly when they share hidden state. Match distribution to the two-owner lane split, not to "one agent per sprint":

- **Parallelizable lanes:** the data pipeline (`sprint-1`) and the scorer (`sprint-2`) are independent and can run as parallel sub-agents from day one. The **external** reference models — R2/R3 only (`yolo26_ref.py`, `locateanything_zs.py`, in `sprint-6`) — are independent of the study spine and may run as a parallel sub-agent. (There is **no pretraining lane** — all four FMs are downloaded — which removes the project's old long-pole background job. The two LS-SSDD supervised-transfer backbones for Arms 4/8 are trained inside `sprint-7-grid` under the serial detector owner, **not** as a parallel lane.)
- **Must be serial — one owner:** every sprint that touches the shared detector (`sprint-3-detector → sprint-4-vit-arms → sprint-5-cnn-arms → sprint-6-floor-refs`) modifies the same `lit_modules.py`/`finetune.py`. In particular `sprint-5-cnn-arms` is Spine-tier CNN backbone/detector integration (see the sprint table) — it is on this serial chain and must **not** be run as an independent parallel sub-agent. The floor arms (1, 5) also live in `sprint-6` but run *through* the shared detector; only the external R2/R3 references are the parallel exception. Do **not** parallelize the detector sprints across sub-agents — they would each edit the shared code and silently diverge, reintroducing the per-arm inconsistency the fairness design exists to prevent. One agent owns the detector; arms run through it sequentially.
- Do not subdivide finer than these lanes; coordination overhead and shared-state drift outgrow the speedup.

### Mandatory STOP conditions (halt and ask a human — not optional)

Stopping is cheap and expected; it is not a sign of failure. An agent that asks ten questions costs minutes; one that silently invents a wrong split costs the study. **Halt and surface the question** when any of these holds:

1. A cited reference (Section 1a) is unavailable, ambiguous, or two references conflict. (ground rule 14)
2. An acceptance criterion fails and the fix is not obvious within one attempt.
3. A result violates a sanity check — e.g. a label-efficiency curve non-monotone beyond seed noise, or an arm scoring implausibly high (suspect leakage).
4. Any change would touch a frozen artifact (see do-not-touch manifest) after it is locked.
5. The agent is tempted to deviate from a cited method "because it seems better." (ground rule 14)
6. A step's measured compute or time exceeds its budget by more than ~2× (catches runaway runs before they burn the node).

Do not optimize to "seem autonomous." Surfacing these is the job.

### Do-not-touch manifest (frozen after their sprint; changes require a STOP)

Once merged, these are locked; modifying them requires an explicit human decision because changing them silently invalidates earlier results:

- `src/eval/scorer.py` — frozen after `sprint-2` (ground rule 1).
- `data/splits.json` — frozen after `sprint-1`; scene-level membership never changes mid-study.
- `data/stats.json` — frozen after `sprint-1`; train-split-only per-polarization mean/std, computed ONCE over the 100%-train scene set and reused unchanged for every label fraction, both tracks, and all seeds (see P1.5).
- `data/lsssdd_split.json` — frozen after `sprint-1`; the LS-SSDD internal train/val partition consumed identically by the Arm-4 (ViT) and Arm-8 (CNN) supervised pretrainings (see P1.5).
- `configs/detector.yaml` — the shared head/optimizer/schedule, frozen after `sprint-3`; this is the fairness contract.
- The verified-scene eval protocol / its lockfile — touched exactly once (ground rule 4).

Each frozen artifact should carry a machine-checkable freeze guard analogous to `test_scorer_immutable` (a pinned sha256 activated at its sprint merge): `test_splits_immutable` (`data/splits.json`, sprint-1), `test_detector_immutable` (`configs/detector.yaml`, sprint-3). This gives "Phase 1 done" / "Phase 3 done" the same binary, cold-agent-checkable signal the scorer freeze has.

### Guard tests (machine-enforced, wired to pre-merge CI)

Prose checklists depend on someone remembering to run them — and the drift problem *is* that agents stop remembering. These four assertions live in `tests/` and run on every PR, converting silent corruption into a failed build. They are the technical embodiment of ground rule 14. See P2.3 and P3.x for where they attach; the canonical list:

1. **`test_split_disjoint`** — no `scene_id` appears in more than one split; also asserted at datamodule construction. Kills leakage.
2. **`test_backbone_parity`** — *within-track* param parity: the four ViT arms instantiate the identical ViT-B/16 param count, and the four CNN arms the identical ConvNeXt-V2-Base param count. **Plus cross-track output-geometry parity:** push a `(1,3,512,512)` tensor through one ViT adapter and one ConvNeXt adapter and assert both emit `(1, C, 128, 128)` with the *same* `C` and the *same* stride-4 — so a ConvNeXt adapter wired to the wrong stride (e.g. one too few upsample blocks → stride-8) or the wrong `C` fails CI even though all four CNN arms remain param-identical to each other (see P3.3). Kills accidental architecture divergence *and* head-adapter geometry drift that would silently void the cross-track (arm *k* vs *k+4*) comparison. (Cross-track *param* sizes are close but not identical — ~86M vs ~89M — reported, not asserted equal.)
3. **`test_fm_checkpoints_load`** — all four downloaded backbones load **value-sensitively**, not merely by key name: SatDINO (ViT-B, via `trust_remote_code`), SARMAE (ViT-B), and BigEarthNet ConvNeXt-V2-B S1 and S2 (via the `configilm`/reBEN loader). Assert `len(missing_keys)==0` **and** that a fixed sample of encoder tensors (e.g. `patch_embed`/`blocks.0`) differs — by norm/hash, a minimum L2 threshold — from a fresh trunc-normal init of the same architecture. A bare `strict=False` "no exception" is **not** sufficient: a prefix/nesting mismatch (SatDINO `ckpt['teacher']`, SARMAE encoder-only) can report a good key match while leaving the backbone substantially random — the exact "arm quietly running on random weights" this guard exists to catch. Split the guard into **(a)** a CPU-offline structural check comparing loaded key names against a committed manifest (runs in CI, no downloads, no `transformers`/`configilm` required) and **(b)** a GPU-machine integration check that actually loads the weights and runs the value-sensitivity assertion (runs on the training boxes, not CI). CI runs **only (a)**.
4. **`test_scorer_immutable`** — a hash of `scorer.py` matches the pinned value recorded at `sprint-2` merge; checked before any eval runs. Kills scorer drift.

A sprint cannot merge with a failing guard test. If a guard test must legitimately change (rare), that is itself a STOP.

These tests are enforced by **GitHub Actions CI** (`.github/workflows/ci.yml`), which runs them on every push and PR — so enforcement does not depend on any agent or human remembering to run them. CI is CPU-only and uses a minimal `requirements-ci.txt` (not the GPU training environment); guard tests must therefore run against tiny in-repo fixtures. A companion **`AGENTS.md`** at the repo root gives any coding agent the short list of non-negotiables (this section is the detail behind it); AGENTS.md is the instruction layer, CI is the enforcement layer, and this DEVPLAN is the single source of truth.

**CI trigger & expected color (cold-start).** `ci.yml` must trigger on **`dev`** (the active branch), not only `main`, or it never runs where work happens. The guard job runs each of the four guard files **only if present**, so on a fresh clone it is **green-with-skips** for the three not-yet-written guards (`test_split_disjoint`, `test_backbone_parity`, `test_fm_checkpoints_load`). `test_scorer_immutable` exists and is **GREEN** (its pin was corrected — see BLOCKER-1 in the cold-start ledger). Once a guard's target file exists it becomes a hard, merge-blocking gate. CI installs `scipy` (imported by `decode.py`) and needs no model downloads (it runs only guard half (a), the structural checkpoint check).

## 2. Phase 0 — Environment and smoke checks

Targets both machines. Do first.

- **P0.1** Create the repo skeleton under the actual repo root `JHU-xView3/` (Section 1): a `.gitignore` covering `data/`, `runs/`, `*.pt`, `*.pth`, `*.ckpt`, `*.tif`, `*.tiff`, `*.parquet`, `__pycache__/`, `.pytest_cache/`; a `pyproject.toml` declaring the `src`-layout package + base deps (`pip install -e .` is the only sanctioned install); imports resolve as absolute `from src.…` from the repo root — guarantee it with either an `__init__.py` in every `src/` subpackage **or** `[tool.pytest.ini_options] pythonpath=["."]`, not pytest's accidental `sys.path` insertion; and a `README.md` documenting install + both GPU boxes. **Commit a minimal `.gitignore` now** — it is zero-risk and is the only thing enforcing ground rule 11 ("never commit data or checkpoints").
- **P0.2** Two lockfiles, because the boxes differ:
  - `locks/env-v100node.txt` — torch/torchvision pinned to a **single resolved cu12x** wheel (pick one and record it, e.g. `torch==2.x.x+cu128 torchvision==0.x.x+cu128 --index-url https://download.pytorch.org/whl/cu128`), plus `timm`, `numpy`, `pandas`, `pyarrow`, `rasterio`, `shapely`, `scipy`, `matplotlib`, `huggingface_hub`, `lightning` (PyTorch Lightning), `ultralytics`, `pytest`, `transformers` (SatDINO's `AutoModel`/`trust_remote_code`), and **`configilm`** (the reBEN loader for the BigEarthNet ConvNeXt arms — pin a version that ships `BigEarthNetv2_0_ImageClassifier`). **Never `pip install torch` bare here** — a cu13 wheel will refuse sm_70.
  - `locks/env-5070ti.txt` — the **same package set**, differing **only** in the `torch`/`torchvision` lines: pin a specific dated Blackwell nightly (e.g. `torch==2.x.x.devYYYYMMDD+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128`) so the sm_120 env is not silently re-resolved each day.
  - **Generation recipe (so the lockfile is a lock, not a moving target):** on each machine install the resolved set and freeze — `pip install <resolved set> && pip freeze > locks/env-<machine>.txt` (or `uv pip compile pyproject.toml --index-url <resolved-url> -o locks/env-<machine>.txt`). Record the CUDA toolkit version each box was built against in the README.
  README documents both and which machine runs what.
- **P0.3** `scripts/gpu_sanity.py`: print device name + capability, run a 4096×4096 fp16 matmul and an `F.scaled_dot_product_attention` call, assert no NaN, report the SDPA backend chosen. Run on both machines; paste both outputs into the README.
- **P0.4** `Makefile` targets — one per phase entrypoint **and** per manifest exp-id: `make env-check`, `make test`, `make data`, `make qa` (renders the P1.5/P3.5 QA galleries `runs/qa/chips.png`, `runs/qa/pred_gallery.png`), `make pretrain-sup-vit`, `make pretrain-sup-cnn`, `make grid`, `make references` (the **non-optional** Phase-4 R2/R3 arms → `yolo26-f100`, `locateanything-zs`), `make final-eval CONFIRM=1` (forwards `--i-am-sure`; the flag must **not** be baked into the recipe — ground rule 4), and `make ref-optional` (the **optional** Phase-8 R1). (No FM-pretraining target — all four FMs are downloaded; the only trainings are the two LS-SSDD supervised backbones.)

**Entry preconditions:** none — this is the bootstrap phase. Work on `sprint-0-env` off `dev`.

**Definition of Done — machine-checkable:** `test -f pyproject.toml && test -f Makefile && test -f .gitignore && test -f locks/env-v100node.txt && test -f locks/env-5070ti.txt && pip install -e . && python -m pytest --collect-only -q` exits 0.

**Human review (non-blocking for an unattended agent):** `make env-check` passes on both machines; README shows both `gpu_sanity` outputs and documents the Volta pins. (These need the two physical GPUs and a human eyeball — do not block the unattended DoD on them.)

## 3. Phase 1 — Data acquisition, chipping, splits

Owner: data owner. Jul 1–14. GPU: none (CPU + disk).

### P1.1 xView3 / SARFish download — `src/data/download_sarfish.py`
- `huggingface_hub.snapshot_download(repo_id="ConnorLuckettDSTG/SARFish", repo_type="dataset", revision="<commit-sha>", allow_patterns=[...])`, patterns restricted to **GRD only, never SLC** (SLC is the bulk of multi-TB). Pin the exact dataset `revision` and record it (with the md5s) in `data/raw/xview3/SOURCE.note` — the frozen `data/splits.json` assumes identical scene contents on any re-pull.
- Inputs: a scene-ID list. Outputs: `data/raw/xview3/GRD/<scene_id>/...` + md5 verification per the SARFish instructions.
- Pull: (a) 75–150 train scenes chosen in P1.3, (b) all ~50 human-verified validation scenes. *(The 150 public GRD products for an SSL corpus are no longer pulled — the self-supervised/challenge scope was cut; see Appendix D.)*
- Labels: fetch SARFish CSVs; schema in Appendix B. Note the dataset card's country-restriction warning in the README.
- Budget guard: print estimated size before downloading; abort if > configurable cap (default 400 GB).

### P1.2 Auxiliary SAR datasets — `src/data/download_aux.py`
Fetch and register, each into `data/raw/<name>/`:
- **LS-SSDD-v1.0** (Arms 4 & 8 supervised source): native Sentinel-1, 15 large scenes already cut into 9,000 800×800 sub-images. GitHub `TianwenZhang0825/LS-SSDD-v1.0-OPEN` — pin a specific **tag/commit** and record it in `data/raw/lsssdd/SOURCE.note`.
- (Removed sources: **OpenSARShip** and the pooled SAR-ship sets HRSID / SAR-Ship-Dataset / SSDD were only for the cut SSL/challenge scope and are **no longer fetched** — see Appendix D. LS-SSDD is now the *only* auxiliary dataset.)
- Record license per source in `data/raw/<name>/LICENSE.note`. Do not proceed to use a set whose license note is missing. (Downloaded model **weights** get their own license gate in P3.2.)

### P1.3 Chipping — `src/data/chipper.py`
- Read VH + VV GeoTIFFs with rasterio in windows; emit 800×800 chips, 100 px overlap, stride 700 (xView3). LS-SSDD already arrives as 800×800, so just normalize + register it.
- Pixel pipeline: raw → dB-like log `x = log10(clip(raw, 1, None))`; store float16 log values; normalize at load time with global per-polarization mean/std computed over the **train split only** (`data/stats.json`).
- **Channel representation (study-wide, fixed for all arms).** Every arm — including SatDINO, which was pretrained on 3-channel RGB — receives the *same* 3-channel SAR input: **[VH, VV, VH−VV]** in normalized dB. Rationale: (a) VH and VV are distinct polarimetric measurements, and their difference (≈ co/cross-pol contrast) is a physically meaningful third channel rather than a zero-pad or a duplicate; (b) using one fixed 3-channel tensor everywhere means the input is *physically identical* across all arms, so channel handling is never a confound between them; (c) 3 channels lets the two **ViT** FMs (SatDINO, SARMAE), each pretrained with a 3-channel patch-embed, load their stem with no channel-count change, while the two **CNN RS** backbones (BigEarthNet S1 = 2-band, S2 = 10-band) adapt their stem to 3 channels via the same `timm` `in_chans` path. This single decision replaces all per-arm channel hacks. Document it once; do not vary it per arm.
- **Channel adaptation — use the named library method, do not invent one.** The two **ViT** FMs load their patch-embed with **no channel-count change** (SatDINO is 3-ch RGB; SARMAE is single-pol→3-ch). The two **CNN RS** arms require a shape-changing stem adaptation to the fixed 3-channel input — BigEarthNet-S1 (2-band VV/VH → 3) and BigEarthNet-S2 (10-band → 3) — done via **`timm.create_model(..., in_chans=3)`** (Repeat-with-rescaling, identical mechanism for both, differing only in the pretrained starting weights; see P3.2). So it is one documented mechanism applied to every arm, but a no-op for the ViT FMs and a real 2→3 / 10→3 expansion for the CNN arms — *not* "no surgery for any arm." If at any point a channel count must change (e.g. an ablation), the ONLY sanctioned mechanism is **`timm.create_model(..., in_chans=N)`**, which implements the field-standard *Repeat* method (tile/average the pretrained RGB projection weights across the new channels and rescale to preserve activation magnitude). Do NOT hand-roll patch-embed weight manipulation, and do NOT derive a "better" scheme. The single documented alternative, allowed only as an explicit ablation, is **RGB+random** (USat: keep pretrained weights for the RGB-equivalent channels, randomly initialize any extra channels). References to follow, not redesign: `timm` (`in_chans` rescaling); USat (2023); the init-strategy survey arXiv 2503.09493 (which documents that *Repeat* and *RGB+random* are the two known options and that neither is universally best — i.e. this is a bounded choice, not an open problem). A third named method, IC-ViT (isolated-channel patchify, arXiv 2503.09826), is noted in Appendix D as considered-but-not-default; do not adopt it without a human decision.
- Drop chips >95% no-data; record land fraction per chip where shoreline vectors exist.
- Label projection: per-chip JSON sidecar with vessel points in chip-pixel coords, `confidence`, `source`, `vessel_length_m`, `distance_from_shore_km`, and bbox fields when present.
- Manifest: one parquet per scene/source — chip path, scene_id, origin row/col, n_vessels, has_low_conf, land_frac.

### P1.4 Centroid conversion — `src/data/to_centroids.py`
- The detector trains on **center points**, so every labeled supervised source is reduced to centroids. Box → center; rotated box → center; instance mask → centroid. This is the move that lets LS-SSDD feed the same heatmap head (for both the ViT Arm 4 and CNN Arm 8 supervised backbones) with no box-format harmonization.
- Intensity normalization of LS-SSDD to the common dB range used study-wide.
- `tests/test_centroids.py`: box/rbox/mask → expected center within 1 px.

### P1.5 Scene selection and splits — `src/data/splits.py`
- Stratify chosen xView3 train scenes by coarse region (cluster scene-center lat/lon into ~6 bins) and shoreline presence (any label `distance_from_shore_km < 5`).
- Scene-level split of xView3 train scenes: **75% train / 15% dev / 10% test** (seeded). The ~50 verified scenes form `eval_final`, excluded from everything. *(There is no `corpus_extra`/unlabeled SSL split — the SSL scope was cut; see Appendix D.)*
- **`data/stats.json` (owned here, frozen).** Compute global per-polarization (VH, VV) mean/std over **only the frozen train-split `scene_id`s** (never dev/test/eval_final). The 100%-train statistics are computed **once** and reused unchanged for every `label_frac` (10/25/50/100%), for **both** the ViT and CNN tracks, and for all seed reruns — `stats.json` is never re-derived per fraction or per track (referenced from P1.3).
- **`data/lsssdd_split.json` (owned here, frozen).** LS-SSDD gets a fixed, seeded internal train/val split (e.g. 90/10) materialized to this file by `src/data/splits.py`. Both the Arm-4 (ViT) and Arm-8 (CNN) supervised pretrainings **read the identical `data/lsssdd_split.json`** (never re-derive it inside the training script), so the matched-role pair cannot diverge. It is a *pretraining* source, never mixed into the xView3 splits.
- `data/splits.json` maps scene_id → split; `tests/test_splits.py` asserts disjointness and counts, that `stats.json`'s contributing scenes are a subset of the train split, and that the LS-SSDD partition is fixed/seeded/disjoint.

**Entry preconditions:** `sprint-0-env` merged (`pyproject.toml`, `.gitignore`, `configs/data.yaml` exist); raw data pulled (P1.1/P1.2) with `SOURCE.note` revision pins present.

**Definition of Done — machine-checkable:** `pytest tests/test_chipper.py tests/test_centroids.py tests/test_splits.py tests/test_split_disjoint.py && test -f data/splits.json && test -f data/stats.json && test -f data/lsssdd_split.json` exits 0. On merge, pin `data/splits.json` via `test_splits_immutable` (do-not-touch manifest) and tag `phase-1-done`.

**Human review (non-blocking for an unattended agent):** per-split + per-source chip counts logged; `make qa` renders a 4×4 gallery of random chips with label points overlaid (`runs/qa/chips.png`) for a human eyeball.

## 4. Phase 2 — Scorer and decode (before any model code)

Owner: detector owner. Jul 1–14 (parallel-safe with Phase 1).

### P2.1 `src/eval/scorer.py` (SACRED — pure scoring only)
- Inputs: predictions `[(x_m, y_m, score)]` and ground truth `[(x_m, y_m, attrs)]` per scene, in meters (chip px × 10 m GSD, offset by chip origin).
- Greedy matching: sort predictions by score desc; each matches the nearest unmatched GT within `tol_m = 200`. Matched → TP; unmatched prediction → FP; unmatched GT → FN.
- Outputs: precision, recall, aggregate F1, plus sliced metrics on attrs: `dark` (`source == "Manual"`, no AIS correlate; reported as recall), `near_shore` (`distance_from_shore_km <= 2`; reported as F1 with unmatched near-shore FPs counted), and a `low_conf_ignore` protocol where LOW-confidence GT neither count as FN nor award TP.
- Near-shore slice FPs are assigned from prediction-side `distance_from_shore_km`; an unmatched FP prediction missing that field raises. This is the sprint-2b resolution of BLOCKER-3.
- The threshold sweep does NOT live in the frozen scorer. Operating-point selection lives in `src/eval/threshold.py` (P2.2b), so `scorer.py` stays pure and hash-pinned.

### P2.2 `src/eval/decode.py`
- `peaks = (heat == maxpool3x3(heat)) & (heat > tau)` → candidates with scores.
- Distance NMS at `d_nms_m = 120` (config): sort by score, suppress candidates within `d_nms` of a kept one. Use `scipy.spatial.cKDTree` so whole-scene decode stays fast.
- **Heatmap-px → meters unit contract (owned by `infer_scene.py`, machine-checked).** The head output is stride-4 (P3.3) and GSD is 10 m, so one output pixel = `4 × 10 = 40 m`. Callers must pass `decode_heatmap(..., output_stride_m = 40.0)` and then add the chip-origin offset before handing points to the scorer; the `output_stride_m=1.0` default in `decode.py` is **for unit tests only**. `src/eval/infer_scene.py` is the single site performing `heatmap-px → chip-px (×4) → meters (×10) → scene-meters (+origin)`. Put `decode_stride_px: 4`, `gsd_m: 10`, `output_stride_m: 40` in `configs/detector.yaml`. A caller left at the `1.0` default feeds the scorer coordinates 40× too small — inside the 200 m tolerance and 120 m NMS radius — producing plausibly-high but meaningless F1 with no error.

### P2.2b `src/eval/threshold.py` (NOT frozen)
- The operating-point logic P2.1 keeps out of the frozen scorer. Given raw scored predictions, return the **F1-maximizing threshold selected on dev**; freeze it for test/eval and apply it verbatim (no re-selection on test/eval). Consumes `scorer.py`'s `score_points()` output. `P5.4`/`P6.1` reference this module as the single home for threshold selection — no per-arm code reimplements it.

### P2.3 Tests
- `tests/test_scorer.py`: exact hit; hit at 199 m (TP); miss at 201 m (FN+FP); two predictions on one GT (1 TP + 1 FP); score-order priority; LOW-confidence ignore behavior; dark recall; near-shore slice FPs; per-scene dataset aggregation so coordinates cannot match across scenes.
- `tests/test_decode.py`: plateau handling, NMS suppression order, empty heatmap, non-finite heatmap rejection, and the 40 m/px unit assertion — a peak at output-pixel `(r,c)` with `output_stride_m=40.0` lands at `(40r, 40c)` m, pinning the decode↔scorer coupling before the freeze.
- `tests/test_threshold.py`: the dev-selected threshold is applied unchanged to a held-out set (assert no re-selection on test).
- `tests/test_scorer_immutable.py`: CWD-independent guard path and pinned scorer hash after sprint-2b eval hardening.

**Scorer freeze protocol (who / when / how — this is BLOCKER-1's resolution).**
1. **Resolve BLOCKER-2 and BLOCKER-3 first** so `scorer.py` is in final form (pure scoring; near-shore slice-FP decision made).
2. The **detector owner** records the pinned sha256 in the **final commit before the `sprint-2` merge** — `python -c "import hashlib;print(hashlib.sha256(open('src/eval/scorer.py','rb').read()).hexdigest())"` — noting the merge SHA beside the pin.
3. From that commit the scorer is frozen (ground rule 1, do-not-touch manifest). A hash **mismatch on a fresh clone is a STOP** to surface, never a silent re-pin.
4. **Current state:** the pin has been intentionally re-recorded to `85dec7ab…` after sprint-2b eval hardening fixed near-shore slice FPs and added per-scene aggregation. A future hash mismatch is a STOP.

**Entry preconditions:** `sprint-0-env` merged (package installable). Phase 2 is *parallel-safe* with Phase 1 (touches no data artifacts).

**Definition of Done — machine-checkable:** `pytest tests/test_scorer.py tests/test_decode.py tests/test_threshold.py tests/test_scorer_immutable.py` exits 0. Scorer reproduces a synthetic scene's known P/R exactly, counts near-shore FPs, and refuses cross-scene matching via `score_dataset()`. Tag `phase-2-done` on merge.

## 5. Phase 3 — Models and the shared fine-tune pipeline

Owner: detector owner. Jul 8–21. Dev card: 5070 Ti. This detector is frozen once Phase 4 begins (ground rule 1–2). It must support **two backbone families** (ViT-B/16 and ConvNeXt-V2-Base) behind one head and one training loop.

### P3.1 `src/models/backbones.py`
- **ViT arm backbone:** ViT-B/16 via timm (`vit_base_patch16_224`, `in_chans=3` on the fixed [VH,VV,VH−VV] input), learnable pos-embed interpolated to 512×512 inputs (32×32 tokens), final norm kept. SatDINO and SARMAE are both this architecture.
- **CNN arm backbone:** ConvNeXt-V2-Base via timm (`convnextv2_base`, `in_chans=3` on the same input). Use `features_only=True` or take the stage-3 feature map; ConvNeXt-V2-B at 512 input gives a stride-32 feature map (16×16, 1024-dim). BigEarthNet arms are this architecture.
- Both expose a uniform `(feature_map, channels, stride)` interface to the head adapter. ViT: reshape the stride-16 token grid to (B, 768, 32, 32). ConvNeXt: the stride-32 (B, 1024, 16, 16) stage-3 map.
- Smaller-variant fallback (ViT-S/16 for the ViT track, ConvNeXt-Tiny for the CNN track) documented in `detector.yaml`; the trigger is **STOP condition #6** (a step's measured compute exceeds ~2× its budget, §1b) and the "Backbone too slow on the node" risk-register row — **not** a "P6 throughput check" (P6 is the once-only Final Eval; never run throughput probing against it). If invoked, drop the whole track to the smaller variant uniformly, preserving within-track parity.

### P3.2 `src/models/init_loaders.py`
Eight loaders behind one enum, four per track. Each prints matched/missing/unexpected key counts after loading. **Within a track all four produce the identical backbone; only the weights differ.**

*ViT track (all `vit_base_patch16`, 768-dim):*
- `vit_random`: timm trunc-normal init.
- `satdino_b`: **SatDINO ViT-B/16** fMoW-RGB checkpoint (`strakajk/satdino-vit_base-16`, DINO; `AutoModel.from_pretrained(..., trust_remote_code=True)`, or manual `vit_base(patch_size=16)` + `load_state_dict(ckpt['teacher'])`). Backbone features (768-dim); drop the DINO projection head. Apache-2.0.
- `sarmae_b`: **SARMAE ViT-B/16** (`Wenquandan777/SARMAE`, file `SARMAE_vitb_checkpoint-last`; CC BY-NC 4.0, accept terms). `vit_base_patch16`, loads with no surgery; encoder weights only, ignore optical-branch/SARC/decoder.
- `vit_supervised`: the Phase 5 ViT supervised-pretrained encoder (LS-SSDD); shapes match.

*CNN track (all `convnextv2_base`, 1024-dim stage-3):*
- `cnn_random`: timm trunc-normal init.
- `bigearthnet_s2`: **BigEarthNet ConvNeXt-V2-B, Sentinel-2 (optical)** (`BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0`; via `configilm` `BigEarthNetv2_0_ImageClassifier.from_pretrained`, take the backbone, drop the multi-label classification head). The optical-RS CNN anchor.
- `bigearthnet_s1`: **BigEarthNet ConvNeXt-V2-B, Sentinel-1 (SAR)** (`...convnextv2_base-s1-v0.2.0`, same loader). The SAR-RS CNN anchor. Pretrained on Sentinel-1 VV/VH (2-band); the stem adapts to the 3-channel [VH,VV,VH−VV] input via `in_chans`.
- `cnn_supervised`: the Phase 5 CNN supervised-pretrained backbone (LS-SSDD); shapes match.

**Matched-role structure:** arm *k* (ViT) and arm *k+4* (CNN) share a pretraining role — floor (`vit_random`/`cnn_random`), optical (`satdino_b`/`bigearthnet_s2`), SAR (`sarmae_b`/`bigearthnet_s1`), supervised (`vit_supervised`/`cnn_supervised`). Within-track contrasts isolate pretraining; cross-track matched-role contrasts isolate architecture family.

**Cross-family caveats (documented):** ViT FM arms are self-supervised (DINO/MAE); CNN RS arms are supervised land-cover classification. The BigEarthNet S1-vs-S2 contrast is the cleanest domain comparison (same model/dataset/task, only modality differs); the SatDINO-vs-SARMAE contrast additionally differs in SSL method. Neither is a within-track confound.

- Assert after loading (`tests/test_fm_checkpoints_load.py`, §1b guard): all four downloaded backbones (SatDINO, SARMAE, BigEarthNet-S1, BigEarthNet-S2) load **value-sensitively** (§1b guard 3) — no silent shape-mismatch, no partial/random load. Within each track all four arms expose an identical backbone interface to the head.
- **Weights license gate (mirrors the P1.2 dataset gate).** On download of each backbone, write `data/weights/<name>/LICENSE.note` recording the exact license and its commercial/redistribution terms (SatDINO Apache-2.0; SARMAE CC BY-NC 4.0; BigEarthNet-S1/S2 — verify at download). Refuse to load a backbone whose `LICENSE.note` is missing. CC BY-NC forbids commercial use and constrains redistribution of derived checkpoints — keep the writeup and any released weights scoped non-commercial (see risk register).

### P3.3 `src/models/heatmap_head.py`
- One head class, attached to either backbone via a small adapter that maps the backbone feature map to a common (B, C, H, W) at the head's input stride. ViT: (B,768,32,32) stride-16. ConvNeXt: (B,1024,16,16) stride-32, so the CNN head uses one extra upsample block to reach the same output stride.
- Upsample blocks `[ConvTranspose2d, GroupNorm(32), GELU]` to output **stride 4** (128×128 for a 512 input), then 3×3 conv → 1 channel; sigmoid at inference, logits at train.
- The head, loss, optimizer, and schedule are identical across both tracks; only the backbone-to-head adapter differs by architecture (this is the minimal, documented cross-track difference, not a per-arm one).

### P3.4 Targets, loss, sampler
- `src/train/losses.py`: penalty-reduced focal (CenterNet form), α=2, β=4, on Gaussian targets σ=2 output px; LOW-confidence labels stamp an ignore disk (radius 3 output px) where loss is zeroed.
- `src/train/sampler.py`: epoch-level weighted sampler so ~50% of sampled chips contain ≥1 HIGH/MED vessel.
- `src/data/transforms.py`: random 512 crop from the 800 chip (vessel-biased: 70% of crops centered within 128 px of a vessel when one exists), flips, 90° rotations, intensity jitter ±0.1 in log space. No augmentation that breaks SAR statistics (no blur, no elastic).

### P3.5 `src/train/finetune.py` — the shared loop
- AdamW, lr 1e-4, layer-wise lr decay 0.65, weight decay 0.05, cosine schedule, 5-epoch warmup, 50 epochs, batch 16 (fp16 + GradScaler, norms fp32), grad clip 1.0.
- `--init {vit_random,satdino_b,sarmae_b,vit_supervised,cnn_random,bigearthnet_s2,bigearthnet_s1,cnn_supervised}` selects the loader and implicitly the backbone family. **Within each track all four fine-tune end-to-end with the identical schedule** (no frozen path, no per-arm optimizer differences). Head, loss, sampler, augmentation, decode, schedule, and seeds are identical across *both* tracks; only the backbone (and its head adapter) differs. Only the loaded weights differ within a track.
- `--label_frac {0.1,0.25,0.5,1.0}` subsamples xView3 train scenes (scene-level, seeded); fractions **nest** (10% ⊂ 25% ⊂ 50% ⊂ 100%) so the curves are monotone in data.
- Every 5 epochs: tiled inference (`infer_scene.py`, 512 windows, stride 384, global NMS) on 8 fixed dev scenes → dev F1. Early-stop patience 4 dev evals. Save best + last.

**Acceptance:** smoke run (`--init vit_random --label_frac 0.1 --epochs 3`) completes on the 5070 Ti, loss decreases, dev decode yields >0 detections, `runs/qa/pred_gallery.png` overlays look sane. Each of `--init satdino_b`, `sarmae_b`, `cnn_random`, `bigearthnet_s2`, `bigearthnet_s1` loads its backbone with the expected key report and trains identically within its track. `tests/test_backbone_parity.py` and `tests/test_fm_checkpoints_load.py` (§1b guards) pass.

### P3.6 Downloaded-backbone load + early-signal check (cheap, before the full grid)
The four downloaded backbones all fine-tune normally, so there is no frozen-feature dependency to de-risk. Early checks before committing the full grid: (1) confirm all four (SatDINO, SARMAE, BigEarthNet-S1, BigEarthNet-S2) load into their track's backbone with the expected key match (the `test_fm_checkpoints_load` guard); (2) confirm a short fine-tune (a few epochs at 100% labels) of each trains stably and beats its track's random-init at equal epochs — a quick sanity that the downloaded weights are a usable start on the [VH,VV,VH−VV] input. The highest-risk unknown is the **channel-format gap** (every backbone pretrained on a 3-channel-ish input that is not dual-pol SAR; BigEarthNet-S1 is closest at Sentinel-1 VV/VH). This check catches a catastrophic stem mismatch early, per track. Record in `runs/decisions.md`. (A downloaded arm underperforming its floor at low fractions is a *finding* about transfer, not a bug.)

**Entry preconditions (Phase 3):** `sprint-1-data` and `sprint-2-scorer` merged to `dev`; `data/splits.json`, `data/stats.json`, `data/lsssdd_split.json` exist; `test_split_disjoint` and `test_scorer_immutable` green (scorer freeze valid — BLOCKER-1 resolved).

**Definition of Done — machine-checkable (Phase 3):** `pytest tests/test_backbone_parity.py tests/test_fm_checkpoints_load.py && test -f configs/detector.yaml && python -m src.train.finetune --init vit_random --label_frac 0.1 --epochs 3 --smoke` exits 0. On merge, pin `configs/detector.yaml` via `test_detector_immutable` and tag `phase-3-done`.

**Human review (non-blocking):** the P3.5 smoke overlays (`runs/qa/pred_gallery.png`) look sane.

## 6. Phase 4 — Both tracks' foundation-model + floor arms, and references

Owner: detector owner. Jul 15–28. One config per V100 per night. **No FM pretraining in this project** (all FMs downloaded), so the old "long pole" is gone; this phase plus the grid is the bulk of the compute, and it is cheap.

**Ordering rationale.** Run the four downloaded-backbone arms (2,3,6,7) early, because the highest-risk unknown is the **channel-format gap** per track: whether the fixed [VH,VV,VH−VV] tensor loads cleanly into each pretrained stem. The P3.6 check catches a catastrophic mismatch before the full grid, once per architecture. Since the channel representation is study-wide, validating it on the FM arms de-risks the shared input for every arm. The two floors (1,5) and references need no special handling and run alongside.

*ViT track:*
- **P4.1** Arm 2 (`satdino_b`) and Arm 3 (`sarmae_b`) — run first at 100% labels via the P3.6 check; confirm stable transfer, then all four label fractions each — 8 runs.
- **P4.2** Arm 1 (`vit_random`) at all four label fractions — 4 runs.

*CNN track:*
- **P4.3** Arm 6 (`bigearthnet_s2`) and Arm 7 (`bigearthnet_s1`) — same pattern: P3.6 load+signal check at 100%, then all four fractions each — 8 runs. Confirm the `configilm`/reBEN load path and the ConvNeXt stem's channel adaptation.
- **P4.4** Arm 5 (`cnn_random`) at all four label fractions — 4 runs.

*References:*
- **P4.5** Arm R2 — `src/references/yolo26_ref.py`: build YOLO boxes from xView3 points+lengths (square side `max(6 px, length_m/10)` centered on the point; SARFish bbox fields where present), train YOLO26 (ultralytics, COCO init) on the 100% train split; score its centers through the SACRED scorer. 1 run.
- **P4.6** Arm R3 — `src/references/locateanything_zs.py`: LocateAnything-3B zero-shot on ~200 dev chips, prompts {"ship","vessel","boat"}; runs on the 5070 Ti (bf16-checkpoint caution, Appendix C.5); centers through the scorer.

**Acceptance:** 24 study runs (arms 1,2,3,5,6,7 × 4 fractions) + 2 reference runs have `final_metrics.json`; a partial two-track label-efficiency plot (6 curves) renders from `curves.py`; the channel representation is confirmed working on real SAR for *both* backbone families via the four FM arms.

**Entry preconditions (Phase 4):** `sprint-3-detector` merged (`configs/detector.yaml` frozen); each downloaded backbone's `data/weights/<name>/LICENSE.note` written (P3.2); the P3.6 channel-format check recorded per track in `runs/decisions.md`.

**Definition of Done — machine-checkable (Phase 4):** a `runs/<exp_id>/final_metrics.json` exists for arms 1,2,3,5,6,7 × 4 fractions + `yolo26-f100` + `locateanything-zs` (or R3 explicitly skipped per its droppable note); `curves.py` renders the partial figure without error. Tag `phase-4-done`.

## 7. Phase 5 — Supervised-transfer arms (4, 8) + the label-efficiency grid

Owner: detector owner. Jul 29–Aug 11. This runs the only two backbone trainings in the project (the ViT and CNN supervised-transfer backbones on LS-SSDD); everything else is downloaded.

- **P5.1** `src/train/pretrain_supervised.py`: pretrain a backbone + heatmap head on **LS-SSDD-v1.0 only**, centroids as targets (P1.4), same loss/sampler/aug as the fine-tune pipeline. Run it **twice** — once with the ViT-B backbone (→ `vit_supervised`, Arm 4) and once with the ConvNeXt-V2-B backbone (→ `cnn_supervised`, Arm 8). Same single matched source keeps each supervised arm's contrast clean within its track. ~10–30 GPU-h each; these are the only backbones we train.
- **P5.2** Arm 4 (`vit_supervised`) and Arm 8 (`cnn_supervised`) fine-tuned at all four label fractions — 8 runs. (Arms 1,2,3,5,6,7 already ran in Phase 4; this completes the eight-arm grid.)
- **P5.3** Seed reruns: the 10%, 25%, and 100% cells of **all eight** study arms with 2 extra seeds — 48 runs — so the scarce-label headline and high-label endpoints carry error bars. One config per card per night; ~6 nights across the 8-GPU node.
- **P5.4** Per run: freeze the dev-tuned threshold, score the test split, append to `runs/summary/grid.csv` via `curves.py`; render the two-track label-efficiency figure (x = label fraction, log-scale; y = test F1; eight curves — solid for ViT, dashed for CNN, one color per pretraining role; shaded seed bands).
- **P5.5** Headline computations: (a) *within-track* — arm ordering at 10% labels for each track (optical vs SAR vs supervised vs floor); (b) *cross-track* — for each pretraining role, the ViT-vs-CNN gap at each fraction (does the SAR-domain advantage hold for both architectures?); (c) the interpolated label budget at which each arm matches the SAR-FM arm @25% within its track.

**Acceptance:** `grid.csv` has 32 core (8 arms × 4 fractions) + 48 seed rows, no NaNs, plus per-fraction scene/vessel/dark-proxy/near-shore counts; the eight-curve two-track figure renders; a `monotonicity_ok` boolean per arm (F1 non-decreasing in label fraction, within seed noise) is emitted into `grid.csv` and must be **true** for every arm — a false value is a STOP (§1b rule 3), not a soft "investigate."

**Entry preconditions (Phase 5):** Phase 4 arms landed; `data/lsssdd_split.json` frozen; `src/eval/threshold.py` exists (P2.2b).

**Definition of Done — machine-checkable (Phase 5):** `test -f runs/summary/grid.csv` and a check that `grid.csv` has 80 rows, zero NaNs, per-fraction count columns populated, and `monotonicity_ok == true` for all arms, exits 0; the eight-curve figure renders. Tag `phase-5-done`.

## 8. Phase 6 — Final eval

- **P6.1** FINAL EVAL (once): best config per study arm at 10%, 25%, and 100% scored on the verified scenes via `final_eval.py --i-am-sure`. These are the study's headline numbers; nothing is tuned after this.

**Entry preconditions (Phase 6):** Phase 5 complete (`grid.csv` full, monotonicity green); dev-tuned thresholds frozen (P2.2b/P5.4); the verified-scene lockfile does **not** yet exist (this eval runs exactly once).

**Acceptance / Definition of Done — machine-checkable (Phase 6):** `test -f runs/summary/final_verified.csv` and the verified-eval lockfile exists. This is the once-only eval — nothing is tuned after it. Tag `phase-6-done`.

## 9. Phase 7 — Error analysis and figures (study)

Owner: detector owner. Aug 12–25 (begin writeup in parallel).

- **P7.1** `error_slices.py`: per-arm dark-vessel recall and near-shore F1 vs label fraction, for **both tracks**; FP taxonomy on ~200 sampled FPs (shoreline clutter / fixed infrastructure / sea clutter / sidelobe). Two headline slices: the **SAR-vs-optical dark-vessel-recall gap within each track**, and the **ViT-vs-CNN gap at matched pretraining roles** (does the SAR-domain advantage generalize across architectures?).
- **P7.2** `qualitative.py`: a fixed gallery of 24 chips (8 dark-vessel hits, 8 misses, 8 FPs) rendered identically for all eight study arms — the money figure beside the two-track curve.
- **P7.3** `architecture_comparison.py`: the cross-track summary — for each of the four pretraining roles, ViT vs CNN F1 across fractions, with the BigEarthNet-S1-vs-S2 contrast (cleanest domain comparison) called out.

**Entry preconditions (Phase 7):** Phase 6 final numbers written (`runs/summary/final_verified.csv`).

**Definition of Done — machine-checkable (Phase 7):** `make qa` plus `curves.py` / `error_slices.py` / `architecture_comparison.py` produce the two-track figure, the dark-vessel / near-shore slice tables, and the 24-chip gallery without error. Tag `phase-7-done`.

**Human review (non-blocking):** figures legible; the money gallery reads clearly.

## 10. Phase 8 — Contingent reference arm (R1, OPTIONAL)

Owner: either. Aug 26–Sep 1, **only if the eight core arms have landed and time remains before the deadline.** Reported in the references section, not on the controlled curves. (The unconstrained challenge/leaderboard arm was removed entirely from the project to protect the deadline; this contingent reference replaces it as the schedule's release valve.)

- **P8.1** Arm R1 — `bigearthnet`-style but ImageNet: load ImageNet-pretrained ConvNeXt-V2-B (`timm` `convnextv2_base`, FCMAE ImageNet weights), fine-tune through the identical detector at all four label fractions — 4 runs. This is the "generic natural-image pretraining" baseline for the CNN track: comparing R1 to Arm 6/7 (BigEarthNet RS pretraining) answers *how much of the CNN's transfer comes from RS-specific pretraining vs. generic visual pretraining?*
- **P8.2** Add R1's curve to the references panel (not the controlled two-track figure). One paragraph on the RS-vs-generic-pretraining gap for CNNs.

**Entry preconditions (Phase 8):** all eight core arms landed (Phases 4–5 done) **and** time remains before Sep 1 — otherwise skip; the study is complete without R1.

**Acceptance (only if run):** R1's four cells in `runs/summary/references.csv`; a note on the RS-vs-generic gap. If skipped for time, the eight-arm study is complete and unaffected.

## 11. Hyperparameter reference

| Component | Setting | Value |
|---|---|---|
| Chips | size / overlap / GSD | 800 px / 100 px / 10 m |
| Train crops | size / vessel-biased frac | 512 / 0.7 |
| Heatmap | stride / sigma / loss | 4 / 2 out-px / penalty-reduced focal α=2 β=4 |
| Decode | tau / d_nms / match tol / output_stride | dev-tuned / 120 m / 200 m / 40 m (= stride-4 × 10 m GSD) |
| Fine-tune | opt / lr / lld / wd / epochs / batch | AdamW / 1e-4 / 0.65 / 0.05 / 50 / 16 (identical both tracks) |
| ViT-track backbone | ViT-B/16 | ~86M params, embed-dim 768, fine-tuned end-to-end |
| CNN-track backbone | ConvNeXt-V2-Base | ~89M params, stage-3 1024-dim, fine-tuned end-to-end |
| Init (ViT, downloaded) | arms 2 / 3 | SatDINO fMoW-RGB (DINO) / SARMAE SAR-1M (MAE) |
| Init (CNN, downloaded) | arms 6 / 7 | BigEarthNet-S2 optical / BigEarthNet-S1 SAR (ConvNeXt-V2-B, supervised) |
| Channel input | fixed all arms | [VH, VV, VH\textminus VV] in dB (3-channel) |
| Supervised pretrain | source / target | LS-SSDD-v1.0 / centroids → heatmap (run twice: ViT + CNN) |
| Ignore regions | LOW-conf radius | 3 out-px |
| Splits | train/dev/test of xView3 train scenes | 75/15/10 scene-level |
| Label fractions | nested | 10 / 25 / 50 / 100% |

## 12. Run manifest (experiment IDs)

Study grid — `exp = {init}-f{frac}-s{seed}`, init ∈ ViT {`vitrand`,`satdino`,`sarmae`,`vitsup`} + CNN {`cnnrand`,`beS2`,`beS1`,`cnnsup`}:
- Core (8 arms × 4 fractions, seed 0): 32 runs — e.g. `satdino-f25-s0`, `beS1-f10-s0`.
- Seed reruns (`-f10-`, `-f25-`, and `-f100-`, seeds 1–2, all 8 arms): 48 runs.

References: `yolo26-f100`, `locateanything-zs`.
Backbone trainings we run (not fine-tunes): `vitsup-lsssdd`, `cnnsup-lsssdd` (the two supervised-transfer backbones).
Contingent reference (optional): `imgnetcnn-f{10,25,50,100}-s0` — only if core arms finish early.

Total = 80 study fine-tunes (32 core + 48 seed) + 2 reference runs + 2 supervised-transfer backbones + up to 4 optional contingent-reference fine-tunes. **No foundation-model pretraining** — all four FMs downloaded. Each fine-tune fits one V100 overnight; the 8-GPU node clears the whole grid in several nights.

### GPU-hour estimates (planning, not measured)

Order-of-magnitude only. Assumes 8×V100, 512px crops; ConvNeXt-V2-B ≈ ViT-B cost at this resolution (CNN cells may run ~1.2–1.5× heavier — treated as noise here).

| Component | Runs | ~GPU-h each | ~GPU-h total |
|---|---|---|---|
| Core grid (8 arms × 4 fractions) | 32 | 3 | ~96 |
| Seed reruns (10% + 25% + 100% cells, 8 arms) | 48 | 3 | ~144 |
| Supervised-transfer backbones (LS-SSDD: ViT + CNN) | 2 | 10–30 | 20–60 |
| YOLO26 reference | 1 | 5–15 | ~10 |
| LocateAnything zero-shot (5070 Ti) | 1 | 1–2 | ~2 |
| SatDINO/SARMAE/BigEarthNet weights | — | 0 (downloaded) | 0 |
| **Core total** | | | **~293–360** |
| Contingent ImageNet-ConvNeXt reference (if run) | 4 | 3 | ~24 |

Controlled eight-arm study ≈ **~293–360 GPU-hours** after adding 10% seed reruns for the scarce-label headline. The point estimate is just over 290 GPU-hours, but the honest high end is ~4 nights once the supervised-backbone spread (2 × 10–30 h = 20–60) and the acknowledged 1.2–1.5× CNN-cell penalty are carried through rather than "treated as noise." Size the schedule's release-valve logic to the **high end**. Still well below a from-scratch-pretraining design (~1,200–2,200 GPU-h); the FM downloads keep it cheap, and the only training we run is the two small supervised-transfer backbones. Adding the contingent reference brings it to ~317–384. Note earlier project drafts quoted ~130 GPU-h for a four-arm ViT-only study; doubling to two tracks plus 10% seeds roughly doubles it again.

## 13. Risk register

| Risk | Signal | Mitigation |
|---|---|---|
| Channel-format gap breaks transfer (either backbone) | an FM arm ≤ its floor at all fractions, or fails to converge | the P3.6 check catches it per track; confirm [VH,VV,VH−VV] load + normalization; if persistent, a *finding*, not necessarily a bug |
| A downloaded backbone loads partially (silent random weights) | `test_fm_checkpoints_load` fails / unexpected key report | the guard blocks merge; fix the loader, never proceed on a partial load — covers SatDINO, SARMAE, BigEarthNet-S1/S2 |
| BigEarthNet ConvNeXt-V2 load quirks (`configilm`/FCMAE stem) | key/shape mismatch on the reBEN loader | the guard exercises this path; may need the `configilm` package pinned + the reBEN model-def repo; STOP and verify rather than hand-patch |
| ConvNeXt-V2 stem won't take 3-channel input cleanly | S1 model is 2-band (VV/VH), S2 is 10-band | `timm` `in_chans=3` Repeat-with-rescaling adapts the stem, same policy as the ViT; verify in P3.6 |
| A downloaded arm underperforms its floor | arm ≤ floor at low fractions | not a bug — report as a transfer finding; verify load + channel handling first |
| Cross-track result is null (ViT≈CNN, or SAR-adv doesn't generalize) | matched-role gaps ~0 | still a result ("the SAR-domain advantage is/ isn't architecture-general"); lean on dark-vessel/near-shore slices |
| Licenses (SARMAE CC BY-NC 4.0, BigEarthNet weights) | venue/redistribution needs NC-incompatible use, or a weight `LICENSE.note` is missing (P3.2 gate) | CC BY-NC 4.0 permits the non-commercial course/workshop use but forbids commercial use and constrains redistribution; release derived checkpoints (if any) only under CC BY-NC with attribution; scope the paper's claims and any released code/weights non-commercial. "Fine for the course" ≠ "fine for a CVPR/ICCV submission" — assert each separately. Verify the BigEarthNet weight license at download |
| Backbone too slow on the node | grid cell > a few GPU-h (STOP #6: >~2× budget) | drop the *track* to a smaller variant uniformly — ViT-S for the ViT track, ConvNeXt-Tiny for the CNN track — preserving within-track parity; document. (This is the trigger P3.1 references — there is no "P6 throughput check".) |
| Heatmap recall stuck low | dev recall ≪ YOLO26 ref | raise σ to 3, lower tau, confirm vessel-biased cropping fires |
| Label noise dominates | LOW-conf ignore shifts F1 > 2 pts | report both protocols; prefer ignore-protocol as primary |
| fp16 divergence | NaN loss | norms in fp32 (already), halve lr for that run, log |
| Download cap blown | size estimate > cap | reduce train scenes toward 75; SLC never fetched; need only *weights* for FMs, not SAR-1M (76 GB) or full BigEarthNet |
| Arm count / analysis overrun | Phase 7 slipping, curves illegible | the contingent reference (Phase 8) is cut first; if needed, demote BigEarthNet modality detail to an appendix |
| R3 (LocateAnything-3B) repo id unverified | 404 / gated / no §1a id at `sprint-6`/P4.6 | R3 is off the controlled curves and **droppable** — resolve the id before P4.6 or skip it; never guess a checkpoint (ground rule 12) |

## Appendix A — Scorer worked example

GT at (0,0) and (500,0) m; predictions (10,0,0.9), (180,0,0.8), (900,0,0.7). Greedy by score: 0.9 matches GT1 (10 m) → TP. 0.8 → nearest unmatched GT2 at 320 m > 200 → FP. 0.7 → 400 m from GT2 → FP. GT2 unmatched → FN. P = 1/3, R = 1/2, F1 = 0.4. `test_scorer.py` asserts exactly this.

## Appendix B — Label schema (SARFish / xView3)

Columns used: `scene_id, detect_scene_row, detect_scene_column, is_vessel, is_fishing, vessel_length_m, confidence{HIGH,MEDIUM,LOW}, source{AIS, AIS/Manual, Manual}, *_distance_from_shore_km, top/left/bottom/right (sparse)`. Dark-vessel proxy: `source == "Manual"` (no AIS correlate). Detection positives: `is_vessel == True` with confidence ∈ {HIGH, MEDIUM}; LOW → ignore region; non-vessel maritime objects (fixed infrastructure) are not targets but are the expected near-shore FP source.

LS-SSDD carries only boxes → reduced to centroids (P1.4); it has no confidence tiers or dark/AIS attributes (it is a supervised-transfer source for Arms 4 and 8, not evaluated).

## Appendix C — Volta (sm_70) gotchas, V100 node

1. Pin torch wheels for **CUDA 12.x**; CUDA 13 dropped Volta — a stray `pip install torch` pulling cu13 will not see the cards. (The 5070 Ti box is separate and needs sm_120 nightly — keep the two lockfiles distinct.)
2. **No bf16.** fp16 + `torch.cuda.amp.GradScaler`; keep LayerNorm/GroupNorm and any logit/softmax math in fp32 (`autocast` handles most; verify with `gpu_sanity.py`).
3. **No FlashAttention** (never supported sm_70). `F.scaled_dot_product_attention` falls back to mem-efficient/math kernels — correct, just slower; relevant for the ViT, irrelevant for YOLO26's CNN.
4. **Avoid bitsandbytes** — int8/4-bit paths are unreliable on Volta and nothing here needs them.
5. bf16-trained public checkpoints (e.g. LocateAnything) can overflow when cast to fp16 — load fp32 on CPU, then `.half()` only verified-safe modules, or just run that probe on the 5070 Ti.

## Appendix D — What is deliberately out of scope

- **Pretraining anything ourselves** — out of scope by design. Both FMs are downloaded; the project is a comparison of pretrained backbones, chosen specifically to remove pretraining compute and novel pretraining code as risks. (An earlier project version pretrained a SAR-MAE from scratch; that was cut for exactly these reasons.)
- **ImageNet as a *core* arm** — SatDINO (ViT) and BigEarthNet (CNN) are the optical anchors. ImageNet appears only as the optional contingent CNN reference R1 (RS-vs-generic-pretraining question), never on the controlled curves.
- **DINOv3-SAT ViT-L / DINOv3-ConvNeXt as study arms** — rejected: ViT-L is 3.5\texttimes\ the params (size confound); DINOv3-ConvNeXt was considered for the CNN optical arm but ImageNet-vs-BigEarthNet was chosen instead (DINOv3-ConvNeXt-Base availability unverified and likely ConvNeXt-V1, a within-track architecture mismatch). ConvNeXt-**V2**-Base keeps the CNN track internally consistent.
- **Self-pretraining any foundation model** (SAR-MAE-from-scratch, FCMAE-on-SAR ConvNeXt, or distilling SARMAE into a CNN) — all cut for the same reason: compute + novel-code risk, and (for distillation) it would make the CNN SAR arm a teacher-dependent copy rather than an independent model. The four FMs are downloaded; the only training is the two cheap LS-SSDD supervised backbones.
- **The unconstrained challenge/leaderboard arm** — removed entirely to protect the Sep 1 deadline. The eight-arm controlled study is the deliverable; the pooled multi-dataset supervised set (HRSID/SAR-Ship/SSDD) it would have used is no longer fetched.
- **The self-supervised (SSL) corpus and OpenSARShip** — the 150 public GRD products (formerly `corpus_extra`) and OpenSARShip were fetched only for the removed self-pretraining/challenge scope and are **no longer downloaded**. There is no SSL/self-pretraining lane in this project — all four FMs are downloaded — so any "SSL corpus"/"SSL pixels" reference is dead scope.
- **IC-ViT (isolated-channel patchify) for channel handling** — considered, not default. It patchifies each polarization separately with no channel-specific parameters (arXiv 2503.09826), a principled "VH and VV as separate streams" approach, but it changes tokenization and complicates the same-input fairness story. The fixed 3-channel [VH,VV,VH−VV] representation is the default; adopt IC-ViT only on an explicit human decision.
- **Image-space super-resolution** — confound-heavy and no HR Sentinel-1 target exists; the only SR-flavored move considered is a stride-2 decoder ablation, and even that is optional.
- **Complex-valued SLC / Doppler** — a separate project (storage + non-square pixels); not in this plan.
- **DETR / box heads as the detector** — the point-native heatmap head is the detector; a box-head comparison is at most a one-off, not a study arm.
