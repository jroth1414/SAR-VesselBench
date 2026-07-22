# DEVPLAN — xView3 Label-Efficient Dark Vessel Detection

Development plan for a coding agent (Claude Code or similar). Execute phases in order; each phase ends with explicit acceptance criteria. Do not start a phase until the previous phase's criteria pass, except where a phase is marked parallel-safe.

**Project question (the study).** Under one fixed point detector, how much labeled xView3 data do generic ImageNet, optical remote-sensing, and SAR pretrained backbones save for dark-vessel detection — does SAR pretraining beat optical and generic transfer in the scarce-label regime, and **does that finding hold across architecture families (ViT and CNN)?** The headline deliverable is a two-track seed-0 label-efficiency curve; the domain contrasts and their architecture-generality are the central results.

**What this is and is not.** This is a *controlled label-efficiency comparison of pretrained backbones across two architecture families*. Every pretrained initialization is downloaded; we train only the shared xView3 detector fine-tunes. Arms 4 and 8 are matched generic-natural-image controls initialized from ImageNet-1K classification checkpoints, not backbones trained by this project. This scope removes self-pretraining compute and novel pretraining code while answering the domain-of-pretraining and architecture-generality questions. The novelty is the systematic comparison — generic ImageNet vs optical remote sensing vs SAR pretraining, ViT vs CNN, on *dark-vessel, scarce-label* detection through a point-native detector — which to our knowledge has not been done.

**Design principle: architecture-matched fairness.** The study has **two tracks — a ViT track (ViT-B/16, ~86M) and a CNN track (ConvNeXt-V2-Base, ~89M, size-matched).** Within each track, all four arms share the same backbone architecture, detector, fine-tuning schedule, and seed 0; only initialization differs. Across tracks, the backbones are size-matched. Every pretraining claim is within-track; matched-role cross-track comparisons assess architecture, with disclosed method-history caveats. We report parameter count, hardware adaptation, and measured GPU-hours per arm.

**Secondary deliverables removed.** The unconstrained challenge/leaderboard arm has been **cut** to protect the Sep 1 deadline; the controlled two-track study is the entire deliverable. The former optional ImageNet-ConvNeXt reference R1 is also removed because ImageNet is now a matched core role in Arms 4 and 8. Only the external R2/R3 references remain.

## ⚑ Current repository state — READ THIS FIRST (cold-start runbook)

> **Purpose.** This section lets any fresh agent or human resume using only the repo. Read it before AGENTS.md and before choosing a phase or branch. Section 1 is an orientation to the implemented tree; this ledger and the repository are the current-state authority.
>
> **Update discipline.** Update the status ledger and add a `phase-N-done` git tag at every sprint merge. The `sprint-2b-eval-hardening` merge must be tagged `phase-2-done`; later phase merges follow the same pattern. If the ledger looks stale, rebuild it with the *state-detection runbook* below — **the repo is ground truth; this table is a cache.**

### Branch model (as actually built — this overrides the "main" phrasing elsewhere)
- The integration / default branch is **`dev`** (GitHub `HEAD → dev`), **not `main`**; `main` currently lags `dev`. Everywhere this plan or AGENTS.md says "land on `main`" / "no direct commits to `main`," read **`dev`**: open each sprint branch off `dev` and PR back into `dev`.
- Current amendment work belongs on `sprint-7b-imagenet-arms`, opened from `dev`; historical sprint branches are not an active-work checklist.
- CI must trigger on `dev` (see the CI-trigger fix in §1b) or it never runs on the active branch.

### Status ledger (ground truth as of this revision)
| Phase | Sprint branch | State | Evidence | Missing to reach DONE |
|---|---|---|---|---|
| 0 Env/scaffold | `sprint-0-env` + `sprint-7b-imagenet-arms` | **DONE — P100 revalidated 2026-07-22** | `locks/env-5070ti.txt` (sm_120) plus verified `locks/env-p100node.txt` (torch 2.11.0+cu126, sm_60); P100 `gpu_sanity.py` PASS; both families passed fp16-autocast backward/AdamW at micro-batch 8 with accumulation 2 preserving effective batch 16 | record actual f10 wall time for the refreshed compute forecast |
| 1 Data/splits | `sprint-1-data` + `sprint-1b/1c` freeze branches | **DONE — all three artifacts frozen** | all P1 code + tests green; labels acquired and profiled (BLOCKER-4); **`data/splits.json` FROZEN** (150 scenes: 111/23/16 + 50 eval_final, seed 0) pinned by `test_splits_immutable`; **`data/lsssdd_split.json` FROZEN** (8,100/900 over the verified 9,000 sub-images) pinned by `test_lsssdd_split_immutable`; **`data/stats.json` FROZEN** (105,408 train chips, 111 scenes, 150/150 chipped with zero failures: VH −26.448/5.951, VV −16.599/6.062 dB) pinned by `test_stats_immutable`; label projection visually verified (QA gallery) | tag `phase-1-done` at the sprint-1c merge |
| 2 Scorer/decode/threshold | `sprint-2-scorer` + `sprint-2b-eval-hardening` | **DONE — scorer re-frozen after eval hardening; tagged `phase-2-done`** | `scorer.py` counts near-shore FPs and exposes per-scene aggregation; `threshold.py` owns dev threshold selection; `decode.py` rejects non-finite heatmaps; Phase-2 tests pass | — |
| 3 Detector | `sprint-3-detector` + `sprint-3b` + `sprint-3c-optimizer-fix` | **DONE — detector frozen + tagged `phase-3-done`; P3.6 PASSED** | the optimizer fix and Option-B plan-literal 50-epoch/early-stop budget decision are merged; historical P3.6 dev F1: ViT floor 0.788 < SatDINO 0.835 < SARMAE 0.858; CNN floor 0.677 < BigEarthNet-S2 0.726 < BigEarthNet-S1 0.819 (`runs/p36_summary.json`) | — |
| 4 FM+floor arms+refs | `sprint-4/5/6` | **PARTIAL** | seed-0 f10 cells are complete for Arms 1,2,3,5,6,7; R2 `yolo26-f100` and R3 `locateanything-zs` are complete; their result exports are under `results/` | f25/f50/f100 for the six retained arms; keep the frozen detector/scorer/splits unchanged |
| 5 ImageNet arms+grid | `sprint-7-grid` + `sprint-7b-imagenet-arms` | **BLOCKED — P100 throughput decision required 2026-07-22** | exact Arms 4/8 implementation and all 88 guards pass; real f10 probes loaded correctly but stabilized at 0.95 steps/s (ViT) and 0.20 steps/s (CNN), projecting 20.50/97.36 training hours at 50 epochs before dev eval | **HUMAN NEXT:** choose unchanged one-GPU jobs on faster hardware, explicitly approve/revalidate distributed training, or accept the P100 schedule; do not resume until then |
| 6 Final eval | `sprint-8-final-eval` | NOT STARTED | once-only tripwire and frozen 50 eval IDs exist; no lockfile has been written | the 50 eval-final raster scenes are not present on this node and must be acquired/extracted before the one allowed evaluation |
| 7 Analysis | `sprint-9-analysis` | NOT STARTED | — | — |
| 8 Contingent ref | — | **REMOVED** | former R1 ImageNet-ConvNeXt role is now represented symmetrically by core Arms 4/8 | — |

### Known blockers — resolve before the affected phase (do not skip)
- **BLOCKER-1 — scorer freeze pin (RESOLVED).** The original pin in `tests/test_scorer_immutable.py` was mis-recorded at birth; `sprint-2b-eval-hardening` intentionally re-pinned the scorer after the BLOCKER-3 metric fix. Future scorer changes require human sign-off and a new pin.
- **BLOCKER-2 (RESOLVED).** Dev-tuned threshold selection lives in non-frozen `src/eval/threshold.py`; `scorer.py` stays pure scoring.
- **BLOCKER-3 (RESOLVED).** Near-shore F1 now counts unmatched FP predictions with `PredictionPoint.distance_from_shore_km <= 2.0`; missing shore distance on an unmatched FP raises. Dark-vessel remains a GT-defined recall slice, not a precision/F1 slice.
- **BLOCKER-4 (RESOLVED 2026-07-05).** Label CSVs acquired by the human: `D:\train.csv` (64,113 rows, 554 scenes), `D:\validation.csv` (19,224 rows, 50 scenes). **Measured label facts every later phase must respect (extends Appendix B):** (a) `source` values are LOWERCASE — `ais` / `manual` / `ais/manual`; the frozen scorer's dark test is `source == "Manual"`, so the eval adapter (P3 `infer_scene.py` / `final_eval.py`) MUST normalize case when building `GroundTruthPoint`s — the scorer itself stays frozen. (b) The **train CSV is 100% `ais`-sourced** — dark vessels (`manual`) exist ONLY in the validation/eval_final scenes (8,022 rows, 6,420 vessels), so the dark-recall slice is structurally empty on dev/test and is measurable only at final eval. (c) Train confidence↔is_vessel is degenerate: HIGH ⇒ is_vessel=False (16,692 fixed objects), MEDIUM ⇒ is_vessel=True (36,375 vessels), LOW ⇒ NaN (11,046, ignore); the standard positives rule (`is_vessel & conf∈{HIGH,MEDIUM}`) therefore selects exactly the 36,375 MEDIUM vessels in train. (d) bbox fields exist only in validation (19,049 rows); YOLO-reference boxes for train are always synthesized from points+lengths. (e) `distance_from_shore_km` uses a `9999.99` far-from-shore sentinel.
- **BLOCKER-5 (RESOLVED historically; source no longer active).** (a) Local DIU-format archives are the canonical xView3 imagery source (the SARFish HF mirror hosts raw `SAFE.zip` products, not xView3 GeoTIFFs, and no labels); `download_sarfish.py --from-local` registers them. (b) LS-SSDD-v1.0 was acquired and verified for the original design: 6,000+3,000 800×800 JPGs + 9,000 VOC XMLs, with a frozen seeded 8,100/900 split. The 2026-07-22 amendment supersedes the random→LS-SSDD Arms 4/8, so LS-SSDD is no longer an active training dependency. **`data/lsssdd_split.json` remains frozen and hash-pinned as a historical provenance artifact; never delete, regenerate, or repurpose it.** A directory named `LS-SSDD` on the current server was audited as OSDataset2.0 and must not be substituted.
- **Design amendment (human, 2026-07-22).** Replace the original random→LS-SSDD Arms 4/8 with matched ImageNet/generic-classification transfer: Arm 4 = `timm/vit_base_patch16_224.augreg_in1k`; Arm 8 = `timm/convnextv2_base.fcmae_ft_in1k`. Both end in supervised ImageNet-1K classification on generic natural images and transfer encoder weights only, but their training histories are not identical: the ViT checkpoint is supervised AugReg, whereas the ConvNeXt-V2 checkpoint includes FCMAE self-supervised pretraining followed by supervised IN1K fine-tuning. This is a disclosed cross-track limitation, not a within-track confound. The old LS f10 results are superseded and excluded from every current table/curve: `vitsup-f10-s0` dev/test F1 0.8518/0.7805 and `cnnsup-f10-s0` 0.7507/0.6644; their LS pretraining val losses (5.69e-10 and 0.0) motivated the overfit concern. Preserve them only under a clearly marked historical archive.
- **Seed-count decision (human, 2026-07-22): seed 0 only.** The study matrix is 8 arms × 4 fractions = **32 core fine-tunes**. With the already-run R2/R3 references, the complete research plan is **34 experiments**. There are no seed reruns, seed bands, or error-bar claims.
- **BLOCKER-6 — RESOLVED 2026-07-22 for f10 launch.** The actual host is 8× Tesla P100 PCIe 12 GB (Pascal sm_60). A repo-local Python 3.11 environment installed the committed `locks/env-p100node.txt` (torch 2.11.0+cu126); `torch.cuda.get_arch_list()` includes sm_60 and `gpu_sanity.py` passed finite fp16 matmul plus non-Flash SDPA. Full fp16-autocast forward/backward/AdamW probes passed at micro-batch 8: ViT allocated/reserved 3.796/4.078 GiB; ConvNeXt-V2 10.193/10.523 GiB. Real runs use accumulation 2, preserving the frozen effective batch 16 without editing `configs/detector.yaml`. At the verification snapshot, five GPUs were free and lockable; availability is dynamic, so re-run `gpu info` immediately before `gpu get --lock 5` (or a smaller available count). Once five are attached, use only the container-local indices 0–4 printed by `nvidia-smi -L`—never physical host IDs. The two replacement f10 wall times remain the throughput measurement gate; STOP if the refreshed forecast exceeds budget by >2×.
- **BLOCKER-7 — OPEN 2026-07-22: P100 throughput.** Real f10 probes on the frozen recipe stabilized at 0.95 steps/s (ViT) and 0.20 steps/s (CNN), with 1,402 batches per epoch. The 50-epoch training-only projections are 20.50 h and 97.36 h, before every-five-epoch whole-scene dev evaluation—about 5.9× and 15.0× slower than comparable completed development-GPU f10 cells. Both processes were interrupted gracefully during epoch 0; no checkpoint or completion marker exists. See `results/throughput/p100_f10_probe_2026-07-22.json`. A human must choose unchanged one-GPU execution on faster hardware, accept this schedule, or explicitly authorize and revalidate a distributed Trainer change.
- **Scene-count decision (human, 2026-07-05): 150 study scenes** of the 554 available, selected stratified at seed 0 and frozen as 111 train / 23 dev / 16 test; all 50 verified IDs form `eval_final`. This split remains immutable. The old compute rationale is historical; current compute is remeasured on P100.
- **Expected CI color.** All freeze/split/parity guards exist and must be green. On this amendment branch, `test_fm_checkpoints_load` must cover six exact pretrained checkpoints; a failure or a still-four-checkpoint manifest is a STOP, not an expected skip.

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

The experiment is a set of downloaded pretrained backbone initializations plus random floors, fed through one fixed detector in two architecture-matched tracks. The initialization is the only variable *within* each track; the architecture is the only variable *across* matched roles. This project performs no backbone pretraining.

| Arm | Track | Backbone | Initialization | On curve? | Role |
|-----|-------|----------|----------------|-----------|------|
| 1 | ViT | ViT-B/16 (~86M) | random init | yes | ViT floor |
| 2 | ViT | ViT-B/16 | **SatDINO** (fMoW-RGB, DINO, downloaded) | yes | optical-domain FM transfer |
| 3 | ViT | ViT-B/16 | **SARMAE** (SAR-1M, MAE, downloaded) | yes | SAR-domain FM transfer |
| 4 | ViT | ViT-B/16 | **ImageNet-1K AugReg** (`timm/vit_base_patch16_224.augreg_in1k`, downloaded) | yes | ImageNet / generic classification transfer |
| 5 | CNN | ConvNeXt-V2-B (~89M) | random init | yes | CNN floor |
| 6 | CNN | ConvNeXt-V2-B | **BigEarthNet-S2** (optical RS, downloaded) | yes | optical-RS transfer |
| 7 | CNN | ConvNeXt-V2-B | **BigEarthNet-S1** (SAR RS, downloaded) | yes | SAR-RS transfer |
| 8 | CNN | ConvNeXt-V2-B | **ImageNet-1K FCMAE→FT** (`timm/convnextv2_base.fcmae_ft_in1k`, downloaded) | yes | ImageNet / generic classification transfer |
| R2 | ref | YOLO26 | COCO-pretrained | **no** | detector reference |
| R3 | ref | LocateAnything-3B | zero-shot | **no** | VLM reference |

The two tracks are matched by *role*: floor (1,5), optical remote-sensing pretraining (2,6), SAR pretraining (3,7), and ImageNet/generic classification transfer (4,8). Comparing arm *k* to arm *k+4* isolates architecture family with the pretraining role held fixed.

**Downloaded pretrained backbones (we do not pretrain them):**
- **SatDINO** (Straka & Gruber 2025, `strakajk/satdino-vit_base-16`): ViT-B/16, DINO self-distillation on fMoW-RGB optical satellite imagery. Apache-2.0. The ViT optical anchor.
- **SARMAE** (Liu et al., CVPR 2026; HF **weights** `Wenquandan777/SARMAE`, file `SARMAE_vitb_checkpoint-last`; **code** `github.com/MiliLab/SARMAE`): ViT-B/16 masked autoencoder pretrained on SAR-1M (million-scale SAR) with speckle-aware enhancement. CC BY-NC 4.0 (gated — accept terms). The ViT SAR anchor. *(The HF org `MiliLab/SARMAE` is the code mirror and is gated; the downloadable ViT-B weights are at `Wenquandan777/SARMAE` — verified. §1a is canonical for this id.)*
- **BigEarthNet ConvNeXt-V2-B** (TU Berlin RSiM/BIFOLD, reBEN; `BIFOLD-BigEarthNetv2-0/convnextv2_base-{s1,s2}-v0.2.0`): ConvNeXt-V2-Base supervised on BigEarthNet v2.0 land-cover classification, in Sentinel-1 (SAR) and Sentinel-2 (optical) variants. The CNN optical/SAR anchors. *These are supervised-classification pretrained, not self-supervised foundation models — a documented distinction from SatDINO/SARMAE; they still provide in-domain SAR/optical feature learning.*
- **ImageNet ViT-B/16, AugReg IN1K** (`timm/vit_base_patch16_224.augreg_in1k`): supervised ImageNet-1K classification checkpoint; drop the classification head and transfer the full encoder. The ViT generic-natural-image anchor.
- **ImageNet ConvNeXt-V2-B, FCMAE FT IN1K** (`timm/convnextv2_base.fcmae_ft_in1k`): FCMAE self-supervised pretraining followed by supervised ImageNet-1K classification fine-tuning; drop the classification head and transfer the full encoder. The CNN generic-natural-image anchor. Do **not** substitute the previously downloaded `fcmae_ft_in22k_in1k` R1 checkpoint.


**Cross-family method caveats (stated openly in the writeup, not hidden):**
- The two *ViT* FM arms differ in SSL method (SatDINO=DINO, SARMAE=MAE) as well as domain — no downloadable optical ViT-B MAE existed to match SARMAE's method, so we compare the best released FM of each domain as-is.
- The two *CNN* RS arms (BigEarthNet S1/S2) are the *cleanest* domain contrast in the study: identical model, identical dataset, identical supervised task, differing *only* in input modality (SAR vs optical). The CNN track is therefore the controlled domain comparison; the ViT track is the best-available-FM comparison.
- ViT FM arms are self-supervised; CNN RS arms are supervised-classification. This SSL-vs-supervised difference is a track-level property, documented, not a within-track confound.
- The matched ImageNet pair has the same final supervision and source domain (ImageNet-1K classification on generic natural images), but not the same training history: Arm 4 is supervised AugReg, while Arm 8 is FCMAE followed by supervised fine-tuning. Treat matched-role cross-track comparisons as architecture comparisons with this disclosed training-history limitation.
- The ViT hands the head a stride-16 (32×32) latent while the ConvNeXt hands it a stride-32 (16×16) latent the CNN head upsamples once more; both decode on the same stride-4 (128×128) map, but the *pre-head* feature resolution and amount of learned upsampling differ by 2×. This is an inherent ViT-vs-CNN architecture-family property, reported alongside the cross-track (arm *k* vs *k+4*) gap as a limitation — not a defect the shared head removes.

**Channel representation (study-wide, all arms):** the fixed 3-channel [VH, VV, VH−VV] in dB. Every downloaded backbone was pretrained on an input that is not dual-pol xView3 SAR (ImageNet/SatDINO=3-channel optical; SARMAE=single-pol SAR→3ch; BigEarthNet-S1=Sentinel-1 VV/VH 2-band; BigEarthNet-S2=10-band Sentinel-2), so each backbone's stem is loaded through the documented `timm` `in_chans=3` path. It is a no-op in shape for the ImageNet, SatDINO, and SARMAE stems and a real 2→3 / 10→3 adaptation for BigEarthNet S1/S2. Bounded, documented, and never varied per arm.

**ImageNet is a matched core role.** Arms 4 and 8 make generic-natural-image transfer available in both architecture tracks; the former CNN-only R1 is removed as redundant.

## 0. Ground rules

1. **The eval scorer is sacred.** Written first (Phase 2), unit-tested, never modified after Phase 3 begins. Every reported number in the project flows through `src/eval/scorer.py`.
2. **Initialization is the only variable *within* a track; architecture is the only variable *across* matched roles.** Two backbone definitions only: ViT-B/16 (arms 1–4) and ConvNeXt-V2-Base (arms 5–8). Within each track, one head, one optimizer config, one augmentation policy, one decode config, one fine-tuning schedule, and the single seed 0 — every arm fine-tunes end-to-end, with no frozen/fine-tuned asymmetry. The head/optimizer/schedule are shared across *both* tracks too (only the backbone and its family adapter differ), so ViT-vs-CNN is fair. If you are tempted to tune something per-arm, stop — that breaks the study. All six pretrained initializations are downloaded and encoder-only; no LS-SSDD or other backbone training remains active.
3. **Scene-level splits only.** No chip from a dev/test/eval scene may appear in any training or pretraining corpus. Membership is keyed on `scene_id`, recorded once in `data/splits.json`; every dataloader asserts membership at construction.
4. **The ~50 human-verified xView3 validation scenes are touched exactly once,** by `src/eval/final_eval.py`, after the grid is complete. Tripwire: the script refuses to run without `--i-am-sure` and writes a timestamped lockfile on first use.
5. **Two machines, distinct roles.** Dev/iteration card: one RTX 5070 Ti (16 GB, Blackwell sm_120). Heavy lifting: one **8× Tesla P100 PCIe node** (12 GB each, Pascal sm_60). The P100 constraints in Appendix C are mandatory: use `locks/env-p100node.txt`, fp16 + GradScaler with norms in fp32, no bf16, no FlashAttention, and no bitsandbytes. Real P100 cells use the verified micro-batch 8 + accumulation 2 to preserve the frozen effective batch 16; never rewrite `configs/detector.yaml` for memory. GPU leases are dynamic: inspect and lock only free cards, then pass the container-local CUDA indices shown after attachment (five attached cards map to 0–4), not physical host IDs from `gpu info`.
6. **Core arms vs. references are reported separately.** The eight core arms (1–8) go in the study tables and the two-track label-efficiency figure. R2 YOLO26 and R3 LocateAnything go in a separate references section for context; they are not on the controlled curves. There is no R1 and no challenge/leaderboard arm.
7. **Determinism.** Every run takes `--seed`; seed torch, numpy, random, and dataloader workers (Lightning's `seed_everything(workers=True)`). Log the resolved config (full YAML), git SHA, and an environment hash into the run directory.
8. **Run directories.** Every experiment writes `runs/<exp_id>/` with `config.yaml`, `metrics.csv` (per-epoch), `final_metrics.json`, `checkpoints/`, `log.txt`. Experiment IDs come from the manifest in Section 12 (Run manifest) — never invent ad-hoc names.
9. **One framework for every arm: PyTorch Lightning.** All eight xView3 fine-tunes use the same `LightningModule`, `LightningDataModule`, and `Trainer` configuration. The P100 runner dispatches one process per reserved container-local GPU and uses the verified micro-batch 8 + accumulation 2 for both families; effective batch, optimizer steps, seeding, and checkpointing remain identical across arms. Any future memory adaptation requires a new STOP/revalidation and must preserve the effective recipe. Backbones load as plain `nn.Module`s inside the LightningModule (timm for both ImageNet checkpoints and both base architectures; SatDINO/SARMAE via their loaders; BigEarthNet ConvNeXt via the `configilm`/reBEN loader). One head class attaches via a small **per-family** adapter: ViT stride-16 tokens (32×32) reshape then upsample to the stride-4 map; the ConvNeXt stride-32 stage-3 map (16×16) needs one **extra** upsample block to reach the same stride-4 map (see P3.3). Both decode on the same stride-4 (128×128) output, but pre-head feature resolution differs by 2× — an inherent architecture-family property reported as a limitation. Keep configs in YAML; avoid hydra.
10. **Fairness accounting.** Every arm logs parameter count and xView3 fine-tuning GPU-hours. Pretraining GPU-hours are 0 on our side for all six downloaded pretrained arms (2,3,4,6,7,8); their external training histories are cited. The random floors (1,5) have no pretraining. Within each track all four arms share architecture and effective fine-tuning compute; across tracks the two backbones are size-matched (~86M vs ~89M). Report P100 hardware, microbatch, accumulation, and measured time explicitly.
11. **Commit per task** with messages referencing the task ID (e.g. `P1.3: scene-level split builder`). Never commit data, checkpoints, or anything under `runs/`.
12. **Follow cited methods; do not invent.** For any component that names a reference method (Section 1a), implement that method's published recipe — do not substitute a "better," "simpler," or "more modern" approach reasoned up independently, however plausible it seems. Novelty in this project lives in the *experimental comparison*, not in re-deriving pretraining or channel-adaptation mechanics. If a reference is ambiguous, unavailable, or two references conflict, STOP and surface the question to a human rather than improvising — a silently-wrong method (e.g. a hand-rolled patch-embed hack or an ad-hoc reconstruction target) corrupts every downstream number without throwing an error.

## 1. Repository layout

This tree is an orientation to the implemented repository. The cold-start ledger above is authoritative for current state; historical LS-SSDD code/artifacts may remain for provenance but are not active experiment paths.

```
JHU-xView3/
  README.md
  DEVPLAN.md                  # this file
  pyproject.toml              # base deps; see two lockfiles below
  locks/
    env-5070ti.txt            # cu12x / nightly for sm_120
    env-p100node.txt          # verified P100/sm_60 torch 2.11.0+cu126 freeze
    env-v100node.txt          # historical V100/sm_70 candidate only
  Makefile                    # one target per phase entrypoint
  configs/
    data.yaml                 # paths, chip size, split fractions, seed
    detector.yaml              # SHARED: head, optimizer, aug, decode (never edited per study-arm)
    arms.yaml                 # the seed-0 run matrix (Section 12)
  data/                       # gitignored; symlink to large storage
    raw/
      xview3/                 # SARFish GRD GeoTIFFs + label CSVs
      lsssdd/                 # historical LS-SSDD payload if present; no active arm consumes it
    chips/                    # 800x800 chips + per-chip JSON sidecars
    splits.json               # scene_id -> split (frozen after sprint-1)
    stats.json                # train-split-only per-pol mean/std (frozen after sprint-1)
    lsssdd_split.json         # frozen historical LS split; retained and hash-pinned
    manifests/                # parquet chip manifests per split/source
  src/
    data/
      download_sarfish.py     # selective HF GRD pulls
      download_aux.py         # historical auxiliary-data registration
      chipper.py              # scene -> chips + label projection
      to_centroids.py         # historical box/mask -> point conversion
      splits.py               # scene-level split builder
      datasets.py             # active FineTuneDataset + retained historical dataset code
      transforms.py           # log-norm, crops, flips
    models/
      backbones.py            # ViT-B/16 + ConvNeXt-V2-B, both in_chans=3 (VH,VV,VH-VV)
      heatmap_head.py         # deconv tower + 1ch sigmoid head + per-family adapter
      init_loaders.py         # 8 loaders: vit_random/satdino_b/sarmae_b/vit_imagenet + cnn_random/bigearthnet_s2/bigearthnet_s1/cnn_imagenet
    train/
      lit_modules.py          # shared fine-tune LightningModule
      datamodule.py           # one LightningDataModule (chips, splits, channel rep)
      pretrain_supervised.py  # superseded historical entrypoint; never invoked by current matrix
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
    test_fm_checkpoints_load.py  # guard: all six downloaded pretrained checkpoints load structurally/value-sensitively (1b)
    test_scorer_immutable.py    # guard: scorer.py hash matches pinned value (1b)
  runs/                       # gitignored
```

## 1a. Reference implementations — adapt these, do not reinvent

Every component below has a canonical paper and/or repository. **The agent's job is to adapt the reference implementation, not to design a new method.** When a detail is unspecified by this plan, copy the reference's choice rather than inventing one. Pin versions; for dataset and downloaded-weight sources a **revision pin is mandatory** (see the note below the table). **This table is the single canonical source for every reference id — any other mention in this plan or in AGENTS.md must match it verbatim.**

| Component | Follow this reference | What to copy |
|---|---|---|
| ViT-B backbone (shared by Arms 1–4) | `timm` `vit_base_patch16_224` | the one ViT architecture every ViT arm uses; downloaded checkpoints load into it |
| Arm-3 weights (ViT SAR FM) | SARMAE — HF **weights** `Wenquandan777/SARMAE`, file `SARMAE_vitb_checkpoint-last` (code `github.com/MiliLab/SARMAE`; Liu et al. CVPR 2026) | ViT-B SAR-1M checkpoint load; encoder only, drop optical-branch/decoder; CC BY-NC 4.0 (gated). **Canonical id — arm list, P3.2, and AGENTS.md must match this verbatim.** |
| Arms 6,7 weights (CNN RS) | BigEarthNet ConvNeXt-V2-B, `BIFOLD-BigEarthNetv2-0/convnextv2_base-{s2,s1}-v0.2.0` (reBEN, Clasen et al. IGARSS 2025) | ConvNeXt-V2-Base; load via `configilm` `BigEarthNetv2_0_ImageClassifier.from_pretrained`, take the backbone, drop the classification head; S2=optical, S1=SAR |
| CNN backbone + stem adaptation | `timm` `convnextv2_base` + `in_chans=3` | ConvNeXt-V2-Base as the CNN arm backbone; stem takes the fixed [VH,VV,VH−VV] input via `in_chans`, same Repeat-with-rescaling policy as the ViT stem |
| Arm-8 weights (CNN ImageNet) | `timm/convnextv2_base.fcmae_ft_in1k` | ConvNeXt-V2-B FCMAE→supervised IN1K checkpoint; encoder only, drop classifier. Exact variant is mandatory; do not use `fcmae_ft_in22k_in1k` |
| Backbone + channel adapt | `timm` (`vit_base_patch16_224`, `in_chans`) | ViT-B/16 definition; `in_chans` Repeat-with-rescaling for any channel change |
| Arm-4 weights (ViT ImageNet) | `timm/vit_base_patch16_224.augreg_in1k` | ViT-B/16 supervised AugReg IN1K checkpoint; encoder only, drop classifier |
| Arm-2 weights (optical FM) | SatDINO, `strakajk/satdino-vit_base-16` (Straka & Gruber 2025, arXiv 2508.21402) | ViT-B/16 fMoW-RGB DINO checkpoint; `AutoModel.from_pretrained(..., trust_remote_code=True)`; Apache-2.0 |
| Heatmap head + decode | CenterNet (Objects as Points); SAR precedent TRANSAR | penalty-reduced focal target, peak + distance-NMS decode |
| Arm R3 weights (VLM reference) | LocateAnything-3B — `nvidia/LocateAnything-3B`, revision `c32291ca5e996f5a7a485845b4f57a233936bba0` | zero-shot grounding on ~200 dev chips with the pinned prompt/parser protocol; centers through the SACRED scorer |

**Source-revision pinning is mandatory (not "where practical").** Every dataset and downloaded-weight source must be pinned to an exact revision — HuggingFace `revision=<commit-sha>`, GitHub tag/commit — recorded in a `SOURCE.note` beside the download. The frozen `data/splits.json` assumes identical scene contents on any re-pull, so an unpinned source silently voids reproducibility (see P1.1/P1.2 and the risk register).

If a referenced repo is unavailable, a citation is ambiguous, or two references conflict, **stop and flag it for a human** — do not paper over the gap with an improvised method (see ground rule 12). This applies to **any** named reference, including the R2/R3 reference arms, not only the §1a table rows.

## 1b. Execution discipline — sprints, gates, and stop conditions

This section governs *how* the plan is executed, not what it builds. Its purpose is to limit agent drift: the failure mode where an agent, run unsupervised against a large spec, gradually optimizes for "make this file work" instead of "serve the study," and silently substitutes plausible-but-wrong choices that never throw an error. Every rule here shrinks the unsupervised interval, makes failures loud, or gates progression behind a human.

### Schedule (calendar)

The project runs **July 1 to September 1, 2026** (about nine weeks); the final project is due **Sep 1**. All pretrained backbones are downloaded, the grid is seed 0 only, and the P100 environment/memory gate has passed. The immediate work is the two replacement ImageNet f10 cells; their measured wall times gate the refreshed forecast and remaining fine-tuning grid. The old V100-based 293–360 GPU-hour estimate is historical and retired.

| Dates (2026) | Focus (phases) |
|---|---|
| Jul 1–14 | Data pipeline + sacred scorer + detector (Phases 0–3, parallel lanes) |
| Jul 15–28 | Both tracks' FM + floor arms (ViT: SatDINO, SARMAE; CNN: BigEarthNet S1/S2); channel-format check; references (Phase 4) |
| Jul 29–Aug 11 | ImageNet/generic-transfer Arms 4/8 + the remaining seed-0 label-fraction grid (Phase 5) |
| Aug 12–25 | Final verified-scene eval + ViT-vs-CNN error analysis; begin writeup (Phases 6–7) |
| Aug 26–Sep 1 | Finish writeup and submit; no contingent arm (Phase 8 removed) |

The writeup deliberately overlaps the Aug 12–25 analysis block rather than starting cold at the end; the final week is for finishing and submission, not first drafting. The eight-arm seed-0 study is the complete deliverable; both the former R1 and the unconstrained challenge arm are removed.

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
| `sprint-7-grid` + `sprint-7b-imagenet-arms` | Phase 5 | Spine | ImageNet Arms 4/8 amendment + full seed-0 label-fraction grid |
| `sprint-8-final-eval` | Phase 6 | **Foundation** | touches the once-only verified-scene eval |
| `sprint-9-analysis` | Phase 7 | Leaf | ViT-vs-CNN figures/slices; read the output, trust the code |

- **Foundation tier** — review every line; slow, careful merge. These define or consume ground truth.
- **Spine tier** — review that the acceptance criteria are genuinely met; spot-check the implementation. These carry the study's validity.
- **Leaf tier** — review the output for sanity; trust the code. These cannot corrupt the controlled comparison.

**Hard branch-ordering constraint:** no model-code sprint (`sprint-3` onward) may open a PR until **`sprint-2-scorer` is merged to `dev`**. Building arms against a still-moving scorer makes every early number untrustworthy. Likewise `sprint-1-data` must merge before any sprint that trains on chips.

### Sub-agent distribution

Sub-agents are safe only on **independent** work; they drift badly when they share hidden state. Match distribution to the two-owner lane split, not to "one agent per sprint":

- **Parallelizable lanes:** the data pipeline (`sprint-1`) and the scorer (`sprint-2`) are independent and can run as parallel sub-agents. The external R2/R3 references are independent of the study spine. There is **no pretraining lane**: all six pretrained core checkpoints are downloaded, and the former LS-SSDD jobs are superseded.
- **Must be serial — one owner:** every sprint that touches the shared detector (`sprint-3-detector → sprint-4-vit-arms → sprint-5-cnn-arms → sprint-6-floor-refs`) modifies the same `lit_modules.py`/`finetune.py`. In particular `sprint-5-cnn-arms` is Spine-tier CNN backbone/detector integration (see the sprint table) — it is on this serial chain and must **not** be run as an independent parallel sub-agent. The floor arms (1, 5) also live in `sprint-6` but run *through* the shared detector; only the external R2/R3 references are the parallel exception. Do **not** parallelize the detector sprints across sub-agents — they would each edit the shared code and silently diverge, reintroducing the per-arm inconsistency the fairness design exists to prevent. One agent owns the detector; arms run through it sequentially.
- Do not subdivide finer than these lanes; coordination overhead and shared-state drift outgrow the speedup.

### Mandatory STOP conditions (halt and ask a human — not optional)

Stopping is cheap and expected; it is not a sign of failure. An agent that asks ten questions costs minutes; one that silently invents a wrong split costs the study. **Halt and surface the question** when any of these holds:

1. A cited reference (Section 1a) is unavailable, ambiguous, or two references conflict. (ground rule 14)
2. An acceptance criterion fails and the fix is not obvious within one attempt.
3. A result violates a sanity check — e.g. a seed-0 label-efficiency curve drops by more than the predeclared 0.02 F1 tolerance, or an arm scores implausibly high (suspect leakage).
4. Any change would touch a frozen artifact (see do-not-touch manifest) after it is locked.
5. The agent is tempted to deviate from a cited method "because it seems better." (ground rule 14)
6. A step's measured compute or time exceeds its budget by more than ~2× (catches runaway runs before they burn the node).

Do not optimize to "seem autonomous." Surfacing these is the job.

### Do-not-touch manifest (frozen after their sprint; changes require a STOP)

Once merged, these are locked; modifying them requires an explicit human decision because changing them silently invalidates earlier results:

- `src/eval/scorer.py` — frozen after `sprint-2` (ground rule 1).
- `data/splits.json` — frozen after `sprint-1`; scene-level membership never changes mid-study.
- `data/stats.json` — frozen after `sprint-1`; train-split-only per-polarization mean/std, computed ONCE over the 100%-train scene set and reused unchanged for every label fraction and both tracks at seed 0 (see P1.5).
- `data/lsssdd_split.json` — frozen after `sprint-1`; retained solely as a hash-pinned historical artifact from the superseded LS-SSDD design. No current arm consumes it; never delete, regenerate, or repurpose it.
- `configs/detector.yaml` — the shared head/optimizer/schedule, frozen after `sprint-3`; this is the fairness contract.
- The verified-scene eval protocol / its lockfile — touched exactly once (ground rule 4).

Each frozen artifact should carry a machine-checkable freeze guard analogous to `test_scorer_immutable` (a pinned sha256 activated at its sprint merge): `test_splits_immutable` (`data/splits.json`, sprint-1), `test_detector_immutable` (`configs/detector.yaml`, sprint-3). This gives "Phase 1 done" / "Phase 3 done" the same binary, cold-agent-checkable signal the scorer freeze has.

### Guard tests (machine-enforced, wired to pre-merge CI)

Prose checklists depend on someone remembering to run them — and the drift problem *is* that agents stop remembering. These four assertions live in `tests/` and run on every PR, converting silent corruption into a failed build. They are the technical embodiment of ground rule 14. See P2.3 and P3.x for where they attach; the canonical list:

1. **`test_split_disjoint`** — no `scene_id` appears in more than one split; also asserted at datamodule construction. Kills leakage.
2. **`test_backbone_parity`** — *within-track* param parity: the four ViT arms instantiate the identical ViT-B/16 param count, and the four CNN arms the identical ConvNeXt-V2-Base param count. **Plus cross-track output-geometry parity:** push a `(1,3,512,512)` tensor through one ViT adapter and one ConvNeXt adapter and assert both emit `(1, C, 128, 128)` with the *same* `C` and the *same* stride-4 — so a ConvNeXt adapter wired to the wrong stride (e.g. one too few upsample blocks → stride-8) or the wrong `C` fails CI even though all four CNN arms remain param-identical to each other (see P3.3). Kills accidental architecture divergence *and* head-adapter geometry drift that would silently void the cross-track (arm *k* vs *k+4*) comparison. (Cross-track *param* sizes are close but not identical — ~86M vs ~89M — reported, not asserted equal.)
3. **`test_fm_checkpoints_load`** — all **six** downloaded pretrained checkpoints load value-sensitively, not merely by key name: SatDINO, SARMAE, BigEarthNet S1/S2, `timm/vit_base_patch16_224.augreg_in1k`, and `timm/convnextv2_base.fcmae_ft_in1k`. Assert full encoder coverage (classification heads are the only sanctioned drops) and that a fixed sample of encoder tensors differs by norm/hash from fresh random initialization. A bare `strict=False` “no exception” is insufficient. Split the guard into **(a)** CPU-offline structural key-manifest coverage for all six (CI, no downloads) and **(b)** a training-box integration check that loads all six exact pinned weights and runs the value-sensitivity assertion. The ImageNet cases must also reject accidental use of the old `fcmae_ft_in22k_in1k` variant. CI runs only (a); training-host half (b) is part of the recorded passed launch gate and must be rerun if the weights or environment change.
4. **`test_scorer_immutable`** — a hash of `scorer.py` matches the pinned value recorded at `sprint-2` merge; checked before any eval runs. Kills scorer drift.

A sprint cannot merge with a failing guard test. If a guard test must legitimately change (rare), that is itself a STOP.

These tests are enforced by **GitHub Actions CI** (`.github/workflows/ci.yml`), which runs them on every push and PR — so enforcement does not depend on any agent or human remembering to run them. CI is CPU-only and uses a minimal `requirements-ci.txt` (not the GPU training environment); guard tests must therefore run against tiny in-repo fixtures. A companion **`AGENTS.md`** at the repo root gives any coding agent the short list of non-negotiables (this section is the detail behind it); AGENTS.md is the instruction layer, CI is the enforcement layer, and this DEVPLAN is the single source of truth.

**CI trigger & expected color (current).** `ci.yml` triggers on **`dev`** and all four guard files exist. The amendment branch includes all six exact pretrained-checkpoint cases and committed structural manifests. CI remains CPU-only and runs structural half (a); the value-sensitive half (b) runs on the training host with local weights. A red guard is a STOP, never a reason to weaken the assertion.

## 2. Phase 0 — Environment and smoke checks

Targets the 5070 Ti and actual P100 host. The original scaffold and amended BLOCKER-6 hardware gate are complete.

- **P0.1** Create the repo skeleton under the actual repo root `JHU-xView3/` (Section 1): a `.gitignore` covering `data/`, `runs/`, `*.pt`, `*.pth`, `*.ckpt`, `*.tif`, `*.tiff`, `*.parquet`, `__pycache__/`, `.pytest_cache/`; a `pyproject.toml` declaring the `src`-layout package + base deps (`pip install -e .` is the only sanctioned install); imports resolve as absolute `from src.…` from the repo root — guarantee it with either an `__init__.py` in every `src/` subpackage **or** `[tool.pytest.ini_options] pythonpath=["."]`, not pytest's accidental `sys.path` insertion; and a `README.md` documenting install + both GPU boxes. **Commit a minimal `.gitignore` now** — it is zero-risk and is the only thing enforcing ground rule 11 ("never commit data or checkpoints").
- **P0.2** Machine-specific environments, because the boxes differ:
  - `locks/env-p100node.txt` is the verified P100 freeze: torch 2.11.0+cu126 explicitly includes sm_60 kernels and pins the full package set. `locks/env-v100node.txt` is historical only. Never use a moving or bare `pip install torch`.
  - `locks/env-5070ti.txt` is the verified Blackwell environment; do not assume its binaries support Pascal.
  - **Generation recipe:** install a resolved set on each machine and freeze it. Record device capability, driver, torch CUDA build, and package hash in README/run metadata.
  README documents which environment ran each experiment.
- **P0.3** `scripts/gpu_sanity.py`: print device name + capability, run a 4096×4096 fp16 matmul and an `F.scaled_dot_product_attention` call, assert no NaN, report the SDPA backend chosen. Run on both machines; paste both outputs into the README.
- **P0.4** Active `Makefile` targets: `make env-check`, `make test`, `make data`, `make qa`, `make grid`, `make references` (R2/R3), and `make final-eval CONFIRM=1`. There are no active supervised-pretraining or optional-R1 targets; all six pretrained core checkpoints are downloaded.

**Entry preconditions:** none — this is the bootstrap phase. Work on `sprint-0-env` off `dev`.

**Definition of Done — machine-checkable:** `test -f pyproject.toml && test -f Makefile && test -f .gitignore && test -f locks/env-p100node.txt && test -f locks/env-5070ti.txt && pip install -e . && python -m pytest --collect-only -q` exits 0.

**P100 review record (2026-07-22):** `gpu_sanity.py` passed on sm_60 and both family probes completed with finite loss at micro-batch 8; accumulation 2 preserves effective batch 16. The GPU-manager snapshot showed five free/lockable cards. Recheck availability before launch; after five-card attachment, address only container-local CUDA indices 0–4.

## 3. Phase 1 — Data acquisition, chipping, splits

Owner: data owner. Jul 1–14. GPU: none (CPU + disk).

### P1.1 xView3 / SARFish download — `src/data/download_sarfish.py`
- `huggingface_hub.snapshot_download(repo_id="ConnorLuckettDSTG/SARFish", repo_type="dataset", revision="<commit-sha>", allow_patterns=[...])`, patterns restricted to **GRD only, never SLC** (SLC is the bulk of multi-TB). Pin the exact dataset `revision` and record it (with the md5s) in `data/raw/xview3/SOURCE.note` — the frozen `data/splits.json` assumes identical scene contents on any re-pull.
- Inputs: a scene-ID list. Outputs: `data/raw/xview3/GRD/<scene_id>/...` + md5 verification per the SARFish instructions.
- Pull: (a) 75–150 train scenes chosen in P1.3, (b) all ~50 human-verified validation scenes. *(The 150 public GRD products for an SSL corpus are no longer pulled — the self-supervised/challenge scope was cut; see Appendix D.)*
- Labels: fetch SARFish CSVs; schema in Appendix B. Note the dataset card's country-restriction warning in the README.
- Budget guard: print estimated size before downloading; abort if > configurable cap (default 400 GB).

### P1.2 Historical auxiliary SAR provenance — `src/data/download_aux.py`
This subsection records completed work from the original design; it is not an active acquisition or training dependency.
- **LS-SSDD-v1.0** was acquired, licensed, verified as 9,000 800×800 sub-images, and split for the superseded design. Keep its `SOURCE.note`, `LICENSE.note`, frozen split, and conversion tests as historical provenance.
- The amended Arms 4/8 do not fetch or consume LS-SSDD. Never substitute the unrelated OSDataset2.0 payload found in the server directory named `LS-SSDD`.
- OpenSARShip, HRSID, SAR-Ship-Dataset, and SSDD remain out of scope. Downloaded model weights retain their separate P3.2 source/license gates.

### P1.3 Chipping — `src/data/chipper.py`
- Read VH + VV GeoTIFFs with rasterio in windows; emit 800×800 chips, 100 px overlap, stride 700 (xView3). Historical LS-SSDD registration is retained but unused by the current arms.
- Pixel pipeline: raw → dB-like log `x = log10(clip(raw, 1, None))`; store float16 log values; normalize at load time with global per-polarization mean/std computed over the **train split only** (`data/stats.json`).
- **Channel representation (study-wide, fixed for all arms).** Every arm — including SatDINO and the ImageNet checkpoints, which were pretrained on 3-channel RGB — receives the *same* 3-channel SAR input: **[VH, VV, VH−VV]** in normalized dB. Rationale: (a) VH and VV are distinct polarimetric measurements, and their difference (≈ co/cross-pol contrast) is a physically meaningful third channel rather than a zero-pad or a duplicate; (b) using one fixed 3-channel tensor everywhere means the input is *physically identical* across all arms, so channel handling is never a confound between them; (c) all 3-channel pretrained stems (SatDINO, SARMAE, and ImageNet Arms 4/8) load with no channel-count change, while the two **CNN RS** backbones (BigEarthNet S1 = 2-band, S2 = 10-band) adapt their stem to 3 channels via the same `timm` `in_chans` path. This single decision replaces all per-arm channel hacks. Document it once; do not vary it per arm.
- **Channel adaptation — use the named library method, do not invent one.** SatDINO, SARMAE, and the two ImageNet checkpoints load their patch-embed/stem with **no channel-count change** (SatDINO and the ImageNet checkpoints are 3-ch RGB; SARMAE is single-pol→3-ch). The two **CNN RS** arms require a shape-changing stem adaptation to the fixed 3-channel input — BigEarthNet-S1 (2-band VV/VH → 3) and BigEarthNet-S2 (10-band → 3) — done via **`timm.create_model(..., in_chans=3)`** (Repeat-with-rescaling, identical mechanism for both, differing only in the pretrained starting weights; see P3.2). So it is one documented mechanism applied to every arm, but a no-op for the 3-channel checkpoints and a real 2→3 / 10→3 expansion for the CNN RS arms — *not* "no surgery for any arm." If at any point a channel count must change (e.g. an ablation), the ONLY sanctioned mechanism is **`timm.create_model(..., in_chans=N)`**, which implements the field-standard *Repeat* method (tile/average the pretrained RGB projection weights across the new channels and rescale to preserve activation magnitude). Do NOT hand-roll patch-embed weight manipulation, and do NOT derive a "better" scheme. The single documented alternative, allowed only as an explicit ablation, is **RGB+random** (USat: keep pretrained weights for the RGB-equivalent channels, randomly initialize any extra channels). References to follow, not redesign: `timm` (`in_chans` rescaling); USat (2023); the init-strategy survey arXiv 2503.09493 (which documents that *Repeat* and *RGB+random* are the two known options and that neither is universally best — i.e. this is a bounded choice, not an open problem). A third named method, IC-ViT (isolated-channel patchify, arXiv 2503.09826), is noted in Appendix D as considered-but-not-default; do not adopt it without a human decision.
- Drop chips >95% no-data; record land fraction per chip where shoreline vectors exist.
- Label projection: per-chip JSON sidecar with vessel points in chip-pixel coords, `confidence`, `source`, `vessel_length_m`, `distance_from_shore_km`, and bbox fields when present.
- Manifest: one parquet per scene/source — chip path, scene_id, origin row/col, n_vessels, has_low_conf, land_frac.

### P1.4 Centroid conversion — `src/data/to_centroids.py`
- Historical conversion support reduces box/rotated-box/mask labels to centroids. It was used by the superseded LS-SSDD path and remains tested for provenance; no current core arm consumes it.
- Historical LS-SSDD normalization code is not an active experiment input.
- `tests/test_centroids.py`: box/rbox/mask → expected center within 1 px.

### P1.5 Scene selection and splits — `src/data/splits.py`
- Stratify chosen xView3 train scenes by coarse region (cluster scene-center lat/lon into ~6 bins) and shoreline presence (any label `distance_from_shore_km < 5`).
- Scene-level split of xView3 train scenes: **75% train / 15% dev / 10% test** (seeded). The ~50 verified scenes form `eval_final`, excluded from everything. *(There is no `corpus_extra`/unlabeled SSL split — the SSL scope was cut; see Appendix D.)*
- **`data/stats.json` (owned here, frozen).** Compute global per-polarization (VH, VV) mean/std over **only the frozen train-split `scene_id`s** (never dev/test/eval_final). The 100%-train statistics are computed once and reused unchanged for every label fraction and both tracks at seed 0; never re-derive them per fraction or track.
- **`data/lsssdd_split.json` (owned here, frozen historical artifact).** Preserve the original seeded 8,100/900 partition and its immutable hash. The amended Arms 4/8 do not read it.
- `data/splits.json` maps scene_id → split; `tests/test_splits.py` asserts disjointness and counts, that `stats.json`'s contributing scenes are a subset of the train split, and that the LS-SSDD partition is fixed/seeded/disjoint.

**Entry preconditions (historical Phase 1):** `sprint-0-env` merged; xView3 raw data pulled with revision pins. The LS artifact remains part of the completed historical gate but is not a current training dependency.

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
- **ViT arm backbone:** ViT-B/16 via timm (`vit_base_patch16_224`, `in_chans=3` on the fixed [VH,VV,VH−VV] input), learnable pos-embed interpolated to 512×512 inputs (32×32 tokens), final norm kept. SatDINO, SARMAE, and the Arm-4 AugReg checkpoint all load into this architecture.
- **CNN arm backbone:** ConvNeXt-V2-Base via timm (`convnextv2_base`, `in_chans=3` on the same input). Take the stage-3 feature map; at 512 input it is stride 32 (16×16, 1024-dim). BigEarthNet and the Arm-8 FCMAE checkpoint all load into this architecture.
- Both expose a uniform `(feature_map, channels, stride)` interface to the head adapter. ViT: (B,768,32,32); ConvNeXt: (B,1024,16,16).
- The architectures and frozen detector are now locked by completed results. A P100 timing or memory failure is a STOP; do not invoke the old smaller-variant fallback or change `detector.yaml` mid-study.

### P3.2 `src/models/init_loaders.py`
Eight loaders behind one enum, four per track. Each prints matched/missing/unexpected key counts after loading. **Within a track all four produce the identical backbone; only the weights differ.**

*ViT track (all `vit_base_patch16`, 768-dim):*
- `vit_random`: timm trunc-normal init.
- `satdino_b`: **SatDINO ViT-B/16** fMoW-RGB checkpoint (`strakajk/satdino-vit_base-16`, DINO; `AutoModel.from_pretrained(..., trust_remote_code=True)`, or manual `vit_base(patch_size=16)` + `load_state_dict(ckpt['teacher'])`). Backbone features (768-dim); drop the DINO projection head. Apache-2.0.
- `sarmae_b`: **SARMAE ViT-B/16** (`Wenquandan777/SARMAE`, file `SARMAE_vitb_checkpoint-last`; CC BY-NC 4.0, accept terms). `vit_base_patch16`, loads with no surgery; encoder weights only, ignore optical-branch/SARC/decoder.
- `vit_imagenet`: `timm/vit_base_patch16_224.augreg_in1k`; supervised AugReg ImageNet-1K, classification head dropped, full encoder required.

*CNN track (all `convnextv2_base`, 1024-dim stage-3):*
- `cnn_random`: timm trunc-normal init.
- `bigearthnet_s2`: **BigEarthNet ConvNeXt-V2-B, Sentinel-2 (optical)** (`BIFOLD-BigEarthNetv2-0/convnextv2_base-s2-v0.2.0`; via `configilm` `BigEarthNetv2_0_ImageClassifier.from_pretrained`, take the backbone, drop the multi-label classification head). The optical-RS CNN anchor.
- `bigearthnet_s1`: **BigEarthNet ConvNeXt-V2-B, Sentinel-1 (SAR)** (`...convnextv2_base-s1-v0.2.0`, same loader). The SAR-RS CNN anchor. Pretrained on Sentinel-1 VV/VH (2-band); the stem adapts to the 3-channel [VH,VV,VH−VV] input via `in_chans`.
- `cnn_imagenet`: `timm/convnextv2_base.fcmae_ft_in1k`; FCMAE→supervised ImageNet-1K, classification head dropped, full encoder required.

**Matched-role structure:** arm *k* (ViT) and arm *k+4* (CNN) share a pretraining role — floor (`vit_random`/`cnn_random`), optical RS (`satdino_b`/`bigearthnet_s2`), SAR (`sarmae_b`/`bigearthnet_s1`), ImageNet/generic classification (`vit_imagenet`/`cnn_imagenet`). Within-track contrasts isolate initialization; cross-track matched-role contrasts compare architecture families subject to the disclosed ImageNet training-history mismatch.

**Cross-family caveats (documented):** ViT RS arms are self-supervised (DINO/MAE); CNN RS arms are supervised land-cover classification. The BigEarthNet S1-vs-S2 contrast is the cleanest domain comparison; SatDINO-vs-SARMAE additionally differs in SSL method. The ImageNet pair matches final dataset/task but AugReg-vs-FCMAE training history differs. These are track-level interpretation limits, not license to alter a recipe.

- Assert after loading (`tests/test_fm_checkpoints_load.py`, §1b guard): all six exact pretrained checkpoints load structurally and value-sensitively, with no partial/random encoder. Within each track all four arms expose an identical backbone interface.
- **Weights source/license gate.** Each of the six pretrained checkpoints has exact model/revision provenance and a `LICENSE.note`. Refuse to load a checkpoint whose source/license record is missing. SARMAE remains CC BY-NC 4.0; verify and report all others.

### P3.3 `src/models/heatmap_head.py`
- One head class, attached to either backbone via a small adapter that maps the backbone feature map to a common (B, C, H, W) at the head's input stride. ViT: (B,768,32,32) stride-16. ConvNeXt: (B,1024,16,16) stride-32, so the CNN head uses one extra upsample block to reach the same output stride.
- Upsample blocks `[ConvTranspose2d, GroupNorm(32), GELU]` to output **stride 4** (128×128 for a 512 input), then 3×3 conv → 1 channel; sigmoid at inference, logits at train.
- The head, loss, optimizer, and schedule are identical across both tracks; only the backbone-to-head adapter differs by architecture (this is the minimal, documented cross-track difference, not a per-arm one).

### P3.4 Targets, loss, sampler
- `src/train/losses.py`: penalty-reduced focal (CenterNet form), α=2, β=4, on Gaussian targets σ=2 output px; LOW-confidence labels stamp an ignore disk (radius 3 output px) where loss is zeroed.
- `src/train/sampler.py`: epoch-level weighted sampler so ~50% of sampled chips contain ≥1 HIGH/MED vessel.
- `src/data/transforms.py`: random 512 crop from the 800 chip (vessel-biased: 70% of crops centered within 128 px of a vessel when one exists), flips, 90° rotations, intensity jitter ±0.1 in log space. No augmentation that breaks SAR statistics (no blur, no elastic).

### P3.5 `src/train/finetune.py` — the shared loop
- AdamW, lr 1e-4, layer-wise lr decay 0.65, weight decay 0.05, cosine schedule, 5-epoch warmup, 50 epochs, **effective batch 16** (verified P100 setting: micro-batch 8 + accumulation 2 for both families; fp16 + GradScaler, norms fp32), grad clip 1.0.
- `--init {vit_random,satdino_b,sarmae_b,vit_imagenet,cnn_random,bigearthnet_s2,bigearthnet_s1,cnn_imagenet}` selects the loader and implicitly the backbone family. **Within each track all four fine-tune end-to-end with the identical schedule**. Head, loss, sampler, augmentation, decode, schedule, and seed 0 are identical across both tracks; only the backbone family/adapter differs across tracks and only loaded weights differ within one.
- `--label_frac {0.1,0.25,0.5,1.0}` subsamples xView3 train scenes (scene-level, seeded); fractions **nest** (10% ⊂ 25% ⊂ 50% ⊂ 100%) so the curves are monotone in data.
- Every 5 epochs: tiled inference (`infer_scene.py`, 512 windows, stride 384, global NMS) on 8 fixed dev scenes → dev F1. Early-stop patience 4 dev evals. Save best + last.

**Acceptance:** the historical detector smoke/gates remain valid. The amendment adds required structural and value-sensitive loads for `vit_imagenet` and `cnn_imagenet`, exact checkpoint-variant assertions, and unchanged parity. The P100 micro-batch-8/accumulation-2 smoke passed for both families on 2026-07-22; the two replacement f10 cells are next.

### P3.6 Downloaded-backbone load + early-signal check (cheap, before the full grid)
The historical P3.6 gate covered SatDINO, SARMAE, and BigEarthNet S1/S2 and remains recorded unchanged. The 2026-07-22 amendment extends the checkpoint guard to the two ImageNet arms and requires their f10 runs under the full frozen recipe; do not rewrite the historical P3.6 summary. A pretrained arm underperforming its floor is a finding after load integrity and protocol parity are confirmed.

**Entry preconditions (amended loader work):** all frozen artifact guards remain green; exact ImageNet source/license records and six-checkpoint structural manifests exist; BLOCKER-6 is recorded passed before training. The historical `data/lsssdd_split.json` pin remains green but is not consumed.

**Definition of Done — machine-checkable (Phase 3):** `pytest tests/test_backbone_parity.py tests/test_fm_checkpoints_load.py && test -f configs/detector.yaml && python -m src.train.finetune --init vit_random --label_frac 0.1 --epochs 3 --smoke` exits 0. On merge, pin `configs/detector.yaml` via `test_detector_immutable` and tag `phase-3-done`.

**Human review (non-blocking):** the P3.5 smoke overlays (`runs/qa/pred_gallery.png`) look sane.

## 6. Phase 4 — Both tracks' foundation-model + floor arms, and references

Owner: detector owner. Jul 15–28. BLOCKER-6 is now passed; dispatch one process per explicitly reserved, container-local P100 index. There is no backbone pretraining; all pretrained checkpoints are downloaded. The old V100 runtime/memory estimate is historical and does not apply.

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

## 7. Phase 5 — ImageNet/generic-transfer arms (4, 8) + the seed-0 grid

Owner: detector owner. Jul 29–Aug 11. Arms 4/8 load exact downloaded ImageNet checkpoints; this project performs no backbone pretraining.

- **P5.1 — DONE 2026-07-22.** Arm 4 `vit_imagenet` is pinned to `timm/vit_base_patch16_224.augreg_in1k`; Arm 8 `cnn_imagenet` is pinned to `timm/convnextv2_base.fcmae_ft_in1k`. Only classification heads are dropped; both are included in the six-checkpoint structural/value-sensitive guard with exact source/license provenance.
- **P5.2 — STOPPED AT THROUGHPUT GATE.** Five P100s were locked and the two replacement f10 cells launched on local CUDA 0/1. Exact loads and finite losses passed, but measured 0.95/0.20 steps/s projected 20.50/97.36 training hours at 50 epochs before dev eval—5.9×/15.0× the comparable development-GPU wall times. Both were interrupted during epoch 0 with no checkpoint or completion marker. Do not resume, score, or launch remaining cells until the owner resolves BLOCKER-7. After that decision, complete Arms 4/8 at f10/f25/f50/f100 under the approved unchanged or revalidated execution path. Do not invoke LS-SSDD pretraining.
- **P5.3** Use **seed 0 only** for every cell. Do not schedule seeds 1/2 and do not report seed bands, variance, confidence intervals, or error bars. Archive the superseded `vitsup-f10-s0`, `cnnsup-f10-s0`, `vitsup-lsssdd`, and `cnnsup-lsssdd` artifacts outside the active run-ID namespace; never delete or mix them into current summaries.
- **P5.4** Per run: freeze the dev-tuned threshold, score test unchanged, append to `runs/summary/grid.csv`, and render eight seed-0 curves (solid ViT, dashed CNN; colors for floor, optical RS, SAR, and ImageNet/generic). No uncertainty shading.
- **P5.5** Headline computations: (a) within-track ordering at 10% (floor vs generic ImageNet vs optical RS vs SAR); (b) within-track generic-vs-RS and optical-vs-SAR gaps; (c) cross-track matched-role gaps with the ImageNet training-history caveat; (d) interpolated label budgets. Describe seed-0 results as point estimates, not uncertainty-aware estimates.

**Acceptance:** `grid.csv` has exactly 32 active rows (8 arms × 4 fractions × seed 0), no superseded IDs, no NaNs, and populated per-fraction scene/vessel/dark-proxy/near-shore counts. The eight-curve point-estimate figure renders without bands. `monotonicity_ok` uses the predeclared 0.02 F1 drop tolerance and must be true for every arm; false is a STOP.

**Entry preconditions (Phase 5):** frozen artifacts/guards green; `src/eval/threshold.py` exists; exact ImageNet checkpoints and source/license notes present; six-checkpoint load guard green; BLOCKER-6 P100 validation recorded passed. These conditions are satisfied for the two replacement f10 launches. The historical LS split is not an input.

**Definition of Done — machine-checkable (Phase 5):** `grid.csv` contains exactly the 32 manifest IDs, all at seed 0, zero NaNs, populated count columns, no `vitsup`/`cnnsup` rows, and `monotonicity_ok == true` for all arms; the eight-curve figure renders. Together with `yolo26-f100` and `locateanything-zs`, 34 total experiment result records exist. Tag `phase-5-done`.

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

## 10. Phase 8 — Removed

The former CNN-only ImageNet reference R1 is permanently removed: ImageNet is now the symmetric core role in Arms 4 and 8. Do not schedule `imgnetcnn-*` experiments or create a references curve for R1. R2/R3 are the only references.


## 11. Hyperparameter reference

| Component | Setting | Value |
|---|---|---|
| Chips | size / overlap / GSD | 800 px / 100 px / 10 m |
| Train crops | size / vessel-biased frac | 512 / 0.7 |
| Heatmap | stride / sigma / loss | 4 / 2 out-px / penalty-reduced focal α=2 β=4 |
| Decode | tau / d_nms / match tol / output_stride | dev-tuned / 120 m / 200 m / 40 m (= stride-4 × 10 m GSD) |
| Fine-tune | opt / lr / lld / wd / effective batch | AdamW / 1e-4 / 0.65 / 0.05 / 50 / 16 (microbatch+accumulation only as P100 hardware adaptation) |
| ViT-track backbone | ViT-B/16 | ~86M params, embed-dim 768, fine-tuned end-to-end |
| CNN-track backbone | ConvNeXt-V2-Base | ~89M params, stage-3 1024-dim, fine-tuned end-to-end |
| Init (ViT, downloaded) | arms 2 / 3 / 4 | SatDINO / SARMAE / `vit_base_patch16_224.augreg_in1k` |
| Init (CNN, downloaded) | arms 6 / 7 / 8 | BigEarthNet-S2 / BigEarthNet-S1 / `convnextv2_base.fcmae_ft_in1k` |
| Channel input | fixed all arms | [VH, VV, VH\textminus VV] in dB (3-channel) |
| Seed | all core cells | 0 only |
| Ignore regions | LOW-conf radius | 3 out-px |
| Splits | train/dev/test of xView3 train scenes | 75/15/10 scene-level |
| Label fractions | nested | 10 / 25 / 50 / 100% |

## 12. Run manifest (experiment IDs)

Study grid — `exp = {init}-f{frac}-s0`, init ∈ ViT {`vitrand`,`satdino`,`sarmae`,`vitin1k`} + CNN {`cnnrand`,`beS2`,`beS1`,`cnnin1k`}:
- Core (8 arms × 4 fractions, seed 0): 32 runs — e.g. `satdino-f25-s0`, `beS1-f10-s0`.
- No seed reruns. Seeds 1/2 are outside the approved task.

References: `yolo26-f100`, `locateanything-zs`.
Backbone trainings: none. Historical `vitsup-lsssdd`/`cnnsup-lsssdd` and `vitsup-f10-s0`/`cnnsup-f10-s0` are superseded archives, never active manifest IDs.
Contingent R1: removed; do not schedule `imgnetcnn-*`.

**Total = 32 core fine-tunes + 2 existing reference experiments = 34.** All six pretrained core initializations are downloaded. Do not describe the study matrix itself as 34 cells: it is 32 cells, while the complete experiment plan is 34 including R2/R3.

### P100 compute measurement gate

The historical V100 estimate is invalid on the actual 12 GB P100 host. The real f10 probes completed the measurement gate but triggered BLOCKER-7. At 1,402 batches/epoch, stabilized 0.95 steps/s (ViT) and 0.20 steps/s (CNN) imply the following training-only ranges; whole-scene dev evaluations add time.

| Component | Runs | Measured/projection | Status |
|---|---|---|---|
| `vitin1k-f10-s0` | 1 | 10.25 h at 25 epochs; 20.50 h at 50 | stopped in epoch 0 |
| `cnnin1k-f10-s0` | 1 | 48.68 h at 25 epochs; 97.36 h at 50 | stopped in epoch 0 |
| Seed-0 core grid | 32 | fraction/family total not approved on P100 | BLOCKED |
| Seed reruns | 0 | 0 | removed |
| Backbone pretraining jobs | 0 | 0 | downloaded weights |
| YOLO26 + LocateAnything | 2 | complete | recorded in result provenance |
| **Active plan total** | **34 experiments** | pending execution decision | **BLOCKED at 26 remaining core cells** |

The probes are approximately 5.9× (ViT) and 15.0× (CNN) slower than comparable completed development-GPU f10 runs, exceeding the declared >2× tripwire. Do not infer a detector/config change. The owner must select faster one-GPU hardware, accept the P100 schedule, or explicitly authorize and revalidate distributed execution before any core cell resumes.

## 13. Risk register

| Risk | Signal | Mitigation |
|---|---|---|
| Channel-format gap breaks transfer (either backbone) | an FM arm ≤ its floor at all fractions, or fails to converge | the P3.6 check catches it per track; confirm [VH,VV,VH−VV] load + normalization; if persistent, a *finding*, not necessarily a bug |
| A downloaded backbone loads partially (silent random weights) | `test_fm_checkpoints_load` fails / unexpected key report | the guard blocks merge; fix the loader, never proceed on a partial load — covers all six exact pretrained checkpoints |
| BigEarthNet ConvNeXt-V2 load quirks (`configilm`/FCMAE stem) | key/shape mismatch on the reBEN loader | the guard exercises this path; may need the `configilm` package pinned + the reBEN model-def repo; STOP and verify rather than hand-patch |
| ConvNeXt-V2 stem won't take 3-channel input cleanly | S1 model is 2-band (VV/VH), S2 is 10-band | `timm` `in_chans=3` Repeat-with-rescaling adapts the stem, same policy as the ViT; verify in P3.6 |
| A downloaded arm underperforms its floor | arm ≤ floor at low fractions | not a bug — report as a transfer finding; verify load + channel handling first |
| Cross-track result is null (ViT≈CNN, or SAR-adv doesn't generalize) | matched-role gaps ~0 | still a result ("the SAR-domain advantage is/ isn't architecture-general"); lean on dark-vessel/near-shore slices |
| Licenses (SARMAE CC BY-NC 4.0, BigEarthNet weights) | venue/redistribution needs NC-incompatible use, or a weight `LICENSE.note` is missing (P3.2 gate) | CC BY-NC 4.0 permits the non-commercial course/workshop use but forbids commercial use and constrains redistribution; release derived checkpoints (if any) only under CC BY-NC with attribution; scope the paper's claims and any released code/weights non-commercial. "Fine for the course" ≠ "fine for a CVPR/ICCV submission" — assert each separately. Verify the BigEarthNet weight license at download |
| P100 memory/throughput fails after the passed gate | OOM, non-finite fp16, or refreshed forecast >2× budget | STOP; retain the verified sm_60 lock and micro-batch 8 + accumulation 2, and diagnose without changing architecture or the frozen detector. Revalidate only if the environment/model runtime changes |
| Heatmap recall stuck low | dev recall ≪ YOLO26 ref | verify load/data/cropping and report; any detector change requires human approval, re-pin, and uniform reruns |
| Label noise dominates | LOW-conf ignore shifts F1 > 2 pts | report the frozen primary protocol and limitation; do not retune scorer/protocol |
| fp16 divergence | NaN loss | STOP and diagnose the hardware/load path; never halve LR for one arm or edit the frozen schedule |
| Download cap blown | size estimate > cap | reduce train scenes toward 75; SLC never fetched; need only *weights* for FMs, not SAR-1M (76 GB) or full BigEarthNet |
| Arm count / analysis overrun | Phase 7 slipping, curves illegible | no new arms or seeds; the 32-cell core is fixed, and secondary prose/figures may move to an appendix |
| R3 limitations | zero-shot detector collapses or returns null predictions on SAR | retain the pinned null result as reference context; do not tune it into a core arm |

## Appendix A — Scorer worked example

GT at (0,0) and (500,0) m; predictions (10,0,0.9), (180,0,0.8), (900,0,0.7). Greedy by score: 0.9 matches GT1 (10 m) → TP. 0.8 → nearest unmatched GT2 at 320 m > 200 → FP. 0.7 → 400 m from GT2 → FP. GT2 unmatched → FN. P = 1/3, R = 1/2, F1 = 0.4. `test_scorer.py` asserts exactly this.

## Appendix B — Label schema (SARFish / xView3)

Columns used: `scene_id, detect_scene_row, detect_scene_column, is_vessel, is_fishing, vessel_length_m, confidence{HIGH,MEDIUM,LOW}, source{AIS, AIS/Manual, Manual}, *_distance_from_shore_km, top/left/bottom/right (sparse)`. Dark-vessel proxy: `source == "Manual"` (no AIS correlate). Detection positives: `is_vessel == True` with confidence ∈ {HIGH, MEDIUM}; LOW → ignore region; non-vessel maritime objects (fixed infrastructure) are not targets but are the expected near-shore FP source.

LS-SSDD carries only boxes and was reduced to centroids under the superseded design. It is not a current Arm-4/8 source; its frozen split and old results are retained only for provenance.

## Appendix C — Pascal (sm_60) gotchas, P100 node

1. The driver may advertise a newer CUDA level than the installed torch wheel supports. Use a torch build that explicitly contains **sm_60** kernels and prove it with `gpu_sanity.py`; the historical V100 lock is not evidence of P100 compatibility.
2. **No bf16.** fp16 + `torch.cuda.amp.GradScaler`; keep LayerNorm/GroupNorm and any logit/softmax math in fp32 (`autocast` handles most; verify with `gpu_sanity.py`).
3. **No FlashAttention.** Use the supported math/memory-efficient SDPA path and verify it; P100 has no Tensor Cores, so fp16 does not imply modern mixed-precision throughput.
4. **Avoid bitsandbytes**; nothing here needs it and Pascal support is not part of the recipe.
5. **Memory gate passed 2026-07-22.** Both ViT and CNN completed finite fp16 forward/backward/AdamW probes at micro-batch 8; use accumulation 2 for effective batch 16 on every P100 cell. Revalidate only after an environment/model-runtime change. Run bf16-sensitive R3 on the 5070 Ti.

## Appendix D — What is deliberately out of scope

- **Pretraining anything ourselves** — out of scope. All six pretrained core checkpoints are downloaded; only xView3 fine-tuning runs locally.
- **Random→LS-SSDD as core Arms 4/8, and CNN-only R1** — superseded by the 2026-07-22 matched ImageNet core role. Preserve old LS artifacts/results historically, but never include them in active curves or counts.
- **DINOv3-SAT ViT-L / DINOv3-ConvNeXt as study arms** — rejected: ViT-L is 3.5\texttimes\ the params (size confound); DINOv3-ConvNeXt was considered for the CNN optical arm but ImageNet-vs-BigEarthNet was chosen instead (DINOv3-ConvNeXt-Base availability unverified and likely ConvNeXt-V1, a within-track architecture mismatch). ConvNeXt-**V2**-Base keeps the CNN track internally consistent.
- **Self-pretraining any foundation model** (SAR-MAE-from-scratch, FCMAE-on-SAR ConvNeXt, or distilling SARMAE into a CNN) — cut for compute, novel-code, and teacher-dependence risk. The six pretrained core checkpoints are downloaded.
- **The unconstrained challenge/leaderboard arm** — removed entirely to protect the Sep 1 deadline. The eight-arm controlled study is the deliverable; the pooled multi-dataset supervised set (HRSID/SAR-Ship/SSDD) it would have used is no longer fetched.
- **The self-supervised (SSL) corpus and OpenSARShip** — no SSL/self-pretraining lane exists; all pretrained core checkpoints are downloaded, so “SSL corpus/pixels” is dead scope.
- **IC-ViT (isolated-channel patchify) for channel handling** — considered, not default. It patchifies each polarization separately with no channel-specific parameters (arXiv 2503.09826), a principled "VH and VV as separate streams" approach, but it changes tokenization and complicates the same-input fairness story. The fixed 3-channel [VH,VV,VH−VV] representation is the default; adopt IC-ViT only on an explicit human decision.
- **Image-space super-resolution** — confound-heavy and no HR Sentinel-1 target exists; the only SR-flavored move considered is a stride-2 decoder ablation, and even that is optional.
- **Complex-valued SLC / Doppler** — a separate project (storage + non-square pixels); not in this plan.
- **DETR / box heads as the detector** — the point-native heatmap head is the detector; a box-head comparison is at most a one-off, not a study arm.
