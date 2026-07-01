# DEVPLAN — xView3 Label-Efficient Dark Vessel Detection

Development plan for a coding agent (Claude Code or similar). Execute phases in order; each phase ends with explicit acceptance criteria. Do not start a phase until the previous phase's criteria pass, except where a phase is marked parallel-safe.

**Project question (the study).** Under one fixed point-detection harness, how much labeled xView3 data does each class of *pretrained backbone* save for dark-vessel detection — does a SAR-domain foundation model beat an optical one in the scarce-label regime, and **does that finding hold across architecture families (ViT and CNN)?** The headline deliverable is a **label-efficiency curve** with two matched tracks (ViT and CNN); the optical-vs-SAR contrast and its architecture-generality are the central results.

**What this is and is not.** This is a *controlled label-efficiency comparison of pretrained backbones across two architecture families*. Foundation-model backbones are downloaded; the only thing we train ourselves is a supervised-detection backbone per family (on LS-SSDD), which is cheap. This scope removes the project's two largest risks (self-pretraining a foundation model — its GPU-hours and novel code) while still answering both the domain-of-pretraining question and the architecture-generality question. The novelty is the systematic comparison — optical vs SAR pretraining, ViT vs CNN, on *dark-vessel, scarce-label* detection through a point-native harness — which to our knowledge has not been done.

**Design principle: architecture-matched fairness.** The study has **two tracks — a ViT track (ViT-B/16, ~86M) and a CNN track (ConvNeXt-V2-Base, ~89M, size-matched).** Within each track, all four arms share the *same* backbone architecture, detection head, fine-tuning schedule, and seeds; only the downloaded initialization differs. Across tracks, the two backbones are deliberately size-matched (~86M vs ~89M) so ViT-vs-CNN is a fair architecture comparison. Every claim about pretraining is made *within* a track (architecture held fixed); every architecture-generality claim is made by comparing *matched roles across* tracks (pretraining role held fixed). We report parameter count and GPU-hours per arm.

**Secondary deliverable removed.** The unconstrained challenge/leaderboard arm has been **cut** to protect the Sep 1 deadline; the controlled two-track study is the entire deliverable. An optional contingent reference arm (ImageNet-ConvNeXt) may be added if the eight core arms complete early (see Section 12).

## The arms

The experiment is a set of **downloaded (or, for the two supervised arms, cheaply-trained) backbone initializations** fed through one fixed detection harness, in two architecture-matched tracks. The initialization is the only variable *within* each track; the architecture is the only variable *across* matched roles.

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
- **SARMAE** (Liu et al., CVPR 2026, `MiliLab/SARMAE`): ViT-B/16 masked autoencoder pretrained on SAR-1M (million-scale SAR) with speckle-aware enhancement. CC BY-NC 4.0. The ViT SAR anchor.
- **BigEarthNet ConvNeXt-V2-B** (TU Berlin RSiM/BIFOLD, reBEN; `BIFOLD-BigEarthNetv2-0/convnextv2_base-{s1,s2}-v0.2.0`): ConvNeXt-V2-Base supervised on BigEarthNet v2.0 land-cover classification, in Sentinel-1 (SAR) and Sentinel-2 (optical) variants. The CNN optical/SAR anchors. *These are supervised-classification pretrained, not self-supervised foundation models — a documented distinction from SatDINO/SARMAE; they still provide in-domain SAR/optical feature learning.*

**Cross-family method caveats (stated openly in the writeup, not hidden):**
- The two *ViT* FM arms differ in SSL method (SatDINO=DINO, SARMAE=MAE) as well as domain — no downloadable optical ViT-B MAE existed to match SARMAE's method, so we compare the best released FM of each domain as-is.
- The two *CNN* RS arms (BigEarthNet S1/S2) are the *cleanest* domain contrast in the study: identical model, identical dataset, identical supervised task, differing *only* in input modality (SAR vs optical). The CNN track is therefore the controlled domain comparison; the ViT track is the best-available-FM comparison.
- ViT FM arms are self-supervised; CNN RS arms are supervised-classification. This SSL-vs-supervised difference is a track-level property, documented, not a within-track confound.

**Channel representation (study-wide, all arms):** the fixed 3-channel [VH, VV, VH−VV] in dB. Every downloaded backbone was pretrained on a 3-channel-ish input that is not dual-pol SAR (SatDINO=optical RGB; SARMAE=single-pol SAR→3ch; BigEarthNet-S1=Sentinel-1 VV/VH 2-band; BigEarthNet-S2=10-band Sentinel-2), so each backbone's stem adapts to the [VH,VV,VH−VV] input during fine-tuning via the `timm` `in_chans` path. Bounded, documented, and applied identically to every arm.

**ImageNet is intentionally NOT a core arm** — it appears only as the optional contingent reference R1 (ConvNeXt-V2-B, ImageNet), gating the "RS-specific vs generic pretraining" question for the CNN track if time permits.

## 0. Ground rules

1. **The eval scorer is sacred.** Written first (Phase 2), unit-tested, never modified after Phase 3 begins. Every reported number in the project flows through `src/eval/scorer.py`.
2. **Initialization is the only variable *within* a track; architecture is the only variable *across* matched roles.** Two backbone definitions only: ViT-B/16 (arms 1–4) and ConvNeXt-V2-Base (arms 5–8). Within each track, one head, one optimizer config, one augmentation policy, one decode config, one fine-tuning schedule, shared seeds — every arm fine-tunes end-to-end, no frozen/fine-tuned asymmetry. The head/optimizer/schedule are shared across *both* tracks too (only the backbone differs), so ViT-vs-CNN is fair. If you are tempted to tune something per-arm, stop — that breaks the study. The two supervised arms (4, 8) additionally involve a cheap LS-SSDD backbone pretraining before the sweep; that is the only training we run.
3. **Scene-level splits only.** No chip from a dev/test/eval scene may appear in any training or pretraining corpus. Membership is keyed on `scene_id`, recorded once in `data/splits.json`; every dataloader asserts membership at construction.
4. **The ~50 human-verified xView3 validation scenes are touched exactly once,** by `src/eval/final_eval.py`, after the grid is complete. Tripwire: the script refuses to run without `--i-am-sure` and writes a timestamped lockfile on first use.
5. **Two machines, distinct roles.** Dev/iteration card: one RTX 5070 Ti (16 GB, Blackwell sm_120 — needs PyTorch nightly / cu12x). Heavy lifting: one **8× V100 node** (32 GB each, Volta sm_70). Volta constraints in Appendix C are mandatory — CUDA 12.x wheels, fp16 + GradScaler with norms in fp32, no bf16, no FlashAttention, no bitsandbytes. The two boxes have *different* CUDA/torch builds; pin each in its own lockfile.
6. **Core arms vs. references are reported separately.** The eight core arms (1–8) go in the study tables and the two-track label-efficiency figure. The reference arms (R1 ImageNet-ConvNeXt if run, R2 YOLO26, R3 LocateAnything) go in a separate "references" section for context; they are not on the controlled curves. There is no challenge/leaderboard arm.
7. **Determinism.** Every run takes `--seed`; seed torch, numpy, random, and dataloader workers. Log the resolved config (full YAML), git SHA, and an environment hash into the run directory.
8. **Run directories.** Every experiment writes `runs/<exp_id>/` with `config.yaml`, `metrics.csv` (per-epoch), `final_metrics.json`, `checkpoints/`, `log.txt`. Experiment IDs come from the manifest in Section 13 — never invent ad-hoc names.
9. **One framework for every arm: PyTorch Lightning.** All training entrypoints (the eight arms' fine-tuning, plus the two supervised-transfer backbone trainings on LS-SSDD for arms 4 and 8) are `LightningModule`s sharing one `LightningDataModule` and one `Trainer` config, so DDP across the 8×V100 node, mixed-precision, seeding, and checkpointing are identical across arms by construction. Backbones load as plain `nn.Module`s inside the LightningModule (timm for both ViT-B and ConvNeXt-V2-B; SatDINO/SARMAE via their loaders; BigEarthNet ConvNeXt via the `configilm`/reBEN loader). One head class attaches to either backbone via a small adapter (ViT tokens → stride-4 map; ConvNeXt stage-3 feature map → stride-4 map). Keep configs in YAML; avoid hydra. This consistency is itself part of the fairness argument.
10. **Fairness accounting.** Every arm logs parameter count and fine-tuning GPU-hours. Pretraining GPU-hours are 0 on our side for the six downloaded/random arms (1,2,3,5,6,7) and external/cited for the four downloaded FMs. The two supervised-transfer arms (4, 8) each run a cheap LS-SSDD backbone pretraining, measured. These populate a fairness table in the writeup. Within each track all four arms share architecture and fine-tuning compute; across tracks the two backbones are size-matched (~86M vs ~89M).
11. **Determinism.** Every run takes `--seed`; seed torch, numpy, random, and dataloader workers (Lightning's `seed_everything(workers=True)`). Log the resolved config (full YAML), git SHA, and an environment hash into the run directory.
12. **Run directories.** Every experiment writes `runs/<exp_id>/` with `config.yaml`, `metrics.csv` (per-epoch), `final_metrics.json`, `checkpoints/`, `log.txt`. Experiment IDs come from the manifest in Section 13 — never invent ad-hoc names.
13. **Commit per task** with messages referencing the task ID (e.g. `P1.3: scene-level split builder`). Never commit data, checkpoints, or anything under `runs/`.
14. **Follow cited methods; do not invent.** For any component that names a reference method (Section 1a), implement that method's published recipe — do not substitute a "better," "simpler," or "more modern" approach reasoned up independently, however plausible it seems. Novelty in this project lives in the *experimental comparison*, not in re-deriving pretraining or channel-adaptation mechanics. If a reference is ambiguous, unavailable, or two references conflict, STOP and surface the question to a human rather than improvising — a silently-wrong method (e.g. a hand-rolled patch-embed hack or an ad-hoc reconstruction target) corrupts every downstream number without throwing an error.

## 1. Repository layout

```
xview3-ssl/
  README.md
  DEVPLAN.md                  # this file
  pyproject.toml              # base deps; see two lockfiles below
  locks/
    env-5070ti.txt            # cu12x / nightly for sm_120
    env-v100node.txt          # cu12x for sm_70 (no bf16/flash)
  Makefile                    # one target per phase entrypoint
  configs/
    data.yaml                 # paths, chip size, split fractions, seed
    harness.yaml              # SHARED: head, optimizer, aug, decode (never edited per study-arm)
    arms.yaml                 # the run matrix (Section 13)
  data/                       # gitignored; symlink to large storage
    raw/
      xview3/                 # SARFish GRD GeoTIFFs + label CSVs
      lsssdd/                 # LS-SSDD-v1.0 (Arm 4 supervised source)
    chips/                    # 800x800 chips + per-chip JSON sidecars
    splits.json
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

Every component below has a canonical paper and/or repository. **The agent's job is to adapt the reference implementation, not to design a new method.** When a detail is unspecified by this plan, copy the reference's choice rather than inventing one. Pin versions where practical.

| Component | Follow this reference | What to copy |
|---|---|---|
| ViT-B backbone (shared by all arms) | `timm` `vit_base_patch16_224` | the one architecture every arm uses; FM checkpoints load into it |
| Arm-3 weights (ViT SAR FM) | SARMAE, `MiliLab/SARMAE` (Liu et al. CVPR 2026) | ViT-B SAR-1M checkpoint load (HF `SARMAE_vitb_checkpoint-last`); encoder only, drop optical-branch/decoder; CC BY-NC |
| Arms 6,7 weights (CNN RS) | BigEarthNet ConvNeXt-V2-B, `BIFOLD-BigEarthNetv2-0/convnextv2_base-{s2,s1}-v0.2.0` (reBEN, Clasen et al. IGARSS 2025) | ConvNeXt-V2-Base; load via `configilm` `BigEarthNetv2_0_ImageClassifier.from_pretrained`, take the backbone, drop the classification head; S2=optical, S1=SAR |
| CNN backbone + stem adaptation | `timm` `convnextv2_base` + `in_chans=3` | ConvNeXt-V2-Base as the CNN arm backbone; stem takes the fixed [VH,VV,VH−VV] input via `in_chans`, same Repeat-with-rescaling policy as the ViT stem |
| CNN floor / supervised / R1 | `timm` `convnextv2_base` | Arm 5 = trunc-normal random init; Arm 8 supervised-pretrains on LS-SSDD like Arm 4; R1 (contingent) = ImageNet FCMAE weights |
| Backbone + channel adapt | `timm` (`vit_base_patch16_224`, `in_chans`) | ViT-B/16 definition; `in_chans` Repeat-with-rescaling for any channel change |
| Arm-2 weights (optical FM) | SatDINO, `strakajk/satdino-vit_base-16` (Straka & Gruber 2025, arXiv 2508.21402) | ViT-B/16 fMoW-RGB DINO checkpoint; `AutoModel.from_pretrained(..., trust_remote_code=True)`; Apache-2.0 |
| Heatmap head + decode | CenterNet (Objects as Points); SAR precedent TRANSAR | penalty-reduced focal target, peak + distance-NMS decode |
| Supervised SAR detection (Arm 4) | LS-SSDD-v1.0 repo conventions | box→centroid conversion, 800px sub-image handling |

If a referenced repo is unavailable, a citation is ambiguous, or two references conflict, **stop and flag it for a human** — do not paper over the gap with an improvised method (see ground rule 14).

## 1b. Execution discipline — sprints, gates, and stop conditions

This section governs *how* the plan is executed, not what it builds. Its purpose is to limit agent drift: the failure mode where an agent, run unsupervised against a large spec, gradually optimizes for "make this file work" instead of "serve the study," and silently substitutes plausible-but-wrong choices that never throw an error. Every rule here shrinks the unsupervised interval, makes failures loud, or gates progression behind a human.

### Schedule (calendar)

The project runs **July 1 to September 1, 2026** (about nine weeks); the final project is due **Sep 1**. All dates in phase headers below are calendar dates on this window. Because foundation models are downloaded (no self-pretraining of an FM), the schedule is gated by engineering and analysis, not training compute (~245 GPU-hours total, about two days on the node). High-level mapping of phases to dates:

| Dates (2026) | Focus (phases) |
|---|---|
| Jul 1–14 | Data pipeline + sacred scorer + harness (Phases 0–3, parallel lanes) |
| Jul 15–28 | Both tracks' FM + floor arms (ViT: SatDINO, SARMAE; CNN: BigEarthNet S1/S2); channel-format check; references (Phase 4) |
| Jul 29–Aug 11 | Supervised-transfer arms (ViT+CNN on LS-SSDD) + full label-fraction grid + seed reruns (Phase 5) |
| Aug 12–25 | Final verified-scene eval + ViT-vs-CNN error analysis; begin writeup (Phases 6–7) |
| Aug 26–Sep 1 | Finish writeup and submit; optional contingent ImageNet-ConvNeXt reference only if time remains (Phase 8) |

The writeup deliberately overlaps the Aug 12–25 analysis block rather than starting cold at the end; the final week is for finishing and submission, not first drafting. The optional contingent reference arm (Phase 8) is the schedule's release valve — cut it first if anything runs late, since the eight-arm study is the deliverable. (The unconstrained challenge arm was removed entirely to protect the deadline.)

### Git and repository

- **Remote:** `https://github.com/jroth1414/JHU-xView3`.
- **Authorship — strict.** All commits are authored by the repository owner (the human). Set `git config user.name` and `git config user.email` to the owner's identity. **Do NOT add `Co-Authored-By:` trailers, "Generated with …", "🤖", or any AI/agent attribution** in commit messages, commit bodies, PR titles, or PR descriptions. The agent is a tool, not a listed author. (Note for the human: some agent CLIs inject attribution by default; disable it in the agent's own settings too, and verify the first few commits land clean — the repo instruction is necessary but may not be sufficient on its own.)
- **Branching:** one branch per sprint, named as below. No direct commits to `main`.
- **Merge = human gate.** Every sprint lands on `main` via a pull request the human reviews. The PR is where drift is caught before it compounds. Keep PRs small enough to actually read line-by-line; if a sprint's diff is growing past a few hundred lines, split it.
- **Commits:** small and frequent, each referencing the task ID (e.g. `P1.3: scene-level split builder`). Small commits make the review diff legible and `git bisect` cheap.

### Sprints (one branch each; mirror the phases, no renumbering)

Each sprint branch carries a short `SPRINT.md` stating its goal, its acceptance criteria (copied from the referenced phase), and its definition of done — so the agent on that branch holds a *small* contract in context, not the whole plan. Sprints are tagged by **review tier**, which sets how much human attention the merge gets — vigilance is finite, so spend it where drift actually corrupts the study.

| Sprint (branch) | Phase | Review tier | Why this tier |
|---|---|---|---|
| `sprint-0-env` | Phase 0 | Leaf | tooling only; can't corrupt results |
| `sprint-1-data` | Phase 1 | **Foundation** | a leaky split silently poisons every run |
| `sprint-2-scorer` | Phase 2 | **Foundation** | defines correctness for every reported number |
| `sprint-3-harness` | Phase 3 | Spine | the harness *is* the fairness guarantee |
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

**Hard branch-ordering constraint:** no model-code sprint (`sprint-3` onward) may open a PR until **`sprint-2-scorer` is merged to `main`**. Building arms against a still-moving scorer makes every early number untrustworthy. Likewise `sprint-1-data` must merge before any sprint that trains on chips.

### Sub-agent distribution

Sub-agents are safe only on **independent** work; they drift badly when they share hidden state. Match distribution to the two-owner lane split, not to "one agent per sprint":

- **Parallelizable lanes:** the data pipeline (`sprint-1`) and the scorer (`sprint-2`) are independent and can run as parallel sub-agents from day one. The reference models (`sprint-5`) are independent of the study spine. (There is no pretraining lane — both FMs are downloaded — which removes the project's old long-pole background job.)
- **Must be serial — one owner:** every sprint that touches the shared harness (`sprint-3 → 4 → 6`) modifies the same `lit_modules.py`/`finetune.py`. Do **not** parallelize these across sub-agents — they would each edit the shared code and silently diverge, reintroducing the per-arm inconsistency the fairness design exists to prevent. One agent owns the harness; arms run through it sequentially.
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
- `configs/harness.yaml` — the shared head/optimizer/schedule, frozen after `sprint-3`; this is the fairness contract.
- The verified-scene eval protocol / its lockfile — touched exactly once (ground rule 4).

### Guard tests (machine-enforced, wired to pre-merge CI)

Prose checklists depend on someone remembering to run them — and the drift problem *is* that agents stop remembering. These four assertions live in `tests/` and run on every PR, converting silent corruption into a failed build. They are the technical embodiment of ground rule 14. See P2.3 and P3.x for where they attach; the canonical list:

1. **`test_split_disjoint`** — no `scene_id` appears in more than one split; also asserted at datamodule construction. Kills leakage.
2. **`test_backbone_parity`** — *within-track* parity: the four ViT arms instantiate the identical ViT-B/16 param count, and the four CNN arms the identical ConvNeXt-V2-Base param count. Kills accidental architecture divergence that would void a within-track fairness claim. (Cross-track sizes are close but not identical — ~86M vs ~89M — and are reported, not asserted equal.)
3. **`test_fm_checkpoints_load`** — all four downloaded backbones load with the expected key match, no silent shape-mismatch or partial load: SatDINO (ViT-B, via `trust_remote_code`), SARMAE (ViT-B), and BigEarthNet ConvNeXt-V2-B S1 and S2 (via the `configilm`/reBEN loader). Exercises each non-standard load path. Kills any arm quietly running on random weights.
4. **`test_scorer_immutable`** — a hash of `scorer.py` matches the pinned value recorded at `sprint-2` merge; checked before any eval runs. Kills scorer drift.

A sprint cannot merge with a failing guard test. If a guard test must legitimately change (rare), that is itself a STOP.

These tests are enforced by **GitHub Actions CI** (`.github/workflows/ci.yml`), which runs them on every push and PR — so enforcement does not depend on any agent or human remembering to run them. CI is CPU-only and uses a minimal `requirements-ci.txt` (not the GPU training environment); guard tests must therefore run against tiny in-repo fixtures. A companion **`AGENTS.md`** at the repo root gives any coding agent the short list of non-negotiables (this section is the detail behind it); AGENTS.md is the instruction layer, CI is the enforcement layer, and this DEVPLAN is the single source of truth.

## 2. Phase 0 — Environment and smoke checks

Targets both machines. Do first.

- **P0.1** Create the repo skeleton; `.gitignore` covering `data/`, `runs/`, `*.pt`, `*.tif`.
- **P0.2** Two lockfiles, because the boxes differ:
  - `locks/env-v100node.txt` — torch/torchvision from a **cu12x** index (cu126 or cu128), `timm`, `numpy`, `pandas`, `pyarrow`, `rasterio`, `shapely`, `scipy`, `matplotlib`, `huggingface_hub`, `lightning` (PyTorch Lightning), `ultralytics`, `pytest`, and `transformers` (for SatDINO's `AutoModel`/`trust_remote_code` load) plus `huggingface_hub` for fetching the gated SARMAE weights. **Never `pip install torch` bare here** — a cu13 wheel will refuse sm_70.
  - `locks/env-5070ti.txt` — same packages but torch built for **sm_120** (PyTorch nightly / cu12x as required by Blackwell).
  README documents both and which machine runs what.
- **P0.3** `scripts/gpu_sanity.py`: print device name + capability, run a 4096×4096 fp16 matmul and an `F.scaled_dot_product_attention` call, assert no NaN, report the SDPA backend chosen. Run on both machines; paste both outputs into the README.
- **P0.4** `Makefile` targets: `make env-check`, `make test`, `make data`, `make pretrain-sup-vit`, `make pretrain-sup-cnn`, `make grid`, `make final-eval`, `make ref-optional`. (No FM-pretraining target — all four FMs are downloaded; the only trainings are the two LS-SSDD supervised backbones.)

**Acceptance:** `make env-check` passes on both machines; `pytest` collects (0 tests OK); README shows both `gpu_sanity` outputs and documents the Volta pins.

## 3. Phase 1 — Data acquisition, chipping, splits

Owner: data owner. Jul 1–14. GPU: none (CPU + disk).

### P1.1 xView3 / SARFish download — `src/data/download_sarfish.py`
- `huggingface_hub.snapshot_download(repo_id="ConnorLuckettDSTG/SARFish", repo_type="dataset", allow_patterns=[...])`, patterns restricted to **GRD only, never SLC** (SLC is the bulk of multi-TB).
- Inputs: a scene-ID list. Outputs: `data/raw/xview3/GRD/<scene_id>/...` + md5 verification per the SARFish instructions.
- Pull: (a) 75–150 train scenes chosen in P1.3, (b) all ~50 human-verified validation scenes, (c) the 150 public products (GRD) for the SSL corpus.
- Labels: fetch SARFish CSVs; schema in Appendix B. Note the dataset card's country-restriction warning in the README.
- Budget guard: print estimated size before downloading; abort if > configurable cap (default 400 GB).

### P1.2 Auxiliary SAR datasets — `src/data/download_aux.py`
Fetch and register, each into `data/raw/<name>/`:
- **LS-SSDD-v1.0** (Arm 4 supervised source): native Sentinel-1, 15 large scenes already cut into 9,000 800×800 sub-images. GitHub `TianwenZhang0825/LS-SSDD-v1.0-OPEN`.
- **OpenSARShip** 1.0/2.0 (SSL pixels only): ~34k Sentinel-1 IW ship chips. SJTU portal (registration-gated; document access in README).
- (The pooled SAR-ship datasets HRSID / SAR-Ship-Dataset / SSDD were only for the removed challenge arm and are **no longer fetched**.)
- Record license per source in `data/raw/<name>/LICENSE.note`. Do not proceed to use a set whose license note is missing.

### P1.3 Chipping — `src/data/chipper.py`
- Read VH + VV GeoTIFFs with rasterio in windows; emit 800×800 chips, 100 px overlap, stride 700 (xView3). LS-SSDD already arrives as 800×800, so just normalize + register it.
- Pixel pipeline: raw → dB-like log `x = log10(clip(raw, 1, None))`; store float16 log values; normalize at load time with global per-polarization mean/std computed over the **train split only** (`data/stats.json`).
- **Channel representation (study-wide, fixed for all arms).** Every arm — including SatDINO, which was pretrained on 3-channel RGB — receives the *same* 3-channel SAR input: **[VH, VV, VH−VV]** in normalized dB. Rationale: (a) VH and VV are distinct polarimetric measurements, and their difference (≈ co/cross-pol contrast) is a physically meaningful third channel rather than a zero-pad or a duplicate; (b) using one fixed 3-channel tensor everywhere means the input is *physically identical* across all arms, so channel handling is never a confound between them; (c) 3 channels lets both downloaded FMs (SatDINO, SARMAE), each pretrained with a 3-channel patch-embed, load unmodified. This single decision replaces all per-arm channel hacks. Document it once; do not vary it per arm.
- **Channel adaptation — use the named library method, do not invent one.** Because the input is fixed at 3 channels, SatDINO's 3-channel patch embed loads with **no modification** and no channel surgery is required for any arm. If at any point a channel count must change (e.g. an ablation), the ONLY sanctioned mechanism is **`timm.create_model(..., in_chans=N)`**, which implements the field-standard *Repeat* method (tile/average the pretrained RGB projection weights across the new channels and rescale to preserve activation magnitude). Do NOT hand-roll patch-embed weight manipulation, and do NOT derive a "better" scheme. The single documented alternative, allowed only as an explicit ablation, is **RGB+random** (USat: keep pretrained weights for the RGB-equivalent channels, randomly initialize any extra channels). References to follow, not redesign: `timm` (`in_chans` rescaling); USat (2023); the init-strategy survey arXiv 2503.09493 (which documents that *Repeat* and *RGB+random* are the two known options and that neither is universally best — i.e. this is a bounded choice, not an open problem). A third named method, IC-ViT (isolated-channel patchify, arXiv 2503.09826), is noted in Appendix D as considered-but-not-default; do not adopt it without a human decision.
- Drop chips >95% no-data; record land fraction per chip where shoreline vectors exist.
- Label projection: per-chip JSON sidecar with vessel points in chip-pixel coords, `confidence`, `source`, `vessel_length_m`, `distance_from_shore_km`, and bbox fields when present.
- Manifest: one parquet per scene/source — chip path, scene_id, origin row/col, n_vessels, has_low_conf, land_frac.

### P1.4 Centroid conversion — `src/data/to_centroids.py`
- The harness trains on **center points**, so every labeled supervised source is reduced to centroids. Box → center; rotated box → center; instance mask → centroid. This is the move that lets LS-SSDD feed the same heatmap head (for both the ViT Arm 4 and CNN Arm 8 supervised backbones) with no box-format harmonization.
- Intensity normalization of LS-SSDD to the common dB range used study-wide.
- `tests/test_centroids.py`: box/rbox/mask → expected center within 1 px.

### P1.5 Scene selection and splits — `src/data/splits.py`
- Stratify chosen xView3 train scenes by coarse region (cluster scene-center lat/lon into ~6 bins) and shoreline presence (any label `distance_from_shore_km < 5`).
- Scene-level split of xView3 train scenes: **75% train / 15% dev / 10% test** (seeded). The ~50 verified scenes form `eval_final`, excluded from everything. The 150 public products form `corpus_extra` (unlabeled).
- LS-SSDD gets its own internal train/val split (it is a *pretraining* source, never mixed into xView3 splits).
- `data/splits.json` maps scene_id → split; `tests/test_splits.py` asserts disjointness and counts.

**Acceptance:** per-split + per-source chip counts logged; `pytest tests/test_chipper.py tests/test_centroids.py tests/test_splits.py tests/test_split_disjoint.py` green (the disjointness guard from §1b is wired here); a QA script renders a 4×4 gallery of random chips with label points overlaid (`runs/qa/chips.png`) for a human eyeball.

## 4. Phase 2 — Scorer and decode (before any model code)

Owner: harness owner. Jul 1–14 (parallel-safe with Phase 1).

### P2.1 `src/eval/scorer.py` (SACRED)
- Inputs: predictions `[(x_m, y_m, score)]` and ground truth `[(x_m, y_m, attrs)]` per scene, in meters (chip px × 10 m GSD, offset by chip origin).
- Greedy matching: sort predictions by score desc; each matches the nearest unmatched GT within `tol_m = 200`. Matched → TP; unmatched prediction → FP; unmatched GT → FN.
- Outputs: precision, recall, aggregate F1, plus sliced metrics by filtering GT (and matched predictions) on attrs: `dark` (`source == "Manual"`, no AIS correlate), `near_shore` (`distance_from_shore_km <= 2`), and a `low_conf_ignore` protocol where LOW-confidence GT neither count as FN nor award TP.
- Threshold sweep: given raw scored predictions, return the F1-maximizing threshold on **dev**; that threshold is frozen for test/eval.

### P2.2 `src/eval/decode.py`
- `peaks = (heat == maxpool3x3(heat)) & (heat > tau)` → candidates with scores.
- Distance NMS at `d_nms_m = 120` (config): sort by score, suppress candidates within `d_nms` of a kept one. Use `scipy.spatial.cKDTree` so whole-scene decode stays fast.

### P2.3 Tests
- `tests/test_scorer.py`: exact hit; hit at 199 m (TP); miss at 201 m (FN+FP); two predictions on one GT (1 TP + 1 FP); score-order priority; LOW-confidence ignore behavior; dark / near-shore slice math (see Appendix A).
- `tests/test_decode.py`: plateau handling, NMS suppression order, empty heatmap.

**Acceptance:** all tests green; scorer reproduces a synthetic scene's known P/R exactly. On merge of `sprint-2`, record the pinned `scorer.py` hash that `tests/test_scorer_immutable.py` (§1b guard) checks before every subsequent eval; the scorer is frozen from this point (ground rule 1, do-not-touch manifest).

## 5. Phase 3 — Models and the shared fine-tune harness

Owner: harness owner. Jul 8–21. Dev card: 5070 Ti. This harness is frozen once Phase 4 begins (ground rule 1–2). It must support **two backbone families** (ViT-B/16 and ConvNeXt-V2-Base) behind one head and one training loop.

### P3.1 `src/models/backbones.py`
- **ViT arm backbone:** ViT-B/16 via timm (`vit_base_patch16_224`, `in_chans=3` on the fixed [VH,VV,VH−VV] input), learnable pos-embed interpolated to 512×512 inputs (32×32 tokens), final norm kept. SatDINO and SARMAE are both this architecture.
- **CNN arm backbone:** ConvNeXt-V2-Base via timm (`convnextv2_base`, `in_chans=3` on the same input). Use `features_only=True` or take the stage-3 feature map; ConvNeXt-V2-B at 512 input gives a stride-32 feature map (16×16, 1024-dim). BigEarthNet arms are this architecture.
- Both expose a uniform `(feature_map, channels, stride)` interface to the head adapter. ViT: reshape the stride-16 token grid to (B, 768, 32, 32). ConvNeXt: the stride-32 (B, 1024, 16, 16) stage-3 map.
- ViT-S/16 fallback documented in `harness.yaml`, gated by the P6 throughput check (applies uniformly within a track if invoked).

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

- Assert after loading (`tests/test_fm_checkpoints_load.py`, §1b guard): all four downloaded backbones (SatDINO, SARMAE, BigEarthNet-S1, BigEarthNet-S2) load with the expected key match — no silent shape-mismatch, no partial load leaving a backbone partly random. Within each track all four arms expose an identical backbone interface to the head.

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

## 6. Phase 4 — Both tracks' foundation-model + floor arms, and references

Owner: harness owner. Jul 15–28. One config per V100 per night. **No FM pretraining in this project** (all FMs downloaded), so the old "long pole" is gone; this phase plus the grid is the bulk of the compute, and it is cheap.

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

## 7. Phase 5 — Supervised-transfer arms (4, 8) + the label-efficiency grid

Owner: harness owner. Jul 29–Aug 11. This runs the only two backbone trainings in the project (the ViT and CNN supervised-transfer backbones on LS-SSDD); everything else is downloaded.

- **P5.1** `src/train/pretrain_supervised.py`: pretrain a backbone + heatmap head on **LS-SSDD-v1.0 only**, centroids as targets (P1.4), same loss/sampler/aug as the fine-tune harness. Run it **twice** — once with the ViT-B backbone (→ `vit_supervised`, Arm 4) and once with the ConvNeXt-V2-B backbone (→ `cnn_supervised`, Arm 8). Same single matched source keeps each supervised arm's contrast clean within its track. ~10–30 GPU-h each; these are the only backbones we train.
- **P5.2** Arm 4 (`vit_supervised`) and Arm 8 (`cnn_supervised`) fine-tuned at all four label fractions — 8 runs. (Arms 1,2,3,5,6,7 already ran in Phase 4; this completes the eight-arm grid.)
- **P5.3** Seed reruns: the 25% and 100% cells of **all eight** study arms with 2 extra seeds — 32 runs — so the headline cells carry error bars. One config per card per night; ~4–5 nights across the 8-GPU node.
- **P5.4** Per run: freeze the dev-tuned threshold, score the test split, append to `runs/summary/grid.csv` via `curves.py`; render the two-track label-efficiency figure (x = label fraction, log-scale; y = test F1; eight curves — solid for ViT, dashed for CNN, one color per pretraining role; shaded seed bands).
- **P5.5** Headline computations: (a) *within-track* — arm ordering at 10% labels for each track (optical vs SAR vs supervised vs floor); (b) *cross-track* — for each pretraining role, the ViT-vs-CNN gap at each fraction (does the SAR-domain advantage hold for both architectures?); (c) the interpolated label budget at which each arm matches the SAR-FM arm @25% within its track.

**Acceptance:** `grid.csv` has 32 core (8 arms × 4 fractions) + 32 seed rows, no NaNs; the eight-curve two-track figure renders; monotonicity sanity check (F1 non-decreasing in label fraction per arm, within seed noise) passes or is investigated.

## 8. Phase 6 — Final eval

- **P6.1** FINAL EVAL (once): best config per study arm at 100% (and the 25% cells) scored on the verified scenes via `final_eval.py --i-am-sure`. These are the study's headline numbers; nothing is tuned after this.

**Acceptance:** lockfile exists; `runs/summary/final_verified.csv` written.

## 9. Phase 7 — Error analysis and figures (study)

Owner: harness owner. Aug 12–25 (begin writeup in parallel).

- **P7.1** `error_slices.py`: per-arm dark-vessel recall and near-shore F1 vs label fraction, for **both tracks**; FP taxonomy on ~200 sampled FPs (shoreline clutter / fixed infrastructure / sea clutter / sidelobe). Two headline slices: the **SAR-vs-optical dark-vessel-recall gap within each track**, and the **ViT-vs-CNN gap at matched pretraining roles** (does the SAR-domain advantage generalize across architectures?).
- **P7.2** `qualitative.py`: a fixed gallery of 24 chips (8 dark-vessel hits, 8 misses, 8 FPs) rendered identically for all eight study arms — the money figure beside the two-track curve.
- **P7.3** `architecture_comparison.py`: the cross-track summary — for each of the four pretraining roles, ViT vs CNN F1 across fractions, with the BigEarthNet-S1-vs-S2 contrast (cleanest domain comparison) called out.

## 10. Phase 8 — Contingent reference arm (R1, OPTIONAL)

Owner: either. Aug 26–Sep 1, **only if the eight core arms have landed and time remains before the deadline.** Reported in the references section, not on the controlled curves. (The unconstrained challenge/leaderboard arm was removed entirely from the project to protect the deadline; this contingent reference replaces it as the schedule's release valve.)

- **P8.1** Arm R1 — `bigearthnet`-style but ImageNet: load ImageNet-pretrained ConvNeXt-V2-B (`timm` `convnextv2_base`, FCMAE ImageNet weights), fine-tune through the identical harness at all four label fractions — 4 runs. This is the "generic natural-image pretraining" baseline for the CNN track: comparing R1 to Arm 6/7 (BigEarthNet RS pretraining) answers *how much of the CNN's transfer comes from RS-specific pretraining vs. generic visual pretraining?*
- **P8.2** Add R1's curve to the references panel (not the controlled two-track figure). One paragraph on the RS-vs-generic-pretraining gap for CNNs.

**Acceptance (only if run):** R1's four cells in `runs/summary/references.csv`; a note on the RS-vs-generic gap. If skipped for time, the eight-arm study is complete and unaffected.

## 11. Hyperparameter reference

| Component | Setting | Value |
|---|---|---|
| Chips | size / overlap / GSD | 800 px / 100 px / 10 m |
| Train crops | size / vessel-biased frac | 512 / 0.7 |
| Heatmap | stride / sigma / loss | 4 / 2 out-px / penalty-reduced focal α=2 β=4 |
| Decode | tau / d_nms / match tol | dev-tuned / 120 m / 200 m |
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
- Seed reruns (`-f25-` and `-f100-`, seeds 1–2, all 8 arms): 32 runs.

References: `yolo26-f100`, `locateanything-zs`.
Backbone trainings we run (not fine-tunes): `vitsup-lsssdd`, `cnnsup-lsssdd` (the two supervised-transfer backbones).
Contingent reference (optional): `imgnetcnn-f{10,25,50,100}-s0` — only if core arms finish early.

Total = 64 study fine-tunes (32 core + 32 seed) + 2 reference runs + 2 supervised-transfer backbones + up to 4 optional contingent-reference fine-tunes. **No foundation-model pretraining** — all four FMs downloaded. Each fine-tune fits one V100 overnight; the 8-GPU node clears the whole grid in a few nights.

### GPU-hour estimates (planning, not measured)

Order-of-magnitude only. Assumes 8×V100, 512px crops; ConvNeXt-V2-B ≈ ViT-B cost at this resolution (CNN cells may run ~1.2–1.5× heavier — treated as noise here).

| Component | Runs | ~GPU-h each | ~GPU-h total |
|---|---|---|---|
| Core grid (8 arms × 4 fractions) | 32 | 3 | ~96 |
| Seed reruns (25% + 100% cells, 8 arms) | 32 | 3 | ~96 |
| Supervised-transfer backbones (LS-SSDD: ViT + CNN) | 2 | 10–30 | ~40 |
| YOLO26 reference | 1 | 5–15 | ~10 |
| LocateAnything zero-shot (5070 Ti) | 1 | 1–2 | ~2 |
| SatDINO/SARMAE/BigEarthNet weights | — | 0 (downloaded) | 0 |
| **Core total** | | | **~245** |
| Contingent ImageNet-ConvNeXt reference (if run) | 4 | 3 | ~24 |

Controlled eight-arm study ≈ **~245 GPU-hours** — roughly two days on the 8×V100 node. Still an order of magnitude below a from-scratch-pretraining design (~1,200–2,200 GPU-h); the FM downloads keep it cheap, and the only training we run is the two small supervised-transfer backbones. Adding the contingent reference brings it to ~270. Note earlier project drafts quoted ~130 GPU-h for a four-arm ViT-only study; doubling to two tracks roughly doubles it.

## 13. Risk register

| Risk | Signal | Mitigation |
|---|---|---|
| Channel-format gap breaks transfer (either backbone) | an FM arm ≤ its floor at all fractions, or fails to converge | the P3.6 check catches it per track; confirm [VH,VV,VH−VV] load + normalization; if persistent, a *finding*, not necessarily a bug |
| A downloaded backbone loads partially (silent random weights) | `test_fm_checkpoints_load` fails / unexpected key report | the guard blocks merge; fix the loader, never proceed on a partial load — covers SatDINO, SARMAE, BigEarthNet-S1/S2 |
| BigEarthNet ConvNeXt-V2 load quirks (`configilm`/FCMAE stem) | key/shape mismatch on the reBEN loader | the guard exercises this path; may need the `configilm` package pinned + the reBEN model-def repo; STOP and verify rather than hand-patch |
| ConvNeXt-V2 stem won't take 3-channel input cleanly | S1 model is 2-band (VV/VH), S2 is 10-band | `timm` `in_chans=3` Repeat-with-rescaling adapts the stem, same policy as the ViT; verify in P3.6 |
| A downloaded arm underperforms its floor | arm ≤ floor at low fractions | not a bug — report as a transfer finding; verify load + channel handling first |
| Cross-track result is null (ViT≈CNN, or SAR-adv doesn't generalize) | matched-role gaps ~0 | still a result ("the SAR-domain advantage is/ isn't architecture-general"); lean on dark-vessel/near-shore slices |
| Licenses (SARMAE CC BY-NC, BigEarthNet weights) | — | academic/research use only; fine for the course; flag in writeup; verify BigEarthNet weight license at download |
| Backbone too slow on the node | grid cell > a few GPU-h | drop the *track* to a smaller variant uniformly (ViT-S / ConvNeXt-Tiny), preserving within-track parity; document |
| Heatmap recall stuck low | dev recall ≪ YOLO26 ref | raise σ to 3, lower tau, confirm vessel-biased cropping fires |
| Label noise dominates | LOW-conf ignore shifts F1 > 2 pts | report both protocols; prefer ignore-protocol as primary |
| fp16 divergence | NaN loss | norms in fp32 (already), halve lr for that run, log |
| Download cap blown | size estimate > cap | reduce train scenes toward 75; SLC never fetched; need only *weights* for FMs, not SAR-1M (76 GB) or full BigEarthNet |
| Arm count / analysis overrun | Phase 7 slipping, curves illegible | the contingent reference (Phase 8) is cut first; if needed, demote BigEarthNet modality detail to an appendix |

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
- **IC-ViT (isolated-channel patchify) for channel handling** — considered, not default. It patchifies each polarization separately with no channel-specific parameters (arXiv 2503.09826), a principled "VH and VV as separate streams" approach, but it changes tokenization and complicates the same-input fairness story. The fixed 3-channel [VH,VV,VH−VV] representation is the default; adopt IC-ViT only on an explicit human decision.
- **Image-space super-resolution** — confound-heavy and no HR Sentinel-1 target exists; the only SR-flavored move considered is a stride-2 decoder ablation, and even that is optional.
- **Complex-valued SLC / Doppler** — a separate project (storage + non-square pixels); not in this plan.
- **DETR / box heads as the harness** — the point-native heatmap head is the harness; a box-head comparison is at most a one-off, not a study arm.
