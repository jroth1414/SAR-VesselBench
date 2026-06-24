# DEVPLAN — xView3 Label-Efficient Dark Vessel Detection

Development plan for a coding agent (Claude Code or similar). Execute phases in order; each phase ends with explicit acceptance criteria. Do not start a phase until the previous phase's criteria pass, except where a phase is marked parallel-safe.

**Project question (the study).** Under one fixed point-detection harness, how much labeled xView3 data does each class of *pretrained backbone* save for dark-vessel detection — and does a SAR-domain foundation model (SARMAE) deliver materially better label efficiency than a strong optical-domain one (SatDINO) in the scarce-label regime? The headline deliverable is a **label-efficiency curve**; the optical-vs-SAR foundation-model contrast is the central result, and arm ordering reads as a domain-distance transferability result.

**What this is and is not.** This is a *controlled label-efficiency comparison of pretrained backbones*, not a study of pretraining methods we run ourselves. Every backbone is downloaded; we pretrain nothing. This is a deliberate scope choice: it removes the project's two largest risks (pretraining GPU-hours and novel pretraining code) and lets all effort go into a clean, metric-matched evaluation. The novelty is the systematic comparison — optical-FM vs SAR-FM transfer, specifically on *dark-vessel, scarce-label* detection through a point-native harness — which to our knowledge has not been done.

**Design principle: maximal fairness, zero pretraining.** All four study arms use the **same architecture (ViT-B/16, ~86M params), the same detection head, the same fine-tuning schedule, and the same seeds** — the only variable is the downloaded initialization. Both foundation-model arms (SatDINO, SARMAE) are ViT-B/16, so swapping them in does not change architecture; the fairness argument is simply "identical ViT-B backbone, fine-tuned identically through one harness, differing only in pretrained weights." We report parameter count and GPU-hours per arm. (A larger optical model, DINOv3-SAT ViT-L, was rejected as a study arm precisely because its 300M params — 3.5× ViT-B — would confound size with the pretraining-domain effect; it may appear only as an optional, separately-reported reference.)

**Secondary deliverable (the challenge).** A single unconstrained run aimed at the xView3 leaderboard, reported *separately* from the study.

## The arms

The experiment is a set of **downloaded ViT-B/16 initializations** fed through one fixed detection harness. The downloaded init is the only variable across study arms.

| Arm | Name | Initialization | On the curve? | Role |
|-----|------|----------------|---------------|------|
| 1 | Random (floor) | none (random init) | yes | no-pretraining baseline |
| 2 | SatDINO-B | **SatDINO ViT-B/16** (fMoW-RGB, DINO, downloaded) | yes | optical-domain FM transfer |
| 3 | SARMAE (core) | **SARMAE ViT-B/16** (SAR-1M, downloaded) | yes | SAR-domain FM transfer; the central comparison |
| 4 | Supervised | LS-SSDD-v1.0 supervised (we train head+backbone) | yes | labeled near-domain transfer |
| 5 | Challenge | best study init → unconstrained tricks | **no** | leaderboard coda, reported apart |
| 6 | References | YOLO26-COCO; LocateAnything-3B zero-shot | **no** | bracket the achievable range |

SARMAE (Liu et al., CVPR 2026; `MiliLab/SARMAE`) is a ViT-B/16 masked autoencoder pretrained on SAR-1M (the first million-scale SAR dataset) with speckle-aware enhancement and optical-prior alignment. We use the **released weights** — we do not pretrain it. It is the SAR-domain counterpart to SatDINO's optical-domain pretraining; the Arm-2-vs-Arm-3 contrast is the study's headline (optical FM vs SAR FM for dark-vessel detection). Note the two FM arms differ in SSL *method* as well as domain — SatDINO is DINO self-distillation, SARMAE is masked autoencoding — so the comparison is between the best available released foundation model of each domain, taken as-is, not a controlled same-method ablation. This is stated openly in the writeup. License CC BY-NC 4.0 (academic use only); SatDINO is Apache-2.0.

Arms 1–4 are the controlled study (each run at every label fraction). Arm 5 is the kitchen-sink leaderboard entry. Arm 6 is two external models run once for context.

**ImageNet-MAE is intentionally NOT an arm** — SatDINO (fMoW-RGB) is the single optical-transfer anchor. Do not add an ImageNet-MAE arm.

**All study arms are ViT-B/16 and all fine-tune end-to-end.** Both foundation-model checkpoints (SatDINO, SARMAE) are fine-tuned end-to-end (no frozen-backbone arm), so there is no frozen/fine-tuned asymmetry — arms 1–4 share architecture, size, fine-tuning regime, and head. The only variable is the downloaded init (none / optical-FM / SAR-FM / supervised). **Channel-format note (state in the writeup, do not hide):** both foundation models were pretrained on 3-channel inputs that are not dual-pol SAR — SatDINO on optical RGB, SARMAE on single-pol SAR PNGs mapped to 3 channels. We feed both the same fixed 3-channel [VH, VV, VH−VV] tensor (so the input is identical across all arms, preserving fairness); each model's patch-embed therefore sees a third channel it did not pretrain on, which the fine-tuning adapts. This is a known, bounded transfer gap shared symmetrically by both FM arms, not a confound between them. DINOv3-SAT ViT-L is rejected as a study arm (3.5× the params); it may return only as an optional, separately-reported reference.

## 0. Ground rules

1. **The eval scorer is sacred.** Written first (Phase 2), unit-tested, never modified after Phase 3 begins. Every reported number in the project flows through `src/eval/scorer.py`.
2. **Pretraining is the only variable across study arms (1–4).** One model definition (ViT-B/16), one head, one optimizer config, one augmentation policy, one decode config, one fine-tuning schedule, shared seeds. Every arm fine-tunes end-to-end; there is no frozen/fine-tuned asymmetry. If you are tempted to tune something per-arm, stop — that breaks the study. (Arm 5 is fully exempt; it is not a controlled arm.)
3. **Scene-level splits only.** No chip from a dev/test/eval scene may appear in any training or pretraining corpus. Membership is keyed on `scene_id`, recorded once in `data/splits.json`; every dataloader asserts membership at construction.
4. **The ~50 human-verified xView3 validation scenes are touched exactly once,** by `src/eval/final_eval.py`, after the grid is complete. Tripwire: the script refuses to run without `--i-am-sure` and writes a timestamped lockfile on first use.
5. **Two machines, distinct roles.** Dev/iteration card: one RTX 5070 Ti (16 GB, Blackwell sm_120 — needs PyTorch nightly / cu12x). Heavy lifting: one **8× V100 node** (32 GB each, Volta sm_70). Volta constraints in Appendix C are mandatory — CUDA 12.x wheels, fp16 + GradScaler with norms in fp32, no bf16, no FlashAttention, no bitsandbytes. The two boxes have *different* CUDA/torch builds; pin each in its own lockfile.
6. **Study vs. challenge are reported separately.** Arms 1–4 + the corpus ablation go in the study tables and the label-efficiency figure. Arm 5 goes in its own "challenge submission" section. They may cite each other; they never share a results table.
7. **Determinism.** Every run takes `--seed`; seed torch, numpy, random, and dataloader workers. Log the resolved config (full YAML), git SHA, and an environment hash into the run directory.
8. **Run directories.** Every experiment writes `runs/<exp_id>/` with `config.yaml`, `metrics.csv` (per-epoch), `final_metrics.json`, `checkpoints/`, `log.txt`. Experiment IDs come from the manifest in Section 13 — never invent ad-hoc names.
9. **One framework for every arm: PyTorch Lightning.** All training entrypoints (the four arms' fine-tuning, plus the supervised-transfer backbone training for Arm 4) are `LightningModule`s sharing one `LightningDataModule` and one `Trainer` config, so DDP across the 8×V100 node, mixed-precision, seeding, and checkpointing are identical across arms by construction. Backbones load as plain `nn.Module`s inside the LightningModule (timm for ViT-B; the SatDINO and SARMAE checkpoints via their loaders). Keep configs in YAML; avoid hydra. This consistency is itself part of the fairness argument.
10. **Fairness accounting.** Every arm logs parameter count and fine-tuning GPU-hours; pretraining GPU-hours are 0 on our side for arms 1–3 (random needs none; SatDINO and SARMAE are downloaded) and external/cited for those two FMs. Arm 4's supervised-transfer backbone training is the only pretraining we run, and its cost is measured. These populate a fairness table in the writeup. Arms 1–4 share architecture and fine-tuning compute.
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
      pool/                   # HRSID, SAR-Ship-Dataset, SSDD (Arm 5 only)
    chips/                    # 800x800 chips + per-chip JSON sidecars
    splits.json
    manifests/                # parquet chip manifests per split/source
  src/
    data/
      download_sarfish.py     # selective HF GRD pulls
      download_aux.py         # LS-SSDD / HRSID / SAR-Ship / SSDD / OpenSARShip fetchers
      chipper.py              # scene -> chips + label projection (xView3 + LS-SSDD)
      to_centroids.py         # boxes/masks -> center points (LS-SSDD, pool sets)
      splits.py               # scene-level split builder
      datasets.py             # FineTuneDataset, SupervisedSARDataset
      transforms.py           # log-norm, crops, flips
    models/
      vit.py                  # ViT-B/16, 3-channel patch embed (VH,VV,VH-VV)
      heatmap_head.py         # deconv tower + 1ch sigmoid head
      init_loaders.py         # random / satdino_b / sarmae_b / supervised loaders
    train/
      lit_modules.py          # LightningModules: supervised-pretrain, finetune
      datamodule.py           # one LightningDataModule (chips, splits, channel rep)
      pretrain_supervised.py  # entrypoint: heatmap pretrain on labeled SAR (Arm 4 source)
      finetune.py             # entrypoint: shared fine-tune loop (all study arms)
      sampler.py              # foreground-balanced chip sampler
      losses.py               # penalty-reduced focal
    eval/
      decode.py               # peaks, distance NMS
      scorer.py               # distance-matched F1 + slices  (SACRED)
      infer_scene.py          # tiled whole-scene inference
      final_eval.py           # verified-scenes tripwire script
    challenge/
      build_pool.py           # harmonize pooled supervised sets -> centroids
      pseudo_label.py         # self-train on noisy/public scenes
      ensemble_tta.py         # YOLO26 ensemble + test-time augmentation + WBF
    references/
      yolo26_ref.py           # boxes from points+lengths, ultralytics run
      locateanything_zs.py    # zero-shot grounding VLM probe
    analysis/
      curves.py               # label-efficiency plots
      error_slices.py         # dark-vessel / shoreline breakdowns
      qualitative.py          # chip galleries with predictions
  tests/
    test_scorer.py
    test_decode.py
    test_chipper.py
    test_centroids.py
    test_splits.py
    test_split_disjoint.py      # guard: no scene_id in two splits (1b)
    test_backbone_parity.py     # guard: all arms identical ViT-B param count (1b)
    test_fm_checkpoints_load.py  # guard: SatDINO + SARMAE load into ViT-B w/ expected keys (1b)
    test_scorer_immutable.py    # guard: scorer.py hash matches pinned value (1b)
  runs/                       # gitignored
```

## 1a. Reference implementations — adapt these, do not reinvent

Every component below has a canonical paper and/or repository. **The agent's job is to adapt the reference implementation, not to design a new method.** When a detail is unspecified by this plan, copy the reference's choice rather than inventing one. Pin versions where practical.

| Component | Follow this reference | What to copy |
|---|---|---|
| ViT-B backbone (shared by all arms) | `timm` `vit_base_patch16_224` | the one architecture every arm uses; FM checkpoints load into it |
| Arm-3 weights (SAR FM) | SARMAE, `MiliLab/SARMAE` (Liu et al. CVPR 2026) | ViT-B SAR-1M checkpoint load (HF `SARMAE_vitb_checkpoint-last`); encoder only, drop optical-branch/decoder; CC BY-NC |
| Backbone + channel adapt | `timm` (`vit_base_patch16_224`, `in_chans`) | ViT-B/16 definition; `in_chans` Repeat-with-rescaling for any channel change |
| Arm-2 weights (optical FM) | SatDINO, `strakajk/satdino-vit_base-16` (Straka & Gruber 2025, arXiv 2508.21402) | ViT-B/16 fMoW-RGB DINO checkpoint; `AutoModel.from_pretrained(..., trust_remote_code=True)`; Apache-2.0 |
| Heatmap head + decode | CenterNet (Objects as Points); SAR precedent TRANSAR | penalty-reduced focal target, peak + distance-NMS decode |
| Supervised SAR detection (Arm 4) | LS-SSDD-v1.0 repo conventions | box→centroid conversion, 800px sub-image handling |

If a referenced repo is unavailable, a citation is ambiguous, or two references conflict, **stop and flag it for a human** — do not paper over the gap with an improvised method (see ground rule 14).

## 1b. Execution discipline — sprints, gates, and stop conditions

This section governs *how* the plan is executed, not what it builds. Its purpose is to limit agent drift: the failure mode where an agent, run unsupervised against a large spec, gradually optimizes for "make this file work" instead of "serve the study," and silently substitutes plausible-but-wrong choices that never throw an error. Every rule here shrinks the unsupervised interval, makes failures loud, or gates progression behind a human.

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
| `sprint-4-fm-arms` | Phase 4 (Arms 2,3) | Spine | the FM arms + channel-format check; the headline contrast |
| `sprint-5-floor-refs` | Phase 4 (rest) | Leaf | floor + external refs; off the controlled curve |
| `sprint-6-grid` | Phase 5 | Spine | supervised arm + label-fraction grid through the shared harness |
| `sprint-7-final-eval` | Phase 6 | **Foundation** | touches the once-only verified-scene eval |
| `sprint-8-analysis` | Phase 7 | Leaf | figures/slices; read the output, trust the code |
| `sprint-9-challenge` | Phase 8 | Leaf | optional, separate from the study |

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
2. **`test_backbone_parity`** — all four study arms instantiate the identical ViT-B/16 parameter count. Kills accidental architecture divergence that would void the fairness claim.
3. **`test_fm_checkpoints_load`** — the SatDINO and SARMAE checkpoints both load into the ViT-B/16 backbone with the expected key match (no silent shape-mismatch or partial load; SatDINO loads via `trust_remote_code`, so exercise that path and verify patch-token extraction). Kills a foundation-model arm quietly running on random weights.
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
- **P0.4** `Makefile` targets: `make env-check`, `make test`, `make data`, `make pretrain-sup`, `make grid`, `make challenge`, `make final-eval`. (No `pretrain-sarmae` target — the SAR FM is downloaded.)

**Acceptance:** `make env-check` passes on both machines; `pytest` collects (0 tests OK); README shows both `gpu_sanity` outputs and documents the Volta pins.

## 3. Phase 1 — Data acquisition, chipping, splits

Owner: data owner. Week 1–2. GPU: none (CPU + disk).

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
- **Pool sets — Arm 5 only:** HRSID (`chaozhong2010/HRSID`), SAR-Ship-Dataset (`CAESAR-Radi/SAR-Ship-Dataset`), SSDD (`TianwenZhang0825/Official-SSDD`).
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
- The harness trains on **center points**, so every labeled supervised source is reduced to centroids. Box → center; rotated box → center; instance mask → centroid. This is the move that lets LS-SSDD (Arm 4) and the pool sets (Arm 5) feed the same heatmap head with no box-format harmonization.
- Per-source intensity normalization to the common dB range (sensors differ across pool sets).
- `tests/test_centroids.py`: box/rbox/mask → expected center within 1 px.

### P1.5 Scene selection and splits — `src/data/splits.py`
- Stratify chosen xView3 train scenes by coarse region (cluster scene-center lat/lon into ~6 bins) and shoreline presence (any label `distance_from_shore_km < 5`).
- Scene-level split of xView3 train scenes: **75% train / 15% dev / 10% test** (seeded). The ~50 verified scenes form `eval_final`, excluded from everything. The 150 public products form `corpus_extra` (unlabeled).
- LS-SSDD gets its own internal train/val split (it is a *pretraining* source, never mixed into xView3 splits).
- `data/splits.json` maps scene_id → split; `tests/test_splits.py` asserts disjointness and counts.

**Acceptance:** per-split + per-source chip counts logged; `pytest tests/test_chipper.py tests/test_centroids.py tests/test_splits.py tests/test_split_disjoint.py` green (the disjointness guard from §1b is wired here); a QA script renders a 4×4 gallery of random chips with label points overlaid (`runs/qa/chips.png`) for a human eyeball.

## 4. Phase 2 — Scorer and decode (before any model code)

Owner: harness owner. Week 1–2. Parallel-safe with Phase 1.

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

## 5. Phase 3 — Model and the shared fine-tune harness

Owner: harness owner. Week 2–3. Dev card: 5070 Ti. This harness is frozen once Phase 4 begins (ground rule 1–2).

### P3.1 `src/models/vit.py`
- ViT-B/16 via timm (`vit_base_patch16_224`): 3-channel patch embed (`in_chans=3`, taking the fixed [VH,VV,VH-VV] input), learnable pos-embed interpolated to 512×512 inputs (32×32 tokens), final norm kept. Both downloaded FMs (SatDINO, SARMAE) are ViT-B/16; SARMAE loads as a plain `vit_base_patch16`, SatDINO via its `trust_remote_code` class, both exposing the same 768-dim backbone interface.
- ViT-S/16 fallback documented in `harness.yaml`, gated by the P6 throughput check (applies uniformly to all arms if invoked).

### P3.2 `src/models/init_loaders.py`
Four loaders behind one enum. Each prints matched/missing/unexpected key counts after loading. **All four produce the identical ViT-B/16 backbone; only the weights differ.**
- `random`: timm trunc-normal init.
- `satdino_b`: load the **SatDINO ViT-B/16** fMoW-RGB checkpoint (`strakajk/satdino-vit_base-16`, DINO self-distillation; `AutoModel.from_pretrained("strakajk/satdino-vit_base-16", trust_remote_code=True)`, or the manual `vit_base(patch_size=16)` + `load_state_dict(ckpt['teacher'])` path from the repo). Encoder loads into the identical ViT-B backbone; the 3-channel patch embed takes the fixed [VH, VV, VH−VV] input directly. SatDINO's default `forward(x)` returns only the CLS vector; call `forward(x, return_all=True)` and use `tokens[:, 1:-1]` (drop CLS and GSD register) as the patch-token sequence, reshaped to `(B, 768, H/16, W/16)` for the shared head. Drop the DINO projection head. Apache-2.0.
- `sarmae_b`: load the **SARMAE ViT-B/16** checkpoint (`Wenquandan777/SARMAE` on HuggingFace, file `SARMAE_vitb_checkpoint-last`; CC BY-NC 4.0, accept terms first). It is `vit_base_patch16`, so the encoder loads into the identical backbone with no surgery; the 3-channel patch embed takes the fixed [VH, VV, VH−VV] input directly. SARMAE was pretrained on single-pol SAR PNGs mapped to 3 channels, so its embed sees the VH−VV third channel as out-of-distribution — adapted during fine-tuning (channel-format note, §"All study arms"). Load encoder weights only; ignore SARMAE's optical-branch / SARC / decoder parameters (we use the SAR encoder).
- Arm 2 (SatDINO) and Arm 3 (SARMAE) are both ViT-B foundation models fine-tuned identically through one harness; their contrast is the study headline: optical-domain FM transfer vs. SAR-domain FM transfer. (They differ in SSL method — DINO vs MAE — as well as domain; this is a documented caveat, since no downloadable optical ViT-B MAE was available to match SARMAE's method. We compare released models as-is.)
- `supervised`: load the Phase 5 supervised-pretrained encoder (LS-SSDD); shapes match.
- Assert after loading (`tests/test_fm_checkpoints_load.py`, §1b guard): SatDINO and SARMAE both load into the ViT-B/16 backbone with the expected key match — no silent shape-mismatch, no partial load leaving the backbone partly random. The test must also pass a tiny tensor through SatDINO's `return_all=True` path and confirm the extracted patch tokens reshape to the expected ViT-B stride-16 grid. All four arms expose an identical ViT-B/16 backbone interface to the head.

### P3.3 `src/models/heatmap_head.py`
- Input: a stride-16 token grid reshaped to (B, 768, 32, 32) for 512 px crops — **uniform across all four arms**, since every backbone is ViT-B/16 (embed dim 768). SatDINO must provide patch tokens via `return_all=True` / `tokens[:, 1:-1]`; SARMAE/timm backbones expose the same patch-token grid. No per-arm head variation.
- Two `[ConvTranspose2d ×2 upsample, GroupNorm(32), GELU]` blocks 768→256→128, then 3×3 conv → 1 channel; sigmoid at inference, logits at train.
- Output stride 4 (128×128 for a 512 input).

### P3.4 Targets, loss, sampler
- `src/train/losses.py`: penalty-reduced focal (CenterNet form), α=2, β=4, on Gaussian targets σ=2 output px; LOW-confidence labels stamp an ignore disk (radius 3 output px) where loss is zeroed.
- `src/train/sampler.py`: epoch-level weighted sampler so ~50% of sampled chips contain ≥1 HIGH/MED vessel.
- `src/data/transforms.py`: random 512 crop from the 800 chip (vessel-biased: 70% of crops centered within 128 px of a vessel when one exists), flips, 90° rotations, intensity jitter ±0.1 in log space. No augmentation that breaks SAR statistics (no blur, no elastic).

### P3.5 `src/train/finetune.py` — the shared loop
- AdamW, lr 1e-4, layer-wise lr decay 0.65, weight decay 0.05, cosine schedule, 5-epoch warmup, 50 epochs, batch 16 (fp16 + GradScaler, norms fp32), grad clip 1.0.
- `--init {random,satdino_b,sarmae_b,supervised}` selects the loader. **All four fine-tune end-to-end with the identical schedule** (no frozen path, no per-arm optimizer differences). Everything — head, loss, sampler, augmentation, decode, schedule, seeds — is identical across arms; only the loaded weights differ.
- `--label_frac {0.1,0.25,0.5,1.0}` subsamples xView3 train scenes (scene-level, seeded); fractions **nest** (10% ⊂ 25% ⊂ 50% ⊂ 100%) so the curve is monotone in data.
- Every 5 epochs: tiled inference (`infer_scene.py`, 512 windows, stride 384, global NMS) on 8 fixed dev scenes → dev F1. Early-stop patience 4 dev evals. Save best + last.

**Acceptance:** smoke run (`--init random --label_frac 0.1 --epochs 3`) completes on the 5070 Ti, loss decreases, dev decode yields >0 detections, `runs/qa/pred_gallery.png` overlays look sane. Switching `--init satdino_b` and `--init sarmae_b` loads each encoder into the same backbone with the expected key report and trains identically. `tests/test_backbone_parity.py` and `tests/test_fm_checkpoints_load.py` (§1b guards) pass.

### P3.6 Foundation-model load + early-signal check (cheap, before the full grid)
Both FMs fine-tune normally, so there is no frozen-feature dependency to de-risk. The early checks worth doing before committing the full grid: (1) confirm **both** SatDINO and SARMAE encoders load into the ViT-B backbone with the expected key match (the `test_fm_checkpoints_load` guard); (2) confirm a short fine-tune (a few epochs at 100% labels) of each trains stably and beats random-init at equal epochs — a quick sanity that the downloaded weights are a usable starting point on the [VH,VV,VH−VV] input before the full label-fraction grid. The highest-risk unknown is the **channel-format gap** (both FMs pretrained on 3-channel inputs that are not dual-pol SAR); this check catches a catastrophic mismatch early. Record in `runs/decisions.md`. (Note: SatDINO or SARMAE underperforming random at low fractions is a *finding* about transfer, not a bug.)

## 6. Phase 4 — All foundation-model + floor arms, and references

Owner: harness owner. Week 2–4. One config per V100 per night. **No pretraining runs in this project** (both FMs are downloaded), so the old "long pole" is gone; this phase and the grid are the bulk of the compute, and it is cheap.

**Ordering rationale.** Run Arm 2 (SatDINO) and Arm 3 (SARMAE) early, because the highest-risk unknown is the **channel-format gap**: whether feeding the fixed [VH, VV, VH−VV] tensor into a patch-embed pretrained on 3-channel optical (SatDINO) or single-pol-mapped SAR (SARMAE) transfers cleanly. The P3.6 check catches a catastrophic mismatch before the full grid. Since the channel representation is study-wide, validating it on the FM arms de-risks the shared input for every arm. Arms 1 and 6 need no special handling and run alongside.

- **P4.1** Arm 2 (`satdino_b`) and Arm 3 (`sarmae_b`) — run these first, at 100% labels, via the P3.6 check; confirm stable transfer for both, then run all four label fractions for each — 8 runs. No pretraining cost on our side (weights downloaded). Same ViT-B backbone, schedule, head. If the channel input fails for either, revisit the [VH,VV,VH−VV] representation before proceeding (it affects all arms).
- **P4.2** Arm 1 (`random`) at all four label fractions — 4 runs.
- **P4.3** Arm 6a — `src/references/yolo26_ref.py`: build YOLO boxes from xView3 points+lengths (square side `max(6 px, length_m/10)` centered on the point; use SARFish bbox fields where present), train YOLO26 (ultralytics, COCO init) on the 100% train split, VH/VV/(VH−VV) → RGB slots; score its centers through the SACRED scorer. 1 run.
- **P4.4** Arm 6b — `src/references/locateanything_zs.py`: LocateAnything-3B zero-shot on ~200 dev chips, prompts {"ship","vessel","boat"}; runs on the 5070 Ti (bf16-checkpoint caution, Appendix C.5); centers through the scorer. Expectation-setting, not a competitor.

**Acceptance:** 12 study runs (arms 1–3 × 4 fractions) + 2 reference runs have `final_metrics.json`; a partial label-efficiency plot (3 curves) renders from `curves.py`; the channel representation is confirmed working on real SAR via both FM arms.

## 7. Phase 5 — Supervised-transfer arm (Arm 4) + the label-efficiency grid

Owner: harness owner. Week 3–5. This is the only backbone training we run (Arm 4's supervised transfer); everything else is downloaded.

- **P5.1** `src/train/pretrain_supervised.py`: pretrain the **same ViT + heatmap head** on **LS-SSDD-v1.0 only**, centroids as targets (P1.4), same loss/sampler/aug as the fine-tune harness. Save the encoder for `init_loaders.supervised`. (Single matched source keeps Arm 4's contrast with the FM arms clean — labeled near-domain SAR vs. downloaded FMs.) This is the only backbone we train.
- **P5.2** Arm 4 (`supervised`) fine-tuned at all four label fractions — 4 runs. (Arms 1–3 already ran in Phase 4; this completes the four-arm grid.)
- **P5.3** Seed reruns: the 25% and 100% cells of **all four** study arms with 2 extra seeds — 16 runs — so the headline cells carry error bars. One config per card per night; ~2–3 nights.
- **P5.4** Per run: freeze the dev-tuned threshold, score the test split, append to `runs/summary/grid.csv` via `curves.py`; render the label-efficiency figure (x = label fraction, log-scale; y = test F1; four curves; shaded seed bands).
- **P5.5** Headline computation: for each fraction f, report what fraction of (100% random-init) F1 each arm reaches, and the interpolated label budget at which random / SatDINO-B / supervised match SARMAE@25%. State the arm ordering at 10% labels (the optical-FM vs SAR-FM transferability readout — the study's central result).

**Acceptance:** `grid.csv` has 16 core (4 arms × 4 fractions) + 16 seed rows, no NaNs; the four-curve figure renders; monotonicity sanity check (F1 non-decreasing in label fraction per arm, within seed noise) passes or is investigated.

## 8. Phase 6 — Final eval

- **P6.1** FINAL EVAL (once): best config per study arm at 100% (and the 25% cells) scored on the verified scenes via `final_eval.py --i-am-sure`. These are the study's headline numbers; nothing is tuned after this.

**Acceptance:** lockfile exists; `runs/summary/final_verified.csv` written.

## 9. Phase 7 — Error analysis and figures (study)

Owner: harness owner. Week 5–7.

- **P7.1** `error_slices.py`: per-arm dark-vessel recall and near-shore F1 vs label fraction; FP taxonomy on ~200 sampled FPs (shoreline clutter / fixed infrastructure / sea clutter / sidelobe) via a small chip-labeling helper. The **optical-FM vs SAR-FM dark-vessel-recall gap** is the headline slice.
- **P7.2** `qualitative.py`: a fixed gallery of 24 chips (8 dark-vessel hits, 8 misses, 8 FPs) rendered identically for all four study arms — the money figure beside the curve.

## 10. Phase 8 — Challenge submission (Arm 5, unconstrained, OPTIONAL)

Owner: either. Week 7+, **only if the study has landed and time remains.** Reported in its own section, never in the study tables. This phase is deliberately exempt from ground rule 2 (single-variable discipline).

- **P8.1** `src/challenge/build_pool.py`: harmonize LS-SSDD + HRSID + SAR-Ship-Dataset + SSDD to centroids (reuse P1.4); per-source dB normalization. This pooled set is for the challenge only.
- **P8.2** Start from the **best study init** (whichever FM won — likely SARMAE), fine-tune on the pooled supervised set, then on xView3. (No self-pretraining; we build on downloaded weights.)
- **P8.3** `src/challenge/pseudo_label.py`: self-train — fit on verified + high-confidence train scenes, pseudo-label the noisy/public scenes with high-confidence predictions, retrain (AI2 4th-place recipe; Zoph et al. 2020).
- **P8.4** `src/challenge/ensemble_tta.py`: ensemble the heatmap model with YOLO26, test-time augmentation (flips/rotations), fuse with WBF/soft-NMS. Tune the decode threshold against the challenge aggregate metric.
- **P8.5** For this run only, training may use train+dev+test (the challenge has its own hidden eval). **Never touch the study's verified-scene protocol or its frozen scorer outputs.**

**Acceptance:** a single challenge entry scored and written to `runs/summary/challenge.csv`, clearly labeled as unconstrained; a one-paragraph note on what each trick contributed.

## 11. Hyperparameter reference

| Component | Setting | Value |
|---|---|---|
| Chips | size / overlap / GSD | 800 px / 100 px / 10 m |
| Train crops | size / vessel-biased frac | 512 / 0.7 |
| Heatmap | stride / sigma / loss | 4 / 2 out-px / penalty-reduced focal α=2 β=4 |
| Decode | tau / d_nms / match tol | dev-tuned / 120 m / 200 m |
| Fine-tune | opt / lr / lld / wd / epochs / batch | AdamW / 1e-4 / 0.65 / 0.05 / 50 / 16 |
| All arm backbones | ViT-B/16 | 86M params, embed-dim 768, fine-tuned end-to-end |
| Init (downloaded) | arms 2 / 3 | SatDINO fMoW-RGB (DINO) / SARMAE SAR-1M (MAE), both ViT-B |
| Channel input | fixed all arms | [VH, VV, VH\textminus VV] in dB (3-channel) |
| Supervised pretrain | source / target | LS-SSDD-v1.0 / centroids → heatmap |
| Ignore regions | LOW-conf radius | 3 out-px |
| Splits | train/dev/test of xView3 train scenes | 75/15/10 scene-level |
| Label fractions | nested | 10 / 25 / 50 / 100% |

## 12. Run manifest (experiment IDs)

Study grid — `exp = {init}-f{frac}-s{seed}`, init ∈ {`rand`,`satdino`,`sarmae`,`sup`}:
- Core (4 arms × 4 fractions, seed 0): 16 runs — `rand-f10-s0` … `sup-f100-s0` (e.g. `sarmae-f25-s0`).
- Seed reruns (`-f25-` and `-f100-`, seeds 1–2, all 4 arms): 16 runs.

References: `yolo26-f100`, `locateanything-zs`.
Backbone training we run (not a fine-tune): `sup-lsssdd` (Arm 4's supervised transfer — the only one).
Challenge (optional, separate): `challenge-final`.

Total = 32 study fine-tunes (16 core + 16 seed) + 2 reference runs + 1 supervised-transfer backbone + 1 optional challenge. **No SAR-MAE pretraining** — both foundation models are downloaded. Each fine-tune fits one V100 overnight; the 8-GPU node clears the whole grid in a few nights.

### GPU-hour estimates (planning, not measured)

Order-of-magnitude only. Assumes 8×V100, ViT-B/16, 512px crops. **The big change from earlier project versions: there is no self-pretraining, so the ~1,000–2,000 GPU-hour SAR-MAE long pole is gone.** Total compute is now dominated by the fine-tuning grid, which is cheap.

| Component | Runs | ~GPU-h each | ~GPU-h total |
|---|---|---|---|
| Fine-tune grid (arms 1–4 × 4 fractions) | 16 | 3 | ~50 |
| Seed reruns (25% + 100% cells) | 16 | 3 | ~50 |
| Supervised-transfer backbone (LS-SSDD, Arm 4) | 1 | 10–30 | ~20 |
| YOLO26 reference | 1 | 5–15 | ~10 |
| LocateAnything zero-shot (5070 Ti) | 1 | 1–2 | ~2 |
| SatDINO + SARMAE weights | — | 0 (downloaded) | 0 |

Controlled study (excluding optional challenge) ≈ **~130 GPU-hours** — roughly a single day on the 8×V100 node, plus the supervised-transfer training and references. This is an order of magnitude below the earlier from-scratch-pretraining design (~1,200–2,200 GPU-h), which is the whole point of the downloaded-FM decision: it removes both the compute long pole and the novel pretraining code.

## 13. Risk register

| Risk | Signal | Mitigation |
|---|---|---|
| Channel-format gap breaks FM transfer | a FM arm ≤ random at all fractions, or fails to converge | the P3.6 check catches it early; confirm [VH,VV,VH−VV] load + normalization; if persistent, it is a *finding* about transfer, not necessarily a bug |
| SARMAE/SatDINO loads partially (silent random weights) | `test_fm_checkpoints_load` fails / unexpected key report | the guard test blocks merge; fix the loader, never proceed on a partial load |
| SatDINO or SARMAE underperforms random | a FM arm ≤ floor at low fractions | not a bug — report as a transfer finding; verify load + channel handling first |
| FM arms perform near-identically | optical ≈ SAR FM on the curve | still a result ("a generic optical FM matches a SAR-specific one on dark vessels"); lean on the dark-vessel/near-shore slices where they may diverge |
| SARMAE license (CC BY-NC) | — | academic/research use only; fine for the course; flag in writeup; no commercial deployment |
| ViT-B too slow on the node | grid run > a few GPU-h each | drop *all* study arms to ViT-S uniformly (incl. a ViT-S optical ckpt); preserve equal-architecture fairness; document |
| Heatmap recall stuck low | dev recall ≪ YOLO26 ref | raise σ to 3, lower tau, confirm vessel-biased cropping fires |
| Label noise dominates | LOW-conf ignore shifts F1 > 2 pts | report both protocols; prefer ignore-protocol as primary |
| fp16 divergence | NaN loss | norms in fp32 (already), halve lr for that run, log |
| Download cap blown | size estimate > cap | reduce train scenes toward 75; SLC must never be fetched; need only SARMAE *weights*, not SAR-1M (76 GB) |
| Challenge eats the semester | Phase 8 slipping | it is optional and last; cut it — the study is the deliverable |

## Appendix A — Scorer worked example

GT at (0,0) and (500,0) m; predictions (10,0,0.9), (180,0,0.8), (900,0,0.7). Greedy by score: 0.9 matches GT1 (10 m) → TP. 0.8 → nearest unmatched GT2 at 320 m > 200 → FP. 0.7 → 400 m from GT2 → FP. GT2 unmatched → FN. P = 1/3, R = 1/2, F1 = 0.4. `test_scorer.py` asserts exactly this.

## Appendix B — Label schema (SARFish / xView3)

Columns used: `scene_id, detect_scene_row, detect_scene_column, is_vessel, is_fishing, vessel_length_m, confidence{HIGH,MEDIUM,LOW}, source{AIS, AIS/Manual, Manual}, *_distance_from_shore_km, top/left/bottom/right (sparse)`. Dark-vessel proxy: `source == "Manual"` (no AIS correlate). Detection positives: `is_vessel == True` with confidence ∈ {HIGH, MEDIUM}; LOW → ignore region; non-vessel maritime objects (fixed infrastructure) are not targets but are the expected near-shore FP source.

LS-SSDD and the pool sets carry only boxes/masks → reduced to centroids (P1.4); they have no confidence tiers or dark/AIS attributes (they are pretraining sources, not evaluated).

## Appendix C — Volta (sm_70) gotchas, V100 node

1. Pin torch wheels for **CUDA 12.x**; CUDA 13 dropped Volta — a stray `pip install torch` pulling cu13 will not see the cards. (The 5070 Ti box is separate and needs sm_120 nightly — keep the two lockfiles distinct.)
2. **No bf16.** fp16 + `torch.cuda.amp.GradScaler`; keep LayerNorm/GroupNorm and any logit/softmax math in fp32 (`autocast` handles most; verify with `gpu_sanity.py`).
3. **No FlashAttention** (never supported sm_70). `F.scaled_dot_product_attention` falls back to mem-efficient/math kernels — correct, just slower; relevant for the ViT, irrelevant for YOLO26's CNN.
4. **Avoid bitsandbytes** — int8/4-bit paths are unreliable on Volta and nothing here needs them.
5. bf16-trained public checkpoints (e.g. LocateAnything) can overflow when cast to fp16 — load fp32 on CPU, then `.half()` only verified-safe modules, or just run that probe on the 5070 Ti.

## Appendix D — What is deliberately out of scope

- **Pretraining anything ourselves** — out of scope by design. Both FMs are downloaded; the project is a comparison of pretrained backbones, chosen specifically to remove pretraining compute and novel pretraining code as risks. (An earlier project version pretrained a SAR-MAE from scratch; that was cut for exactly these reasons.)
- **ImageNet-MAE arm** — SatDINO (fMoW-RGB, ViT-B) is the optical-domain anchor; a third generic-pretraining arm adds little.
- **DINOv3-SAT ViT-L as a study arm** — rejected: 3.5\texttimes\ the params of the other arms confounds size with pretraining domain. Allowed only as an optional separately-reported frontier reference.
- **Choosing the SAR arm's SSL method ourselves** — moot now that Arm 3 is downloaded SARMAE (a masked autoencoder with speckle-aware enhancement, already trained). For the record, the SAR-SSL literature finds contrastive/augmentation-based pretext tasks mismatched to speckle physics and favors masked modeling with gradient targets — which is consistent with SARMAE's design.
- **IC-ViT (isolated-channel patchify) for channel handling** — considered, not default. It patchifies each polarization separately with no channel-specific parameters (arXiv 2503.09826), a principled "VH and VV as separate streams" approach, but it changes tokenization and complicates the same-input fairness story. The fixed 3-channel [VH,VV,VH−VV] representation is the default; adopt IC-ViT only on an explicit human decision.
- **Image-space super-resolution** — confound-heavy and no HR Sentinel-1 target exists; the only SR-flavored move considered is a stride-2 decoder ablation, and even that is optional.
- **Complex-valued SLC / Doppler** — a separate project (storage + non-square pixels); not in this plan.
- **DETR / box heads as the harness** — the point-native heatmap head is the harness; a box-head comparison is at most a one-off, not a study arm.
