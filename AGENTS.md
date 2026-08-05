# AGENTS.md

Instructions for any coding agent (Claude Code, Codex, Cursor, sub-agents) working in this repository.

**This file is the short list of non-negotiables. `DEVPLAN.md` is the single source of truth** for the full design, phases, and rationale. When this file says "see DEVPLAN §X," read it before acting. If this file and DEVPLAN ever conflict, DEVPLAN wins and you should flag the conflict.

**Before choosing a phase or a branch, read the "⚑ Current repository state — READ THIS FIRST" cold-start runbook at the top of `DEVPLAN.md`.** It holds the status ledger (what is DONE / PARTIAL / NOT STARTED), the branch model (the integration branch is **`dev`**, not `main`), and the known blockers. Do not re-do frozen work or start a phase whose entry preconditions are unmet.

---

## What this project is (one paragraph)

A controlled label-efficiency comparison of **downloaded pretrained backbones** for dark-vessel detection in Sentinel-1 SAR (xView3), in **two architecture-matched tracks**: a **ViT track** (arms 1–4, ViT-B/16, ~86M) and a **CNN track** (arms 5–8, ConvNeXt-V2-Base, ~89M, size-matched). Within each track the four arms — random floor, optical remote-sensing pretraining, SAR pretraining, and generic ImageNet pretraining — are fine-tuned identically; **initialization is the only variable *within* a track, architecture the only variable *across* matched roles** (floor 1/5, optical 2/6, SAR 3/7, ImageNet 4/8). Every non-random core initialization is a downloaded checkpoint; there is **no active LS-SSDD pretraining**. The study uses one fixed run seed (`0`): 32 core fine-tunes (8 arms × 4 label fractions) plus R2/R3, for **34 reported experiments**. The deliverable is a **two-track label-efficiency curve**; the optical-vs-SAR contrast and its architecture-generality are the headline. There is **no challenge/leaderboard arm** (removed). Validity depends on the arms being truly comparable, so most rules below exist to protect that comparability.

## The prime directive

**Novelty lives in the experimental comparison, not in the mechanics.** Do not invent methods. For any component with a cited reference (DEVPLAN §1a), implement that reference's published recipe. If a reference is ambiguous, unavailable, or two conflict — **STOP and ask a human.** A silently-wrong method (a hand-rolled patch-embed hack, an ad-hoc reconstruction target) produces plausible numbers with no error and corrupts the whole study. (DEVPLAN ground rule 14.)

## Hard rules (do not violate)

1. **Do not invent methods.** Use the cited reference implementations in DEVPLAN §1a (that table is the canonical source for every id):
   - Arm-3 weights (ViT SAR FM) → **SARMAE**: HF **weights** `Wenquandan777/SARMAE`, file `SARMAE_vitb_checkpoint-last` (code `github.com/MiliLab/SARMAE`; Liu et al. CVPR 2026). Downloaded ViT-B, encoder only, CC BY-NC 4.0 (gated). We do NOT pretrain it.
   - Arm-2 weights (ViT optical FM) → **SatDINO** ViT-B/16 fMoW-RGB (`strakajk/satdino-vit_base-16`, `trust_remote_code=True`; Apache-2.0). DINO, not MAE — differs from SARMAE in method as well as domain (documented caveat).
   - Arm-4 weights (ViT ImageNet) → `timm/vit_base_patch16_224.augreg_in1k`: supervised AugReg training on ImageNet-1K; drop the classification head and transfer the encoder.
   - Arms 6,7 weights (CNN RS) → **BigEarthNet ConvNeXt-V2-B** (`BIFOLD-BigEarthNetv2-0/convnextv2_base-{s2,s1}-v0.2.0`, reBEN): load via `configilm` `BigEarthNetv2_0_ImageClassifier.from_pretrained`, drop the classification head; S2=optical, S1=SAR.
   - Arm-8 weights (CNN ImageNet) → `timm/convnextv2_base.fcmae_ft_in1k`: FCMAE pretraining followed by supervised ImageNet-1K classification fine-tuning; drop the classification head and transfer the encoder.
   - Backbones + any channel change → `timm` `vit_base_patch16_224` (ViT) and `timm` `convnextv2_base` (CNN), both `in_chans=3` on the fixed `[VH, VV, VH−VV]` input (Repeat-with-rescaling). Never hand-roll patch-embed surgery.
   - Head/decode → CenterNet / TRANSAR (heatmap, peak + distance-NMS).

   Arms 4 and 8 are matched on generic source dataset and final classification supervision, **not on full training history**: Arm 4 is supervised AugReg, while Arm 8 is FCMAE then supervised fine-tuning. Treat this as a documented cross-track limitation; never describe them as a matched MAE/FCMAE pair.

2. **Initialization is the only variable *within* a track; architecture is the only variable *across* matched roles.** Within each track (ViT arms 1–4, CNN arms 5–8): same backbone, head, optimizer/schedule, shared strict-IEEE-FP32 Lightning `32-true` training and model-forward inference, fixed run seed `0`, and fixed 3-channel input `[VH, VV, VH−VV]`. The head/optimizer/schedule/precision and execution-hardware class are shared across both tracks too — only the backbone (and its head adapter) differs. On H100, “strict IEEE FP32” additionally means TF32 is disabled for both CUDA matmul and cuDNN; `32-true` alone is not sufficient evidence. Never tune something per-arm. (There is no challenge arm.)

3. **The scorer is sacred.** `src/eval/scorer.py` was written in sprint-2 and intentionally re-frozen in `sprint-2b-eval-hardening` after the near-shore FP fix. Never modify it after that lock without a human STOP and re-pin. Every reported number flows through it.

4. **Scene-level splits only.** No `scene_id` may appear in more than one split. No dev/test/eval scene may appear in any training or pretraining corpus. The ~50 human-verified scenes are touched exactly once, at final eval.

5. **One framework: PyTorch Lightning.** All training entrypoints share one `LightningModule` pattern, one `LightningDataModule`, one `Trainer` config. This is part of the fairness guarantee — do not write a bespoke loop for one arm.

6. **Do-not-touch after their sprint** (changing any of these requires a human STOP): `src/eval/scorer.py`, `data/splits.json`, `data/stats.json`, `data/lsssdd_split.json`, `configs/detector.yaml`, the verified-eval lockfile. The owner approved and `sprint-7c-fp32-grid` intentionally re-pinned `configs/detector.yaml` once on 2026-07-26 for shared `32-true`; that does not unlock any future edit. `data/lsssdd_split.json` remains a retired historical provenance artifact.

7. **The corrected H100 campaign is all-or-none, and V100 remains diagnostic.** All 32 core cells must restart from scratch on one uniformly recorded H100 hardware/environment class after every Sprint-7f acceptance and control-plane gate passes. Never combine V100 and H100 core cells in one curve, table, summary, resume decision, or completion namespace. R2/R3 remain separate V100 references. The owner elected on 2026-08-04 to let the live V100 campaign continue for diagnostic value: do not stop, signal, pause, reconfigure, or otherwise mutate it, and never let one of its markers suppress an H100 cell.

8. **Evaluation GT means vessel HIGH/MEDIUM positives plus LOW ignores.** Every evaluation caller must use the centralized converter: include HIGH/MEDIUM only when `is_vessel` is explicitly true, exclude HIGH/MEDIUM non-vessels as background, retain LOW rows as ignore points, preserve source/shore fields, and reject ambiguous HIGH/MEDIUM booleans. Never edit the frozen scorer to implement this contract, and never inspect `validation.csv` before the once-only final gate.

9. **Operating points are checkpoint-bound and held-out scoring is cohort-gated.** A reportable training marker must use exact result schema 2 and bind the complete best-dev operating point to the actual best-checkpoint relative path, SHA-256, and epoch; `last_dev` is diagnostic only. Validate all 32 markers and atomically freeze one immutable `TRAINING_COHORT.json` before reading any test raster, label, or metric. Test outputs live in separate immutable `test_metrics.json` files and must never mutate `final_metrics.json`.

## The four guard tests (CI enforces these — see `.github/workflows/ci.yml`)

A PR **cannot merge** if any of these fail (once its target file exists). They are not optional and they encode study validity:

- `test_split_disjoint` — no scene in two splits (anti-leakage).
- `test_backbone_parity` — within each track all four arms share an identical param count (ViT-B/16 for arms 1–4, ConvNeXt-V2-B for arms 5–8), **and** both tracks' adapters emit the same stride-4 / 128×128 / C output (anti-architecture-drift + anti-adapter-geometry-drift).
- `test_fm_checkpoints_load` — all six downloaded pretrained core backbones load **value-sensitively** (not just by key name): SatDINO, SARMAE, and ImageNet AugReg (ViT-B), plus BigEarthNet-S1, BigEarthNet-S2, and ImageNet FCMAE→IN1K (ConvNeXt-V2-B). Asserts loaded encoder tensors differ from a fresh random init (anti-silent-random-weights). CI runs the CPU-offline structural half; the value-sensitive load runs on the GPU boxes.
- `test_scorer_immutable` — `scorer.py` hash matches the sprint-2b pinned value (anti-scorer-drift). *(The scorer was intentionally re-pinned after the near-shore FP fix; see the DEVPLAN cold-start runbook.)*

If you believe a guard test must change, that is itself a STOP — surface it, don't edit the test to pass.

## When to STOP and ask a human (mandatory, not optional)

Stopping is cheap and expected. It is not a failure. Halt and ask when:

1. A cited reference is unavailable/ambiguous, or two conflict.
2. An acceptance criterion fails and the fix isn't obvious in one attempt.
3. A result violates a sanity check (non-monotone label-efficiency curve beyond the declared tolerance; an arm scoring implausibly high → suspect leakage). With one seed, do not claim empirical seed-noise estimates or error bars.
4. A change would touch a do-not-touch artifact after it's locked.
5. You're tempted to deviate from a cited method "because it seems better."
6. A step's measured compute/time exceeds its budget by more than ~2×.

Do not optimize to seem autonomous. Asking is the job.

## Workflow

- **Branch per sprint** (see DEVPLAN §1b for the list and review tiers). The integration branch is `dev`; no direct commits to `dev` or `main`.
- **PR per sprint**, reviewed and merged by the human. Keep diffs small enough to read line-by-line; if a sprint's diff grows past a few hundred lines, split it.
- **Branch ordering:** no model-code sprint opens until `sprint-2-scorer` is merged; no chip-training sprint until `sprint-1-data` is merged.
- **Commits:** small, frequent, each referencing the task ID (e.g. `P1.3: scene-level split builder`).
- **Sub-agents:** parallelize only independent lanes — data, scorer, and the **external** reference models (R2/R3). There is **no pretraining lane**: all six pretrained core checkpoints are downloaded, and LS-SSDD is retired from the active study. Anything touching the shared detector (including `sprint-5-cnn-arms`) is serial under one owner — parallel edits to `lit_modules.py`/`finetune.py` silently diverge and break arm comparability.

## Git authorship — strict

- Commits are authored by the **repository owner**. Set `git config user.name` / `user.email` to the owner's identity.
- **Do NOT add** `Co-Authored-By:` trailers, "Generated with …", "🤖", or any AI/agent attribution — not in commit messages, commit bodies, PR titles, or PR descriptions.
- You are a tool, not a listed author.

## Environment notes

- The currently running diagnostic campaign is on this server's **8× Tesla V100-SXM2-32GB** pool (Volta sm_70), verified by `locks/env-v100node.txt`: Python 3.11 and torch 2.11.0+cu126. It continues unchanged by owner decision. Its existing and future core outputs are non-reportable diagnostics because their model selection and early stopping used the superseded evaluation contract. No V100 core cell may enter or suppress the corrected H100 curve.
- The owner-approved Judy H100 runtime is a **native, sealed Python venv only**. First prove the complete base-Python runtime fingerprint is identical on at least two eligible DGX compute nodes; then build one fresh venv inside a Judy compute allocation, directly at a new final persistent path, with Judy's exact Python 3.11.13 executable and `python -m venv --copies`. Never qualify a login-node-built receipt for H100 execution. Install entirely offline with `--no-index` from the verified Sprint-7d wheelhouse; require `pip check`, an exact normalized lock match, a read-only tree, a deterministic tree digest, the exact Sprint-7d extraction/wheelhouse identity, and executable plus full base-Python-runtime/build-receipt hashes. Snapshot Judy's canonical libpython directory and use it as the complete `LD_LIBRARY_PATH` under Slurm `--export=NONE`; never inherit a mutable submit-host loader path. Invoke the verifier only as `python -B -m scripts.h100.build_venv`; Judy's endpoint controls kill the literal direct `scripts/h100/build_venv.py` execution path. Invoke `venv/bin/python` directly under a clean environment. Do not normalize away compute-node provenance differences and do not use Apptainer, Enroot, Pyxis, SIF, OCI, shell activation, user-site packages, BF16, TF32, or FP16 for the H100 core lane.
- Every H100 core cell uses shared Lightning `32-true` for training and dev/test/final model forwards, CUDA-matmul TF32 disabled, cuDNN TF32 disabled, **micro-batch 16, accumulation 1, effective batch 16**, and exactly one process per GPU with no DDP. All 32 core cells restart from scratch after cutover. R2/R3 remain on V100 under their independently pinned recipes.
- The already uploaded immutable Sprint-7d base payload is `xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a`: 150 chip archives, 39 dev/test raster archives, train labels, six checkpoint archives, the exact wheelhouse, one Sprint-7d Git bundle, and its historical Apptainer definition. The definition remains base-payload provenance and is **not executed on Judy**. Never rebuild or mutate that READY package to express any later runtime amendment.
- Sprint 7f travels as a separate, small schema-2 runtime-amendment package containing exactly the `sprint-7f-eval-contract` Git bundle, the deterministic source-audited 13,911-row TRAIN+fixed-DEV8 label CSV, and `manifest.json` / `SHA256SUMS` / `READY.json` controls. It contains no TEST/eval-final rows or imagery. Its manifest binds the full Sprint-7d base-payload receipt and requires the accepted Sprint-7e native-venv ancestor. Use a separate initially empty Box folder so the base payload and earlier amendments remain immutable. The destination exists only in runtime `BOX_FOLDER_ID`; JWTs, folder IDs, tokens, and URLs never belong in code, docs, logs, bundles, or receipts. The generated hash-pinned standalone runtime bootstrap is the mandatory Judy download/clone path. The full-base `download-runtime`, `verify-runtime`, and `extract-runtime` CLI commands are source/setup-only and are forbidden in H100 pre-cohort allocations because they open and hash held-out base archives.
- Judy and the live V100 host have completely separate filesystems. The Judy site contract must set `H100_V100_CONTROL_PLANE=box-transfer-v1`; never create a dummy Judy path for the V100 runs tree. Smoke and H100 acceptance use Judy-local inputs only. Narrow, content-addressed Box packages carry finalized R2/R3 JSON evidence to Judy, `CUTOVER_READY.json` to the V100 operator, and the external `V100_DIAGNOSTIC_ISOLATION.json` attestation back to Judy. Campaign execution must not depend on a mounted V100/reference filesystem or on V100 being stopped.
- Judy scratch is an explicit site contract: set `H100_SCRATCH_ROOT` to one canonical, existing, non-symlink, writable filesystem root visible on every eligible compute node; never assume Slurm exports `SLURM_TMPDIR`. Each allocation owns only its mode-`0700` `xview3-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT}` child, and campaign allocations require at least 500,000,000,000 free bytes there. Cleanup is guarded to that exact allocation child; requeues reconstruct scratch while checkpoints and cohort state remain under persistent `H100_RUNS_ROOT`. Judy smoke job 540200 failed before GPU discovery or training because `SLURM_TMPDIR` was unset, wrote no readiness marker or canonical campaign state, and did not affect the live V100 diagnostic.
- Judy jobs 541320 and 541333 showed the direct verifier path being killed with exit 137; module invocation is therefore mandatory. `dgx09` and `dgx18` subsequently matched full runtime closure `a4af214a...`, and compute job 541574 built and verified the sealed venv at tree `d1237904...` with receipt `d84d60c8...`. The login-built tree remains diagnostic evidence only. Transfer the corrected Sprint-7f runtime and pass a fresh full smoke before H100 acceptance or training.
- The dynamic control packages are JSON-only, separately allowlisted, verified by manifest plus SHA-256, and published with READY last. A manually copied directory or the control-plane literal alone is not transfer proof. The diagnostic-isolation attestation is a human assertion about namespace and reporting isolation; it is not an archive receipt and authorizes no process-control action.
- Exclude `validation.csv` and all eval-final material, `runs/`, virtual environments/caches, JWTs/tokens, raw LS-SSDD, preprocessing manifests, YOLO/reference payloads, superseded results/checkpoints, and obsolete weights from both forward packages. The reverse content-addressed result bundle contains only core metrics, resolved configs, logs, schema-2 provenance, and best/last checkpoints.
- The V100 fp32 batch-16 gates passed on 2026-07-26, but they are not H100 acceptance. Judy must independently pass the complete suite, six value-sensitive loads, strict-FP32 assertions, both-family batch-16 train/full-scene inference probes, external `SIGUSR1` requeue/resume smoke, corrected evaluation-GT receipt, and a 200-step forecast that still beats the recorded V100 comparison. A failed gate leaves V100 running and triggers a STOP.
- Judy H100 acceptance binds its allocation-private phase allowlist as
  `H100_READY.json:venv.staged_data_view`. Never restore a scratch
  `staged_base_extraction` or expose the full base payload before the frozen
  all-32 cohort. The persistent base-extraction path lives only in the sealed
  native-venv build receipt; readiness carries its pathless identity.
- CI is CPU-only and uses `requirements-ci.txt` — a minimal set, *not* the training environment. CI runs guard tests on tiny fixtures with **no GPU and no model downloads** — so `test_fm_checkpoints_load`'s value-sensitive weight load runs on the GPU boxes, and CI runs only its CPU-offline structural (key-manifest) half. CI triggers on `dev` (the active branch), not only `main`.
- Working scratch goes outside version control; never commit data, checkpoints, or `runs/`.
