# AGENTS.md

Instructions for any coding agent (Claude Code, Codex, Cursor, sub-agents) working in this repository.

**This file is the short list of non-negotiables. `DEVPLAN.md` is the single source of truth** for the full design, phases, and rationale. When this file says "see DEVPLAN §X," read it before acting. If this file and DEVPLAN ever conflict, DEVPLAN wins and you should flag the conflict.

---

## What this project is (one paragraph)

A controlled label-efficiency comparison of **downloaded pretrained backbones** for dark-vessel detection in Sentinel-1 SAR (xView3): random floor, optical foundation model (SatDINO), SAR foundation model (SARMAE), and labeled SAR transfer. **Four study arms, all ViT-B/16, all fine-tuned identically — the downloaded initialization is the only variable.** We pretrain nothing ourselves (this removes compute and novel-code risk). The deliverable is a label-efficiency curve; the optical-FM-vs-SAR-FM contrast is the headline. Validity depends on the arms being truly comparable, so most rules below exist to protect that comparability.

## The prime directive

**Novelty lives in the experimental comparison, not in the mechanics.** Do not invent methods. For any component with a cited reference (DEVPLAN §1a), implement that reference's published recipe. If a reference is ambiguous, unavailable, or two conflict — **STOP and ask a human.** A silently-wrong method (a hand-rolled patch-embed hack, an ad-hoc reconstruction target) produces plausible numbers with no error and corrupts the whole study. (DEVPLAN ground rule 14.)

## Hard rules (do not violate)

1. **Do not invent methods.** Use the cited reference implementations in DEVPLAN §1a:
   - MAE skeleton → `facebookresearch/mae`
   - Arm-3 weights (SAR FM) → **SARMAE** (`MiliLab/SARMAE`, file `SARMAE_vitb_checkpoint-last`; Liu et al. CVPR 2026): downloaded ViT-B, encoder only, CC BY-NC. We do NOT pretrain it.
   - Backbone + any channel change → `timm` `vit_base_patch16_224`, `in_chans=` (Repeat-with-rescaling). Never hand-roll patch-embed surgery.
   - Arm-2 weights (optical FM) → SatDINO ViT-B/16 fMoW-RGB (`strakajk/satdino-vit_base-16`, `trust_remote_code=True`; Apache-2.0). DINO, not MAE — differs from SARMAE in method as well as domain (documented caveat). SatDINO's default forward returns CLS; the harness must request all tokens and use patch tokens only.
   - Head/decode → CenterNet / TRANSAR (heatmap, peak + distance-NMS).

2. **Pretraining is the only variable across arms 1–4.** Same ViT-B/16, same head, same optimizer/schedule/seeds, same fixed 3-channel input `[VH, VV, VH−VV]`. Never tune something per-arm. (Arm 5 / challenge is exempt and reported separately.)

3. **The scorer is sacred.** `src/eval/scorer.py` is written first (sprint-2), then frozen. Never modify it after it's locked. Every reported number flows through it.

4. **Scene-level splits only.** No `scene_id` may appear in more than one split. No dev/test/eval scene may appear in any training or pretraining corpus. The ~50 human-verified scenes are touched exactly once, at final eval.

5. **One framework: PyTorch Lightning.** All training entrypoints share one `LightningModule` pattern, one `LightningDataModule`, one `Trainer` config. This is part of the fairness guarantee — do not write a bespoke loop for one arm.

6. **Do-not-touch after their sprint** (changing any of these requires a human STOP): `src/eval/scorer.py`, `data/splits.json`, `configs/harness.yaml`, the verified-eval lockfile.

## The four guard tests (CI enforces these — see `.github/workflows/ci.yml`)

A PR **cannot merge** if any of these fail. They are not optional and they encode study validity:

- `test_split_disjoint` — no scene in two splits (anti-leakage).
- `test_backbone_parity` — all four arms = identical ViT-B param count (anti-architecture-drift).
- `test_fm_checkpoints_load` — SatDINO and SARMAE both load into ViT-B with expected keys, exercising SatDINO's `trust_remote_code` path and patch-token extraction (anti-silent-random-weights).
- `test_scorer_immutable` — `scorer.py` hash matches the pinned value (anti-scorer-drift).

If you believe a guard test must change, that is itself a STOP — surface it, don't edit the test to pass.

## When to STOP and ask a human (mandatory, not optional)

Stopping is cheap and expected. It is not a failure. Halt and ask when:

1. A cited reference is unavailable/ambiguous, or two conflict.
2. An acceptance criterion fails and the fix isn't obvious in one attempt.
3. A result violates a sanity check (non-monotone label-efficiency curve beyond seed noise; an arm scoring implausibly high → suspect leakage).
4. A change would touch a do-not-touch artifact after it's locked.
5. You're tempted to deviate from a cited method "because it seems better."
6. A step's measured compute/time exceeds its budget by more than ~2×.

Do not optimize to seem autonomous. Asking is the job.

## Workflow

- **Branch per sprint** (see DEVPLAN §1b for the list and review tiers). No direct commits to `main`.
- **PR per sprint**, reviewed and merged by the human. Keep diffs small enough to read line-by-line; if a sprint's diff grows past a few hundred lines, split it.
- **Branch ordering:** no model-code sprint opens until `sprint-2-scorer` is merged; no chip-training sprint until `sprint-1-data` is merged.
- **Commits:** small, frequent, each referencing the task ID (e.g. `P1.3: scene-level split builder`).
- **Sub-agents:** parallelize only independent lanes (data, scorer, pretraining, references). Anything touching the shared harness is serial under one owner — parallel edits to `lit_modules.py`/`finetune.py` silently diverge and break arm comparability.

## Git authorship — strict

- Commits are authored by the **repository owner**. Set `git config user.name` / `user.email` to the owner's identity.
- **Do NOT add** `Co-Authored-By:` trailers, "Generated with …", "🤖", or any AI/agent attribution — not in commit messages, commit bodies, PR titles, or PR descriptions.
- You are a tool, not a listed author.

## Environment notes

- Two GPU machines, different builds: RTX 5070 Ti (Blackwell sm_120, PyTorch nightly / cu12x) and an 8×V100 node (Volta sm_70, cu12x, **fp16 only — no bf16, no FlashAttention, no bitsandbytes**). See `locks/` for the two lockfiles and DEVPLAN Appendix C for the Volta constraints.
- CI is CPU-only and uses `requirements-ci.txt` — a minimal set, *not* the training environment. Guard tests must run on tiny fixtures, no GPU or model downloads.
- Working scratch goes outside version control; never commit data, checkpoints, or `runs/`.
