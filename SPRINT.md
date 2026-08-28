# Sprint 10d — BigEarthNet-S2 readiness-shape correction

Branch: `sprint-10d-bes2-readiness-shape-fix` (Spine review)
Base: `0458fbedb3a6afd618847e443251aeb4ef009f14` (`dev`)
Phase: 5 diagnostic — TRAIN+fixed-DEV8 only

## Goal

Correct the fail-closed readiness-shape mismatch exposed by the verified Sprint
10c Judy audit. Job 573446 completed in 28:19 on `dgx22` and wrote an
immutable marker whose `strict_fp32` field contained the full launch contract;
the existing downstream validator requires the canonical three direct IEEE
backend fields. No probe or full training was submitted.

This sprint never acquires or controls a V100 lease, never mutates the live
V100 diagnostic campaign, never reads TEST or verified-final data, and changes
no scientific or training behavior.

## Authorized change

- The readiness writer cross-checks the launch backend against canonical H100
  acceptance and writes only that mapping; the bound audit JSON retains the
  complete autocast/process/backend evidence.
- Regression tests cover writer-to-validator shape, backend mismatch, branch
  parity, and Git executable modes for both diagnostic launch scripts.
- Builder, submitter, and compute clone bind Sprint 10d and the reviewed Sprint
  10c ancestor. A fresh package/folder/namespace supersedes Sprint 10c.
- Fractions, variants, model loading, recipe, data view, evidence, decision
  thresholds, strict FP32, requeue behavior, and result packaging are
  unchanged.

## Judy execution contract

The schema-2 amendment contains only the reviewed Git bundle and controls. It
binds the accepted canonical `H100_READY.json`, sealed native venv, checkpoint,
source SHA, and prospective TRAIN+fixed-DEV8 data view. It is transferred
through a fresh Box folder, verified by manifest and SHA-256, and publishes
`READY.json` last. Runtime credentials, folder identifiers, tokens, and URLs
are never recorded.

The Sprint 10c package, job 573446 log, and readiness SHA-256
`b3375624af2f8d1cb60359dc930c1e0efc53d47508313713f6d6eb71812a45ce`
remain immutable diagnostic evidence and cannot satisfy Sprint 10d readiness.
The fresh audit must validate its generated readiness before probe submission.

Audit runs the complete sealed-venv tests, value-sensitive S2 loading, strict
IEEE-FP32 checks, batch-16 forward/backward probes, deterministic input
covariance, activations, initialization hashes, and update-drift prerequisites.
The f50 five-epoch probe pair and fresh full f10/f50 pairs retain the original
50-epoch schedule horizon. Variants execute concurrently within each pair on
two H100s, one process per GPU, without DDP. Persistent diagnostic checkpoints
remain on Judy; the narrow return package contains only validated JSON, the
reproducible Markdown report, and necessary logs.

The conservative site forecast is 190 GPU-hours. Values above 190 remain
blocked, and the mandatory compute STOP remains near 250 hours.

## Scientific contract

Sprint 10d changes no causal rule. Sprint 10a still classifies replay
reproducibility and reset recovery independently at f10 and f50, never pools
the two fractions, and requires concordant determinate stem-dominant decisions
before reset-based replacement can become conditionally eligible. Evidence
remains single-seed TRAIN/DEV point estimation with no significance,
seed-variance, confidence-interval, or error-bar claim.

## Definition of done

Source completion requires review and merge: generated readiness passes its
existing validator, both launch scripts are tracked executable, builder/submit/
compute clone bind Sprint 10d, and the regression tests plus records pass.
Operational completion requires a fresh verified Sprint 10d transfer and the
unchanged Sprint 10a Judy audit/probe/full/result sequence.

Sprint 10d cannot approve or implement a replacement. A stem-reset core arm
still requires the predeclared diagnostic result and a new explicit owner
amendment. Any replacement, fresh all-32 campaign, or prospective joint
64-cell final-evaluation registry remains outside this sprint.

## Frozen and out of scope

The scorer, detector config, splits, statistics, historical LS-SSDD split,
existing H100 cohort/results, existing TEST results, verified-data locks, and
all V100 state remain unchanged. BigEarthNet-S1 is recorded as a separate
shared-helper compliance concern but is not modified or tested here.
