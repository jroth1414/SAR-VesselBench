# Sprint 10c — BigEarthNet-S2 diagnostic clone-ref correction

Branch: `sprint-10c-bes2-clone-ref-fix` (Spine review)
Base: `6467872d184ba9a672b956a9452012060c9a3b11` (`dev`)
Phase: 5 diagnostic — TRAIN+fixed-DEV8 only

## Goal

Correct the pre-submission branch/ref mismatch found after the verified Sprint
10b Judy pull: its bundle exposed Sprint 10b, while the compute script requested
Sprint 10a. No Slurm job or GPU was requested before the mismatch was found.

This sprint never acquires or controls a V100 lease, never mutates the live
V100 diagnostic campaign, never reads TEST or verified-final data, and changes
no scientific or training behavior.

## Authorized change

- The amendment builder, login-node submit guard, and compute-node clone all
  bind the same Sprint 10c branch and reviewed Sprint 10b ancestor.
- A regression test fails if the builder/submit branch and compute clone branch
  differ again.
- A fresh content-addressed Sprint 10c package and fresh empty Box folder
  supersede the immutable Sprint 10b transfer for execution.
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

Sprint 10c changes no causal rule. Sprint 10a still classifies replay
reproducibility and reset recovery independently at f10 and f50, never pools
the two fractions, and requires concordant determinate stem-dominant decisions
before reset-based replacement can become conditionally eligible. Evidence
remains single-seed TRAIN/DEV point estimation with no significance,
seed-variance, confidence-interval, or error-bar claim.

## Definition of done

Source completion is merged: the builder, submitter, and compute clone bind one
branch, and the regression test, DEVPLAN record, and runbook are committed.
Operational completion requires a fresh verified Sprint 10c transfer and the
unchanged Sprint 10a Judy audit/probe/full/result sequence.

Sprint 10c cannot approve or implement a replacement. A stem-reset core arm
still requires the predeclared diagnostic result and a new explicit owner
amendment. Any replacement, fresh all-32 campaign, or prospective joint
64-cell final-evaluation registry remains outside this sprint.

## Frozen and out of scope

The scorer, detector config, splits, statistics, historical LS-SSDD split,
existing H100 cohort/results, existing TEST results, verified-data locks, and
all V100 state remain unchanged. BigEarthNet-S1 is recorded as a separate
shared-helper compliance concern but is not modified or tested here.
