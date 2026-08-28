# Sprint 10b — BigEarthNet-S2 diagnostic budget amendment

Branch: `sprint-10b-bes2-budget-amendment` (Spine review)
Base: `a62daae80a00a6b34878b774a2ede89fb98425a5` (`dev`)
Phase: 5 diagnostic — TRAIN+fixed-DEV8 only

## Goal

Encode the owner's 2026-08-28 approval to raise the Judy BigEarthNet-S2
diagnostic ceiling from 125 to 190 GPU-hours after the reduced f10/f50 matrix
was forecast from accepted H100 timing evidence.

This sprint never acquires or controls a V100 lease, never mutates the live
V100 diagnostic campaign, and never reads TEST or verified-final data.

## Authorized change

- The Python readiness contract and Judy submitter accept a finite, positive
  forecast no greater than 190 GPU-hours and reject any larger value.
- The schema-2 amendment explicitly records the 190-hour ceiling and binds the
  reviewed Sprint 10a source as its required ancestor.
- A fresh content-addressed Sprint 10b package and fresh empty Box folder
  supersede the already READY-published Sprint 10a transfer.
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

Sprint 10b changes no causal rule. Sprint 10a still classifies replay
reproducibility and reset recovery independently at f10 and f50, never pools
the two fractions, and requires concordant determinate stem-dominant decisions
before reset-based replacement can become conditionally eligible. Evidence
remains single-seed TRAIN/DEV point estimation with no significance,
seed-variance, confidence-interval, or error-bar claim.

## Definition of done

Source completion requires the 190-hour Python and shell boundaries, package
identity, tests, DEVPLAN decision record, and runbook to be reviewed and
merged. Operational completion additionally requires a fresh verified transfer
and the unchanged Sprint 10a Judy audit/probe/full/result sequence.

Sprint 10b cannot approve or implement a replacement. A stem-reset core arm
still requires the predeclared diagnostic result and a new explicit owner
amendment. Any replacement, fresh all-32 campaign, or prospective joint
64-cell final-evaluation registry remains outside this sprint.

## Frozen and out of scope

The scorer, detector config, splits, statistics, historical LS-SSDD split,
existing H100 cohort/results, existing TEST results, verified-data locks, and
all V100 state remain unchanged. BigEarthNet-S1 is recorded as a separate
shared-helper compliance concern but is not modified or tested here.
