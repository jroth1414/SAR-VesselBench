# Sprint 10a — BigEarthNet-S2 root-cause diagnostic

Branch: `sprint-10a-bes2-root-cause` (Spine review)
Base: `322dea060a37ec793f1df3d49a7513dfd90b324f` (`dev`)
Phase: 5 diagnostic — TRAIN+fixed-DEV8 only

## Goal

Determine whether Arm 6's BigEarthNet-S2 deficit is primarily caused by the
current 10-to-3-channel stem conversion. Run one fresh, seed-0, full-`f100`
pair on Judy H100s: an exact production replay and an exact reset of only
`stem.0.weight`.

This sprint never acquires or controls a V100 lease, never mutates the live
V100 diagnostic campaign, and never reads TEST or verified-final data.

## Exact intervention

- `current_replay` invokes the unchanged production `bigearthnet_s2` loader.
- `first_conv_reset` constructs a seeded three-channel target and loads every
  BigEarthNet-S2 backbone tensor except `stem.0.weight`.
- The reset variant loads stem bias, stem normalization, and all post-stem
  tensors, asserts the missing set is exactly `{stem.0.weight}`, and proves the
  retained convolution is byte-identical to a fresh seed-0 target.
- Detector head initialization, parameter count, adapter geometry, data order,
  schedule, optimizer, layer decay, precision, and scoring are paired.
- The reset name is diagnostic-only; it is absent from core arm manifests and
  production initialization choices.

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
Five-epoch probes retain the 50-epoch schedule horizon. Fresh full runs execute
concurrently on two H100s, one process per GPU, without DDP. Persistent
diagnostic checkpoints remain on Judy; the narrow return package contains only
validated JSON, the reproducible Markdown report, and necessary logs.

The expected forecast is 103 GPU-hours. A forecast above 125 hours requires
owner approval before submission; the mandatory compute STOP applies above
approximately 250 hours.

## Cause decision

Replay must be within 0.02 of frozen S2 DEV F1 `0.8635917566`; otherwise the
result is indeterminate. If replay does not trail frozen random DEV F1
`0.8919449902`, the deficit did not reproduce. Otherwise:

`recovery = (F1_reset - F1_replay) / (0.8919449902 - F1_replay)`

- `recovery >= 0.50`: stem-dominant.
- `recovery <= 0.20`: reset insufficient/post-stem primary.
- `0.20 < recovery < 0.50`: mixed mechanism.
- Negative recovery: stem conversion is not the performance cause, while the
  compliance concern remains.

The evidence includes DEV-only precision/recall, exact candidate-threshold
curves, score distributions, trajectories, activation/drift evidence, and
leave-one-DEV-scene-out sensitivity. It makes no significance, seed-variance,
confidence-interval, or error-bar claim.

## Definition of done

Source work is complete when the diagnostic loader, evidence contracts, Judy
controller, strict-FP32 diagnostic gate, Slurm entrypoints, content-addressed
forward/return packages, tests, and runbook are reviewed and merged.
Operational completion additionally requires Judy audit, paired probes, paired
full runs, verified return transfer, and an immutable causal classification.

Sprint 10a cannot approve or implement a replacement. A stem-reset core arm
requires a stem-dominant result plus a new explicit owner amendment. Mixed or
reset-insufficient findings require an approved downloaded, licensed,
native-three-channel optical-RS ConvNeXt-V2-Base checkpoint. Any replacement
and fresh all-32 campaign belongs to a later reviewed sprint. A prospective
joint 64-cell final-evaluation registry belongs to a still later sprint and is
legal only if both exact-32 cohorts and TEST grids freeze before the first and
only final-data access.

## Frozen and out of scope

The scorer, detector config, splits, statistics, historical LS-SSDD split,
existing H100 cohort/results, existing TEST results, verified-data locks, and
all V100 state remain unchanged. BigEarthNet-S1 is recorded as a separate
shared-helper compliance concern but is not modified or tested here.
