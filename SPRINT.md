# Sprint 10i — Arm-6 production fallback

Branch: `sprint-10i-bes2-production-fallback`
Base: integration `dev` at `d409a4a`.
Task: P10.34 owner amendment; P10.35 exact post-stem transfer; P10.36 validation.
Review tier: Spine. Owner review and merge are required before runtime packaging.

The approved Arm-6 initialization retains the ordinary seeded three-channel
ConvNeXt-V2-Base first-convolution weight and loads every other compatible
BigEarthNet-S2 backbone tensor. Production and historical diagnostic reset use
one semantic definition. Historical replay retains the original conversion.

Acceptance requires exact single-weight exclusion, complete value-sensitive
post-stem loading, seed-0 stem/head determinism and unchanged RNG, parameter
parity and stride-4/128×128 geometry, the unchanged strict-FP32 training recipe,
no changes to other initializations, diagnostic/core isolation, all frozen
guards unchanged and passing, and the full offline CPU suite. GPU acceptance
must cover S2 value-sensitive transfer and batch-16 forward/backward on Judy.
No S1 checkpoint is loaded or tested by this sprint.

Do not touch frozen scorer/splits/statistics/detector configuration, guards,
original cohort/TEST evidence, V100 processes, credentials, or final data.
No replacement cell starts before separate fresh smoke and acceptance pass.

The complete task remains active beyond this narrow review: a separate fresh
code/control-only schema-2 runtime package, user-mediated Box/Judy transfer,
new 32-cell H100 cohort from initialization, frozen cohort then bounded TEST
and sanity gates, and preparation (never execution) of a prospective 64-cell
final-evaluation orchestrator. Each later implementation sprint starts from
reviewed `dev`; do not absorb unmerged diagnostic branches without review.

Validation complete: 528 passed/8 skipped in the full offline CPU suite;
one separate official S2-only value-sensitive test passed. Review evidence and
prepared (unexecuted) Judy GPU commands are in
`docs/BES2_PRODUCTION_FALLBACK_REVIEW.md`. Owner review/merge is pending; no
replacement package, job, readiness, cohort, TEST or final result is claimed.
