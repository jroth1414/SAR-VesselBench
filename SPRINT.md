# Sprint 7c — V100 full-fp32 grid amendment

Branch: `sprint-7c-fp32-grid` (Spine review)
Phase: 5 — 32-cell core grid plus fresh R2/R3 references

## Goal

Replace the superseded mixed-precision core campaign with one uniformly
full-fp32 campaign on the locked 8x V100-SXM2-32GB pool. Preserve every
scientific invariant except the explicitly owner-approved precision change.

## Authorized amendment

- Core training precision: shared Lightning `32-true` for all 32 cells.
- Core model-forward inference: the same shared precision for dev, test, and
  final evaluation; the sacred scorer and established decode canvas are
  unchanged.
- Effective batch remains 16 (micro-batch 16, accumulation 1).
- The detector freeze hash is intentionally re-pinned once on this branch.
- The owner explicitly overrode the greater-than-2x and 1,300-GPU-hour STOPs.
- All mixed-precision core results are superseded; no per-cell precision mix.
- R2/R3 are rerun fresh but retain their independent reference recipes.

## Acceptance criteria

1. `configs/detector.yaml` and every core inference caller resolve to
   `32-true`; completion markers validate precision and detector hash.
2. The full CPU test suite and all guard tests pass.
3. All six downloaded checkpoints pass the V100 value-sensitive load gate.
4. Both backbone families pass finite fp32 batch-16 train and inference
   probes without OOM.
5. A clean committed SHA and exact environment/config hashes are recorded in
   a fresh campaign manifest before launch.
6. Every canonical run namespace starts empty; the replacement launcher
   schedules exactly 32 core cells plus R2/R3 and fail-stops on any error.
7. The new 32-row grid must independently pass the 0.02 monotonicity guard
   before Phase 6. The fp32 amendment does not waive that scientific STOP.

## Definition of done

The fresh campaign is launched and health-checked on all reserved V100s;
completion requires 32 recipe-matched core markers, both fresh reference
records, a green monotonicity check, and updated result exports. Final
verified-scene evaluation remains untouched until those conditions hold.
