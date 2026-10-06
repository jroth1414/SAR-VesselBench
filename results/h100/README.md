# H100 result evidence

## Evidence tree (`evidence/`)

`evidence/` is the manuscript's input. It carries the replacement32 cohort:
all 32 cells retrained from initialization on one H100 node after the
BigEarthNet-S2 first-convolution fix (code `06b2e61`, cohort `a8996efd`,
strict IEEE FP32, seed 0), their held-out TEST results, and the once-only
50-scene human-verified evaluation (Slurm job 671477). The earlier August
cohort (code `1a82d508`, cohort `9b1ba03e`) remains in git history.

- `TRAINING_COHORT.json` — the frozen cohort, byte-exact. It binds the
  committed `configs/detector.yaml` by SHA-256 and records, for every cell,
  the completion-marker hash, best-checkpoint hash and epoch, and the
  dev-selected operating threshold.
- `<exp_id>/final_metrics.json` — each cell's completion marker, byte-exact;
  its SHA-256 must equal the cohort binding.
- `<exp_id>/metrics.csv` — each cell's full training curve, or, for the two
  cells whose last dev evaluation ran in a zero-step resume,
  `<exp_id>/terminal_recovery.json`, hash-bound by the marker.
- `<exp_id>/test_metrics.json` — the immutable 16-scene TEST result.
- `<exp_id>/final_verified_metrics.json` — the once-only 50-scene result,
  hash-bound by `FINAL_EVAL_COMPLETE.json`.
- `<exp_id>/runtime_provenance.json` — sanitized copy (private cluster paths
  replaced; `REDACTIONS.json` records each original SHA-256).

Checkpoint bytes stay outside the repository; their SHA-256 bindings are
published so the operator archive can re-verify them.

Build the tree from a delivered run tree, then generate and validate the
paper inputs:

```bash
python scripts/stage_evidence.py --run-root <runs tree> --out <new dir> --logs-out <excerpt dir>
python -m src.analysis.heldout_results --output-dir docs/results/generated
```

The generator fails closed: markers must hash to their cohort bindings, every
metric must be TP/FP/FN-consistent, and each marker's best-dev F1 and epoch
must equal its training-curve maximum (or its bound zero-step recovery
record). Held-out 16-scene TEST macros render only when all 32 immutable
`test_metrics.json` results are present and revalidate against the cohort.
The 50-scene macros render only when all 32 final results bind the cohort,
their TEST result, and their threshold. Both stages are all-or-nothing; until
they validate the report shows dashes, never substitutes.

## Logs (`logs/`)

- `h100_excerpts/<exp_id>.log` — head/tail excerpts of each cell's raw H100
  training log (full-log SHA-256 recorded in each header; site paths
  redacted). This is the only committed log content; operator-status
  transcripts and local scoring-attempt logs are regenerable build
  artifacts and are not committed.

## Operator status snapshot

`h100_campaign_snapshot.json` is the sanitized deadline status record of the
August cohort (13 done / 8 started / 11 pending at capture). It is
historical and does not describe the replacement32 cohort.
`python -m src.analysis.h100_results generate` still renders it, and its
`import-complete` / `import-deadline` machinery remains available for a
fully receipted reverse handback.

Never copy a value from a running cell, backfill an H100 value from another
hardware class, or mix hardware classes in one comparison.
