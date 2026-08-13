# H100 result evidence

## Evidence tree (`evidence/`)

`evidence/` is the manuscript's input. It carries, from the completed 32-cell
H100 campaign (code `1a82d508`, strict IEEE FP32, seed 0):

- `TRAINING_COHORT.json` — the frozen cohort, byte-exact. It binds the
  committed `configs/detector.yaml` by SHA-256 and records, for every cell,
  the completion-marker hash, best-checkpoint hash and epoch, and the
  dev-selected operating threshold.
- `<exp_id>/final_metrics.json` — each cell's completion marker, byte-exact;
  its SHA-256 must equal the cohort binding.
- `<exp_id>/metrics.csv` — each cell's full training curve.
- `<exp_id>/runtime_provenance.json` — sanitized copy (private cluster paths
  replaced; `REDACTIONS.json` records each original SHA-256).

Checkpoint bytes stay outside the repository; their SHA-256 bindings are
published so the operator archive can re-verify them.

Generate and validate the paper inputs with:

```bash
python -m src.analysis.heldout_results --output-dir docs/results/generated
```

The generator fails closed: markers must hash to their cohort bindings, every
metric must be TP/FP/FN-consistent, and each marker's best-dev F1 and epoch
must equal its training-curve maximum. Held-out 16-scene TEST macros render
only when all 32 immutable `test_metrics.json` results are present and
revalidate against the cohort — all-or-nothing. The sealed 50-scene
human-verified evaluation renders only from a committed `final_verified.csv`.
Until those artifacts exist the report shows dashes, never substitutes.

## Logs (`logs/`)

- `campaign_status.txt` — the sanitized operator status transcript.
- `h100_excerpts/<exp_id>.log` — head/tail excerpts of each training log
  (full-log SHA-256 recorded in each header; site paths redacted).
- `test_scoring_rtx5070ti.log` — the complete local TEST-scoring record from
  2026-08-12, including the two failed protocol attempts, the owner-approved
  Windows fsync deviation, two sealed cross-hardware results
  (`cnnrand-f100-s0` 0.8124, `beS2-f100-s0` 0.7948), and the owner's stop
  directive that moved TEST scoring to the H100 node.

## Operator status snapshot

`h100_campaign_snapshot.json` is the sanitized deadline status record
(13 done / 8 started / 11 pending at capture). It is historical: the campaign
subsequently finished all 32 training cells, which is what the evidence tree
records. `python -m src.analysis.h100_results generate` still renders it, and
its `import-complete` / `import-deadline` machinery remains available for a
fully receipted reverse handback.

Never copy a value from a running cell, backfill an H100 value from another
hardware class, or mix hardware classes in one comparison. Published F1
values are corrected development-selection metrics until the held-out results
land through the fail-closed path above.
