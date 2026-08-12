# H100 result evidence

`h100_campaign_snapshot.json` is the single public input to the paper result
generator. The committed deadline snapshot is an operator-status record, not a
scientific result bundle. Its `DONE`, `STARTED`, and `PENDING` values describe
campaign progress only. In particular, its status-only cells contain no loss,
development F1, threshold, precision, or recall values.

Generate and validate the current paper inputs with:

```bash
python -m src.analysis.h100_results generate \
  --snapshot results/h100/h100_campaign_snapshot.json \
  --arms-config configs/arms.yaml \
  --output-dir docs/results/generated
```

The command also writes the sanitized, metric-free transcript at
`results/h100/logs/campaign_status.txt` and binds its SHA-256 in the generated
manifest.

A deadline cohort may replace the status-only snapshot only through
`import-deadline`. The command requires a status-only 32-cell ledger. It opens
and hashes each completed cell marker, runtime record, and best and last
checkpoint. The importer reports a fraction after all eight cells pass these
checks.

```bash
python -m src.analysis.h100_results import-deadline \
  --extracted-root /path/to/verified/reverse-package \
  --status-snapshot results/h100/h100_campaign_snapshot.json \
  --arms-config configs/arms.yaml \
  --frozen-root . \
  --source-sha256 SHA256_OF_REVERSE_PACKAGE \
  --captured-utc 2026-08-13T00:00:00+00:00 \
  --output results/h100/h100_campaign_snapshot.json
```

A `complete` snapshot may replace the deadline snapshot only through
`import-complete`. The importer consumes a verified H100 result package. It
checks the exact 32-cell matrix, strict IEEE FP32 H100 identity, frozen hashes,
marker and runtime bindings, and the best and last checkpoint bytes. The
generator then admits development-selection values to tables and curves.

```bash
python -m src.analysis.h100_results import-complete \
  --extracted-root /path/to/verified/reverse-package \
  --arms-config configs/arms.yaml \
  --frozen-root . \
  --source-sha256 SHA256_OF_REVERSE_PACKAGE \
  --captured-utc 2026-08-13T00:00:00+00:00 \
  --output results/h100/h100_campaign_snapshot.json
```

If R2 and R3 are both complete, `--reference-receipt` may name a separately
closed receipt for that exact pair. The importer rejects a singleton,
non-`DONE` record, code-revision mismatch, duplicate experiment identity, or
unbound marker/runtime hashes. Reference results remain separate from the
32-cell core curves.

The committed public log contains campaign progress. It contains no
acceptance-test evidence. A future submission may include acceptance artifacts
after the importer verifies a result handback.

Never copy a value from a running cell, backfill a missing H100 value with a
V100 diagnostic, or combine hardware classes. The 50-scene human-verified set
remains sealed; public F1 values, when available, are corrected
development-selection metrics.
