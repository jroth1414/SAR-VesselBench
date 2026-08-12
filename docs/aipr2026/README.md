# Supplemental AIPR manuscript

This LNCS manuscript is a work-in-progress research companion to the class
report. It consumes the same corrected H100 snapshot and contains no V100
result, held-out test score, or human-verified evaluation claim.

From the repository root:

```bash
python -m src.analysis.h100_results generate \
  --snapshot results/h100/h100_campaign_snapshot.json \
  --output-dir docs/results/generated
cd docs/aipr2026
tectonic -X compile --keep-intermediates paper.tex
```

Use Tectonic 0.17.0.
The generator emits a deadline profile until all 32 H100 cells finish. It
admits a label-fraction comparison only when all eight arms at that fraction
have verified completion markers. Regenerating after the final reverse export
updates both manuscripts from one source.

The vendored LNCS class and bibliography style remain under Springer's terms.
