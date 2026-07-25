# Preliminary 10%-label analysis

These artifacts summarize the complete seed-0, 10%-label (`f10`) wave for the
active eight-arm study. They are preliminary point estimates, not the final
label-efficiency result: the 25%, 50%, and 100% cells are still outstanding,
and the once-only 50-scene final evaluation remains untouched.

Every test score below was evaluated at the threshold selected on dev and then
frozen. Test thresholds were not retuned.

| Track | Initialization | Dev F1 | Frozen threshold | Test F1 |
|---|---|---:|---:|---:|
| ViT | Random floor | 0.8554 | 0.7373 | 0.7881 |
| ViT | SatDINO optical | 0.8766 | 0.5835 | 0.8158 |
| ViT | SARMAE SAR | 0.8803 | 0.4023 | 0.8154 |
| ViT | ImageNet-1K | 0.8940 | 0.6245 | 0.8073 |
| CNN | Random floor | 0.7530 | 0.7378 | 0.6559 |
| CNN | BigEarthNet-S2 optical | 0.7678 | 0.8008 | 0.6675 |
| CNN | BigEarthNet-S1 SAR | 0.8337 | 0.8169 | 0.3047 |
| CNN | ImageNet-1K | 0.8966 | 0.1304 | 0.8208 |

## Figures and tables

- [`grid.csv`](grid.csv) is the active eight-arm f10 result table.
- [`label_efficiency.png`](label_efficiency.png) plots the eight available
  test points. It is not yet a label-efficiency curve because only one label
  fraction is complete.
- [`threshold_transfer.png`](threshold_transfer.png) compares best dev F1 with
  test F1 at the frozen dev threshold. BigEarthNet-S1 is the only flagged
  transfer gap.
- [`slice_metrics.csv`](slice_metrics.csv) and
  [`role_deltas.csv`](role_deltas.csv) contain the refreshed active-arm
  diagnostics and within-track contrasts.
- [`chip_gallery.png`](chip_gallery.png) is a ground-truth SAR-chip gallery.
- [`prediction_galleries/`](prediction_galleries/) contains operating-threshold
  overlays for the seven weights-only checkpoints available in the analysis
  package. Green circles are accepted ground-truth vessels and red crosses are
  predictions. The same four curated dev chips are used for every model.

The ImageNet CNN result is included in the quantitative tables, but it has no
prediction gallery because its completed weights-only checkpoint was not in
the package used for rendering.

## Preliminary reading

- The strongest reportable f10 test point is ImageNet CNN at `0.8208`.
- SatDINO and SARMAE are effectively tied on test (`0.8158` and `0.8154`) and
  each improves on the ViT random floor by about `0.028` F1.
- BigEarthNet-S2 improves on the CNN random floor by about `0.012` F1.
- BigEarthNet-S1's reportable `0.3047` test F1 is a threshold-transfer warning,
  not evidence by itself that its detections are intrinsically poor. Its dev
  confidence scale did not transfer to test; the frozen-threshold score remains
  the protocol-compliant result.
- Dark-vessel support is zero on the train-derived dev/test split by design.
  Dark-vessel recall can only be measured during final evaluation. Near-shore
  support is sparse, so those slice values should not be overinterpreted.

## Reproduction

The quantitative summaries can be rebuilt from the committed result exports:

```bash
python -m src.analysis.curves all \
  --runs-root results \
  --out-csv results/summary/grid.csv \
  --out-png results/summary/label_efficiency.png

python -m src.analysis.error_slices \
  --results-root results \
  --out-dir results/summary
```

Prediction galleries are produced by `src/analysis/qualitative.py` from the
uncommitted SAR chips and weights-only checkpoints. Checkpoints, chips, the
10+ GB source archive, and local inference dependencies are intentionally not
versioned.
