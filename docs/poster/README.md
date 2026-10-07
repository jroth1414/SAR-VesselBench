# AIPR 2026 poster

One 48 × 36 in landscape poster in two formats built from one source of
figures and numbers:

- `poster.pdf`: tikzposter source `poster.tex` (TeX Gyre Heros, vector figures).
- `poster.pptx`: editable PowerPoint built by `build_pptx.py` (Arial; text,
  the detector pipeline and the stat tiles are native shapes).

Every number on the poster comes from `generated/poster_numbers.json`, which
`src.analysis.poster_figures` derives from `results/h100/evidence` through the
fail-closed `heldout_results` validator. `generated/poster_macros.tex` carries
the same values as `\PN...` macros for the PDF. The 95% intervals come from a
paired scene bootstrap (`src.analysis.scene_bootstrap`, 10,000 resamples,
seed 0) over the per-scene counts of the 16 test and 50 verified scenes; they
cover scene sampling, not training seeds.

The near-shore figures (`poster_recall_by_distance`, `poster_detection_examples`)
come from `src.analysis.poster_coastal` and need three local xView3 inputs that
are not in the repository. Each is checked before use:

| Input | Check |
|---|---|
| `train.csv` (xView3 training labels) | SHA-256 must equal the binding in `EVAL_GROUND_TRUTH_VALIDATED.json` |
| `validation.csv` (labels of the 50 verified scenes) | re-scoring every cell's stored predictions must reproduce its TP/FP/FN |
| `<scene>.tar.gz` validation archives | the two example scenes' `VH_dB.tif` is extracted once into a cache |

Build from the repository root:

```bash
git archive 481200e results/h100/evidence | tar -x -C <tmp>   # August cohort, for the rerun band
python -m src.analysis.poster_figures --rerun-reference <tmp>/results/h100/evidence \
  --train-labels <xview3>/train.csv --verified-labels <xview3>/validation.csv \
  --verified-imagery <xview3>/validation --imagery-cache <cache>
cd docs/poster && tectonic -X compile poster.tex && cd ../..
python docs/poster/build_pptx.py
python docs/poster/check_poster.py
```

Without the three near-shore inputs the generator still writes the other two
figures and their numbers, but the poster needs all four.

`check_poster.py` fails if either build quotes a three-decimal number that is
not generated, if the two builds quote different numbers, or if the PDF is
not one 48 × 36 in page. The layout follows prior AIPR posters: centered
section headings over a thin rule, a full-width answer strip, and a footer
bar with the venue and contact.
