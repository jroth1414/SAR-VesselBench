# AIPR 2026 poster (v2: more images)

One 48 × 36 in landscape poster in two formats built from one source of
figures and numbers, plus a reference handout:

- `poster.pdf`: tikzposter source `poster.tex` (TeX Gyre Heros, vector figures).
- `poster.pptx`: editable PowerPoint built by `build_pptx.py` (Arial; text,
  the detector pipeline and the arms table are native shapes).
- `references.pdf`: letter-size handout from `references.tex`, the same list
  as the poster's two-column References block with full titles and page
  ranges.

Version 2 (branch `poster-v2`) keeps the v1 layout and replaces text with
images: a dark-vessel radar crop (Motivation), a scene map (Data), a nested
budget waffle (arms block), a win/tie/loss scoreboard (Findings), detector
heatmaps ("Inside the detector"), and the YOLO26 reference on the
label-efficiency chart. Limits and open questions are one line each.

Every number on the poster comes from `generated/poster_numbers.json`, which
`src.analysis.poster_figures` derives from `results/h100/evidence` through the
fail-closed `heldout_results` validator. `generated/poster_macros.tex` carries
the same values as `\PN...` macros for the PDF. The 95% intervals come from a
paired scene bootstrap (`src.analysis.scene_bootstrap`, 10,000 resamples,
seed 0) over the per-scene counts of the 16 test and 50 verified scenes; they
cover scene sampling, not training seeds. The reference detectors come from
`results/h100/references` (copied from the replacement32 delivery); each file
must match its line in that folder's `SHA256SUMS`.

Several figures need local inputs that are not in the repository. Each is
checked before use:

| Figure | Input | Check |
|---|---|---|
| `poster_recall_by_distance` | `train.csv` (xView3 training labels) | SHA-256 must equal the binding in `EVAL_GROUND_TRUTH_VALIDATED.json` |
| `poster_recall_by_distance`, `poster_detection_examples`, `poster_detector_heatmaps` | `validation.csv` (labels of the 50 verified scenes) | re-scoring every cell's stored predictions must reproduce its TP/FP/FN |
| `poster_detection_examples`, `poster_dark_vessels`, `poster_detector_heatmaps` | `<scene>.tar.gz` validation archives | the example scenes' `VH_dB.tif` (and `VV_dB.tif` for heatmaps) are extracted once into a cache |
| `poster_detector_heatmaps` | run tree with `<exp_id>/checkpoints/best.ckpt` | the checkpoint's SHA-256 must equal the one the evidence binds |
| `poster_scene_map` | Natural Earth `ne_50m_land.geojson` (public domain) | SHA-256 `e874b27a…826b` |
| `poster_scene_context` | scene directory `835f7629c3a3a9abt` with `VH_dB.tif` | none; the figure is descriptive only |

`poster_coastal`, `poster_extras`, `poster_heatmaps` and `poster_scene` draw
these figures. The heatmaps repeat `infer_scene`'s 512 px / stride-384 tiling
over each window, so they show the confidences the scorer saw.

Build from the repository root:

```bash
git archive 481200e results/h100/evidence | tar -x -C <tmp>   # August cohort, for the rerun band
python -m src.analysis.poster_figures --rerun-reference <tmp>/results/h100/evidence \
  --train-labels <xview3>/train.csv --verified-labels <xview3>/validation.csv \
  --verified-imagery <xview3>/validation --imagery-cache <cache> \
  --context-scene <xview3>/GRD/835f7629c3a3a9abt \
  --land-geojson <basemap>/ne_50m_land.geojson --runs-root <replacement32 run tree>
cd docs/poster && tectonic -X compile poster.tex && tectonic -X compile references.tex && cd ../..
python docs/poster/build_pptx.py
python docs/poster/check_poster.py
```

Without the local inputs the generator still writes the result charts, the
scoreboard and the waffle, but the poster needs every figure.

`check_poster.py` fails if either build quotes a three-decimal number that is
not generated, if the two builds quote different numbers, or if the PDF is
not one 48 × 36 in page. Figure text therefore uses at most two decimals. The
layout follows prior AIPR posters: centered section headings over a thin
rule, a full-width answer strip, and a footer bar with the venue and contact.
