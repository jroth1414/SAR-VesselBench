# AIPR 2026 poster

One 48 × 36 in landscape poster in two formats built from one source of
figures and numbers:

- `poster.pdf`: tikzposter source `poster.tex` (Roboto, vector figures).
- `poster.pptx`: editable PowerPoint built by `build_pptx.py` (Arial; text,
  the detector pipeline and the protocol steps are native shapes).

Every number on the poster comes from `generated/poster_numbers.json`, which
`src.analysis.poster_figures` derives from `results/h100/evidence` through the
fail-closed `heldout_results` validator. `generated/poster_macros.tex` carries
the same values as `\PN...` macros for the PDF.

Build from the repository root:

```bash
git archive 481200e results/h100/evidence | tar -x -C <tmp>   # August cohort, for the rerun band
python -m src.analysis.poster_figures --rerun-reference <tmp>/results/h100/evidence
cd docs/poster && tectonic -X compile poster.tex && cd ../..
python docs/poster/build_pptx.py
python docs/poster/check_poster.py
```

`check_poster.py` fails if either build quotes a three-decimal number that is
not generated, if the two builds quote different numbers, or if the PDF is
not one 48 × 36 in page. `assets/` holds the two QR codes for the PowerPoint
(the PDF draws its own with the `qrcode` package).
