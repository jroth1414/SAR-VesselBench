# AIPR 2026 LNCS manuscript

Paper source for the label-efficiency study, in the official Springer LNCS
class (vendored unmodified in vendor/). This build reports the completed
32-cell V100 campaign's best-development values as its results, with the
evaluation-contract audit disclosed in Sect. 4.2 and the development-selection
scope carried through the abstract, results, and limitations. The held-out
test matrix and the once-only 50-scene human-verified evaluation remain
sealed and are deferred to follow-on work.

## Build

All result numbers, tables, and figures are generated from the campaign
archive; nothing is hand-copied. From this directory:

    python tools/make_results.py --runs-root <extracted-archive>/runs
    latexmk -pdf -output-directory=build paper.tex

`make_results.py` cross-checks every extracted value against the audited
development-score record and re-verifies every comparative claim made in the
prose (leader pattern, contrast signs, floor crossings, monotonicity
declines) before writing `generated/`. It fails closed on any mismatch.

On the Windows dev box, bibtex needs the vendored style and bib on its search
path, and the build keeps local copies in build/ as a fallback:

    BSTINPUTS="<this-dir>/vendor;" BIBINPUTS="<this-dir>;" \
      latexmk -pdf -output-directory=build paper.tex

The campaign archive is `xview3-s7f-diagnostic-isolation.zip` (repo root on
the dev box); extract the inner zip and point `--runs-root` at its `runs/`
directory. `generated/paper_data.json` records every value the build
consumed.

Springer LNCS instructions:
https://link.springer.com/series/558/information-for-authors-and-editors

Official template archive SHA-256:
7cc8efaa4f6e7ea8d17069c37a192c6023170f1e60f59509f3bb00591dcaf5de
