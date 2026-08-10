# AIPR 2026 LNCS manuscript

This directory contains the paper source for the 2026 Applied Imagery and
Pattern Recognition Workshop. The manuscript uses the official Springer LNCS
class and bibliography style copied without modification into vendor/.

The default paper.tex build is an internal results-audit draft. It records all
32 completed V100 best-development values and the completed 34-run execution
campaign for provenance. Those values were produced before the corrected
Sprint 7f evaluation contract and are not reportable evidence. They must never
enter the abstract, finding statements, conclusion, or a submitted artifact.

The paper_submission.tex wrapper is fail closed. It refuses to compile unless
generated/h100_result_macros.tex declares the exact provenance string
corrected-h100-sprint7f and supplies every reviewed finding and disclosure
macro. Final tables and arithmetic should be generated from the machine-readable
32-cell H100 campaign manifest; do not hand-copy numbers into prose.

## Build

From this directory, use the pinned Tectonic executable available on the
development host:

    export LD_LIBRARY_PATH=/home/johnroth/miniconda3/envs/qac-core-env/lib:/home/johnroth/miniconda3/envs/giga/lib
    /home/johnroth/miniconda3/pkgs/tectonic-0.16.9-ha39f199_0/bin/tectonic \
      --keep-logs --keep-intermediates --outdir build paper.tex

The submission guard can be tested with the same command and
paper_submission.tex. Until the corrected results are installed, failure is
the expected outcome.

## Required cutover before submission

1. Finish and validate all 32 corrected H100 core cells under strict IEEE FP32.
2. Freeze each best checkpoint and its checkpoint-bound development threshold.
3. Run the held-out 16-scene test matrix exactly once and review the declared
   monotonicity diagnostics without tuning.
4. Unlock and run the 50-scene human-verified evaluation once.
5. Generate result macros, tables, curves, registered qualitative panels, and
   their SHA-256 provenance from the reviewed result bundle.
6. Confirm the target label-budget rule. The current draft uses only discrete
   tested scene-count crossings; interpolation is intentionally prohibited
   until an owner-approved target and interpolation rule exist.
7. Replace acknowledgements and disclosure placeholders with author-approved
   text; complete employer/public-release review and select the appropriate
   Springer publishing agreement.
8. Confirm AIPR's final page limit and author metadata against the current call
   before producing the archival package.

Workshop author information: https://www.aipr-workshop.org/author-info

Springer LNCS instructions:
https://link.springer.com/series/558/information-for-authors-and-editors

Official template archive SHA-256:
7cc8efaa4f6e7ea8d17069c37a192c6023170f1e60f59509f3bb00591dcaf5de
