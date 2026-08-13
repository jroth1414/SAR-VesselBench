# Class final report

`final_report.tex` follows the class submission structure. Its main body runs
from the `mainbody:start` label at Introduction through the `mainbody:end`
label at Conclusion. That inclusive span must not exceed five rendered pages.
References and appendices begin after explicit `\clearpage` boundaries.

Generate both macro files before typesetting:

```bash
cd ../..
python -m src.analysis.h100_results generate \
  --snapshot results/h100/h100_campaign_snapshot.json \
  --output-dir docs/results/generated
python -m src.analysis.heldout_results \
  --output-dir docs/results/generated
cd docs/class_report
tectonic -X compile --keep-intermediates final_report.tex
```

Use Tectonic 0.17.0. The report inputs `h100_results.tex` (the sanitized
operator status snapshot) and `heldout_results.tex` (the fail-closed
evidence-tree render, which supplies the development matrix, figures, and
held-out slots).
