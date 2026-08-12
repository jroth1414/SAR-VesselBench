# Class final report

`final_report.tex` follows the class submission structure. Its main body runs
from the `mainbody:start` label at Introduction through the `mainbody:end`
label at Conclusion. That inclusive span must not exceed five rendered pages.
References and appendices begin after explicit `\\clearpage` boundaries.

Generate H100 artifacts before typesetting:

```bash
cd ../..
python -m src.analysis.h100_results generate \\
  --snapshot results/h100/h100_campaign_snapshot.json \\
  --output-dir docs/results/generated
cd docs/class_report
tectonic -X compile --keep-intermediates final_report.tex
```

Use Tectonic 0.17.0. The build resolves both page labels. Validate the span and
output filename from the repository root:

```bash
python tools/check_report.py docs/class_report/final_report.pdf \\
  --aux docs/class_report/final_report.aux
```
