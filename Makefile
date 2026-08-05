# One target per phase entrypoint (DEVPLAN P0.4). Entrypoints for phases that
# are not built yet will fail until their sprint lands — the targets exist so
# every phase is driven the same way from day one.

PYTHON ?= python

.PHONY: env-check test data qa grid references final-eval

# Phase 0 — device + kernel sanity on the current machine (P0.3).
env-check:
	$(PYTHON) scripts/gpu_sanity.py

# Full test suite, guards included.
test:
	$(PYTHON) -m pytest

# Phase 1 — download + chip + splits (P1.1–P1.5).
data:
	$(PYTHON) -m src.data.download_sarfish
	$(PYTHON) -m src.data.download_aux
	$(PYTHON) -m src.data.chipper
	$(PYTHON) -m src.data.splits

# QA galleries for human eyeballs (P1.5 / P3.5).
qa:
	$(PYTHON) -m src.analysis.qualitative --qa

# Phase 4/5 — the label-fraction grid (arms x fractions x seeds; Section 12).
grid:
	$(PYTHON) scripts/run_grid_queue.py

# Phase 4 — the non-optional external references (R2 yolo26-f100, R3 locateanything-zs).
references:
	@test -n "$(REFERENCE_ACTION)" || { echo "Set REFERENCE_ACTION=manifest|r2|r3 and REFERENCE_ARGS='...'" >&2; exit 2; }
	$(PYTHON) -m scripts.run_corrected_references $(REFERENCE_ACTION) $(REFERENCE_ARGS)

# Phase 6 — the ONCE-ONLY verified-scene eval (ground rule 4). The
# --i-am-sure flag is deliberately NOT baked in: require CONFIRM=1.
final-eval:
ifndef CONFIRM
	$(error final-eval touches the once-only verified scenes; run "make final-eval CONFIRM=1" when you mean it)
endif
	$(PYTHON) -m src.eval.final_eval --i-am-sure
