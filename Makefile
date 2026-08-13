PYTHON ?= python
TECTONIC ?= tectonic
SNAPSHOT ?= results/h100/h100_campaign_snapshot.json
GENERATED ?= docs/results/generated
TRAIN_ARGS ?=

.PHONY: test guards data-check train-cell results class-paper submission final-eval

test:
	$(PYTHON) -m pytest -q

guards:
	$(PYTHON) -m pytest -q \
		tests/test_split_disjoint.py \
		tests/test_splits_immutable.py \
		tests/test_lsssdd_split_immutable.py \
		tests/test_stats_immutable.py \
		tests/test_detector_immutable.py \
		tests/test_backbone_parity.py \
		tests/test_fm_checkpoints_load.py \
		tests/test_scorer_immutable.py

# Validate the committed split/statistics contract without acquiring data.
data-check:
	$(PYTHON) -m pytest -q \
		tests/test_split_disjoint.py tests/test_splits.py \
		tests/test_splits_immutable.py tests/test_stats_immutable.py

# Example: make train-cell TRAIN_ARGS='--init sarmae_b --label_frac 0.10 --seed 0'
train-cell:
	@test -n "$(TRAIN_ARGS)" || { echo "Set TRAIN_ARGS to one core-cell recipe" >&2; exit 2; }
	$(PYTHON) -m src.runtime.train $(TRAIN_ARGS)

results:
	$(PYTHON) -m src.analysis.h100_results generate \
		--snapshot $(SNAPSHOT) --output-dir $(GENERATED)

class-paper: results
	cd docs/class_report && $(TECTONIC) -X compile --keep-intermediates final_report.tex
	$(PYTHON) tools/check_report.py docs/class_report/final_report.pdf \
		--aux docs/class_report/final_report.aux

submission: class-paper
	$(PYTHON) tools/build_submission.py

# The final evaluator has a once-only confirmation gate. It remains outside
# every paper and submission target.
final-eval:
ifndef CONFIRM
	$(error final-eval touches the sealed human-verified scenes; pass CONFIRM=1)
endif
	$(PYTHON) -m src.eval.final_eval --i-am-sure
