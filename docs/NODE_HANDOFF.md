# Handoff to the V100-node agent

You are the agent on the 8×V100 node (Volta sm_70, 32 GB each, fp16-only).
This drive carries the project data; the repo of record is
`github.com/jroth1414/JHU-xView3`, branch **`dev`**. Read `AGENTS.md` and the
DEVPLAN cold-start runbook FIRST — they bind you exactly as they bind the
dev-box agent. This file only adds the node-specific facts.

## State when the drive left the dev box

- Phases 0–3 DONE and tagged (`phase-1-done`, `phase-2-done`, `phase-3-done`).
- All frozen artifacts committed + sha256-pinned: scorer, splits.json (150
  scenes: 111/23/16 + 50 eval_final), stats.json, lsssdd_split.json,
  detector.yaml. **Do not regenerate ANY of them. A pin mismatch is a STOP.**
- P3.6 gate passed: every downloaded arm beats its floor; SAR > optical in
  both tracks.
- Option B is decided (human): plan-literal epochs, early stopping patience
  4 REAL dev evals (patience×cadence fix — do not "simplify" it), 50-epoch
  safety ceiling.
- Some grid cells are already finished — their `runs/<exp>/final_metrics.json`
  are on this drive. The queue skips them. Never rerun a finished cell.

## Your job

Run the remaining core-grid cells 8-wide with
`python scripts/run_grid_node.py --gpus 0 1 2 3 4 5 6 7`
(dependency-aware, resumable; pretrainings first if not already done).
Afterwards: seed reruns (DEVPLAN §12; ask the human before starting them).

## Node bring-up (in order, no skipping)

1. Clone the repo (branch dev) onto NODE-LOCAL disk. Python 3.11 venv,
   `pip install -r locks/env-v100node.txt --extra-index-url
   https://download.pytorch.org/whl/cu126`, then `pip install -e .`.
   NEVER `pip install torch` bare (cu13 dropped Volta).
2. `python scripts/gpu_sanity.py` — must print sm_70, fp16 matmul ok, and a
   NON-flash SDPA backend. Paste the output into the README via PR.
   Re-freeze `locks/env-v100node.txt` (keep its header) in the same PR.
3. **COPY the data off this drive onto node-local storage** (~455 GB):
   `data/chips/`, `data/raw/xview3/GRD/` (dev+test scenes), 
   `data/raw/xview3/labels/`, `data/raw/lsssdd/`, `data/weights/`, `runs/`.
   Do NOT train against the external drive — 8 parallel dataloaders over USB
   will starve the GPUs. eval_final rasters are deliberately absent.
4. `pytest tests/ -q` — ALL green (75+), including the value-sensitive FM
   load half. A red guard is a STOP: surface it, do not patch it.
5. Launch the node queue (step above). Log lives in `runs/logs/node/`.

## Known gotchas you would otherwise rediscover

- Dev evals read whole scene rasters into RAM (~5 GB per scene per worker) —
  `infer_scene.py` is written for that; do not "optimize" it back to
  windowed reads (100× slower on striped GeoTIFFs).
- `data/*.json` freeze pins assume LF bytes; the repo's `.gitattributes`
  handles it — do not re-encode files.
- ConvNeXt at recipe batch 16 fits in 32 GB — the `--micro-batch` flag is a
  dev-card-only adaptation; do not pass it on the node.
- The `source` column in label CSVs is lowercase; `infer_scene.py`
  normalizes it for the frozen scorer. Dark-vessel GT exists ONLY in
  eval_final scenes — dark recall is expected to be 0-support on dev/test.

## Reporting back

After each wave, run `python -m src.analysis.curves` and commit nothing —
push `runs/summary/grid.csv` + the figure to the humans via PR comment or
copy them onto the drive. Watch `monotonicity_ok`: a False is a STOP
(DEVPLAN §1b sanity rule), not a thing to investigate quietly.
Test-split scoring (`scripts/score_test_split.py`) can run on any free GPU
between waves.
