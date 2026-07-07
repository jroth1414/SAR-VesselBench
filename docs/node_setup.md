# V100-node setup and grid handoff (one-time, ~1–2 h + data transfer)

Everything the 8×V100 node needs to take over the study grid. Order matters.

## 1. Repo + environment (Appendix C constraints)

```bash
git clone https://github.com/jroth1414/JHU-xView3 && cd JHU-xView3   # branch dev
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r locks/env-v100node.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e .
python scripts/gpu_sanity.py    # MUST show sm_70, fp16 matmul PASS, and a
                                # NON-flash SDPA backend (mem_efficient/math)
pip freeze > locks/env-v100node.txt   # re-freeze; keep the header; commit via PR
```

If `torch.cuda.is_available()` is false or sanity fails: the wheel is not the
cu126 build — never `pip install torch` bare here (cu13 dropped Volta).

## 2. Data sync (from the dev box, ~455 GB total)

| What | Dev-box path | Size | Needed for |
|---|---|---|---|
| Chips + sidecars | `D:\JHU-xView3\data\chips\` (150 scene dirs) | ~353 GB | training |
| Dev+test scene rasters | `D:\JHU-xView3\data\raw\xview3\GRD\<39 scenes in splits.json dev+test>` | ~94 GB | dev evals + test scoring |
| Label CSVs | `D:\JHU-xView3\data\raw\xview3\labels\` | 14 MB | evals |
| FM weights | `D:\JHU-xView3\data\weights\` | ~2.4 GB | arms 2,3,6,7 |
| LS-SSDD | `D:\JHU-xView3\data\raw\lsssdd\` | ~3.7 GB | pretrainings (arms 4,8) |
| Finished runs | `D:\JHU-xView3\runs\` (final_metrics.json + checkpoints) | varies | skip-if-done handoff |

The frozen artifacts (`data/splits.json`, `data/stats.json`,
`data/lsssdd_split.json`) come from git. eval_final rasters are NOT synced —
final eval happens once, wherever the human decides.

## 3. Verify the guards ON THE NODE (GPU half)

```bash
pytest tests/ -q        # includes test_fm_checkpoints_load value-sensitive
                        # half (needs data/weights synced) + all freeze pins
```

All green (75+) before any cell runs. A red here is a STOP, not a retry.

## 4. Handoff protocol (zero duplication)

1. On the dev box: stop the queue (`Get-Process python | Stop-Process`),
   note the last completed cell in `runs/logs/grid_queue.log`.
2. rsync `runs/` dev-box -> node (finished cells carry final_metrics.json).
3. On the node: `nohup python scripts/run_grid_node.py --gpus 0 1 2 3 4 5 6 7
   > runs/logs/node_queue.log 2>&1 &` — it skips finished cells, runs the two
   pretrainings first, then fills all 8 GPUs, expensive fractions first.
4. The dev box keeps the reference lane (YOLO26 train + LocateAnything) and
   test-split scoring (`scripts/score_test_split.py`) for finished cells.

Expected node wall-clock for the remaining core grid: ~2–3 nights at the
50-epoch ceiling; early stopping (patience 4 real dev evals) usually less.
Seed reruns (48 cells, DEVPLAN §12) are another ~2 nights afterwards —
launch with the same script after editing FRACS/seeds or ask the agent.
```
