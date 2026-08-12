# Agent Instructions

Read `DEVPLAN.md` before changing this repository. It defines the public
scientific contract and current result state.

## Non-negotiable rules

1. Do not invent methods. Use the checkpoint sources and model recipes in
   `DEVPLAN.md` and the cited reference implementations.
2. Preserve within-track fairness. An arm cannot receive a different detector,
   optimizer, schedule, input, batch size, seed, or precision recipe.
3. Do not modify these frozen files without explicit owner approval:
   `src/eval/scorer.py`, `configs/detector.yaml`, `data/splits.json`,
   `data/stats.json`, and `data/lsssdd_split.json`.
4. Keep scene partitions disjoint. No development, test, or final-evaluation
   scene may enter training.
5. Use the shared PyTorch Lightning training path. Do not add a bespoke loop
   for one arm.
6. Treat the 50 human-verified scenes as sealed. Only the once-only held-out
   command may access them.
7. Publish core metrics from verified H100 completion records only. V100 runs
   are diagnostics and cannot fill an H100 table or curve.
8. Keep downloaded data, labels, weights, checkpoints, runs, environments,
   credentials, and site-specific paths out of Git.

## Guard tests

The following checks protect study validity:

- `test_split_disjoint`: scene-level anti-leakage.
- `test_backbone_parity`: architecture and adapter parity.
- `test_fm_checkpoints_load`: structural and value-sensitive checkpoint loads.
- `test_scorer_immutable`: scorer byte hash.
- The split, statistics, detector, and LS-SSDD provenance hash tests.
- H100 result-contract tests: complete matrix, finite metrics, homogeneous
  provenance, and completion-closed reporting.

Do not weaken a guard to make a change pass. Stop and report the conflict.

## Workflow

- Develop from `dev` on a topic branch. `final-submission` is the public
  release branch.
- Keep commits small and use the repository owner identity. Do not add agent or
  coauthor trailers.
- Preserve unrelated work in dirty worktrees.
- Use external scratch for generated data, checkpoints, and runs.
- Run focused tests during implementation and the complete suite before a
  handoff.

Stop and ask the owner if a reference is ambiguous, a frozen artifact would
change, a validity test fails without one obvious correction, or a curve
violates the declared monotonicity tolerance.
