# Sprint 7f — Evaluation Contract Correction and Judy H100 Relaunch

Status: owner-approved implementation amendment (2026-08-04). The owner
explicitly selected continued V100 execution for diagnostic value; the V100
controller and children must not be stopped, signaled, or otherwise mutated by
this amendment. Reportable H100 launch remains gated on every corrected source,
runtime, transfer, and acceptance check below.

Owner intent assumed by this plan:

- Correct the vessel/non-vessel ground-truth contract without modifying the
  frozen scorer.
- Bind every selected operating threshold to the exact checkpoint on which it
  was selected.
- Treat the existing V100 core results as diagnostic because the incorrect dev
  contract affected checkpoint selection and early stopping.
- Run the corrected 32-cell core grid uniformly from scratch on Judy H100s.
- Preserve the once-only verified final evaluation until every earlier gate is
  green.

## 1. Why this amendment is required

Two independent evaluation defects were found while reviewing the provisional
V100 label-efficiency curve.

### 1.1 Non-vessel HIGH/MEDIUM annotations enter the scorer as positives

The training target contract is correct:

- `is_vessel == True` and confidence HIGH/MEDIUM: positive vessel target.
- confidence LOW: ignore target.
- `is_vessel == False` and confidence HIGH/MEDIUM: background/hard negative.

`src/train/losses.py::gaussian_radius_centers` implements that contract.
However, `src/eval/infer_scene.py::ground_truth_from_labels` currently converts
every label row into a `GroundTruthPoint`. The frozen scorer then correctly
interprets every non-LOW point it receives as a positive, because filtering is
the caller's responsibility.

Measured on the eight dev scenes used during training:

| Ground-truth category | Count |
|---|---:|
| Vessel HIGH/MEDIUM | 517 |
| Non-vessel HIGH/MEDIUM | 107 |
| LOW-confidence ignore rows | 118 |
| Positives currently seen by the scorer | 624 |
| Positives required by the study contract | 517 |

The provisional `vitin1k-f10-s0` result has 566 TPs. Since there are only 517
valid vessel positives, at least 49 reported TPs necessarily match annotations
that the training and study contracts define as non-vessel background.

This is a caller/conversion defect. **Do not edit `src/eval/scorer.py` or change
its pinned hash.**

### 1.2 Test/final scoring pairs `best.ckpt` with the last-dev threshold

Training records `best_dev_f1` but stores the complete operating point only in
`last_dev`. Both `scripts/score_test_split.py` and `src/eval/final_eval.py`
currently load `best.ckpt` while applying `last_dev.threshold`.

Observed examples:

| Run | Threshold selected with best checkpoint | Stored last-dev threshold |
|---|---:|---:|
| `vitin1k-f10-s0` | 0.5894 | 0.4150 |
| `vitin1k-f50-s0` | 0.8066 | 0.8999 |
| `cnnin1k-f100-s0` | 0.8848 | 0.4458 |

This mismatch does not create the existing `best_dev_f1` gap, but it would make
held-out test and final results invalid and can strongly distort near-shore F1.

### 1.3 Scientific impact

The incorrect dev target set controlled:

- periodic dev F1;
- threshold selection;
- `best.ckpt` selection; and
- early stopping.

Only `best.ckpt` and `last.ckpt` are retained, so the correct best checkpoint
cannot be reconstructed uniformly from all historical epochs. Post-hoc
rescoring is valuable diagnostically, but it cannot repair reportable model
selection. The scientifically clean remedy is a uniform from-scratch rerun.

No core test scores or verified-final scores have been produced, so those
held-out resources remain unconsumed.

## 2. Non-negotiable scope

### In scope

- Correct label-to-scorer input filtering in the shared evaluation path.
- Persist a complete best-dev operating point tied to the exact best
  checkpoint.
- Make test and final evaluation refuse unbound or legacy thresholds.
- Add CPU tests for both contracts.
- Add a non-reportable corrected-dev diagnostic for existing checkpoints.
- Document the owner-approved amendment and disposition of V100 artifacts.
- Produce a versioned, checksummed Box runtime amendment for Judy.
- Relaunch all 32 H100 core cells uniformly from empty canonical namespaces.
- Correctly rescore or rerun R2/R3 as required by their reference recipes.

### Out of scope

- Any modification to `src/eval/scorer.py`.
- Any modification to `configs/detector.yaml`, `data/splits.json`,
  `data/stats.json`, `data/lsssdd_split.json`, or the verified-eval lock.
- Per-arm or per-fraction tuning.
- Changing precision, batch size, optimizer, LR schedule, architecture,
  augmentation, sampler, seed count, decode settings, or label fractions.
- Using the test or verified-final set to select checkpoints or thresholds.
- Smoothing, clamping, or otherwise forcing a monotone curve.
- Mixing corrected H100 cells with existing V100 cells in one reportable curve.

## 3. Branch and worktree strategy

1. Read the `DEVPLAN.md` cold-start ledger and verify the current integration
   state before choosing a base.
2. Do not edit the live V100 checkout.
3. Create a separate worktree and branch named
   `sprint-7f-eval-contract` from the latest owner-accepted Judy runtime branch.
   The expected predecessor is `sprint-7e-judy-venv`; verify the actual remote
   SHA rather than assuming it. Historical SHA `13110cec...` is evidence only,
   not an instruction to discard later accepted commits.
4. Confirm that the branch contains the native Python 3.11.13 Judy venv path,
   strict H100 FP32 assertions, Box runtime-amendment builder, and Slurm pack.
5. Keep all training data, checkpoints, `runs/`, venvs, Box JWT files, and site
   configuration untracked.

## 4. Implementation work

### 4.1 Centralize the valid evaluation-GT contract

Update `src/eval/infer_scene.py::ground_truth_from_labels` or introduce a small
adjacent helper used by it. Required behavior:

1. Normalize confidence case.
2. Include HIGH/MEDIUM rows only when `is_vessel` is explicitly true.
3. Exclude HIGH/MEDIUM rows when `is_vessel` is explicitly false.
4. Retain LOW-confidence rows as ignore points, matching the training target
   contract.
5. Treat pandas/NumPy booleans and canonical string booleans consistently.
6. Fail clearly on an unparseable HIGH/MEDIUM `is_vessel` value rather than
   silently turning it into a positive.
7. Preserve `source` and `distance_from_shore_km` for vessel positives and LOW
   ignores so dark and near-shore slices remain available.

All evaluation callers must flow through this one conversion path:

- training-time dev evaluation;
- P5.4 test scoring;
- verified final evaluation; and
- R2 scoring.

Do not duplicate filtering independently in each caller.

### 4.2 Persist a checkpoint-bound best-dev record

Extend `DevSceneEval` so its state contains both:

- `best`: the best F1 scalar; and
- `best_result`: the complete result from the same evaluation, including
  epoch, threshold, precision, recall, TP, FP, FN, and candidate count.

Requirements:

- Update `best_result` only when the best F1 improves under the existing
  tie behavior.
- Save and restore `best_result` through callback checkpoint state so Slurm
  requeue/resume preserves it.
- Record the evaluation epoch explicitly.
- Preserve `last_dev` for diagnostics, but never use it as the reportable
  operating point for `best.ckpt`.
- At training completion, obtain the actual best checkpoint path from the
  `ModelCheckpoint` callback and compute its SHA-256.
- Write an explicit artifact schema version.

The new completion marker should include at least:

```json
{
  "result_schema": 2,
  "best_dev_f1": 0.0,
  "best_dev": {
    "epoch": 0,
    "f1": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "tp": 0,
    "fp": 0,
    "fn": 0,
    "threshold": 0.0,
    "n_candidates": 0
  },
  "best_checkpoint": {
    "relative_path": "checkpoints/best.ckpt",
    "sha256": "...",
    "epoch": 0
  },
  "last_dev": {}
}
```

Before publishing `final_metrics.json`, hard-fail if:

- `best_dev` is missing or non-finite;
- the best checkpoint is absent;
- best-dev epoch and best-checkpoint epoch disagree;
- checkpoint hashing fails; or
- the threshold is outside the score range.

Write completion markers atomically so a partial schema cannot be mistaken for
a finished run.

### 4.3 Make held-out scoring refuse threshold/checkpoint mismatches

Update `scripts/score_test_split.py` and `src/eval/final_eval.py` to:

1. Require `result_schema == 2` exactly.
2. Load `best_dev.threshold`, never `last_dev.threshold`.
3. Verify the actual `best.ckpt` SHA-256 against the completion marker before
   inference.
4. Verify the stored best-dev epoch agrees with checkpoint metadata.
5. Verify code SHA, detector hash, precision, micro-batch, accumulation, and
   effective batch as already required by the campaign contract.
6. Refuse legacy completion markers with a clear message; do not silently
   infer or fall back.
7. Apply the frozen dev threshold verbatim to test/final predictions without
   inspecting held-out ground truth.
8. Record threshold source, checkpoint SHA, code SHA, and inference precision
   in every test/final row.

### 4.4 Add a diagnostic-only corrected-dev rescoring command

Create a read-only diagnostic command for existing V100 checkpoints. It must:

- accept an explicit list of run IDs;
- support both `best.ckpt` and `last.ckpt`;
- use only the existing eight dev scenes;
- use the corrected vessel-only GT conversion;
- select a threshold independently for each exact checkpoint;
- record checkpoint SHA, threshold, aggregate metrics, near-shore metrics, and
  per-scene metrics;
- write only beneath a gitignored diagnostic namespace such as
  `runs/diagnostics/eval-contract-<timestamp>/`;
- never modify canonical `final_metrics.json`, checkpoints, or campaign state;
- refuse dev/test/final scene overlap; and
- never read `validation.csv` or any `eval_final` raster.

First diagnostic matrix:

- `vitin1k-f10-s0`: best and last;
- `vitin1k-f50-s0`: best and last;
- `vitin1k-f100-s0`: best and last;
- `cnnin1k-f10-s0`: best and last; and
- `cnnin1k-f100-s0`: best and last.

This diagnostic answers whether corrected scoring shrinks the provisional gap.
It is not a substitute for the uniform reportable rerun.

### 4.5 Audit external references

- R2 YOLO26 scoring calls the shared GT converter, so its existing dev/test
  metrics are invalid. Determine from the reference recipe whether its model
  training/checkpoint selection used the project dev scorer. If not, preserve
  the trained weights and rescore dev/test under the corrected contract. If
  yes, rerun R2 under its unchanged published recipe.
- Audit R3 LocateAnything label filtering. If it uses the same unfiltered
  annotation contract, rerun its scoring. If its sampled chip labels already
  require valid vessels, record evidence and retain it.
- Keep R2/R3 precision recipes independent from core strict FP32.

## 5. Tests and acceptance gates

### 5.1 CPU unit tests

Add focused tests that prove:

- HIGH vessel is included as a positive.
- MEDIUM vessel is included as a positive.
- HIGH/MEDIUM non-vessel is excluded.
- LOW rows remain ignores, including LOW rows with missing `is_vessel`.
- Boolean, NumPy boolean, and canonical string representations behave
  identically.
- NaN/missing HIGH/MEDIUM vessel state is rejected.
- source and shore distance survive conversion.
- the frozen scorer hash is unchanged.

Add checkpoint/threshold tests that construct different best and last dev
thresholds and prove test/final select only the best-bound threshold. Include
negative tests for checkpoint hash drift, epoch mismatch, legacy schema, and
missing best-dev state.

### 5.2 Dataset-scale source acceptance

Using `data/raw/xview3/labels/train.csv`, assert and record:

| Split used before final eval | Vessel H/M positives | Non-vessel H/M excluded | LOW ignores |
|---|---:|---:|---:|
| Training-time dev subset (8 scenes) | 517 | 107 | 118 |
| Full dev split (23 scenes) | 1,479 | 804 | 441 |
| Frozen test split (16 scenes) | 1,165 | 420 | 325 |

The frozen test split currently contains only two valid near-shore vessel GTs;
record this support and interpret its near-shore F1 as a sparse diagnostic.
Do not inspect or count verified-final labels during source acceptance.

### 5.3 Scientific invariants

All of the following must remain byte-identical:

- `src/eval/scorer.py` and its pinned test hash;
- `configs/detector.yaml`;
- `data/splits.json`;
- `data/stats.json`;
- `data/lsssdd_split.json`; and
- the verified-eval lock state (still absent/unconsumed).

The full CPU suite and four guard tests must pass. On a GPU node, rerun all six
value-sensitive checkpoint-load tests and finite strict-FP32 ViT/CNN probes.

## 6. V100 disposition (owner decision, 2026-08-04)

The owner elected to let the V100 campaign continue for diagnostic value.
Therefore:

1. Do not stop, signal, pause, reconfigure, or otherwise mutate the live V100
   controller or any of its children.
2. Classify every existing and subsequently produced V100 core artifact as
   `non-reportable-diagnostic`; the superseded evaluation contract affected
   model selection and early stopping even though gradients used the intended
   FP32 recipe.
3. Preserve all artifacts, logs, completion markers, controller records, and
   OOM-recovery evidence in place while the campaign runs.
4. Before H100 campaign launch, require an external human-authored immutable
   diagnostic-isolation attestation. It binds the V100 and H100 namespaces and
   confirms that V100 marker existence can never suppress, resume, or satisfy
   a corrected H100 cell.
5. Transfer that attestation to Judy through the content-addressed Box control
   plane; Judy must verify its manifest, hashes, and canonical receipt.
6. Never merge V100 and corrected H100 core cells in one curve, table, resume
   decision, or reportable completion namespace.
7. Preserve the original acceptance-time throughput comparison. At cutover,
   if the current remaining V100 wall time is positive, require the conservative
   H100 projection to be strictly smaller and record
   `continues-running-non-reportable-diagnostic`. If the diagnostic has already
   finished, permit exactly zero remaining hours only with the explicit
   `complete-non-reportable-diagnostic` status. Completion never makes the V100
   outputs reportable and never removes the scientifically mandatory uniform
   H100 rerun.

Stopping or archiving V100 later is optional operational work and is not a
precondition for H100 launch. This amendment contains no process-control action
against the live V100 campaign.

## 7. Corrected Judy H100 campaign

Status note (2026-08-05): Slurm smoke 540200 exposed the absent
`SLURM_TMPDIR`; corrected smoke 541320 and probe 541333 then exposed Judy's
host-killed literal verifier path. Ladder 541341 completed the full venv-tree
hash, module probe 541353 reached the strict provenance check, and job 541358
proved the login runtime receipt differs from `dgx18`. Subsequent probes proved
`dgx09` and `dgx18` share full runtime closure `a4af214a...`, and compute
job 541574 built and strictly reverified the sealed venv at tree `d1237904...`
with receipt `d84d60c8...`. Scratch portability, module-form verifier entry,
and compute-runtime qualification are therefore resolved. None of these jobs
produced an H100 acceptance marker or canonical campaign state or affected the
live V100 diagnostic; the corrected runtime must still be repackaged and pass
fresh smoke before H100 acceptance.

After code and source acceptance pass:

1. Retain the login-built Judy Python 3.11.13 venv as diagnostic evidence only.
   Use the compute-built sealed venv from job 541574, whose runtime closure
   matched on `dgx09` and `dgx18`. Every Slurm allocation must reverify it
   with `python -B -m scripts.h100.build_venv`; never execute the literal
   script path or normalize away a host-runtime mismatch.
2. Clone the corrected Git bundle into a new SHA-addressed Judy bootstrap
   directory. Do not overwrite the older Apptainer or native-venv bootstraps.
3. Verify an explicit canonical, existing, non-symlink, compute-node-writable
   `H100_SCRATCH_ROOT`; never assume `SLURM_TMPDIR`. Create only the owned
   mode-`0700` job/restart child and require at least 500,000,000,000 free bytes
   there before campaign extraction. Then verify eight H100s, compute capability
   9.0, strict IEEE FP32/TF32-disabled backend state, code SHA, venv receipt,
   data receipt, and frozen hashes.
4. Require empty corrected-campaign canonical namespaces.
5. Run the full suite, six checkpoint loads, both-family finite train/backward
   probes, and full-scene inference.
6. Run one short corrected-dev diagnostic and confirm exactly 517 positive GTs
   are scored on the eight training-time dev scenes.
7. Run Slurm interruption/requeue/resume smoke and prove best-dev callback state
   survives resume. The requeued allocation must receive fresh reconstructible
   scratch while resume state remains exclusively under persistent
   `H100_RUNS_ROOT`; guarded cleanup must remove only the exact prior allocation
   child.
8. Launch all 32 core cells uniformly from scratch with the unchanged owner-
   approved H100 recipe: `32-true`, TF32 disabled, micro/effective batch 16,
   accumulation 1, seed 0, one process per GPU, no DDP.
9. Complete and validate the training phase for all 32 cells before freezing a
   single immutable `TRAINING_COHORT.json`. No held-out test raster, label, or
   metric may be read before this all-cell barrier.
10. After freezing the cohort, return exit 75 without scoring. The host requeues
    the job; only that fresh allocation may validate the immutable cohort and
    stage the score-only TEST view.
11. Score the frozen cohort in a separate phase and write immutable
    `test_metrics.json` artifacts; never append held-out fields to the training
    completion marker.
12. Record best checkpoint SHA and checkpoint-bound dev threshold in campaign
    provenance for every completed cell.
13. Fail-stop the queue on the first invalid completion marker, non-finite
    value, hash mismatch, or callback-state mismatch.

The compute allocation uses a physical allowlisted data view beneath its
explicit allocation-private
`$H100_SCRATCH_ROOT/xview3-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT}` child,
not an assumed `SLURM_TMPDIR` and not a full base extraction. Acceptance
receives 111 TRAIN chip directories, the fixed sorted eight DEV raster
directories, six weight directories, the offline wheelhouse, and a source-built TRAIN+DEV8 CSV (13,911
rows). Training allocations receive the same scientific inputs without
re-extracting the already installed wheelhouse. Neither view contains a TEST
chip, TEST raster, or TEST label row. After the cohort barrier, the new
allocation receives 16 TEST raster directories, no chips, six weight
directories, and an audited TRAIN+TEST CSV (15,079 rows). The full combined
label archive is opened and filtered only after the cohort validates.

Acceptance records that physical allowlist as
`H100_READY.json:venv.staged_data_view`, binding the canonical
`H100_DATA_VIEW.json` SHA-256 and embedded receipt. Cutover validates the
embedded receipt after allocation cleanup; the scratch path is provenance and
is not required to persist. This replaces, rather than supplements, the
retired full-scratch `staged_base_extraction` contract. The persistent
wheelhouse/base-extraction path remains only in the sealed `venv_build.json`;
the acceptance marker carries its pathless identity, and final result
packaging must require the two identities to match before using that
persistent path.

This is allocation-view isolation, not a claim of host-global ACL isolation.
Judy's immutable Sprint-7d package already exists as plaintext under the same
Unix identity, so a malicious same-UID process could bypass the canonical
launcher and inspect it. The enforced contract prevents accidental or
study-code access by making held-out bytes absent from the canonical pre-cohort
view and by requiring every production launch to validate that view before any
label/raster access.

Monotonicity is an acceptance diagnostic, not a training objective. Do not
alter results or tune a cell to make the curve monotone.

## 8. Held-out evaluation sequence

After all corrected core cells complete:

1. Validate all 32 exact-schema training completion markers, then atomically
   create one immutable `TRAINING_COHORT.json` binding every run ID, code SHA,
   detector/recipe provenance, best checkpoint SHA/epoch, and best-dev
   threshold. The cohort must not be incrementally extended or replaced.
2. Only after that cohort exists, run P5.4 once over the 16 frozen test scenes
   for every cohort member.
3. Write aggregate test F1/P/R, dark recall, and near-shore F1 with support
   counts and provenance to a separate immutable `test_metrics.json` beside
   each run. Never mutate `final_metrics.json` after cohort freeze.
4. Require exact test support (1,165 vessel positives, zero dark-vessel
   positives, two near-shore vessel positives, and all 16 frozen scene IDs) in
   every test result.
5. Build the complete 32-row `grid.csv`; because every test score is present,
   the curve must use test F1 rather than dev F1.
6. Apply the predeclared 0.02 monotonicity diagnostic.
7. If any arm fails, STOP for human interpretation. Do not rerun a favored
   fraction, change a threshold, add a seed, or inspect verified-final data.
8. Only after Phase 5 acceptance and explicit owner confirmation run the
   once-only verified final evaluation.

## 9. Documentation updates

Update, on the amendment branch:

- `DEVPLAN.md`: cold-start ledger, blocker, Phase 5 entry/acceptance state, and
  corrected H100 campaign provenance.
- `AGENTS.md`: short non-negotiable GT conversion and checkpoint-threshold
  binding rules.
- `docs/decisions.md`: evidence, owner decision, V100 diagnostic disposition,
  unchanged scorer hash, and from-scratch H100 rerun rationale.
- Judy handoff/runbook documentation: corrected bundle identity, bootstrap
  directory, validation commands, and rollback path.

Do not describe the change as a scorer modification. It corrects the inputs to
the frozen scorer and the provenance binding between model and operating point.

## 10. Box runtime-amendment handoff to Judy

Package this as a new versioned runtime amendment rather than rebuilding or
overwriting the large base data payload.

Required contents:

- Git bundle containing `sprint-7f-eval-contract` and its history;
- one deterministic source-built TRAIN+fixed-DEV8 label CSV (13,911 rows,
  exactly 111 TRAIN plus eight sorted DEV scenes) and no TEST/eval-final row;
- source GT audit metadata binding the immutable combined-label member without
  requiring Judy to open it before cohort freeze;
- bootstrap/pull script pinned to the exact bundle SHA and branch;
- amendment manifest with code SHA, bundle SHA-256, base payload identity,
  unchanged environment-lock identity, and the native Python/torch contract;
  exact compute-venv root and receipt hashes bind later through ignored
  `site.env`, its immutable submit snapshot, and allocation acceptance;
- `SHA256SUMS`; and
- `READY.json` written last.

Transfer requirements:

- Use the existing Box transfer venv and runtime environment variables.
- Require `BOX_JWT_CONFIG` and a dedicated amendment `BOX_FOLDER_ID` at runtime.
- Keep JWT material outside the repo, mode `0600`, and never log/archive it.
- Upload to a new child folder; do not overwrite prior runtime amendments.
- Use chunked uploads where required and verify Box size/SHA-1 plus local
  SHA-256.
- On Judy, download to `.partial`, verify, rename atomically, and clone into a
  new SHA-addressed bootstrap directory.
- Preserve the existing base-extracted payload. Preserve the login-built
  read-only venv only as diagnostic evidence; production uses the fresh
  compute-built venv after cross-DGX runtime equality and exact receipt
  verification. The amendment contains code/control metadata plus only the
  narrowly derived TRAIN+DEV8 label artifact above.
- Run package verification before changing any Judy launch pointer.

Dynamic cross-site evidence uses the same verified, content-addressed package
protocol with narrow allowlists and separate Box child folders:

1. finalized R2/R3 reference receipts travel V100 to Judy;
2. Judy's immutable `CUTOVER_READY.json` travels Judy to the V100 operator;
3. the owner-approved `V100_DIAGNOSTIC_ISOLATION.json` travels V100 to Judy; and
4. Judy verifies all three canonical receipts before campaign launch.

These control packages contain JSON evidence only. They never contain V100
checkpoints, runs, credentials, or a command capable of stopping the campaign.

## 11. Definition of done

The amendment is ready for reportable H100 training only when all are true:

- The frozen scorer and all frozen scientific artifacts retain their pinned
  hashes.
- Evaluation counts only vessel HIGH/MEDIUM rows as positives and retains LOW
  ignores.
- Dataset-scale dev/test counts match Section 5.2.
- Completion schema binds a full best-dev result to an exact best-checkpoint
  hash and epoch.
- Test/final paths refuse last-dev thresholds and legacy unbound artifacts.
- Full CPU and H100 GPU acceptance suites pass.
- At least two eligible DGX nodes have identical preserved full base-runtime
  fingerprints; the selected build node is remeasured immediately before the
  build; and a fresh compute-built venv at a new persistent path has matching
  tree, build, Python-runtime, wheelhouse, extraction, and lock receipts.
- The external-signal Slurm smoke passes under the explicit
  `H100_SCRATCH_ROOT` contract, proves fresh scratch on requeue and persistent
  checkpoint resume, and leaves no allocation child after guarded cleanup.
- Existing and future V100 core artifacts are explicitly classified as
  diagnostic, the live campaign remains untouched, and Judy has verified the
  external diagnostic-isolation attestation.
- The corrected H100 namespace is empty and all 32 cells are scheduled from
  scratch under one uniform recipe.
- All 32 training markers are validated and frozen into one immutable cohort
  before any held-out test access; held-out outputs are separate immutable
  `test_metrics.json` artifacts.
- The Box amendment has a verified round trip and Judy reports the expected
  code, package, venv, and data hashes.
- No test result has influenced training or threshold selection.
- The verified-final lock remains absent until Phase 5 and human gates pass.

## 12. Rollback and preservation

- Keep the prior Judy bootstrap directories, runtime packages, base extraction,
  and venv receipts intact.
- If amendment verification fails, leave Judy launch pointers unchanged and
  continue no reportable training.
- If corrected probes fail, preserve logs and stop; do not fall back to the
  superseded evaluation contract.
- Existing V100 checkpoints remain recoverable diagnostic artifacts but can
  never satisfy corrected H100 completion markers.
