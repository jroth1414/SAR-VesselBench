# Sprint 7f — Evaluation-contract correction and Judy H100 relaunch

Branch: `sprint-7f-eval-contract` (Foundation review)
Base: `26bece168cd3b9b262ffec5939b836df21b352cd`
Phase: 5 — corrected, gated, uniform-H100 restart of the 32-cell core grid

## Goal

Correct the held-out evaluation contract before Judy training, reuse the
verified Sprint-7d payload and Sprint-7e native venv, and deliver the exact
Sprint-7f code and control receipts through Box. The live V100 campaign remains
untouched as a non-reportable diagnostic while all 32 reportable core cells
restart uniformly on Judy H100s.

## Owner-approved runtime amendment

- Native `python -m venv --copies` is the H100 runtime. Apptainer, Enroot,
  Pyxis, SIF, OCI, BF16, TF32, FP16, DDP, and per-arm exceptions are not part
  of the Judy H100 execution path.
- The Judy base interpreter is exactly Python 3.11.13 for every H100 cell.
  Packages install entirely offline from Sprint 7d's verified wheelhouse;
  torch remains `2.11.0+cu126` and the normalized freeze must match the exact
  lock. Sprint 7d's Python-3.11.15 OCI and wheelhouse-resolution metadata stay
  immutable base-payload provenance and are not the Judy runtime contract.
- The venv is built at its final persistent path, bytecode-cleaned, sealed
  read-only, tree-hashed, and bound to its executable/full base-Python-runtime,
  verified Sprint-7d wheelhouse/extraction, and build-receipt hashes.
  Judy's canonical libpython directory is a required, snapshotted site input
  and the complete `LD_LIBRARY_PATH` under Slurm `--export=NONE`. Jobs invoke
  the venv's `bin/python` directly under a clean environment; activation,
  inherited loader paths, and user-site packages are forbidden.
- All core training and model-forward inference remain Lightning `32-true`
  with CUDA matmul/cuDNN TF32 disabled before CUDA initialization,
  micro-batch 16, accumulation 1, effective batch 16, one process per GPU,
  eight H100s, and no DDP.
- Existing and future V100 core cells are non-reportable diagnostics. The owner
  elected to leave that campaign running untouched; no Sprint-7f command may
  stop, signal, pause, reconfigure, or archive it. Corrected R2/R3 references
  remain separate V100 experiments. All 32 reportable core cells restart from
  empty H100 namespaces only after the atomic cutover.
- Judy and V100 have completely separate filesystems. Every Judy submission
  uses `H100_V100_CONTROL_PLANE=box-transfer-v1`; no live-V100 or dummy path
  is accepted. Smoke and H100 acceptance are Judy-local. Corrected-reference,
  cutover, and diagnostic-isolation evidence crosses Box at the declared gates;
  campaign has no mounted V100/reference path dependency. A stop/archive
  receipt is not required for H100 launch.
- Frozen detector, scorer, splits, stats, and historical LS-SSDD split are not
  modified.

## Transfer identities

The immutable base payload is
`xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a`. Its data,
weights, labels, wheelhouse, and historical Apptainer definition remain
byte-for-byte Sprint 7d artifacts; the definition is not executed on Judy.

Sprint 7f adds a separate schema-2 package containing one exact Git bundle,
the deterministic source-audited 13,911-row TRAIN+fixed-DEV8 label CSV, and
`manifest.json`, `SHA256SUMS`, and `READY.json`. It contains no TEST/eval-final
row or imagery. Its identity binds the complete base-payload receipt, the
required Sprint-7e ancestor, and the Sprint-7f commit. It is uploaded to a new
empty Box child folder so prior payloads and runtime amendments remain
immutable. The generated hash-pinned standalone runtime bootstrap is the
mandatory Judy download/clone path. Full-base `download-runtime`,
`verify-runtime`, and `extract-runtime` are source/setup-only and must never
run in an H100 pre-cohort allocation.

Dynamic control evidence crosses Box in separate narrow, content-addressed
JSON-only packages: five corrected-reference files travel V100 to Judy,
`CUTOVER_READY.json` travels Judy to the V100 operator, and the human-authored
`V100_DIAGNOSTIC_ISOLATION.json` returns to Judy. Judy must verify all three
canonical receipts before campaign launch. These packages never include V100
checkpoints, runs, credentials, or process-control commands, and no V100
stop/archive receipt is required.

## Acceptance criteria

1. Source/setup acceptance re-verifies the Sprint-7d package unchanged,
   including all 150 chip, 39 raster, label, six checkpoint, wheelhouse, and
   control-file hashes. H100 pre-cohort allocations use control-only base
   verification and phase-authorized archive hashes, never this full pass.
2. The runtime amendment round-trips the exact clean Sprint-7f branch and
   required Sprint-7e ancestor; contains exactly its Git bundle, audited
   13,911-row TRAIN+fixed-DEV8 CSV, and three controls; contains no TEST/
   eval-final rows or imagery, weights, wheelhouse, runs, environments, or
   secrets; and publishes `READY.json` last.
3. Judy builds the final-path venv only from the verified wheelhouse with
   `--no-index`; `pip check`, exact freeze, base-Python identity, read-only
   modes, receipt hash, and deterministic tree digest all pass before every
   allocation.
4. Exactly eight H100s at compute capability 9.0 pass strict-FP32 parent/child
   assertions, the complete test suite, all six value-sensitive loads,
   both-family batch-16 train/backward/optimizer/full-scene probes, and the
   200-step timing gate.
5. Slurm invokes the venv Python directly with `--export=NONE`, stages the
   base payload and Sprint-7f bundle separately, proves batch-to-child
   `SIGUSR1`, requeues, and resumes a durable checkpoint without a duplicate
   cell.
   The smoke and H100 acceptance snapshots contain no V100/reference filesystem
   path and enforce the exact Box control-plane literal.
6. The conservative H100 forecast, including per-allocation staging, beats the
   current V100 forecast at acceptance and again at cutover; corrected schema-2
   R2/R3 evidence and the external diagnostic-isolation attestation complete
   verified Box round trips. Stopping or archiving the V100 diagnostic is not
   required.
7. Campaign, per-cell, cutover, and reverse-result receipts bind the code SHA,
   base payload, runtime amendment, venv tree/build/base Python, strict-FP32
   backend, H100 UUIDs, throughput, and per-cell runtime. Schema-1 SIF receipts
   cannot satisfy these schema-2 gates.
8. All 32 training markers validate and freeze into one immutable
   `TRAINING_COHORT.json` before any held-out test raster, label, or metric is
   accessed. A separate scoring phase writes immutable `test_metrics.json`
   artifacts; only then may monotonicity be checked. The reverse Box bundle
   contains only validated metrics, configs, logs, provenance, cohort/control
   records, and best/last checkpoints.

## Definition of done

Source work is complete only after clean tests, exact bundle round trips, and
reviewable commits. Operational cutover remains incomplete until the runtime
amendment is independently uploaded/downloaded, Judy builds and verifies the
venv (or re-verifies the already sealed matching venv), every H100 acceptance
gate passes, the measured speed advantage still holds, transferred corrected
R2/R3 evidence validates, and Judy verifies the human-authored diagnostic-
isolation attestation against the actual empty canonical H100 runs root. All 32
training markers must then freeze into one cohort before held-out test access.
Until those gates pass, no H100 training/result is claimed and V100 continues
untouched as a non-reportable diagnostic.

## Review scope

Sprint 7f is an evaluation/control amendment stacked on the owner-accepted
Sprint-7e native-venv commit and immutable Sprint-7d payload. It changes no
frozen artifact or training recipe. Review covers corrected GT and checkpoint-
bound operating points, held-out isolation, content-addressed cross-site
controls, Judy namespace binding, schema-2 provenance/result contracts, tests,
and documentation. Merge remains a human gate into `dev`.
