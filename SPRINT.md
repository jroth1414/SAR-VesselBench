# Sprint 7e — Judy H100 native-venv amendment

Branch: `sprint-7e-judy-venv` (Spine review)
Base: `2726199efcebbebc89156e708b89df2a3415468a`
Phase: 5 — gated, uniform-H100 restart of the 32-cell core grid

## Goal

Reuse the already verified Sprint-7d Box payload without retransmitting its
294 GB tree, deliver the exact Sprint-7e code as a small content-addressed
runtime amendment, and qualify Judy through a sealed native Python venv. The
live V100 campaign remains untouched until every H100 and cutover gate passes.

## Owner-approved runtime amendment

- Native `python -m venv --copies` is the H100 runtime. Apptainer, Enroot,
  Pyxis, SIF, OCI, BF16, TF32, FP16, DDP, and per-arm exceptions are not part
  of the Sprint-7e H100 execution path.
- The base interpreter is exactly Python 3.11.15. Packages install entirely
  offline from Sprint 7d's verified wheelhouse; torch remains
  `2.11.0+cu126` and the normalized freeze must match the exact lock.
- The venv is built at its final persistent path, bytecode-cleaned, sealed
  read-only, tree-hashed, and bound to its base-Python hash and build receipt.
  Jobs invoke its `bin/python` directly under a clean environment; activation
  and user-site packages are forbidden.
- All core training and model-forward inference remain Lightning `32-true`
  with CUDA matmul/cuDNN TF32 disabled before CUDA initialization,
  micro-batch 16, accumulation 1, effective batch 16, one process per GPU,
  eight H100s, and no DDP.
- V100 remains the unchanged fallback and continues to host R2/R3. All 32 core
  cells restart from empty H100 namespaces only after the atomic cutover.
- Frozen detector, scorer, splits, stats, and historical LS-SSDD split are not
  modified.

## Transfer identities

The immutable base payload is
`xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a`. Its data,
weights, labels, wheelhouse, and historical Apptainer definition remain
byte-for-byte Sprint 7d artifacts; the definition is not executed on Judy.

Sprint 7e adds a separate package containing only one exact Git bundle plus
`manifest.json`, `SHA256SUMS`, and `READY.json`. Its identity binds the complete
base-payload receipt and the Sprint-7e commit. It is uploaded to a new empty
Box folder so an active 7d transfer is never mutated or interrupted.

## Acceptance criteria

1. The Sprint-7d package re-verifies unchanged, including all 150 chip, 39
   raster, label, six checkpoint, wheelhouse, and control-file hashes.
2. The runtime amendment round-trips the exact clean Sprint-7e branch and
   required Sprint-7d ancestor, contains no data/weights/wheelhouse/runs/
   environment/secrets, and publishes `READY.json` last.
3. Judy builds the final-path venv only from the verified wheelhouse with
   `--no-index`; `pip check`, exact freeze, base-Python identity, read-only
   modes, receipt hash, and deterministic tree digest all pass before every
   allocation.
4. Exactly eight H100s at compute capability 9.0 pass strict-FP32 parent/child
   assertions, the complete test suite, all six value-sensitive loads,
   both-family batch-16 train/backward/optimizer/full-scene probes, and the
   200-step timing gate.
5. Slurm invokes the venv Python directly with `--export=NONE`, stages the
   base payload and Sprint-7e bundle separately, proves batch-to-child
   `SIGUSR1`, requeues, and resumes a durable checkpoint without a duplicate
   cell.
6. The conservative H100 forecast, including per-allocation staging, beats the
   current V100 forecast at acceptance and again at cutover; fresh valid R2/R3
   markers and the external V100 archive attestation remain mandatory.
7. Campaign, per-cell, cutover, and reverse-result receipts bind the code SHA,
   base payload, runtime amendment, venv tree/build/base Python, strict-FP32
   backend, H100 UUIDs, throughput, and per-cell runtime. Schema-1 SIF receipts
   cannot satisfy these schema-2 gates.
8. After all 32 cells pass the grid and monotonicity guards, the reverse Box
   bundle contains only validated metrics, configs, logs, provenance, and
   best/last checkpoints.

## Definition of done

Source work is complete only after clean tests, exact bundle round trips, and
reviewable commits. Operational cutover remains incomplete until the runtime
amendment is independently uploaded/downloaded, Judy builds and verifies the
venv, every H100 acceptance gate passes, the measured speed advantage still
holds, R2/R3 validate, and the human operator supplies the V100 archive
attestation. Until then, no H100 training/result is claimed and V100 continues.

## Review scope

Sprint 7e is a runtime-only amendment stacked on the immutable Sprint-7d
handoff commit. It changes no scientific recipe or frozen artifact. Review
covers the native-venv builder/verifier, dual transfer receipts, host-isolated
Slurm launch/signal path, schema-2 provenance/cutover/result contracts, tests,
and documentation. Merge remains a human gate into `dev`.
