# Sprint 7d — H100 strict-FP32 Box handoff

Branch: `sprint-7d-h100-fp32` (Spine review)
Base: `48e10534a8c7baf0662acd548f52928da69f23c8`
Phase: 5 — gated, uniform-H100 restart of the 32-cell core grid

## Goal

Transfer the exact Sprint-7c code, offline Python environment, 150 chipped
scenes, 39 dev/test raster scenes, train labels, and six core checkpoints to
the separate eight-H100 platform through Box. Qualify that target without
interrupting the live V100 fallback, then restart all 32 core cells uniformly
in strict IEEE FP32 only after the atomic cutover barrier passes.

## Authorized amendment

- All core training and model-forward inference remain Lightning `32-true`.
- H100 CUDA matmul and cuDNN use IEEE FP32 with TF32 disabled before CUDA
  initialization and asserted in every subprocess.
- Micro-batch 16, accumulation 1, effective batch 16, one process per GPU,
  eight H100s, and no DDP.
- All 32 core cells restart from empty H100 namespaces. V100 core results
  become non-reportable diagnostics only after successful cutover.
- R2/R3 remain separate V100 references under their independent recipes.
- Frozen detector, scorer, splits, stats, and historical LS-SSDD split are not
  modified by this sprint.

## Acceptance criteria

1. A clean, full-SHA-addressed production package contains one exact Git
   bundle, 150 chip archives, 39 dev/test raster archives, train labels, six
   exact checkpoint archives with source/license notes, and the verified
   Python 3.11.15/cu126 wheelhouse plus digest-pinned Apptainer definition.
2. Box preflight reports at least 500 GB free and a positive maximum file
   size; uploads are size/SHA-1 verified, large files use chunked sessions,
   downloads use verified `.partial` files, and `READY.json` is published
   last only for a complete tree.
3. The target SIF builds with `apptainer build --fakeroot`, installs entirely
   offline from the wheelhouse, and matches the exact normalized lock.
4. Exactly eight H100s at compute capability 9.0 pass strict-FP32 parent/child
   identity probes, the full tests, all six value-sensitive loads, both-family
   batch-16 train/optimizer/full-scene probes, and the Slurm requeue/resume
   smoke.
5. The 200-step projection includes measured staging per projected Slurm
   allocation and beats the measured V100 forecast at acceptance and again
   when refreshed by `cutover-check`.
6. Fresh valid R2/R3 markers exist before the read-only cutover guard writes
   `CUTOVER_READY.json`. A human then gracefully stops and archives V100 core
   diagnostics and provides the external attestation before campaign launch.
7. The campaign schedules every core ID exactly once, expensive fractions
   first, records complete hardware/container/package/runtime provenance,
   fail-stops after the first invalid process, and resumes only durable
   checkpoints after preemption.
8. After all 32 cells pass the grid and monotonicity guards, the reverse Box
   bundle contains only validated core metrics, configs, logs, provenance,
   and best/last checkpoints.

## Definition of done

This sprint's code is done when the source-host suites pass from a clean
commit and the exact Git bundle round-trips. Operational cutover is done only
after the real Box receipt, target SIF, H100 acceptance receipt, current
throughput advantage, R2/R3 barrier, and external V100 archive attestation all
validate. Until then, the V100 campaign continues and no H100 run is claimed.

## Review scope

The owner-directed Sprint 7d handoff is intentionally carried on the single
`sprint-7d-h100-fp32` branch named in the handoff plan. Its package format,
target acceptance, campaign state machine, cutover barrier, and reverse
handoff share signed receipts and therefore land as one atomic contract. To
keep that contract reviewable despite its size, the implementation is split
into four ordered commits: deterministic source packaging and Box transfer;
strict-FP32 target acceptance; fail-stop campaign/cutover/result handback; and
the policy/documentation amendment. A fifth operational-hardening commit
corrects the Box co-owner root-permission preflight without changing the
package contract. A sixth resolver-hardening commit limits the wheelhouse to
the fully enumerated lock; bootstrap tooling remains supplied only by the
pinned OCI base. Review and merge remain human gates.
