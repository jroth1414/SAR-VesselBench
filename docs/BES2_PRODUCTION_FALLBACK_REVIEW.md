# Sprint 10i — Arm-6 production fallback review

Arm 6 now retains the ordinary seeded three-channel ConvNeXt-V2-Base
`backbone.model.stem.0.weight` and transfers every other compatible S2 tensor,
including the stem bias and normalization. The target, head, RNG stream,
parameter count, adapter, and training recipe stay unchanged. `load_post_stem`
is the single production/diagnostic reset implementation. Historical replay
explicitly restores the original converted stem, and never enters the core
arm registry.

This is the owner-approved adoption of timm 1.0.27's unsupported-convolution
fallback semantics. The library's builder itself does not enter its channel
adaptation branch for `in_chans=3`; we do not claim a literal 10→3 call would
select the fallback. See the pinned upstream `timm/models/_builder.py`
`load_pretrained` exception handler and `timm/models/_manipulate.py`
`adapt_input_conv`, also inspected in the existing local training venv.

S2 remains bound to checkpoint SHA-256
`b09d0e41cc683878243a9128a6f4724d6a71d562318beeae716f0dce9cbbf454`.
Its existing downloaded safetensors path and encoder key mapping are preserved;
no new configilm/download path is introduced. Each production load checks that
identity before deserialization and logs a pathless initialization description,
checkpoint digest, retained weight hash, and complete loaded-target list.

## Validation (2026-09-14)

- Complete offline CPU suite: **528 passed, 8 skipped**, 63.36 seconds.
  Six skips are the existing downloaded-weight half; two are the explicit
  S2/H100 acceptance opt-ins. All four frozen guards ran unchanged.
- The official S2-only CPU acceptance ran separately: **1 passed**, 13.51
  seconds. It compares every compatible tensor with the downloaded checkpoint,
  checks a post-stem representation norm against seed-0 random initialization,
  and verifies byte-identical seed-0 stem/head plus unchanged post-head RNG.
- New negative checks reject other missing keys, unexpected/duplicate targets,
  shape/dtype drift, substituted checkpoints, and diagnostic artifacts copied
  into the core namespace, including an attempted schema-only promotion.
- The first full run had 20 failures solely because `zstd` was absent from
  PATH. Reusing the existing `/opt/conda/bin/zstd` 1.5.5 through temporary
  test tooling resolved them. No package or training-venv change was made.
- Source comparison confirms every pre-existing loader helper and Arm-7's
  loader body are AST-identical to `dev`; no S1 checkpoint was opened or tested.
  The existing offline key-manifest/parity guards still cover all eight arms.
- Before/after SHA-256 comparison covers the frozen scorer, splits, stats,
  historical LS split, detector config, four guard files, backbone, shared head,
  Lightning module, and fine-tuning entrypoint: all 13 files are byte-identical.
  Original cohort/TEST evidence and V100 processes were not opened or mutated.
- No local final-consumption control path exists in the inspected runs root.
  Packaged diagnostic evidence reports historical nonconsumption; current Judy
  state must be rechecked before every authorized transfer/allocation.

CPU suite log SHA-256:
`2ad678c49e36c69bf9a00d71c014a547a15088ef2ec1af83d7bd5cdd53d77a85`.
Official S2 test log SHA-256:
`78bf2177ac2d147c9e248d154f977580b6f5bd571338300a3ff4c9ecd85af91a`.
Frozen snapshot SHA-256:
`2bc0f41571442f13c2afec2b41c81f02bc2d0da31c305482c099056968389a14`.
Logs and snapshot are working evidence under `/tmp`, outside Git.

## Prepared Judy acceptance steps — not executed

Integrate these commands into the next reviewed replacement Slurm launcher,
after clean-environment sealed-venv verification, source/receipt binding, fresh
smoke, canonical TRAIN/DEV staging, and final-nonconsumption checks. Invoke
`H100_VENV_ROOT/bin/python` directly. The launcher must set the existing
sitecustomize PYTHONPATH, `NVIDIA_TF32_OVERRIDE=0`, and the complete snapshotted
libpython loader path under `--export=NONE`, and expose exactly one allocated
H100 per worker. `XVIEW3_BES2_WEIGHTS_ROOT` names only the authorized staged
weights root. Do not run these commands from a login node or the live V100 host.

```bash
XVIEW3_BES2_WEIGHTS_ROOT="$H100_STAGED_WEIGHTS_ROOT" \
  "$H100_VENV_ROOT/bin/python" -B -m pytest -q -s \
  tests/test_bes2_production.py::test_official_s2_complete_value_sensitive_transfer
XVIEW3_BES2_WEIGHTS_ROOT="$H100_STAGED_WEIGHTS_ROOT" \
XVIEW3_BES2_H100_ACCEPTANCE=1 \
  "$H100_VENV_ROOT/bin/python" -B -m pytest -q -s \
  tests/test_bes2_production.py::test_h100_s2_batch16_lightning_forward_backward
```

The second test refuses non-H100, multiple visible GPUs, a different torch
version, or inactive IEEE backend guards. It uses the shared Lightning module,
optimizer and schedule for one batch-16 step, asserts finite positive stem,
post-stem and head gradients plus an actual stem update, then checks finite
FP32 batch-16 stride-4 output. It emits measured GPU-hours. This synthetic
acceptance test supplies no training marker, checkpoint, readiness, or cohort;
it supplements the required canonical TRAIN/DEV family probes and forecast.

## Remaining boundary

The source is ready for owner review. No runtime amendment was built or
published and no replacement Judy job ran. After review/merge, a separate
narrow runtime sprint must bind the accepted venv and readiness evidence,
package only code and allowlisted controls, transfer through Box with READY
last, and run fresh smoke and acceptance in separate namespaces. The new
32-cell cohort starts from initialization, freezes only after complete marker
validation, then enters the unchanged bounded TEST/sanity gates. No old or
diagnostic checkpoint is reused.

The prospective final orchestrator remains a later reviewed implementation:
validate the two independent frozen cohort identities, construct exactly 64
namespaced cells, bind original failed TEST status and replacement TEST gates,
retain reference/control barriers, require a new owner authorization over both
hashes, and perform one no-requeue access. Neither the old all-32 authorization
nor this loader amendment authorizes final-data consumption.
