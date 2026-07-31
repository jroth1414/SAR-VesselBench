# H100 strict-FP32 Slurm lane

This lane changes execution hardware only. Every core cell still calls the
unchanged `src.train.finetune` entrypoint with shared `32-true`, micro-batch
16, accumulation 1, and effective batch 16. No script here stops, signals, or
otherwise mutates the live V100 campaign.

## 1. Verify the two-layer Box handoff

Keep the completed Sprint 7d data/wheelhouse payload immutable. The smaller
Sprint 7e runtime amendment is a separate content-addressed package. For each,
verify its out-of-band package id, source Git SHA, `manifest.json`,
`SHA256SUMS`, and `READY.json` before use:

```bash
python3 -m scripts.handoff verify --package-root /path/to/sprint-7d-base-payload
python3 -m scripts.handoff verify-runtime \
  --package-root /path/to/sprint-7e-runtime-amendment \
  --base-package-root /path/to/sprint-7d-base-payload
python3 -m scripts.handoff extract-runtime \
  --package-root /path/to/sprint-7e-runtime-amendment \
  --base-package-root /path/to/sprint-7d-base-payload \
  --destination /an/empty/runtime-amendment-directory
python3 -m scripts.handoff extract \
  --package-root /path/to/sprint-7d-base-payload \
  --destination /an/empty/data-staging-directory
```

The historical bundle inside the base payload is independently rehashed but
never cloned for H100. Only the runtime amendment bundle is cloned to
`$SLURM_TMPDIR/repo`. The batch job extracts the base payload to the separate
empty `$SLURM_TMPDIR/payload` and links only ignored `data/chips`, `data/raw`,
and `data/weights` into the Sprint 7e clone. Committed frozen JSON is never
replaced. Both transfer identities remain bound in every readiness receipt.

## 2. Build the persistent native venv

Provide an executable Python 3.11.15 installation on Judy, record the SHA-256
of its resolved executable, and build directly at the final persistent path
from the base payload's verified offline wheelhouse:

```bash
/path/to/requirements-transfer-python -m scripts.h100.build_venv build \
  --repo /path/to/cloned/repo \
  --wheelhouse /path/to/extracted/environment/wheelhouse \
  --base-extraction-receipt /path/to/extracted/HANDOFF_EXTRACTED.json \
  --expected-base-payload-package-id xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --expected-base-payload-manifest-sha256 fccb0b505c89836a148afec709bb799f7af4908d955ea1b142e153154d830896 \
  --base-python /path/to/python-3.11.15/bin/python3.11 \
  --output /persistent/venvs/xview3-h100-fp32
```

The builder uses `venv --copies`, installs only from the wheelhouse with
`pip --no-index`, checks the normalized freeze against
`locks/env-v100node.txt`, removes bytecode, seals the tree read-only, and
writes `<venv>.sha256` plus `<venv>.build.json`; the build JSON is published
last. The venv is non-relocatable. Every allocation rehashes and verifies the
complete tree, receipt, environment lock, `pip check`, wheelhouse tree, canonical
base-extraction receipt, and full base-Python runtime closure (resolved
executable, stdlib/platstdlib without mutable caches or base site packages,
libpython, and the probed mapped system libraries) before running the exact
`<venv>/bin/python` path. Acceptance also compares a fresh scratch extraction
to the persistent build extraction. There is no activation.

The native process starts under `env -i` with an allocation-local home/cache,
offline package/model settings, `PYTHONNOUSERSITE=1`, and only enumerated CUDA
and Slurm values. `NVIDIA_TF32_OVERRIDE=0` is exported before any CUDA-capable
command. Box variables never enter the child process.

## 3. Configure the site

Copy `site.env.example` to ignored `site.env` and fill every hash/path.
Set `H100_V100_RUNS_ROOT` to the existing live V100 campaign directory; it is
an immutable, read-only isolation boundary, not an H100 output. Submission
canonicalizes that path plus `H100_RUNS_ROOT` and `H100_JOB_LOG_DIR`, then
rejects equality or ancestor overlap in either direction before its first
directory creation, receipt write, or Slurm submission. Account, partition,
reservation, log directory, mail, project name, reference paths, Box JWT
path/folder, and the preflight result-part limit are site interfaces.
Submission uses exact `--export=NONE`; mode and site-file path are validated
positional batch arguments, not exported environment variables.

## 4. Slurm smoke, acceptance, and cutover

Run the lightweight one-GPU Slurm acceptance first:

```bash
slurm/h100/submit.sh smoke
```

The first allocation's exact venv Python publishes its PID. The outer batch
verifies that direct-child identity, sends a real `SIGUSR1` across the
batch-to-native-process boundary, and the child atomically promotes a
synthetic HPC checkpoint before exiting with the reserved host-requeue code.
The batch authorizes and performs exactly one real `scontrol requeue`; the
requeued allocation resumes from `last.ckpt` and writes
`runs/.h100/slurm-smoke/SLURM_SMOKE_READY.json`. That receipt binds source,
detector, venv/build/base-Python, base-payload, and runtime-amendment identity.
Acceptance and cutover reject an absent, stale, or modified smoke receipt.

```bash
slurm/h100/submit.sh acceptance
```

Acceptance requires eight identical H100s at CC 9.0, at least 500 GB
(500,000,000,000 bytes) of scratch before
extraction, exact source/frozen/venv/dual-package hashes, all tests and six
real checkpoint loads, strict IEEE FP32 in child processes, both probes,
and a 200-step CNN projection that conservatively beats the recorded
remaining V100 wall time. The forecast includes the measured package
verification/clone/extraction cost once per projected Slurm allocation. It
writes `runs/.h100/H100_READY.json`.

Once that marker and fresh, provenance-bound R2/R3 results exist, run the
read-only cutover guard:

```bash
slurm/h100/submit.sh cutover-check
```

Refresh `H100_REMAINING_V100_WALL_HOURS` immediately before this command. The
guard rechecks the accepted H100 wall clock against that current forecast and
writes `CUTOVER_READY.json`; it never stops or signals V100. Only
after that succeeds, the human operator gracefully stops the V100 core
processes, archives their core artifacts as non-reportable diagnostics, and
creates the external `V100_CORE_ARCHIVED.json` attestation plus its archive
manifest using the tracked schemas. Record those two file hashes and the
`CUTOVER_READY.json` hash in the untracked `site.env`, then launch:

```bash
slurm/h100/submit.sh campaign
```

Campaign submission validates and persists all three operator evidence files
before it submits the 32-cell H100 job.

## 5. Campaign, preemption, and resume

The queue launches 32 cells expensive-first (`f100`, `f50`, `f25`, `f10`),
one unchanged trainer per GPU. Every allocation repeats the strict hardware
probe and must match the accepted name, memory, CC, driver, torch/CUDA, and
IEEE backend class; actual allocation UUIDs go to the controller. A failure
prevents new launches while already running cells finish. Fifteen minutes
before the Slurm limit, USR1 is forwarded to every trainer; Lightning writes
HPC checkpoints, deferred requeue waits for all eight, and each checkpoint is
promoted atomically to `last.ckpt`. The controller exits with a reserved code
and only the outer host batch calls real `scontrol`; native children never do.

After training, the per-GPU wrapper runs the existing frozen test-split scorer.
A cell is complete only with finite test metrics, both checkpoints, and fully
bound runtime provenance. Campaign completion additionally requires exactly
32 `test_f1` rows and `monotonicity_ok=true` in `runs/summary/grid.csv`.
Box credentials are removed at submission and compute-job boundaries. The
native child uses an explicit empty environment, and package/reference inputs
remain immutable evidence.

Static smoke checks:

```bash
bash -n slurm/h100/submit.sh
bash -n slurm/h100/campaign.sbatch
bash -n slurm/h100/smoke.sbatch
python -m pytest -q tests/test_h100_runtime.py
```

## 6. Return results

After all 32 recipe-matched markers exist:

```bash
python -m scripts.h100.reverse_results \
  --repo /path/to/repo \
  --runs-root /persistent/h100-runs \
  --campaign-manifest /persistent/h100-runs/.h100/campaign_manifest.json \
  --output /path/to/outgoing/xview3-h100-results-FULL40CHARSHA-IDENTITY64HEX \
  --max-part-bytes "$H100_MAX_PART_BYTES"
```

This delegates to `scripts.handoff build-results`, which includes readiness,
runtime, projection, venv receipt, campaign provenance, and validated
cell outputs in the shared handoff format.
