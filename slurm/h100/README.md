# H100 strict-FP32 Slurm lane

This lane changes execution hardware only. Every core cell still calls the
unchanged `src.train.finetune` entrypoint with shared `32-true`, micro-batch
16, accumulation 1, and effective batch 16. No script here stops, signals,
pauses, reconfigures, or otherwise mutates the live V100 campaign. Its outputs
remain non-reportable diagnostics and can never suppress, resume, or satisfy
an H100 cell.

## 1. Prepare the two-layer Box handoff outside allocations

Keep the completed Sprint 7d data/wheelhouse payload immutable. The smaller
Sprint 7f evaluation-contract runtime amendment is a separate schema-2
content-addressed package containing one Git bundle, the deterministic audited
13,911-row TRAIN+fixed-DEV8 CSV, and package controls, with no TEST/eval-final
row or imagery.

The generated hash-pinned standalone runtime bootstrap is the mandatory Judy
download/clone path. Verify its SHA-256 against the trusted source-side result,
then run it from the Judy login/setup environment before submitting any H100
allocation:

```bash
source /outside/repo/sprint7f-runtime-transfer.env
export TRANSFER_PYTHON=/path/to/xview3-box-transfer/bin/python
export XVIEW3_TARGET_ROOT=/persistent/path/xview3-handoff
export BOX_JWT_CONFIG=/outside/repo/xview3-jwt.json
export BOX_FOLDER_ID=${BOX_FOLDER_ID:?provided by the external environment}
sha256sum -- /outside/repo/pull_runtime_amendment.sh
# Compare that digest with the trusted build-runtime-bootstrap result.
chmod 700 -- /outside/repo/pull_runtime_amendment.sh
/outside/repo/pull_runtime_amendment.sh
```

The script checks the exact Box tree, Box SHA-1 and size, and every pinned
SHA-256 before it atomically publishes the runtime package, reconstructed Git
bundle, and clean SHA-addressed checkout. Point `H100_RUNTIME_PACKAGE_ROOT`
and `H100_RUNTIME_BUNDLE` at the printed package and bundle paths.

The Sprint 7d base is fully verified and extracted once during source/setup
staging for data and venv construction; that pass is never run by a compute
allocation. General `download-runtime`, `verify-runtime`, and
`extract-runtime` commands invoke the full base verifier, are source/setup-only,
and are forbidden inside H100 smoke, acceptance, campaign, or any other
pre-cohort allocation. The historical bundle inside the base payload is
independently rehashed but never cloned for H100. Only the Sprint 7f runtime
amendment bundle is cloned to `$SLURM_TMPDIR/repo`.

Each batch allocation verifies the pinned base control files/Git bundle and
the small runtime amendment, then builds an empty phase-specific
`$SLURM_TMPDIR/payload`. Acceptance and training views contain 111 TRAIN chips,
fixed DEV8 rasters, six weights, and the amendment's filtered TRAIN+DEV8 labels;
acceptance alone includes the offline wheelhouse. They contain no TEST asset.
After all 32 training markers freeze, the controller exits 75. The requeued
allocation validates that cohort before hashing/extracting 16 TEST rasters and
the combined label archive, filters labels to TRAIN+TEST, exposes no chip, and
then scores. Committed frozen JSON is never replaced. Both transfer identities
and the phase-view receipt remain bound in readiness/campaign provenance.

This protects the canonical allocation view, not the whole Judy account. The
immutable base package is already plaintext-readable by the same Unix user;
arbitrary same-UID code outside the approved launcher is outside this
isolation boundary.

## 2. Build the persistent native venv

Provide Judy's exact Python 3.11.13 installation and canonical libpython
directory, record the SHA-256 of its resolved executable, and build directly
at the final persistent path from the base payload's verified offline
wheelhouse. The explicit loader path is required because Slurm submits with
`--export=NONE`:

```bash
export H100_BASE_PYTHON_LIB_DIR=/cm/shared/mitre-apps/python/3.11.13/build/lib
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"
/path/to/requirements-transfer-python -m scripts.h100.build_venv build \
  --repo /path/to/cloned/repo \
  --wheelhouse /path/to/extracted/environment/wheelhouse \
  --base-extraction-receipt /path/to/extracted/HANDOFF_EXTRACTED.json \
  --expected-base-payload-package-id xview3-h100-fp32-2726199efcebbebc89156e708b89df2a3415468a \
  --expected-base-payload-manifest-sha256 fccb0b505c89836a148afec709bb799f7af4908d955ea1b142e153154d830896 \
  --base-python /cm/shared/mitre-apps/python/3.11.13/build/bin/python3.11 \
  --output /persistent/venvs/xview3-h100-fp32
```

The builder uses `venv --copies`, installs only from the wheelhouse with
`pip --no-index`, checks the normalized freeze against
`locks/env-v100node.txt`, removes bytecode, seals the tree read-only, and
writes `<venv>.sha256` plus `<venv>.build.json`; the build JSON is published
last. The venv is non-relocatable. Every allocation rehashes and verifies the
complete tree, receipt, environment lock, `pip check`, wheelhouse tree,
canonical base-extraction receipt, and full base-Python runtime closure before
running the exact `<venv>/bin/python` path. There is no activation.

The native process starts under `env -i` with an allocation-local home/cache,
offline package/model settings, `PYTHONNOUSERSITE=1`, and only enumerated CUDA
and Slurm values. `NVIDIA_TF32_OVERRIDE=0` is exported before any CUDA-capable
command. Box variables never enter the child process.

## 3. Configure the site

Copy `site.env.example` to ignored `site.env` and fill only the site and
content-identity fields required for the next mode. Box credentials do not
belong there: source `BOX_JWT_CONFIG` and `BOX_FOLDER_ID` from a separate,
untracked transfer environment only while running a Box command. Folder IDs
are runtime inputs and are never embedded in a package, receipt, bootstrap,
Slurm snapshot, or committed file.

Judy and the live V100 host have completely separate filesystems, so
`H100_V100_CONTROL_PLANE` must be exactly `box-transfer-v1`; a dummy Judy path
for the V100 runs tree is forbidden. Submission canonicalizes
`H100_RUNS_ROOT` and `H100_JOB_LOG_DIR`, then rejects equality or ancestor
overlap between them and the immutable repo, base/runtime packages,
wheelhouse, sealed venv, and the control package used by the selected mode.

Smoke and acceptance require no V100 or reference path. `cutover-check`
requires the verified corrected-R2/R3 references package and all seven exact
identity fields printed by `build-control`/`verify-control`: package root,
package ID, producer Git SHA, identity SHA-256, manifest SHA-256, READY
SHA-256, and SHA256SUMS SHA-256. `campaign` instead requires the same seven
fields for the verified diagnostic-isolation package and the exact
`CUTOVER_READY.json` SHA-256. Both control-package producer SHAs must equal the
Sprint 7f H100 source SHA. See `scripts/handoff/README.md` for the exact
control-package commands and closed binding sets.

Submission uses exact `--export=NONE`. The mode and immutable site snapshot
are positional batch arguments, not inherited environment variables.

## 4. Slurm smoke, acceptance, and external control barrier

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
`runs/.h100/slurm-smoke/SLURM_SMOKE_READY.json`. Acceptance and cutover reject
an absent, stale, or modified smoke receipt.

```bash
slurm/h100/submit.sh acceptance
```

Acceptance requires eight identical H100s at CC 9.0, at least 500 GB
(500,000,000,000 bytes) of scratch before extraction, exact
source/frozen/venv/dual-package hashes, the complete split test suite and six
real checkpoint loads, strict IEEE FP32 in child processes, both model-family
probes, the corrected evaluation-GT receipt, and a 200-step CNN projection
that conservatively beats the recorded remaining V100 wall time. It writes
`runs/.h100/H100_READY.json` without touching V100.

Next, use the JSON-only control-package commands documented in
`scripts/handoff/README.md`:

1. On the V100 side, package the immutable `REFERENCE_CAMPAIGN.json` plus
   corrected R2/R3 `final_metrics.json` and `runtime_provenance.json` as kind
   `references`; upload it to its own empty Box child folder.
2. On Judy, download and verify that package against every out-of-band
   identity value, fill the `H100_REFERENCES_PACKAGE_*` fields, refresh
   `H100_REMAINING_V100_WALL_HOURS`, and set
   `H100_CURRENT_V100_DIAGNOSTIC_STATUS`. Positive remaining time requires
   `continues-running-non-reportable-diagnostic`; exactly zero requires
   `complete-non-reportable-diagnostic`. Then run:

   ```bash
   slurm/h100/submit.sh cutover-check
   ```

   This writes `runs/.h100/CUTOVER_READY.json`; it submits no job and performs
   no V100 process action.
3. Package that exact marker as kind `cutover-ready` and transfer it to the
   V100 operator. The human operator verifies it, authors the schema-checked
   `V100_DIAGNOSTIC_ISOLATION.json`, and leaves the V100 campaign running
   unchanged as a non-reportable diagnostic.
4. Package the attestation as kind `diagnostic-isolation`, transfer it back to
   Judy, verify every out-of-band identity value, and fill the
   `H100_DIAGNOSTIC_ISOLATION_PACKAGE_*` fields plus the exact
   `H100_CUTOVER_READY_SHA256`.

Stopping or archiving V100 is optional after its diagnostic work; neither is
a prerequisite for H100 launch. The attestation instead proves that the V100
and H100 namespaces are physically disjoint, V100 outputs are non-reportable,
and no V100 completion/checkpoint can suppress or resume H100 work.

Launch only after that barrier is complete:

```bash
slurm/h100/submit.sh campaign
```

Campaign submission revalidates the exact diagnostic-isolation package,
persists canonical read-only copies of `CUTOVER_READY.json` and
`V100_DIAGNOSTIC_ISOLATION.json` beneath `runs/.h100`, removes transfer-package
paths and Box variables from the compute snapshot, and then submits the
32-cell H100 job.

## 5. Campaign, cohort barrier, preemption, and resume

The controller first launches all 32 training cells expensive-first (`f100`,
`f50`, `f25`, `f10`), one unchanged trainer per GPU. Every allocation repeats
the strict hardware probe and must match the accepted name, memory, CC,
driver, torch/CUDA, and IEEE backend class. A failure prevents new launches
while already running cells finish. Fifteen minutes before the Slurm limit,
USR1 is forwarded to every trainer; Lightning writes HPC checkpoints,
deferred requeue waits for all eight, and each checkpoint is promoted
atomically to `last.ckpt`.

No held-out test scoring is allowed during this training phase. Each of the 32
cells must first publish a valid schema-2 training marker binding its complete
best-dev operating point to the exact `best.ckpt`. Only after all 32 training
cells validate does the controller atomically publish the read-only,
content-bound `runs/.h100/TRAINING_COHORT.json`, return exit 75, and require the
host to requeue into a fresh allocation with a separate `score-test` view.

The scoring phase revalidates that immutable all-32 cohort and its SHA-256
before each cell applies the frozen best-dev threshold to the test split.
Each cell publishes a separate immutable `test_metrics.json`; it never
rewrites the training marker. Resume is phase-aware: a pre-cohort run can only
resume training, while post-cohort work must retain the same cohort SHA-256.
Campaign completion requires all 32 training markers, all 32 separate finite
test metrics, exactly 32 `test_f1` rows, and `monotonicity_ok=true` in
`runs/summary/grid.csv`. The once-only verified-final set is not read.

Static checks:

```bash
bash -n slurm/h100/submit.sh
bash -n slurm/h100/campaign.sbatch
bash -n slurm/h100/smoke.sbatch
python -m pytest -q tests/test_h100_runtime.py
```

## 6. Return results

After the complete 32-cell H100 cohort and its separate test results validate:

```bash
# In the external Box transfer venv, record maximum_file_bytes first:
"$H100_TRANSFER_PYTHON" -B -m scripts.handoff preflight --repo /path/to/repo

# In the sealed H100 venv (no Box credentials or boxsdk required):
"$H100_VENV_ROOT/bin/python" -I -B -m scripts.h100.reverse_results \
  --repo /path/to/repo \
  --runs-root /persistent/h100-runs \
  --campaign-manifest /persistent/h100-runs/.h100/campaign_manifest.json \
  --output-dir /path/to/outgoing \
  --max-part-bytes "$H100_MAX_PART_BYTES"
```

This delegates to `scripts.handoff build-results`. The reverse package binds
the corrected GT audit, immutable `TRAINING_COHORT.json`, 32 schema-2 training
markers, 32 separate `test_metrics.json` artifacts, configs, logs,
best/last checkpoints, strict-FP32 hardware/venv evidence, and all package and
control provenance. The generated directory name is the exact package ID and
binds the complete validated archive/member digest index. Record all printed
identity/control hashes out of band, then upload the exact directory through
its own initially empty Box child folder using `H100_TRANSFER_PYTHON`; source
Box credentials only from the external transfer environment. Standalone R2/R3
files are excluded, while their exact accepted metrics/provenance remain
embedded in the archived `CUTOVER_READY.json`.

At the receiving site, run `scripts.handoff download` with the exact package
ID and READY/manifest/SHA256SUMS hashes, then `scripts.handoff verify` and
`scripts.handoff extract` into a new destination. See
`scripts/handoff/README.md` for the complete commands.
