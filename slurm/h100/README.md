# H100 strict-FP32 Slurm lane

This lane changes execution hardware only. Every core cell still calls the
unchanged `src.train.finetune` entrypoint with shared `32-true`, micro-batch
16, accumulation 1, and effective batch 16. No script here stops, signals, or
otherwise mutates the live V100 campaign.

## 1. Verify and stage the handoff package

The package is owned by `scripts.handoff`; this lane does not define another
archive format. On the target, first verify the out-of-band hashes for
`manifest.json`, `SHA256SUMS`, `READY.json`, and `code/xview3.bundle`. Clone
the bundle, then use:

```bash
python3 -m scripts.handoff verify --package-root /path/to/package
python3 -m scripts.handoff extract \
  --package-root /path/to/package \
  --destination /an/empty/staging/directory
```

The batch job repeats those checks, clones to `$SLURM_TMPDIR/repo`, extracts
to the separate empty `$SLURM_TMPDIR/payload`, and links only ignored
`data/chips`, `data/raw`, and `data/weights` into the clone. Committed frozen
data JSON files are never replaced.

## 2. Build the persistent SIF

Use the package's verified offline wheelhouse:

```bash
python -m scripts.h100.build_container \
  --repo /path/to/cloned/repo \
  --wheelhouse /path/to/extracted/environment/wheelhouse \
  --output /persistent/path/xview3-h100-fp32.sif
```

The definition pins Python 3.11.15 by OCI digest. The build performs a
separate digest-pinned OCI access check, installs packages with `--no-index`,
uses `apptainer build --fakeroot`, compares the normalized SIF freeze against
the exact environment lock, and writes atomic `.sha256` and `.build.json`
receipts.

## 3. Configure the site

Copy `site.env.example` to ignored `site.env` and fill every hash/path.
Keep H100 runs in a persistent root distinct from V100 runs. Account,
partition, reservation, log directory, mail, project name, reference paths,
Box JWT path/folder, and the preflight result-part limit are site interfaces.

## 4. Slurm smoke, acceptance, and cutover

Run the lightweight one-GPU Slurm acceptance first:

```bash
slurm/h100/submit.sh smoke
```

The first allocation handles a real `SIGUSR1`, writes and atomically promotes
a synthetic HPC checkpoint, then exits with the reserved host-requeue code.
The outer batch authorizes and performs exactly one real `scontrol requeue`;
the requeued allocation resumes the same synthetic cell from `last.ckpt` and
writes `runs/.h100/slurm-smoke/SLURM_SMOKE_READY.json`. That receipt binds the
full source SHA, detector, SIF/build receipt, and package control hashes.
Acceptance and cutover reject an absent, stale, or modified smoke receipt.

```bash
slurm/h100/submit.sh acceptance
```

Acceptance requires eight identical H100s at CC 9.0, at least 500 GB
(500,000,000,000 bytes) of scratch before
extraction, exact source/frozen/SIF/package hashes, all tests and six real
checkpoint loads, strict IEEE FP32 in child processes, both backbone probes,
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
and only the outer host batch calls real `scontrol`; the slim SIF never does.

After training, the per-GPU wrapper runs the existing frozen test-split scorer.
A cell is complete only with finite test metrics, both checkpoints, and fully
bound runtime provenance. Campaign completion additionally requires exactly
32 `test_f1` rows and `monotonicity_ok=true` in `runs/summary/grid.csv`.
Box credentials are removed at submission and compute-job boundaries,
Apptainer uses `--cleanenv --containall`, and package/reference inputs are
read-only.

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
runtime, projection, container receipt, campaign provenance, and validated
cell outputs in the shared handoff format.
