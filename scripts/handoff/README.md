# Sprint 7d H100 Box handoff

These commands build and transfer the core-only, full-SHA-addressed handoff.
They never read Box credentials from arguments or repository files. A real
wheelhouse/package build fails closed unless the Box JWT preflight reports at
least 500 GB (500,000,000,000 bytes) free and a positive maximum file size.

The bootstrap host requires Python 3.11.15, `git`, and `zstd`. Create the
transfer environment separately from training and install only the exact,
read-only `requirements-transfer.txt` contract (`boxsdk[jwt]==4.13.0`,
`packaging==26.2`, `PyYAML==6.0.3`, and `pytest==9.1.1`):

```bash
python3 -m venv /outside/repo/xview3-box-transfer
/outside/repo/xview3-box-transfer/bin/pip install -r requirements-transfer.txt
chmod 0600 /outside/repo/box-jwt.json
export BOX_JWT_CONFIG=/outside/repo/box-jwt.json
export BOX_FOLDER_ID=REPLACE_WITH_DEDICATED_BOX_FOLDER_ID
```

On the source host, run the quota gate before any expensive work, resolve the
exact Python 3.11.15/cu126 lock, build, verify, and upload:

```bash
export XVIEW3_SOURCE_DATA_ROOT=/path/to/live/source-checkout/data
python -m scripts.handoff preflight --repo "$PWD"
python -m scripts.handoff wheelhouse \
  --repo "$PWD" --python /path/to/python3.11.15 \
  --output /outside/repo/wheelhouse-cu126
python -m scripts.handoff build \
  --repo "$PWD" --data-root "$XVIEW3_SOURCE_DATA_ROOT" \
  --wheelhouse /outside/repo/wheelhouse-cu126 \
  --apptainer-definition "$PWD/containers/h100-strict-fp32.def" \
  --output-dir /outside/repo/handoff
python -m scripts.handoff verify \
  --package-root /outside/repo/handoff/xview3-h100-fp32-FULL40CHARSHA
python -m scripts.handoff upload --repo "$PWD" \
  --package-root /outside/repo/handoff/xview3-h100-fp32-FULL40CHARSHA \
  --receipt /outside/repo/receipts/xview3-h100-fp32-FULL40CHARSHA.upload.json
```

The wheelhouse resolver uses binary-only, no-dependency downloads because the
lock already enumerates every non-bootstrap distribution. `pip`, `setuptools`,
and `wheel` come only from the digest-pinned Python base and are the sole
normalized-freeze extras accepted on the H100 target.

`--repo` must be the clean isolated Sprint-7d worktree. `--data-root` may be
the live V100 checkout's untracked data tree (or an equivalent read-only
source) so the branch worktree does not need data duplication; the builder
uses an exact allowlist and never writes that source tree.

The target acceptance suite is deliberately split across its two exact
environments. The Python 3.11.15 transfer environment runs the git/zstd/Box
slice, `python -m pytest -q tests/test_h100_handoff.py
tests/test_experiment_manifest.py`, and records a source-receipt-bound
host-test receipt and log. The second file stays on the host because its legacy
runner checks invoke `git`, which is intentionally absent from the slim SIF.
The verified SIF then runs `python -m pytest -q
--ignore=tests/test_h100_handoff.py
--ignore=tests/test_experiment_manifest.py`. Acceptance combines the two
non-overlapping slices in `runs/.h100/PYTEST_ACCEPTANCE.json` and requires
their aggregate coverage to equal the entire repository test suite; neither
slice alone is target acceptance.

Production build accepts only the committed
`containers/h100-strict-fp32.def`; a byte-identical copy at another path is
rejected. Box preflight proves upload, update, rename, and delete on a
disposable child file and verifies cleanup. It intentionally does not require
`can_delete` on the collaborated root itself: Box reports that value as false
for co-owners because only the owner can delete the shared root. The probe
refuses to run beside a published `READY.json`. Files larger than the literal
50,000,000-byte boundary use chunked upload.

On the H100 host, use an outside-repository destination that does not yet
exist. Download is built and fully verified in a private sibling staging
directory, then the complete tree is atomically renamed into place; an
incomplete final root can therefore never expose `READY.json`. Extraction
streams verified split parts into `DEST/data`, `DEST/environment`, and
`DEST/code/xview3.bundle`; the Slurm pack clones the bundle and symlinks the
individual data roots:

```bash
python -m scripts.handoff download --repo /path/to/bootstrap/repo \
  --package-root /scratch/xview3-h100-fp32-FULL40CHARSHA \
  --expected-ready-sha256 READY64HEX \
  --expected-manifest-sha256 MANIFEST64HEX \
  --expected-sha256sums-sha256 SHA256SUMS64HEX \
  --expected-package-id xview3-h100-fp32-FULL40CHARSHA
python -m scripts.handoff verify \
  --package-root /scratch/xview3-h100-fp32-FULL40CHARSHA
python -m scripts.handoff extract \
  --package-root /scratch/xview3-h100-fp32-FULL40CHARSHA \
  --destination "$SLURM_TMPDIR/xview3-payload"
```

After all 32 H100 cells have valid completion markers, copy the container build
receipt to `runs/.h100/container_build.json`, build the reverse package, then
use the same verified upload command:

```bash
python -m scripts.handoff build-results \
  --repo "$PWD" --runs-root /persistent/runs \
  --campaign-manifest /persistent/runs/.h100/campaign_manifest.json \
  --output /outside/repo/xview3-h100-results-FULL40CHARSHA-IDENTITY64HEX \
  --max-part-bytes BYTES_FROM_BOX_PREFLIGHT
python -m scripts.handoff upload --repo "$PWD" \
  --package-root /outside/repo/xview3-h100-results-FULL40CHARSHA-IDENTITY64HEX \
  --receipt /outside/repo/receipts/xview3-h100-results-FULL40CHARSHA-IDENTITY64HEX.upload.json
```

`READY.json` is written and uploaded last. If its upload response is
ambiguous, or final remote verification fails, the uploader removes any
remote `READY.json` before returning nonzero. The atomic upload receipt binds
the package ID, the READY/manifest/SHA256SUMS SHA-256 values, remote file
count/bytes, and UTC; it never records the Box folder ID. Preserve those
control hashes through a trusted out-of-band channel for target receipt.
Downloads use `.partial` files within their private staging tree and publish
only after size, Box SHA-1, package SHA-256, production-contract, bundled
source, and archive verification.
The JWT path must remain outside the repository at mode `0600`; it is never
printed, archived, or written into a manifest.

The public `verify`, `download`, and `extract` commands accept only
`contract.production: true` packages and enforce the full source/count/base/
checkpoint gates. Non-production packages exist only in tests and are
accepted through underscored internal fixture helpers; they are not a target
handoff interface.

Use one dedicated, initially empty `BOX_FOLDER_ID` for each forward or reverse
package; never mix two package trees in one Box folder. Large-file resume uses
the Box SDK upload session within one running process. Across process restarts,
already completed files are SHA-1/size verified and skipped, while an
interrupted file starts a fresh chunk session.

Status: deterministic fixture build/Box round-trip/extraction and the complete
source-host handoff test slice are implemented and green. The real
approximately 474 GB uncompressed source package, exact wheelhouse, Box quota
query/upload/receipt, target SIF build, H100 tests and numerical probes,
throughput decision, and manual cutover remain pending.
