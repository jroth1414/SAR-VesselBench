#!/usr/bin/env bash
#SBATCH --job-name=xview3-h100-fp32
#SBATCH --account=geofam
#SBATCH --partition=minor-use-case
#SBATCH --reservation=geofam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=1-12:30:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue

set -euo pipefail

# This must precede Python, nvidia-smi, or any CUDA import.
export NVIDIA_TF32_OVERRIDE=0

H100_MODE="${1:?submit.sh must pass acceptance or campaign mode}"
compute_site="${2:?submit.sh must pass the immutable compute-site snapshot}"
compute_site_sha256="${3:?submit.sh must pass the compute-site SHA-256}"
if [[ "$H100_MODE" != "acceptance" && "$H100_MODE" != "campaign" ]]; then
  echo "campaign batch received invalid mode: $H100_MODE" >&2
  exit 2
fi
if [[ ! "$compute_site_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "compute-site SHA-256 must be lowercase 64-hex" >&2
  exit 2
fi
if [[ -L "$compute_site" || ! -f "$compute_site" || "$(stat -c '%a' "$compute_site")" != "444" ]]; then
  echo "compute-site snapshot must be a read-only regular non-symlink file" >&2
  exit 2
fi
actual_compute_site_sha256="$(sha256sum "$compute_site" | awk '{print $1}')"
if [[ "$actual_compute_site_sha256" != "$compute_site_sha256" ]]; then
  echo "compute-site snapshot SHA-256 mismatch" >&2
  exit 2
fi
readonly H100_MODE compute_site compute_site_sha256
# shellcheck disable=SC1090
source "$compute_site"
if [[ "${H100_V100_CONTROL_PLANE:-}" != "box-transfer-v1" ]]; then
  echo "compute-site V100 control plane must be box-transfer-v1" >&2
  exit 2
fi
# Transfer credentials are never needed by a compute allocation.
unset BOX_JWT_CONFIG BOX_FOLDER_ID
if [[ "$H100_BASE_PYTHON_LIB_DIR" != /* ||
      ! -d "$H100_BASE_PYTHON_LIB_DIR" ||
      "$(realpath -e -- "$H100_BASE_PYTHON_LIB_DIR")" != "$H100_BASE_PYTHON_LIB_DIR" ||
      ! -r "$H100_BASE_PYTHON_LIB_DIR/libpython3.11.so.1.0" ]]; then
  echo "snapshotted H100_BASE_PYTHON_LIB_DIR is not a canonical Judy libpython directory" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"
readonly H100_BASE_PYTHON_LIB_DIR LD_LIBRARY_PATH

# Repeat the submission-time Judy write-root isolation check from the verified
# snapshot before the allocation writes campaign metadata. This protects
# queued jobs and every requeue independently of the original mutable site.env.
canonical_write_root() {
  local name="$1"
  local raw="${!name}"
  local canonical
  if [[ "$raw" != /* ]]; then
    echo "$name must be an absolute path" >&2
    return 2
  fi
  canonical="$(realpath -m -- "$raw")"
  if [[ "$canonical" == "/" ]]; then
    echo "$name must not resolve to /" >&2
    return 2
  fi
  printf '%s\n' "$canonical"
}

assert_disjoint_from_protected_root() {
  local name="$1"
  local candidate="$2"
  local protected_name="$3"
  local protected_root="$4"
  if [[ "$candidate" == "$protected_root" ||
        "$candidate" == "$protected_root/"* ||
        "$protected_root" == "$candidate/"* ]]; then
    echo "$name overlaps $protected_name $protected_root: $candidate" >&2
    return 2
  fi
}

h100_runs_root="$(canonical_write_root H100_RUNS_ROOT)"
h100_job_log_root="$(canonical_write_root H100_JOB_LOG_DIR)"
assert_disjoint_from_protected_root \
  H100_JOB_LOG_DIR "$h100_job_log_root" "H100 runs root" "$h100_runs_root"
for protected_name in \
  H100_PROJECT_ROOT H100_BASE_PACKAGE_ROOT H100_RUNTIME_PACKAGE_ROOT \
  H100_WHEELHOUSE H100_VENV_ROOT
do
  protected_value="${!protected_name}"
  if [[ "$protected_value" != /* ]]; then
    echo "$protected_name must be an absolute protected path" >&2
    exit 2
  fi
  protected_root="$(realpath -m -- "$protected_value")"
  assert_disjoint_from_protected_root \
    H100_RUNS_ROOT "$h100_runs_root" "$protected_name" "$protected_root"
  assert_disjoint_from_protected_root \
    H100_JOB_LOG_DIR "$h100_job_log_root" "$protected_name" "$protected_root"
done
H100_RUNS_ROOT="$h100_runs_root"
H100_JOB_LOG_DIR="$h100_job_log_root"
readonly H100_RUNS_ROOT H100_JOB_LOG_DIR

meta_root="$H100_RUNS_ROOT/.h100"
source_validation="$meta_root/SOURCE_VALIDATED.json"
if [[ "${H100_MODE:?H100_MODE missing}" == "campaign" ]]; then
  # Submission persisted byte-identical external evidence at these stable
  # paths. The allocation ignores external source paths from the verified
  # compute-site snapshot.
  H100_CUTOVER_READY="$meta_root/CUTOVER_READY.json"
  H100_V100_CORE_ARCHIVED="$meta_root/V100_CORE_ARCHIVED.json"
  H100_V100_ARCHIVE_MANIFEST="$meta_root/V100_CORE_ARCHIVE_MANIFEST.json"
fi

check_sha() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: expected $expected, got $actual" >&2
    exit 1
  fi
}

staging_started_epoch="$(date +%s)"
base_repo_bundle="$H100_BASE_PACKAGE_ROOT/code/xview3.bundle"
check_sha "$H100_BASE_REPO_BUNDLE_SHA256" "$base_repo_bundle"
check_sha "$H100_BASE_MANIFEST_SHA256" "$H100_BASE_PACKAGE_ROOT/manifest.json"
check_sha "$H100_BASE_READY_SHA256" "$H100_BASE_PACKAGE_ROOT/READY.json"
check_sha "$H100_BASE_SHA256SUMS_SHA256" "$H100_BASE_PACKAGE_ROOT/SHA256SUMS"
# source_validation runs the complete immutable Sprint 7d verifier below; do
# not add a redundant 294 GB checksum pass before that verifier.
check_sha "$H100_RUNTIME_BUNDLE_SHA256" "$H100_RUNTIME_BUNDLE"
check_sha "$H100_RUNTIME_MANIFEST_SHA256" "$H100_RUNTIME_PACKAGE_ROOT/manifest.json"
check_sha "$H100_RUNTIME_READY_SHA256" "$H100_RUNTIME_PACKAGE_ROOT/READY.json"
check_sha "$H100_RUNTIME_SHA256SUMS_SHA256" "$H100_RUNTIME_PACKAGE_ROOT/SHA256SUMS"
(cd "$H100_RUNTIME_PACKAGE_ROOT" && sha256sum --check SHA256SUMS)
check_sha "$H100_VENV_BUILD_SHA256" "$H100_VENV_BUILD_JSON"
check_sha "$H100_BASE_PYTHON_SHA256" "$H100_BASE_PYTHON"
if [[ "${H100_MODE:?H100_MODE missing}" == "campaign" ]]; then
  check_sha "$H100_CUTOVER_READY_SHA256" "$H100_CUTOVER_READY"
  check_sha "$H100_V100_CORE_ARCHIVED_SHA256" "$H100_V100_CORE_ARCHIVED"
  check_sha "$H100_V100_ARCHIVE_MANIFEST_SHA256" "$H100_V100_ARCHIVE_MANIFEST"
fi

scratch="${SLURM_TMPDIR:?SLURM_TMPDIR is required}"
scratch_free="$(df -PB1 "$scratch" | awk 'NR==2 {print $4}')"
if (( scratch_free < 500000000000 )); then
  echo "SLURM_TMPDIR has less than 500 GB free before extraction" >&2
  exit 1
fi
repo="$scratch/repo"
payload="$scratch/payload"
[[ ! -e "$repo" && ! -e "$payload" ]] || {
  echo "scratch repo/payload destinations must start empty" >&2
  exit 1
}
git clone "$H100_RUNTIME_BUNDLE" "$repo"
git -C "$repo" checkout --detach "$H100_EXPECTED_GIT_SHA"
check_sha "$H100_ENV_LOCK_SHA256" "$repo/locks/env-v100node.txt"
PYTHONNOUSERSITE=1 PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" \
  "$repo/scripts/h100/build_venv.py" verify \
  --repo "$repo" \
  --venv-root "$H100_VENV_ROOT" \
  --base-python "$H100_BASE_PYTHON" \
  --wheelhouse "$H100_WHEELHOUSE" \
  --base-extraction-receipt "$H100_BASE_EXTRACTION_RECEIPT" \
  --expected-venv-sha256 "$H100_VENV_SHA256" \
  --expected-receipt-sha256 "$H100_VENV_BUILD_SHA256" \
  --expected-base-python-sha256 "$H100_BASE_PYTHON_SHA256" \
  --expected-base-python-runtime-sha256 "$H100_BASE_PYTHON_RUNTIME_SHA256" \
  --expected-wheelhouse-sha256 "$H100_WHEELHOUSE_SHA256" \
  --expected-base-extraction-receipt-sha256 "$H100_BASE_EXTRACTION_RECEIPT_SHA256" \
  --expected-base-payload-package-id "$H100_BASE_PACKAGE_ID" \
  --expected-base-payload-manifest-sha256 "$H100_BASE_MANIFEST_SHA256"
mkdir -p "$meta_root/acceptance-logs"
frozen_args=(
  --frozen-sha256 "$H100_DETECTOR_SHA256"
  --frozen-sha256 "$H100_SCORER_SHA256"
  --frozen-sha256 "$H100_SPLITS_SHA256"
  --frozen-sha256 "$H100_STATS_SHA256"
  --frozen-sha256 "$H100_LSSSDD_SHA256"
)
(
  cd "$repo"
  PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" -m scripts.h100.source_validation \
    --repo "$repo" \
    --expected-git-sha "$H100_EXPECTED_GIT_SHA" \
    --base-payload-root "$H100_BASE_PACKAGE_ROOT" \
    --runtime-amendment-root "$H100_RUNTIME_PACKAGE_ROOT" \
    "${frozen_args[@]}" \
    --base-payload-package-id "$H100_BASE_PACKAGE_ID" \
    --base-payload-git-sha "$H100_BASE_GIT_SHA" \
    --base-payload-manifest-sha256 "$H100_BASE_MANIFEST_SHA256" \
    --base-payload-ready-sha256 "$H100_BASE_READY_SHA256" \
    --base-payload-sha256sums-sha256 "$H100_BASE_SHA256SUMS_SHA256" \
    --base-payload-repo-bundle-sha256 "$H100_BASE_REPO_BUNDLE_SHA256" \
    --runtime-amendment-package-id "$H100_RUNTIME_PACKAGE_ID" \
    --runtime-amendment-git-sha "$H100_RUNTIME_GIT_SHA" \
    --runtime-amendment-manifest-sha256 "$H100_RUNTIME_MANIFEST_SHA256" \
    --runtime-amendment-ready-sha256 "$H100_RUNTIME_READY_SHA256" \
    --runtime-amendment-sha256sums-sha256 "$H100_RUNTIME_SHA256SUMS_SHA256" \
    --runtime-amendment-bundle-sha256 "$H100_RUNTIME_BUNDLE_SHA256" \
    --output "$source_validation"
)
source_validation_sha256="$(sha256sum "$source_validation" | awk '{print $1}')"
host_test_receipt="$meta_root/HOST_HANDOFF_TESTS.json"
host_test_receipt_sha256=""
if [[ "${H100_MODE:?H100_MODE missing}" == "acceptance" ]]; then
  (
    cd "$repo"
    PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" -m scripts.h100.host_test_gate \
      --repo "$repo" \
      --log "$meta_root/acceptance-logs/pytest-handoff-host.log" \
      --receipt "$host_test_receipt" \
      --source-validation-sha256 "$source_validation_sha256"
  )
  host_test_receipt_sha256="$(sha256sum "$host_test_receipt" | awk '{print $1}')"
fi
# Handoff owns archive verification/extraction and requires an empty target.
# Run it on the host before entering the minimal native runtime environment.
# The persistent venv intentionally has neither transfer credentials nor
# authority to mutate either package.
(
  cd "$repo"
  PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" -m scripts.handoff extract \
    --package-root "$H100_BASE_PACKAGE_ROOT" \
    --destination "$payload"
)
staging_seconds="$(( $(date +%s) - staging_started_epoch ))"

# Preserve committed frozen data files; link only ignored payload subtrees.
for relative in chips raw weights; do
  [[ ! -e "$repo/data/$relative" ]] || {
    echo "refusing to replace existing repo/data/$relative" >&2
    exit 1
  }
  [[ -e "$payload/data/$relative" ]] || {
    echo "payload is missing data/$relative" >&2
    exit 1
  }
  ln -s "$payload/data/$relative" "$repo/data/$relative"
done
[[ ! -e "$repo/runs" ]] || {
  echo "fresh clone unexpectedly contains runs" >&2
  exit 1
}
mkdir -p "$H100_RUNS_ROOT"
ln -s "$H100_RUNS_ROOT" "$repo/runs"
mkdir -p "$meta_root"
receipt_tmp="$meta_root/.venv_build.json.$$"
cp "$H100_VENV_BUILD_JSON" "$receipt_tmp"
check_sha "$H100_VENV_BUILD_SHA256" "$receipt_tmp"
mv "$receipt_tmp" "$meta_root/venv_build.json"

# Execute the exact verified venv interpreter without activation or inherited
# login/user-site state.  Only the CUDA/Slurm values needed by the native
# runtime cross this boundary; Box credentials and submission-host state do
# not.  Cache and HOME writes are confined to allocation scratch.
runtime_home="$scratch/runtime-home"
mkdir -m 700 "$runtime_home"
mkdir -m 700 "$runtime_home/cache" "$runtime_home/torch" "$runtime_home/triton"
native_env=(
  /usr/bin/env -i
  "HOME=$runtime_home"
  "XDG_CACHE_HOME=$runtime_home/cache"
  "TORCH_HOME=$runtime_home/torch"
  "TRITON_CACHE_DIR=$runtime_home/triton"
  "TMPDIR=$scratch"
  "PATH=$H100_VENV_ROOT/bin:/usr/local/bin:/usr/bin:/bin"
  "PYTHONPATH=$repo/scripts/h100:$repo"
  "PYTHONNOUSERSITE=1"
  "PYTHONDONTWRITEBYTECODE=1"
  "PIP_NO_INDEX=1"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "NVIDIA_TF32_OVERRIDE=0"
  "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"
  "LD_LIBRARY_PATH=$H100_BASE_PYTHON_LIB_DIR"
  "SLURM_JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
  "SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
  "SLURM_CLUSTER_NAME=${SLURM_CLUSTER_NAME:-unknown}"
  "SLURM_NTASKS=${SLURM_NTASKS:-1}"
  "SLURM_PROCID=${SLURM_PROCID:-0}"
  "SLURM_LOCALID=${SLURM_LOCALID:-0}"
  "SLURM_NODEID=${SLURM_NODEID:-0}"
  "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-8}"
  "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-48}"
  "SLURMD_NODENAME=${SLURMD_NODENAME:-unknown}"
  "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unknown}"
)
run_venv() {
  "${native_env[@]}" "$H100_VENV_ROOT/bin/python" "$@"
}

if [[ "${H100_MODE:?H100_MODE missing}" == "acceptance" ]]; then
  (
    cd "$repo"
    run_venv -m scripts.h100.acceptance \
      --repo "$repo" \
      --runs-root "$H100_RUNS_ROOT" \
      --scratch "$scratch" \
      --scratch-free-before-extraction "$scratch_free" \
      --ready "$H100_RUNS_ROOT/.h100/H100_READY.json" \
      --smoke-ready "$H100_RUNS_ROOT/.h100/slurm-smoke/SLURM_SMOKE_READY.json" \
      --venv-root "$H100_VENV_ROOT" \
      --venv-sha256 "$H100_VENV_SHA256" \
      --venv-build-sha256 "$H100_VENV_BUILD_SHA256" \
      --base-python "$H100_BASE_PYTHON" \
      --base-python-sha256 "$H100_BASE_PYTHON_SHA256" \
      --base-python-runtime-sha256 "$H100_BASE_PYTHON_RUNTIME_SHA256" \
      --wheelhouse "$H100_WHEELHOUSE" \
      --wheelhouse-sha256 "$H100_WHEELHOUSE_SHA256" \
      --base-extraction-receipt "$H100_BASE_EXTRACTION_RECEIPT" \
      --base-extraction-receipt-sha256 "$H100_BASE_EXTRACTION_RECEIPT_SHA256" \
      --expected-git-sha "$H100_EXPECTED_GIT_SHA" \
      --source-validation-json "$source_validation" \
      --source-validation-sha256 "$source_validation_sha256" \
      --host-test-receipt "$host_test_receipt" \
      --host-test-receipt-sha256 "$host_test_receipt_sha256" \
      "${frozen_args[@]}" \
      --base-payload-package-id "$H100_BASE_PACKAGE_ID" \
      --base-payload-git-sha "$H100_BASE_GIT_SHA" \
      --base-payload-manifest-sha256 "$H100_BASE_MANIFEST_SHA256" \
      --base-payload-ready-sha256 "$H100_BASE_READY_SHA256" \
      --base-payload-sha256sums-sha256 "$H100_BASE_SHA256SUMS_SHA256" \
      --base-payload-repo-bundle-sha256 "$H100_BASE_REPO_BUNDLE_SHA256" \
      --runtime-amendment-package-id "$H100_RUNTIME_PACKAGE_ID" \
      --runtime-amendment-git-sha "$H100_RUNTIME_GIT_SHA" \
      --runtime-amendment-manifest-sha256 "$H100_RUNTIME_MANIFEST_SHA256" \
      --runtime-amendment-ready-sha256 "$H100_RUNTIME_READY_SHA256" \
      --runtime-amendment-sha256sums-sha256 "$H100_RUNTIME_SHA256SUMS_SHA256" \
      --runtime-amendment-bundle-sha256 "$H100_RUNTIME_BUNDLE_SHA256" \
      --remaining-v100-wall-hours "$H100_REMAINING_V100_WALL_HOURS" \
      --staging-seconds "$staging_seconds"
  )
  exit 0
fi

allocation_hardware="$H100_RUNS_ROOT/.h100/h100_runtime-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}.json"
allocation_hardware_tmp="${allocation_hardware}.tmp.$$"
(
  cd "$repo"
  run_venv -m scripts.h100.strict_fp32_probe --expected-gpus 8 \
    > "$allocation_hardware_tmp"
)
mv "$allocation_hardware_tmp" "$allocation_hardware"

(
  cd "$repo"
  run_venv -m scripts.h100.operator_cutover \
    --cutover-ready "$H100_CUTOVER_READY" \
    --cutover-ready-sha256 "$H100_CUTOVER_READY_SHA256" \
    --receipt "$H100_V100_CORE_ARCHIVED" \
    --receipt-sha256 "$H100_V100_CORE_ARCHIVED_SHA256" \
    --archive-manifest "$H100_V100_ARCHIVE_MANIFEST" \
    --archive-manifest-sha256 "$H100_V100_ARCHIVE_MANIFEST_SHA256" \
    --expected-h100-git-sha "$H100_EXPECTED_GIT_SHA" \
    --expected-venv-sha256 "$H100_VENV_SHA256" \
    --expected-base-payload-package-id "$H100_BASE_PACKAGE_ID" \
    --expected-base-payload-git-sha "$H100_BASE_GIT_SHA" \
    --expected-base-payload-manifest-sha256 "$H100_BASE_MANIFEST_SHA256" \
    --expected-base-payload-ready-sha256 "$H100_BASE_READY_SHA256" \
    --expected-base-payload-sha256sums-sha256 "$H100_BASE_SHA256SUMS_SHA256" \
    --expected-base-payload-repo-bundle-sha256 "$H100_BASE_REPO_BUNDLE_SHA256" \
    --expected-runtime-amendment-package-id "$H100_RUNTIME_PACKAGE_ID" \
    --expected-runtime-amendment-git-sha "$H100_RUNTIME_GIT_SHA" \
    --expected-runtime-amendment-manifest-sha256 "$H100_RUNTIME_MANIFEST_SHA256" \
    --expected-runtime-amendment-ready-sha256 "$H100_RUNTIME_READY_SHA256" \
    --expected-runtime-amendment-sha256sums-sha256 "$H100_RUNTIME_SHA256SUMS_SHA256" \
    --expected-runtime-amendment-bundle-sha256 "$H100_RUNTIME_BUNDLE_SHA256" \
    --expected-reference-git-sha "$H100_EXPECTED_REFERENCE_GIT_SHA" \
    --expected-reference-campaign-id "$H100_REFERENCE_CAMPAIGN_ID"
)

controller_pid=""
forward_usr1() {
  if [[ -n "$controller_pid" ]]; then
    kill -USR1 "$controller_pid" 2>/dev/null || true
  fi
}
trap forward_usr1 USR1

cd "$repo"
"${native_env[@]}" "$H100_VENV_ROOT/bin/python" -m scripts.h100.campaign \
    --repo "$repo" \
    --runs-root "$H100_RUNS_ROOT" \
    --campaign-id "$H100_CAMPAIGN_ID" \
    --expected-git-sha "$H100_EXPECTED_GIT_SHA" \
    --venv-root "$H100_VENV_ROOT" \
    --venv-sha256 "$H100_VENV_SHA256" \
    --venv-build-sha256 "$H100_VENV_BUILD_SHA256" \
    --hardware-json "$allocation_hardware" \
    --acceptance-json "$H100_RUNS_ROOT/.h100/H100_READY.json" \
    --source-validation-json "$source_validation" \
    --source-validation-sha256 "$source_validation_sha256" \
    --cutover-ready-json "$H100_CUTOVER_READY" \
    --cutover-ready-sha256 "$H100_CUTOVER_READY_SHA256" \
    --v100-core-archived-json "$H100_V100_CORE_ARCHIVED" \
    --v100-core-archived-sha256 "$H100_V100_CORE_ARCHIVED_SHA256" \
    --v100-archive-manifest-json "$H100_V100_ARCHIVE_MANIFEST" \
    --v100-archive-manifest-sha256 "$H100_V100_ARCHIVE_MANIFEST_SHA256" \
    --workers-per-gpu 6 &
controller_pid="$!"
controller_rc=0
while true; do
  set +e
  wait "$controller_pid"
  controller_rc="$?"
  set -e
  if kill -0 "$controller_pid" 2>/dev/null; then
    continue
  fi
  break
done
if [[ "$controller_rc" -eq 75 ]]; then
  # The native child never receives scontrol.  This host call occurs only after every
  # running cell has a promoted last.ckpt and the manifest requests requeue.
  "${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$controller_rc"
