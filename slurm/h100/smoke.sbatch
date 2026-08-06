#!/usr/bin/env bash
#SBATCH --job-name=xview3-h100-slurm-smoke
#SBATCH --account=geofam
#SBATCH --partition=minor-use-case
# An optional reservation is supplied by submit.sh; there is no script default.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --requeue

set -euo pipefail

# This must precede Python, nvidia-smi, or any CUDA import.
export NVIDIA_TF32_OVERRIDE=0

H100_MODE="${1:?submit.sh must pass smoke mode}"
compute_site="${2:?submit.sh must pass the immutable compute-site snapshot}"
compute_site_sha256="${3:?submit.sh must pass the compute-site SHA-256}"
if [[ "$H100_MODE" != "smoke" ]]; then
  echo "smoke batch received invalid mode: $H100_MODE" >&2
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
# snapshot before the allocation writes smoke metadata. This protects queued
# jobs and every requeue independently of the original mutable site.env.
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
h100_scratch_root="$(canonical_write_root H100_SCRATCH_ROOT)"
assert_disjoint_from_protected_root \
  H100_JOB_LOG_DIR "$h100_job_log_root" "H100 runs root" "$h100_runs_root"
assert_disjoint_from_protected_root \
  H100_SCRATCH_ROOT "$h100_scratch_root" "H100 runs root" "$h100_runs_root"
assert_disjoint_from_protected_root \
  H100_SCRATCH_ROOT "$h100_scratch_root" H100_JOB_LOG_DIR "$h100_job_log_root"
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
  assert_disjoint_from_protected_root \
    H100_SCRATCH_ROOT "$h100_scratch_root" "$protected_name" "$protected_root"
done
H100_RUNS_ROOT="$h100_runs_root"
H100_JOB_LOG_DIR="$h100_job_log_root"
H100_SCRATCH_ROOT="$h100_scratch_root"
readonly H100_RUNS_ROOT H100_JOB_LOG_DIR H100_SCRATCH_ROOT

allocation_scratch=""
remove_allocation_scratch() {
  local actual=""
  local expected=""
  if [[ -z "$allocation_scratch" ]]; then
    return 0
  fi
  expected="$H100_SCRATCH_ROOT/xview3-${SLURM_JOB_ID:-invalid}-r${SLURM_RESTART_COUNT:-0}"
  if [[ ! -e "$allocation_scratch" && ! -L "$allocation_scratch" ]]; then
    allocation_scratch=""
    return 0
  fi
  actual="$(realpath -e -- "$allocation_scratch" 2>/dev/null || true)"
  if [[ "$allocation_scratch" != "$expected" ||
        "$actual" != "$allocation_scratch" ||
        -L "$allocation_scratch" ||
        ! -d "$allocation_scratch" ||
        "$(stat -c '%u' "$allocation_scratch" 2>/dev/null || true)" != "$(id -u)" ]]; then
    echo "refusing unsafe allocation-scratch cleanup: $allocation_scratch" >&2
    return 1
  fi
  find "$allocation_scratch" -xdev -depth -mindepth 1 -delete || return 1
  rmdir -- "$allocation_scratch" || return 1
  allocation_scratch=""
}

cleanup_allocation_scratch() {
  local original_rc="$?"
  trap - EXIT
  if ! remove_allocation_scratch && (( original_rc == 0 )); then
    original_rc=1
  fi
  exit "$original_rc"
}

cleanup_allocation_scratch_before_requeue() {
  if ! remove_allocation_scratch; then
    echo "allocation scratch cleanup failed before requeue" >&2
    return 1
  fi
  trap - EXIT
}

create_allocation_scratch() {
  local job_id="${SLURM_JOB_ID:-}"
  local restart_count="${SLURM_RESTART_COUNT:-0}"
  local expected
  local probe
  if [[ ! "$job_id" =~ ^[1-9][0-9]*$ ||
        ! "$restart_count" =~ ^[0-9]+$ ]]; then
    echo "Slurm job/restart identity is invalid for allocation scratch" >&2
    return 2
  fi
  if [[ -L "$H100_SCRATCH_ROOT" ||
        ! -d "$H100_SCRATCH_ROOT" ||
        "$(realpath -e -- "$H100_SCRATCH_ROOT" 2>/dev/null || true)" != "$H100_SCRATCH_ROOT" ||
        ! -w "$H100_SCRATCH_ROOT" || ! -x "$H100_SCRATCH_ROOT" ]]; then
    echo "H100_SCRATCH_ROOT must be a canonical writable non-symlink directory on the compute node" >&2
    return 2
  fi
  expected="$H100_SCRATCH_ROOT/xview3-${job_id}-r${restart_count}"
  if [[ -e "$expected" || -L "$expected" ]]; then
    echo "allocation scratch destination must start absent: $expected" >&2
    return 1
  fi
  allocation_scratch="$expected"
  trap cleanup_allocation_scratch EXIT
  mkdir -m 700 -- "$allocation_scratch"
  chmod 0700 -- "$allocation_scratch"
  if [[ -L "$allocation_scratch" ||
        ! -d "$allocation_scratch" ||
        "$(realpath -e -- "$allocation_scratch")" != "$allocation_scratch" ||
        "$(stat -c '%a' "$allocation_scratch")" != "700" ||
        "$(stat -c '%u' "$allocation_scratch")" != "$(id -u)" ]]; then
    echo "allocation scratch directory failed ownership/mode validation" >&2
    return 1
  fi
  probe="$allocation_scratch/.write-probe"
  (umask 077; : > "$probe")
  if [[ -L "$probe" || ! -f "$probe" || "$(stat -c '%a' "$probe")" != "600" ]]; then
    echo "allocation scratch write probe failed" >&2
    return 1
  fi
  rm -f -- "$probe"
  printf 'allocation_scratch=%s\n' "$allocation_scratch"
}

create_allocation_scratch
scratch="$allocation_scratch"
readonly scratch

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

base_repo_bundle="$H100_BASE_PACKAGE_ROOT/code/xview3.bundle"
check_sha "$H100_BASE_REPO_BUNDLE_SHA256" "$base_repo_bundle"
check_sha "$H100_BASE_MANIFEST_SHA256" "$H100_BASE_PACKAGE_ROOT/manifest.json"
check_sha "$H100_BASE_READY_SHA256" "$H100_BASE_PACKAGE_ROOT/READY.json"
check_sha "$H100_BASE_SHA256SUMS_SHA256" "$H100_BASE_PACKAGE_ROOT/SHA256SUMS"
check_sha "$H100_RUNTIME_BUNDLE_SHA256" "$H100_RUNTIME_BUNDLE"
check_sha "$H100_RUNTIME_MANIFEST_SHA256" "$H100_RUNTIME_PACKAGE_ROOT/manifest.json"
check_sha "$H100_RUNTIME_READY_SHA256" "$H100_RUNTIME_PACKAGE_ROOT/READY.json"
check_sha "$H100_RUNTIME_SHA256SUMS_SHA256" "$H100_RUNTIME_PACKAGE_ROOT/SHA256SUMS"
(cd "$H100_RUNTIME_PACKAGE_ROOT" && sha256sum --check SHA256SUMS)
check_sha "$H100_VENV_BUILD_SHA256" "$H100_VENV_BUILD_JSON"
check_sha "$H100_BASE_PYTHON_SHA256" "$H100_BASE_PYTHON"

repo="$scratch/repo"
[[ ! -e "$repo" ]] || {
  echo "scratch repo destination must start empty" >&2
  exit 1
}
git clone "$H100_RUNTIME_BUNDLE" "$repo"
git -C "$repo" checkout --detach "$H100_EXPECTED_GIT_SHA"
actual_git_sha="$(git -C "$repo" rev-parse HEAD)"
if [[ "$actual_git_sha" != "$H100_EXPECTED_GIT_SHA" ]]; then
  echo "smoke clone did not resolve to the expected full git SHA" >&2
  exit 1
fi
check_sha "$H100_DETECTOR_SHA256" "$repo/configs/detector.yaml"
check_sha "$H100_ENV_LOCK_SHA256" "$repo/locks/env-v100node.txt"
PYTHONNOUSERSITE=1 PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" \
  -B -m scripts.h100.build_venv verify \
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

mkdir -p "$H100_RUNS_ROOT/.h100/slurm-smoke"
runtime_home="$scratch/runtime-home"
mkdir -m 700 "$runtime_home" "$runtime_home/cache"
native_env=(
  /usr/bin/env -i
  "HOME=$runtime_home"
  "XDG_CACHE_HOME=$runtime_home/cache"
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
  "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-1}"
)

binding_args=(
  --runs-root "$H100_RUNS_ROOT"
  --git-sha "$H100_EXPECTED_GIT_SHA"
  --detector-sha256 "$H100_DETECTOR_SHA256"
  --venv-sha256 "$H100_VENV_SHA256"
  --venv-build-sha256 "$H100_VENV_BUILD_SHA256"
  --base-python-sha256 "$H100_BASE_PYTHON_SHA256"
  --base-python-runtime-sha256 "$H100_BASE_PYTHON_RUNTIME_SHA256"
  --wheelhouse-sha256 "$H100_WHEELHOUSE_SHA256"
  --base-extraction-receipt-sha256 "$H100_BASE_EXTRACTION_RECEIPT_SHA256"
  --base-payload-package-id "$H100_BASE_PACKAGE_ID"
  --base-payload-git-sha "$H100_BASE_GIT_SHA"
  --base-payload-manifest-sha256 "$H100_BASE_MANIFEST_SHA256"
  --base-payload-ready-sha256 "$H100_BASE_READY_SHA256"
  --base-payload-sha256sums-sha256 "$H100_BASE_SHA256SUMS_SHA256"
  --base-payload-repo-bundle-sha256 "$H100_BASE_REPO_BUNDLE_SHA256"
  --runtime-amendment-package-id "$H100_RUNTIME_PACKAGE_ID"
  --runtime-amendment-git-sha "$H100_RUNTIME_GIT_SHA"
  --runtime-amendment-manifest-sha256 "$H100_RUNTIME_MANIFEST_SHA256"
  --runtime-amendment-ready-sha256 "$H100_RUNTIME_READY_SHA256"
  --runtime-amendment-sha256sums-sha256 "$H100_RUNTIME_SHA256SUMS_SHA256"
  --runtime-amendment-bundle-sha256 "$H100_RUNTIME_BUNDLE_SHA256"
)

smoke_root="$H100_RUNS_ROOT/.h100/slurm-smoke"
signal_ready="$smoke_root/SLURM_SMOKE_SIGNAL_READY.pid"
smoke_state="$smoke_root/SLURM_SMOKE_STATE.json"
smoke_ready="$smoke_root/SLURM_SMOKE_READY.json"
needs_external_signal=0
[[ -e "$smoke_state" || -e "$smoke_ready" ]] || needs_external_signal=1
cd "$repo"
set +e
"${native_env[@]}" "$H100_VENV_ROOT/bin/python" -m scripts.h100.slurm_smoke run \
  "${binding_args[@]}" \
  --job-id "$SLURM_JOB_ID" \
  --restart-count "${SLURM_RESTART_COUNT:-0}" \
  --external-signal-ready "$signal_ready" &
smoke_pid="$!"
set -e

if [[ "$needs_external_signal" -eq 1 ]]; then
  for _ in $(seq 1 30); do
    [[ -s "$signal_ready" ]] && break
    kill -0 "$smoke_pid" 2>/dev/null || break
    sleep 1
  done
  [[ -s "$signal_ready" ]] || {
    echo "native smoke did not publish its signal-ready PID" >&2
    kill "$smoke_pid" 2>/dev/null || true
    wait "$smoke_pid" 2>/dev/null || true
    exit 1
  }
  signal_pid="$(<"$signal_ready")"
  if [[ ! "$signal_pid" =~ ^[1-9][0-9]*$ || "$signal_pid" != "$smoke_pid" ]]; then
    echo "native smoke PID receipt does not identify the direct venv process" >&2
    kill "$smoke_pid" 2>/dev/null || true
    wait "$smoke_pid" 2>/dev/null || true
    exit 1
  fi
  kill -0 "$signal_pid"
  kill -USR1 "$signal_pid"
fi

set +e
wait "$smoke_pid"
smoke_rc="$?"
set -e

if [[ "$smoke_rc" -eq 75 ]]; then
  "${native_env[@]}" "$H100_VENV_ROOT/bin/python" -m scripts.h100.slurm_smoke authorize-requeue \
    "${binding_args[@]}" \
    --job-id "$SLURM_JOB_ID" \
    --restart-count "${SLURM_RESTART_COUNT:-0}"
  # The native child never receives scontrol.  This is the smoke's sole real
  # requeue call, protected by the persistent one-shot authorization above.
  cleanup_allocation_scratch_before_requeue
  "${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$smoke_rc"
