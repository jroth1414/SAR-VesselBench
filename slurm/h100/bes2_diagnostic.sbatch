#!/usr/bin/env bash
#SBATCH --job-name=xview3-bes2-diagnostic
#SBATCH --account=geofam
#SBATCH --partition=minor-use-case
# An optional reservation is supplied by submit_bes2_diagnostic.sh.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-18:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue

set -euo pipefail

# This must precede Python, nvidia-smi, or any CUDA import.
export NVIDIA_TF32_OVERRIDE=0

mode="${1:?submit_bes2_diagnostic.sh must pass audit, probe, or full}"
compute_site="${2:?submit_bes2_diagnostic.sh must pass the immutable snapshot}"
compute_site_sha256="${3:?submit_bes2_diagnostic.sh must pass the snapshot hash}"
if [[ "$mode" != "audit" && "$mode" != "probe" && "$mode" != "full" ]]; then
  echo "invalid BES2 diagnostic mode: $mode" >&2
  exit 2
fi
if [[ ! "$compute_site_sha256" =~ ^[0-9a-f]{64}$ ||
      -L "$compute_site" || ! -f "$compute_site" ||
      "$(stat -c '%a' "$compute_site")" != "444" ||
      "$(sha256sum "$compute_site" | awk '{print $1}')" != "$compute_site_sha256" ]]; then
  echo "BES2 compute-site snapshot is not the exact immutable file" >&2
  exit 2
fi
readonly mode compute_site compute_site_sha256
# shellcheck disable=SC1090
source "$compute_site"
unset BOX_JWT_CONFIG BOX_FOLDER_ID

if [[ "$H100_BASE_PYTHON_LIB_DIR" != /* ||
      -L "$H100_BASE_PYTHON_LIB_DIR" ||
      ! -d "$H100_BASE_PYTHON_LIB_DIR" ||
      "$(realpath -e -- "$H100_BASE_PYTHON_LIB_DIR")" != "$H100_BASE_PYTHON_LIB_DIR" ||
      ! -r "$H100_BASE_PYTHON_LIB_DIR/libpython3.11.so.1.0" ]]; then
  echo "invalid canonical Judy libpython directory" >&2
  exit 2
fi
export LD_LIBRARY_PATH="$H100_BASE_PYTHON_LIB_DIR"
readonly LD_LIBRARY_PATH

canonical_root() {
  local name="$1"
  local raw="${!name}"
  local canonical
  if [[ "$raw" != /* ]]; then
    echo "$name must be absolute" >&2
    return 2
  fi
  canonical="$(realpath -m -- "$raw")"
  if [[ "$canonical" == "/" ]]; then
    echo "$name must not resolve to /" >&2
    return 2
  fi
  printf '%s\n' "$canonical"
}

assert_disjoint() {
  local left_name="$1"
  local left="$2"
  local right_name="$3"
  local right="$4"
  if [[ "$left" == "$right" || "$left" == "$right/"* || "$right" == "$left/"* ]]; then
    echo "$left_name overlaps $right_name: $left / $right" >&2
    return 2
  fi
}

diagnostic_root="$(canonical_root BES2_DIAGNOSTIC_ROOT)"
original_runs_root="$(canonical_root BES2_ORIGINAL_RUNS_ROOT)"
scratch_root="$(canonical_root H100_SCRATCH_ROOT)"
job_log_root="$(canonical_root BES2_JOB_LOG_DIR)"
assert_disjoint BES2_DIAGNOSTIC_ROOT "$diagnostic_root" \
  BES2_ORIGINAL_RUNS_ROOT "$original_runs_root"
assert_disjoint H100_SCRATCH_ROOT "$scratch_root" \
  BES2_DIAGNOSTIC_ROOT "$diagnostic_root"
assert_disjoint H100_SCRATCH_ROOT "$scratch_root" \
  BES2_ORIGINAL_RUNS_ROOT "$original_runs_root"
assert_disjoint BES2_JOB_LOG_DIR "$job_log_root" \
  BES2_DIAGNOSTIC_ROOT "$diagnostic_root"
for protected_name in \
  BES2_PACKAGE_ROOT BES2_PROJECT_ROOT H100_PROJECT_ROOT \
  H100_BASE_PACKAGE_ROOT H100_RUNTIME_PACKAGE_ROOT H100_WHEELHOUSE H100_VENV_ROOT
do
  protected="$(canonical_root "$protected_name")"
  assert_disjoint BES2_DIAGNOSTIC_ROOT "$diagnostic_root" \
    "$protected_name" "$protected"
  assert_disjoint H100_SCRATCH_ROOT "$scratch_root" \
    "$protected_name" "$protected"
done
BES2_DIAGNOSTIC_ROOT="$diagnostic_root"
BES2_ORIGINAL_RUNS_ROOT="$original_runs_root"
H100_SCRATCH_ROOT="$scratch_root"
BES2_JOB_LOG_DIR="$job_log_root"
readonly BES2_DIAGNOSTIC_ROOT BES2_ORIGINAL_RUNS_ROOT
readonly H100_SCRATCH_ROOT BES2_JOB_LOG_DIR

assert_final_unconsumed() {
  local relative
  for relative in \
    final_eval.lock \
    .h100/FINAL_DATA_VIEW.json \
    .h100/FINAL_GROUND_TRUTH_CONSUMED.json \
    .h100/FINAL_NORMALIZED_GROUND_TRUTH.json \
    .h100/FINAL_EVAL_COMPLETE.json
  do
    if [[ -e "$BES2_ORIGINAL_RUNS_ROOT/$relative" ||
          -L "$BES2_ORIGINAL_RUNS_ROOT/$relative" ]]; then
      echo "BES2 diagnostic forbidden after final access: $relative" >&2
      return 1
    fi
  done
  if compgen -G "$BES2_ORIGINAL_RUNS_ROOT/*/final_verified_metrics.json" >/dev/null; then
    echo "BES2 diagnostic forbidden after per-cell final output" >&2
    return 1
  fi
}

check_sha() {
  local expected="$1"
  local path="$2"
  local observed
  if [[ -L "$path" || ! -f "$path" ]]; then
    echo "missing regular immutable input: $path" >&2
    return 1
  fi
  observed="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$observed" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path: $observed != $expected" >&2
    return 1
  fi
}

h100_ready="$BES2_ORIGINAL_RUNS_ROOT/.h100/H100_READY.json"
if [[ -L "$h100_ready" || ! -f "$h100_ready" ||
      "$(stat -c '%a' "$h100_ready")" != "444" ]]; then
  echo "canonical H100_READY is absent or mutable" >&2
  exit 2
fi
check_sha "$BES2_H100_READY_SHA256" "$h100_ready"
assert_final_unconsumed

allocation_scratch=""
remove_allocation_scratch() {
  local expected actual
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
        -L "$allocation_scratch" || ! -d "$allocation_scratch" ||
        "$(stat -c '%u' "$allocation_scratch" 2>/dev/null || true)" != "$(id -u)" ]]; then
    echo "refusing unsafe BES2 scratch cleanup: $allocation_scratch" >&2
    return 1
  fi
  find "$allocation_scratch" -xdev -type d -exec chmod u+rwx -- {} + || return 1
  find "$allocation_scratch" -xdev -depth -mindepth 1 -delete || return 1
  rmdir -- "$allocation_scratch" || return 1
  allocation_scratch=""
}

cleanup() {
  local rc="$?"
  trap - EXIT
  if ! remove_allocation_scratch && (( rc == 0 )); then
    rc=1
  fi
  if [[ -n "${h100_ready:-}" ]] &&
     ! check_sha "$BES2_H100_READY_SHA256" "$h100_ready" && (( rc == 0 )); then
    rc=1
  fi
  exit "$rc"
}

create_allocation_scratch() {
  local expected probe free
  if [[ ! "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ||
        ! "${SLURM_RESTART_COUNT:-0}" =~ ^[0-9]+$ ||
        -L "$H100_SCRATCH_ROOT" || ! -d "$H100_SCRATCH_ROOT" ||
        "$(realpath -e -- "$H100_SCRATCH_ROOT" 2>/dev/null || true)" != "$H100_SCRATCH_ROOT" ||
        ! -w "$H100_SCRATCH_ROOT" || ! -x "$H100_SCRATCH_ROOT" ]]; then
    echo "invalid Judy allocation scratch contract" >&2
    return 2
  fi
  expected="$H100_SCRATCH_ROOT/xview3-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}"
  if [[ -e "$expected" || -L "$expected" ]]; then
    echo "BES2 allocation scratch must start absent: $expected" >&2
    return 1
  fi
  allocation_scratch="$expected"
  trap cleanup EXIT
  mkdir -m 700 -- "$allocation_scratch"
  chmod 0700 -- "$allocation_scratch"
  if [[ -L "$allocation_scratch" || ! -d "$allocation_scratch" ||
        "$(realpath -e -- "$allocation_scratch")" != "$allocation_scratch" ||
        "$(stat -c '%a' "$allocation_scratch")" != "700" ||
        "$(stat -c '%u' "$allocation_scratch")" != "$(id -u)" ]]; then
    echo "BES2 allocation scratch failed validation" >&2
    return 1
  fi
  probe="$allocation_scratch/.write-probe"
  (umask 077; : > "$probe")
  [[ ! -L "$probe" && -f "$probe" && "$(stat -c '%a' "$probe")" == "600" ]]
  rm -f -- "$probe"
  free="$(df -PB1 "$allocation_scratch" | awk 'NR==2 {print $4}')"
  if (( free < 500000000000 )); then
    echo "BES2 allocation scratch has less than 500 GB free" >&2
    return 1
  fi
}

create_allocation_scratch
scratch="$allocation_scratch"
readonly scratch

check_sha "$BES2_BUNDLE_SHA256" "$BES2_BUNDLE"
check_sha "$BES2_MANIFEST_SHA256" "$BES2_PACKAGE_ROOT/manifest.json"
check_sha "$BES2_READY_SHA256" "$BES2_PACKAGE_ROOT/READY.json"
check_sha "$BES2_SHA256SUMS_SHA256" "$BES2_PACKAGE_ROOT/SHA256SUMS"
check_sha "$H100_BASE_MANIFEST_SHA256" "$H100_BASE_PACKAGE_ROOT/manifest.json"
check_sha "$H100_RUNTIME_MANIFEST_SHA256" "$H100_RUNTIME_PACKAGE_ROOT/manifest.json"
check_sha "$H100_RUNTIME_READY_SHA256" "$H100_RUNTIME_PACKAGE_ROOT/READY.json"
check_sha "$H100_RUNTIME_SHA256SUMS_SHA256" "$H100_RUNTIME_PACKAGE_ROOT/SHA256SUMS"
check_sha "$H100_VENV_BUILD_SHA256" "$H100_VENV_BUILD_JSON"
check_sha "$H100_BASE_PYTHON_SHA256" "$H100_BASE_PYTHON"
(cd "$BES2_PACKAGE_ROOT" && sha256sum --check SHA256SUMS)
(cd "$H100_RUNTIME_PACKAGE_ROOT" && sha256sum --check SHA256SUMS)

repo="$scratch/repo"
payload="$scratch/payload"
git clone --single-branch --branch sprint-10a-bes2-root-cause "$BES2_BUNDLE" "$repo"
git -C "$repo" checkout --detach "$BES2_EXPECTED_GIT_SHA"
if [[ "$(git -C "$repo" rev-parse HEAD)" != "$BES2_EXPECTED_GIT_SHA" ||
      -n "$(git -C "$repo" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "BES2 scratch checkout did not reproduce the exact clean source" >&2
  exit 1
fi
check_sha "$H100_ENV_LOCK_SHA256" "$repo/locks/env-v100node.txt"
check_sha "$H100_DETECTOR_SHA256" "$repo/configs/detector.yaml"
check_sha "$H100_SCORER_SHA256" "$repo/src/eval/scorer.py"
check_sha "$H100_SPLITS_SHA256" "$repo/data/splits.json"
check_sha "$H100_STATS_SHA256" "$repo/data/stats.json"
check_sha "$H100_LSSSDD_SHA256" "$repo/data/lsssdd_split.json"

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
  --expected-base-extraction-receipt-sha256 \
    "$H100_BASE_EXTRACTION_RECEIPT_SHA256" \
  --expected-base-payload-package-id "$H100_BASE_PACKAGE_ID" \
  --expected-base-payload-manifest-sha256 "$H100_BASE_MANIFEST_SHA256"

PYTHONNOUSERSITE=1 PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" \
  -B -m scripts.handoff verify-bes2-amendment \
  --package-root "$BES2_PACKAGE_ROOT" \
  --expected-h100-ready-sha256 "$BES2_H100_READY_SHA256"

staging_guard="$BES2_DIAGNOSTIC_ROOT/.staging-guard"
if [[ -L "$staging_guard" ]]; then
  echo "BES2 staging guard must not be a symlink" >&2
  exit 1
fi
mkdir -m 700 -p "$staging_guard"
if [[ -n "$(find "$staging_guard" -mindepth 1 -print -quit)" ]]; then
  echo "BES2 staging guard must remain an empty no-cohort namespace" >&2
  exit 1
fi
PYTHONNOUSERSITE=1 PYTHONPATH="$repo" "$H100_TRANSFER_PYTHON" \
  -B -m scripts.h100.data_staging \
  --repo "$repo" \
  --runs-root "$staging_guard" \
  --base-package-root "$H100_BASE_PACKAGE_ROOT" \
  --runtime-package-root "$H100_RUNTIME_PACKAGE_ROOT" \
  --destination "$payload" \
  --expected-git-sha "$H100_EXPECTED_GIT_SHA" \
  --expected-base-package-id "$H100_BASE_PACKAGE_ID" \
  --expected-base-manifest-sha256 "$H100_BASE_MANIFEST_SHA256" \
  --expected-runtime-package-id "$H100_RUNTIME_PACKAGE_ID" \
  --expected-runtime-manifest-sha256 "$H100_RUNTIME_MANIFEST_SHA256"
data_view_receipt="$payload/H100_DATA_VIEW.json"
data_view_receipt_sha256="$(sha256sum "$data_view_receipt" | awk '{print $1}')"

for relative in chips raw weights; do
  if [[ -e "$repo/data/$relative" || -L "$repo/data/$relative" ||
        ! -e "$payload/data/$relative" ]]; then
    echo "unsafe or missing BES2 staged data subtree: $relative" >&2
    exit 1
  fi
  ln -s "$payload/data/$relative" "$repo/data/$relative"
done

runtime_home="$scratch/runtime-home"
mkdir -m 700 "$runtime_home"
mkdir -m 700 "$runtime_home/cache" "$runtime_home/torch" "$runtime_home/triton"
expected_gpus=2
if [[ "$mode" == "audit" ]]; then
  expected_gpus=1
fi
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
  "SLURM_JOB_ID=${SLURM_JOB_ID:?}"
  "SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}"
  "SLURM_CLUSTER_NAME=${SLURM_CLUSTER_NAME:-unknown}"
  "SLURM_NTASKS=${SLURM_NTASKS:-1}"
  "SLURM_PROCID=${SLURM_PROCID:-0}"
  "SLURM_LOCALID=${SLURM_LOCALID:-0}"
  "SLURM_NODEID=${SLURM_NODEID:-0}"
  "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
  "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-$expected_gpus}"
  "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}"
  "SLURMD_NODENAME=${SLURMD_NODENAME:-unknown}"
  "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unknown}"
)
run_venv() {
  "${native_env[@]}" "$H100_VENV_ROOT/bin/python" "$@"
}

control="$BES2_DIAGNOSTIC_ROOT/.control"
mkdir -m 700 -p "$control"
hardware="$control/h100_runtime-${SLURM_JOB_ID}-r${SLURM_RESTART_COUNT:-0}.json"
hardware_tmp="${hardware}.tmp.$$"
(
  cd "$repo"
  run_venv -B -m scripts.h100.strict_fp32_probe \
    --diagnostic \
    --expected-gpus "$expected_gpus" > "$hardware_tmp"
)
mv "$hardware_tmp" "$hardware"

view_ready="$payload/BES2_DATA_VIEW_READY.json"
(
  cd "$repo"
  run_venv -B -m scripts.h100.bes2_diagnostic prepare-view \
    --repo "$repo" \
    --diagnostic-root "$BES2_DIAGNOSTIC_ROOT" \
    --original-runs-root "$BES2_ORIGINAL_RUNS_ROOT" \
    --staging-runs-root "$staging_guard" \
    --data-view-root "$payload" \
    --data-config "$repo/configs/data.yaml" \
    --output "$view_ready" \
    --h100-ready "$h100_ready" \
    --h100-ready-sha256 "$BES2_H100_READY_SHA256" \
    --staging-receipt-sha256 "$data_view_receipt_sha256" \
    --expected-git-sha "$BES2_EXPECTED_GIT_SHA" \
    --expected-base-package-id "$H100_BASE_PACKAGE_ID" \
    --expected-base-manifest-sha256 "$H100_BASE_MANIFEST_SHA256" \
    --expected-runtime-package-id "$H100_RUNTIME_PACKAGE_ID" \
    --expected-runtime-manifest-sha256 "$H100_RUNTIME_MANIFEST_SHA256"
)

if [[ "$mode" == "audit" ]]; then
  (
    cd "$repo"
    run_venv -B -m scripts.h100.bes2_diagnostic test-gate \
      --repo "$repo" \
      --expected-git-sha "$BES2_EXPECTED_GIT_SHA" \
      --receipt "$control/BES2_TEST_GATE.json" \
      --log "$control/BES2_TEST_GATE.log"
    run_venv -B -m src.analysis.bes2_root_cause audit \
      --repo "$repo" \
      --diagnostic-root "$BES2_DIAGNOSTIC_ROOT" \
      --expected-git-sha "$BES2_EXPECTED_GIT_SHA" \
      --h100-ready "$h100_ready" \
      --h100-ready-sha256 "$BES2_H100_READY_SHA256" \
      --original-runs-root "$BES2_ORIGINAL_RUNS_ROOT" \
      --data-view-root "$payload" \
      --data-view-receipt "$view_ready" \
      --data-config "$repo/configs/data.yaml" \
      --test-receipt "$control/BES2_TEST_GATE.json" \
      --weights-root "$payload/data/weights" \
      --forecast-gpu-hours "$BES2_FORECAST_GPU_HOURS"
  )
  assert_final_unconsumed
  exit 0
fi

if [[ ! -f "$control/BES2_DIAGNOSTIC_READY.json" ||
      -L "$control/BES2_DIAGNOSTIC_READY.json" ]]; then
  echo "probe/full requires the immutable BES2 audit readiness receipt" >&2
  exit 1
fi
if [[ "$mode" == "full" && ( ! -f "$control/BES2_PROBES_COMPLETE.json" ||
      -L "$control/BES2_PROBES_COMPLETE.json" ) ]]; then
  echo "full diagnostic requires both finite f50 five-epoch probes" >&2
  exit 1
fi

controller_pid=""
preemption_requested=0
forward_usr1() {
  preemption_requested=1
  if [[ -n "$controller_pid" ]]; then
    kill -USR1 "$controller_pid" 2>/dev/null || true
  fi
}
trap forward_usr1 USR1

request_outer_requeue() {
  assert_final_unconsumed
  check_sha "$BES2_H100_READY_SHA256" "$h100_ready"
  remove_allocation_scratch
  trap - EXIT
  "${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue "$SLURM_JOB_ID"
  exit 0
}

run_controller_wave() {
  local fraction="${1:?fraction wave is required}"
  if [[ "$preemption_requested" -eq 1 ]]; then
    request_outer_requeue
  fi
  cd "$repo"
  "${native_env[@]}" "$H100_VENV_ROOT/bin/python" -B \
    -m scripts.h100.bes2_diagnostic controller \
    --repo "$repo" \
    --diagnostic-root "$BES2_DIAGNOSTIC_ROOT" \
    --original-runs-root "$BES2_ORIGINAL_RUNS_ROOT" \
    --expected-git-sha "$BES2_EXPECTED_GIT_SHA" \
    --stage "$mode" \
    --fraction "$fraction" \
    --data-view-root "$payload" \
    --data-view-receipt "$view_ready" \
    --data-config "$repo/configs/data.yaml" \
    --weights-root "$payload/data/weights" \
    --workers 6 \
    --checkpoint-timeout 600 &
  controller_pid="$!"
  if [[ "$preemption_requested" -eq 1 ]]; then
    kill -TERM "$controller_pid" 2>/dev/null || true
    set +e
    wait "$controller_pid"
    set -e
    controller_pid=""
    request_outer_requeue
  fi
  set +e
  wait "$controller_pid"
  controller_rc="$?"
  set -e
  controller_pid=""
  if [[ "$controller_rc" -eq 0 && "$preemption_requested" -eq 0 ]]; then
    return 0
  fi
  if [[ "$controller_rc" -ne 75 ]]; then
    if [[ "$controller_rc" -eq 0 && "$preemption_requested" -eq 1 ]]; then
      request_outer_requeue
    fi
    return "$controller_rc"
  fi
  request_outer_requeue
}

if [[ "$mode" == "probe" ]]; then
  run_controller_wave 0.5
else
  run_controller_wave 0.1
  run_controller_wave 0.5
fi
assert_final_unconsumed
exit 0
