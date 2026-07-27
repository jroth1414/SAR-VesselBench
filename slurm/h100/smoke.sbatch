#!/usr/bin/env bash
#SBATCH --job-name=xview3-h100-slurm-smoke
#SBATCH --account=geofam
#SBATCH --partition=minor-use-case
#SBATCH --reservation=geofam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --requeue

set -euo pipefail

# This must precede Python, nvidia-smi, Apptainer --nv, or any CUDA import.
export NVIDIA_TF32_OVERRIDE=0

site_env="${H100_SITE_ENV:?submit.sh must export H100_SITE_ENV}"
# shellcheck disable=SC1090
source "$site_env"
# Transfer credentials are never needed by a compute allocation.
unset BOX_JWT_CONFIG BOX_FOLDER_ID

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

check_sha "$H100_REPO_BUNDLE_SHA256" "$H100_REPO_BUNDLE"
check_sha "$H100_MANIFEST_SHA256" "$H100_PACKAGE_ROOT/manifest.json"
check_sha "$H100_READY_SHA256" "$H100_PACKAGE_ROOT/READY.json"
check_sha "$H100_SHA256SUMS_SHA256" "$H100_PACKAGE_ROOT/SHA256SUMS"
check_sha "$H100_SIF_SHA256" "$H100_SIF"
check_sha "$H100_SIF_BUILD_SHA256" "$H100_SIF_BUILD_JSON"

scratch="${SLURM_TMPDIR:?SLURM_TMPDIR is required}"
repo="$scratch/repo"
[[ ! -e "$repo" ]] || {
  echo "scratch repo destination must start empty" >&2
  exit 1
}
git clone "$H100_REPO_BUNDLE" "$repo"
git -C "$repo" checkout --detach "$H100_EXPECTED_GIT_SHA"
actual_git_sha="$(git -C "$repo" rev-parse HEAD)"
if [[ "$actual_git_sha" != "$H100_EXPECTED_GIT_SHA" ]]; then
  echo "smoke clone did not resolve to the expected full git SHA" >&2
  exit 1
fi
check_sha "$H100_DETECTOR_SHA256" "$repo/configs/detector.yaml"
check_sha "$H100_ENV_LOCK_SHA256" "$repo/locks/env-v100node.txt"

mkdir -p "$H100_RUNS_ROOT/.h100/slurm-smoke"
apptainer_bin="${H100_APPTAINER:-apptainer}"
export APPTAINERENV_NVIDIA_TF32_OVERRIDE=0
export APPTAINERENV_PYTHONPATH="$repo/scripts/h100:$repo"
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_SLURM_JOB_ID="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
export APPTAINERENV_SLURM_RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
export APPTAINERENV_SLURM_CLUSTER_NAME="${SLURM_CLUSTER_NAME:-unknown}"

run_sif() {
  "$apptainer_bin" exec --cleanenv --containall --nv \
    --bind "$repo:$repo:ro,$H100_RUNS_ROOT:$H100_RUNS_ROOT" \
    "$H100_SIF" "$@"
}

binding_args=(
  --runs-root "$H100_RUNS_ROOT"
  --git-sha "$H100_EXPECTED_GIT_SHA"
  --detector-sha256 "$H100_DETECTOR_SHA256"
  --sif-sha256 "$H100_SIF_SHA256"
  --container-build-sha256 "$H100_SIF_BUILD_SHA256"
  --package-manifest-sha256 "$H100_MANIFEST_SHA256"
  --package-ready-sha256 "$H100_READY_SHA256"
  --package-sha256sums-sha256 "$H100_SHA256SUMS_SHA256"
  --package-repo-bundle-sha256 "$H100_REPO_BUNDLE_SHA256"
)

set +e
(
  cd "$repo"
  run_sif python -m scripts.h100.slurm_smoke run \
    "${binding_args[@]}" \
    --job-id "$SLURM_JOB_ID" \
    --restart-count "${SLURM_RESTART_COUNT:-0}"
)
smoke_rc="$?"
set -e

if [[ "$smoke_rc" -eq 75 ]]; then
  (
    cd "$repo"
    run_sif python -m scripts.h100.slurm_smoke authorize-requeue \
      "${binding_args[@]}" \
      --job-id "$SLURM_JOB_ID" \
      --restart-count "${SLURM_RESTART_COUNT:-0}"
  )
  # The container never receives scontrol.  This is the smoke's sole real
  # requeue call, protected by the persistent one-shot authorization above.
  "${H100_REAL_SCONTROL:-/usr/bin/scontrol}" requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$smoke_rc"
