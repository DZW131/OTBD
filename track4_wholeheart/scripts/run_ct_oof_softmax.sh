#!/usr/bin/env bash
set -euo pipefail

CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
FOLDS="${FOLDS:-0 1 2 3 4}"
CHECKPOINTS="${CHECKPOINTS:-final best}"
GPUS="${GPUS:-0 1 2 3}"
RUN_ROOT="${RUN_ROOT:-}"
SKIP_DONE="${SKIP_DONE:-1}"
MIN_NPZ_PER_FOLD="${MIN_NPZ_PER_FOLD:-1}"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DATASET_ID=401
DATASET_NAME="Dataset401_CARE2026_WholeHeart_CT"
RESULT_ROOT="${nnUNet_results}/${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIGURATION}"
RUN_ROOT="${RUN_ROOT:-${TRACK4_ROOT}/outputs/internal_val_ct_oof_softmax}"
mkdir -p "${RUN_ROOT}" "${TRACK4_ROOT}/outputs/logs"

if [[ "${TRAINER}" == nnUNetTrainerWholeHeart* ]]; then
  python "${TRACK4_ROOT}/scripts/install_wholeheart_trainer.py"
fi

read -r -a FOLD_ARRAY <<< "${FOLDS}"
read -r -a CHECKPOINT_ARRAY <<< "${CHECKPOINTS}"
read -r -a GPU_ARRAY <<< "${GPUS}"

if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "ERROR: GPUS must contain at least one GPU id." >&2
  exit 1
fi

echo "CT OOF softmax generation"
echo "TRACK4_ROOT=${TRACK4_ROOT}"
echo "nnUNet_results=${nnUNet_results}"
echo "dataset=${DATASET_ID} ${DATASET_NAME}"
echo "trainer=${TRAINER}"
echo "configuration=${CONFIGURATION}"
echo "folds=${FOLDS}"
echo "checkpoints=${CHECKPOINTS}"
echo "gpus=${GPUS}"
echo "run_root=${RUN_ROOT}"
echo "tta=enabled"
echo "save_probabilities=enabled"

validate_checkpoint_name() {
  case "$1" in
    final|best) ;;
    *)
      echo "ERROR: unsupported checkpoint '$1'. Use 'final' and/or 'best'." >&2
      exit 1
      ;;
  esac
}

safe_remove_validation_dir() {
  local validation_dir="$1"
  case "${validation_dir}" in
    "${RESULT_ROOT}"/fold_*/validation)
      rm -rf "${validation_dir}"
      ;;
    *)
      echo "ERROR: refusing to remove unexpected validation directory: ${validation_dir}" >&2
      exit 1
      ;;
  esac
}

run_one_fold() {
  local checkpoint="$1"
  local fold="$2"
  local gpu="$3"
  local fold_dir="${RESULT_ROOT}/fold_${fold}"
  local validation_dir="${fold_dir}/validation"
  local target_dir="${RUN_ROOT}/${checkpoint}/fold_${fold}/validation"
  local log_file="${TRACK4_ROOT}/outputs/logs/ct_oof_${checkpoint}_fold${fold}.log"
  local done_marker="${target_dir}/.done"

  if [[ ! -d "${fold_dir}" ]]; then
    echo "ERROR: missing fold directory: ${fold_dir}" >&2
    return 1
  fi

  if [[ "${SKIP_DONE}" == "1" && -f "${done_marker}" ]]; then
    local npz_count
    npz_count="$(find "${target_dir}" -maxdepth 1 -name '*.npz' | wc -l)"
    if [[ "${npz_count}" -ge "${MIN_NPZ_PER_FOLD}" ]]; then
      echo "[${checkpoint} fold ${fold}] skip existing target (${npz_count} npz): ${target_dir}"
      return 0
    fi
  fi

  mkdir -p "${target_dir}"
  rm -f "${done_marker}"
  safe_remove_validation_dir "${validation_dir}"

  local val_args=(nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${fold}" -tr "${TRAINER}" --val --npz)
  if [[ "${checkpoint}" == "best" ]]; then
    val_args+=(--val_best)
  fi

  {
    echo "[start] $(date)"
    echo "checkpoint=${checkpoint}"
    echo "fold=${fold}"
    echo "gpu=${gpu}"
    echo "command=CUDA_VISIBLE_DEVICES=${gpu} ${val_args[*]}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${val_args[@]}"
    echo "[copy] ${validation_dir} -> ${target_dir}"
    rsync -a --delete "${validation_dir}/" "${target_dir}/"
    local npz_count nii_count
    npz_count="$(find "${target_dir}" -maxdepth 1 -name '*.npz' | wc -l)"
    nii_count="$(find "${target_dir}" -maxdepth 1 -name '*.nii.gz' | wc -l)"
    if [[ "${npz_count}" -lt "${MIN_NPZ_PER_FOLD}" ]]; then
      echo "ERROR: expected at least ${MIN_NPZ_PER_FOLD} npz files, got ${npz_count}" >&2
      exit 1
    fi
    {
      echo "checkpoint=${checkpoint}"
      echo "fold=${fold}"
      echo "gpu=${gpu}"
      echo "npz_count=${npz_count}"
      echo "nii_count=${nii_count}"
      echo "created_at=$(date -Iseconds)"
    } > "${done_marker}"
    safe_remove_validation_dir "${validation_dir}"
    echo "[done] $(date)"
  } > "${log_file}" 2>&1
}

run_checkpoint() {
  local checkpoint="$1"
  validate_checkpoint_name "${checkpoint}"
  echo "=== checkpoint=${checkpoint} ==="

  local launched=0
  for fold in "${FOLD_ARRAY[@]}"; do
    local gpu="${GPU_ARRAY[$((launched % ${#GPU_ARRAY[@]}))]}"
    run_one_fold "${checkpoint}" "${fold}" "${gpu}" &
    launched=$((launched + 1))
    if [[ "$(jobs -rp | wc -l)" -ge "${#GPU_ARRAY[@]}" ]]; then
      wait -n
    fi
  done
  wait
}

for checkpoint in "${CHECKPOINT_ARRAY[@]}"; do
  run_checkpoint "${checkpoint}"
done

echo "CT OOF softmax generation complete."
find "${RUN_ROOT}" -maxdepth 4 -name '.done' -print | sort
