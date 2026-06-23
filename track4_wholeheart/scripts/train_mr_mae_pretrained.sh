#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  GPU=0 FOLD=0 bash track4_wholeheart/scripts/train_mr_mae_pretrained.sh

Fine-tune MR Dataset402 from an nnSSL/MAE checkpoint with the TaWald nnU-Net
branch. Run preprocess_mr_mae_pretrained.sh first.

Common environment overrides:
  FOLD=0|1|2|3|4
  GPU=0
  MAE_PLANS_NAME=<ptPlans__... name without .json>
  MAE_TRAINER=PretrainedTrainer
  CONTINUE=1
  SAVE_NPZ=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/env_mae.sh"

DATASET_ID="${DATASET_ID:-402}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLD="${FOLD:-0}"
GPU="${GPU:-0}"
MAE_TRAINER="${MAE_TRAINER:-PretrainedTrainer}"
SAVE_NPZ="${SAVE_NPZ:-1}"
CONTINUE="${CONTINUE:-0}"

DATASET_DIR="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset${DATASET_ID}_*" | head -n 1)"
if [[ -z "${DATASET_DIR}" || ! -d "${DATASET_DIR}" ]]; then
  echo "ERROR: Could not find preprocessed Dataset${DATASET_ID}_* under ${nnUNet_preprocessed}" >&2
  exit 1
fi

if [[ -z "${MAE_PLANS_NAME:-}" ]]; then
  MAE_PLANS_JSON="$(find "${DATASET_DIR}" -maxdepth 1 -type f -name "ptPlans__${MAE_PRETRAINING_NAME}*.json" | sort | tail -n 1)"
  if [[ -z "${MAE_PLANS_JSON}" ]]; then
    echo "ERROR: No MAE plans found. Run preprocess_mr_mae_pretrained.sh first." >&2
    exit 1
  fi
  MAE_PLANS_NAME="$(basename "${MAE_PLANS_JSON}" .json)"
fi

EXTRA_ARGS=()
if [[ "${SAVE_NPZ}" == "1" ]]; then
  EXTRA_ARGS+=(--npz)
fi
if [[ "${CONTINUE}" == "1" ]]; then
  EXTRA_ARGS+=(--c)
fi

cat <<EOF
MR MAE pretrained fine-tune
dataset_id=${DATASET_ID}
configuration=${CONFIGURATION}
fold=${FOLD}
gpu=${GPU}
trainer=${MAE_TRAINER}
plans=${MAE_PLANS_NAME}
nnUNet_results=${nnUNet_results}
continue=${CONTINUE}
save_npz=${SAVE_NPZ}
EOF

CUDA_VISIBLE_DEVICES="${GPU}" nnUNetv2_train_pretrained \
  "${DATASET_ID}" \
  "${CONFIGURATION}" \
  "${FOLD}" \
  -tr "${MAE_TRAINER}" \
  -p "${MAE_PLANS_NAME}" \
  "${EXTRA_ARGS[@]}"
