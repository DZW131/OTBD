#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  bash track4_wholeheart/scripts/preprocess_mr_mae_pretrained.sh

Prepare Dataset402 MR for MAE-pretrained ResEnc fine-tuning using the
TaWald/nnU-Net nnSSL adaptation branch.

Common environment overrides:
  MAE_CHECKPOINT=/path/to/checkpoint_final.pth
  MAE_PRETRAINING_NAME=ResEncL_OpenMind_MAE
  MAE_ADAPTATION_MODE=default_nnunet|like_pretrained|no_resample|fixed
  MAE_NUM_PROCESSES=4
  MAE_FORCE_PLAN=1             rerun nnUNet planning even if nnUNetPlans.json exists
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/env_mae.sh"

DATASET_ID="${DATASET_ID:-402}"
MAE_ADAPTATION_MODE="${MAE_ADAPTATION_MODE:-default_nnunet}"
MAE_NUM_PROCESSES="${MAE_NUM_PROCESSES:-4}"
MAE_FORCE_PLAN="${MAE_FORCE_PLAN:-0}"

if [[ ! -d "${MAE_NNUNET_ROOT}" ]]; then
  echo "ERROR: MAE_NNUNET_ROOT does not exist: ${MAE_NNUNET_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MAE_CHECKPOINT}" ]]; then
  echo "ERROR: MAE_CHECKPOINT does not exist: ${MAE_CHECKPOINT}" >&2
  exit 1
fi

DATASET_DIR="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset${DATASET_ID}_*" | head -n 1)"

cat <<EOF
MR MAE preprocessing
dataset_id=${DATASET_ID}
mae_nnunet_root=${MAE_NNUNET_ROOT}
checkpoint=${MAE_CHECKPOINT}
pretraining_name=${MAE_PRETRAINING_NAME}
adaptation_mode=${MAE_ADAPTATION_MODE}
num_processes=${MAE_NUM_PROCESSES}
force_plan=${MAE_FORCE_PLAN}
nnUNet_preprocessed=${nnUNet_preprocessed}
nnUNet_results=${nnUNet_results}
EOF

if [[ "${MAE_FORCE_PLAN}" == "1" || -z "${DATASET_DIR}" || ! -f "${DATASET_DIR}/nnUNetPlans.json" ]]; then
  nnUNetv2_plan_and_preprocess -d "${DATASET_ID}" --no_pp
  DATASET_DIR="$(find "${nnUNet_preprocessed}" -maxdepth 1 -type d -name "Dataset${DATASET_ID}_*" | head -n 1)"
else
  echo "Found existing nnUNet plans, skipping planning: ${DATASET_DIR}/nnUNetPlans.json"
fi

nnUNetv2_preprocess_like_nnssl \
  -d "${DATASET_ID}" \
  -n "${MAE_PRETRAINING_NAME}" \
  -pc "${MAE_CHECKPOINT}" \
  -am "${MAE_ADAPTATION_MODE}" \
  -np "${MAE_NUM_PROCESSES}" \
  --verbose

echo "Available MAE plans:"
find "${DATASET_DIR}" -maxdepth 1 -type f -name "ptPlans__${MAE_PRETRAINING_NAME}*.json" -printf "%f\n" | sort
