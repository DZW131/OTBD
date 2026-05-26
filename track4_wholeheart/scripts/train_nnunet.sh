#!/usr/bin/env bash
set -euo pipefail

MODALITY="${1:-}"
FOLD="${2:-}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"

if [[ "${MODALITY}" != "ct" && "${MODALITY}" != "mr" ]]; then
  echo "Usage: $0 {ct|mr} {0|1|2|3|4|all}" >&2
  exit 1
fi

if [[ -z "${FOLD}" ]]; then
  echo "Usage: $0 {ct|mr} {0|1|2|3|4|all}" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ "${MODALITY}" == "ct" ]]; then
  DATASET_ID=401
  DATASET_NAME="Dataset401_CARE2026_WholeHeart_CT"
else
  DATASET_ID=402
  DATASET_NAME="Dataset402_CARE2026_WholeHeart_MR"
fi

echo "Training ${MODALITY^^} dataset ${DATASET_ID}, configuration=${CONFIGURATION}, fold=${FOLD}"
echo "trainer=${TRAINER}"
echo "TRACK4_ROOT=${TRACK4_ROOT}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"

PREPROCESSED_DATASET="${nnUNet_preprocessed}/${DATASET_NAME}"
PREPROCESSED_CONFIGURATION="${PREPROCESSED_DATASET}/nnUNetPlans_${CONFIGURATION}"
if [[ ! -d "${PREPROCESSED_DATASET}" ]]; then
  echo "ERROR: Missing preprocessed dataset directory: ${PREPROCESSED_DATASET}" >&2
  echo "Run plan/preprocess with the same nnUNet_preprocessed path first." >&2
  exit 1
fi
if [[ ! -f "${PREPROCESSED_DATASET}/nnUNetPlans.json" ]]; then
  echo "ERROR: Missing plans file: ${PREPROCESSED_DATASET}/nnUNetPlans.json" >&2
  echo "Run: bash track4_wholeheart/scripts/plan_preprocess.sh ${MODALITY}" >&2
  exit 1
fi
if [[ ! -d "${PREPROCESSED_CONFIGURATION}" ]]; then
  echo "ERROR: Missing preprocessed configuration directory: ${PREPROCESSED_CONFIGURATION}" >&2
  echo "Run: CONFIGURATION=${CONFIGURATION} bash track4_wholeheart/scripts/plan_preprocess.sh ${MODALITY}" >&2
  exit 1
fi

if [[ "${TRAINER}" == nnUNetTrainerWholeHeart* ]]; then
  python "${TRACK4_ROOT}/scripts/install_wholeheart_trainer.py"
fi

nnUNetv2_train "${DATASET_ID}" "${CONFIGURATION}" "${FOLD}" -tr "${TRAINER}" --npz
