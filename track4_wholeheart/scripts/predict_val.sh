#!/usr/bin/env bash
set -euo pipefail

MODALITY="${1:-}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLDS="${FOLDS:-}"
POSTPROCESS="${POSTPROCESS:-0}"
MIN_COMPONENT_SIZE="${MIN_COMPONENT_SIZE:-0}"

if [[ "${MODALITY}" != "ct" && "${MODALITY}" != "mr" ]]; then
  echo "Usage: $0 {ct|mr}" >&2
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

INPUT_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_nnunet"
OFFICIAL_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_official_labels"
POSTPROCESSED_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_nnunet_postprocessed"
MAPPING_JSON="${nnUNet_raw}/${DATASET_NAME}/conversion_mapping.json"

mkdir -p "${PRED_DIR}" "${OFFICIAL_DIR}"

echo "Predicting ${MODALITY^^} validation set"
echo "Input: ${INPUT_DIR}"
echo "Prediction dir: ${PRED_DIR}"

if [[ -n "${FOLDS}" ]]; then
  # shellcheck disable=SC2086
  nnUNetv2_predict -i "${INPUT_DIR}" -o "${PRED_DIR}" -d "${DATASET_ID}" -c "${CONFIGURATION}" -f ${FOLDS} --save_probabilities
else
  nnUNetv2_predict -i "${INPUT_DIR}" -o "${PRED_DIR}" -d "${DATASET_ID}" -c "${CONFIGURATION}" --save_probabilities
fi

RESTORE_SOURCE_DIR="${PRED_DIR}"
if [[ "${POSTPROCESS}" == "1" ]]; then
  python "${TRACK4_ROOT}/scripts/postprocess_predictions.py" \
    --input-dir "${PRED_DIR}" \
    --output-dir "${POSTPROCESSED_DIR}" \
    --label-space train \
    --min-component-size "${MIN_COMPONENT_SIZE}"
  RESTORE_SOURCE_DIR="${POSTPROCESSED_DIR}"
  OFFICIAL_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_official_labels_postprocessed"
fi

python "${TRACK4_ROOT}/scripts/restore_label_values.py" \
  --pred-dir "${RESTORE_SOURCE_DIR}" \
  --mapping-json "${MAPPING_JSON}" \
  --output-dir "${OFFICIAL_DIR}"

echo "Done. Official label predictions: ${OFFICIAL_DIR}"
