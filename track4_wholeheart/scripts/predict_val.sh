#!/usr/bin/env bash
set -euo pipefail

MODALITY="${1:-}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainer}"
FOLDS="${FOLDS:-}"
POSTPROCESS="${POSTPROCESS:-0}"
POSTPROCESS_PRESET="${POSTPROCESS_PRESET:-legacy}"
MIN_COMPONENT_SIZE="${MIN_COMPONENT_SIZE:-0}"
WHOLEHEART_DISTANCE_MM="${WHOLEHEART_DISTANCE_MM:-25}"
VESSEL_MIN_COMPONENT_SIZE="${VESSEL_MIN_COMPONENT_SIZE:-20}"
SANITY_CHECK="${SANITY_CHECK:-0}"
CT_AOPA_SOFT="${CT_AOPA_SOFT:-0}"
CT_AOPA_THRESHOLD_MODE="${CT_AOPA_THRESHOLD_MODE:-adaptive}"
CT_AOPA_P_CLASS_THRESHOLD="${CT_AOPA_P_CLASS_THRESHOLD:-0.25}"
CT_AOPA_STRONG_OTHER_THRESHOLD="${CT_AOPA_STRONG_OTHER_THRESHOLD:-0.65}"
CT_AOPA_BBOX_MARGIN_MM="${CT_AOPA_BBOX_MARGIN_MM:-15}"
CT_AOPA_MAX_DISTANCE_MM="${CT_AOPA_MAX_DISTANCE_MM:-15}"
CT_AOPA_COMPONENT_SCORE="${CT_AOPA_COMPONENT_SCORE:-0}"
CT_AOPA_MIN_VOXELS="${CT_AOPA_MIN_VOXELS:-30}"
CT_AOPA_MIN_MEAN_PROB="${CT_AOPA_MIN_MEAN_PROB:-0.35}"
CT_AOPA_MIN_P10_PROB="${CT_AOPA_MIN_P10_PROB:-0.15}"
CT_AOPA_MAX_MEAN_ENTROPY="${CT_AOPA_MAX_MEAN_ENTROPY:-}"
CT_AOPA_ENABLE_CLOSING="${CT_AOPA_ENABLE_CLOSING:-0}"
CT_AOPA_CLOSING_RADIUS_VOXEL="${CT_AOPA_CLOSING_RADIUS_VOXEL:-1}"

if [[ "${MODALITY}" != "ct" && "${MODALITY}" != "mr" ]]; then
  echo "Usage: $0 {ct|mr}" >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [[ "${MODALITY}" == "ct" ]]; then
  DATASET_ID=401
  DATASET_NAME="Dataset401_CARE2026_WholeHeart_CT"
  EXPECTED_POSTPROCESS_PRESET="legacy"
  EXPECTED_POSTPROCESS_NOTE="CT current best postprocess preset is legacy"
else
  DATASET_ID=402
  DATASET_NAME="Dataset402_CARE2026_WholeHeart_MR"
  EXPECTED_POSTPROCESS_PRESET="class-aware-hd"
  EXPECTED_POSTPROCESS_NOTE="MR current best postprocess preset is class-aware-hd"
fi

NORMALIZED_FOLDS="$(echo "${FOLDS}" | xargs || true)"

INPUT_DIR="${nnUNet_raw}/${DATASET_NAME}/imagesTs"
PRED_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_nnunet"
OFFICIAL_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_official_labels"
POSTPROCESSED_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_nnunet_postprocessed"
CT_AOPA_POSTPROCESSED_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_nnunet_postprocessed_aopa_soft"
MAPPING_JSON="${nnUNet_raw}/${DATASET_NAME}/conversion_mapping.json"

mkdir -p "${PRED_DIR}" "${OFFICIAL_DIR}"

echo "Predicting ${MODALITY^^} validation set"
echo "Input: ${INPUT_DIR}"
echo "Prediction dir: ${PRED_DIR}"
echo "dataset_id=${DATASET_ID}"
echo "dataset_name=${DATASET_NAME}"
echo "configuration=${CONFIGURATION}"
echo "trainer=${TRAINER}"
echo "folds=${NORMALIZED_FOLDS:-<nnUNet default>}"
echo "tta=enabled"
echo "save_probabilities=enabled"
echo "postprocess=${POSTPROCESS}"
echo "postprocess_preset=${POSTPROCESS_PRESET}"
echo "expected_postprocess_preset=${EXPECTED_POSTPROCESS_PRESET} (${EXPECTED_POSTPROCESS_NOTE})"
echo "sanity_check=${SANITY_CHECK}"
echo "ct_aopa_soft=${CT_AOPA_SOFT}"
if [[ "${MODALITY}" == "ct" ]]; then
  echo "ct_aopa_threshold_mode=${CT_AOPA_THRESHOLD_MODE}"
  echo "ct_aopa_component_score=${CT_AOPA_COMPONENT_SCORE}"
  echo "ct_aopa_enable_closing=${CT_AOPA_ENABLE_CLOSING}"
fi

if [[ "${NORMALIZED_FOLDS}" != "0 1 2 3 4" ]]; then
  echo "WARNING: ${MODALITY^^} prediction is not using full 5-fold ensemble (FOLDS=\"${NORMALIZED_FOLDS:-}\"). Use FOLDS=\"0 1 2 3 4\" for submission-oriented inference." >&2
fi

if [[ "${POSTPROCESS}" != "1" ]]; then
  echo "WARNING: postprocess is disabled. This is fine for raw ablations but not the current submission-oriented ${MODALITY^^} baseline." >&2
elif [[ "${POSTPROCESS_PRESET}" != "${EXPECTED_POSTPROCESS_PRESET}" ]]; then
  echo "WARNING: ${MODALITY^^} current best postprocess preset is ${EXPECTED_POSTPROCESS_PRESET}, but POSTPROCESS_PRESET=${POSTPROCESS_PRESET}." >&2
fi

if [[ "${CT_AOPA_SOFT}" == "1" ]]; then
  if [[ "${MODALITY}" != "ct" ]]; then
    echo "WARNING: CT_AOPA_SOFT is CT-only and will be ignored for ${MODALITY^^}." >&2
  elif [[ "${POSTPROCESS}" != "1" || "${POSTPROCESS_PRESET}" != "legacy" ]]; then
    echo "ERROR: CT AOPA soft patch requires POSTPROCESS=1 and POSTPROCESS_PRESET=legacy." >&2
    exit 1
  fi
fi

if [[ "${TRAINER}" == nnUNetTrainerWholeHeart* ]]; then
  python "${TRACK4_ROOT}/scripts/install_wholeheart_trainer.py"
fi

if [[ -n "${FOLDS}" ]]; then
  # shellcheck disable=SC2086
  nnUNetv2_predict -i "${INPUT_DIR}" -o "${PRED_DIR}" -d "${DATASET_ID}" -c "${CONFIGURATION}" -tr "${TRAINER}" -f ${FOLDS} --save_probabilities
else
  nnUNetv2_predict -i "${INPUT_DIR}" -o "${PRED_DIR}" -d "${DATASET_ID}" -c "${CONFIGURATION}" -tr "${TRAINER}" --save_probabilities
fi

RESTORE_SOURCE_DIR="${PRED_DIR}"
if [[ "${POSTPROCESS}" == "1" ]]; then
  python "${TRACK4_ROOT}/scripts/postprocess_predictions.py" \
    --input-dir "${PRED_DIR}" \
    --output-dir "${POSTPROCESSED_DIR}" \
    --label-space train \
    --preset "${POSTPROCESS_PRESET}" \
    --wholeheart-distance-mm "${WHOLEHEART_DISTANCE_MM}" \
    --vessel-min-component-size "${VESSEL_MIN_COMPONENT_SIZE}" \
    --min-component-size "${MIN_COMPONENT_SIZE}"
  RESTORE_SOURCE_DIR="${POSTPROCESSED_DIR}"
  OFFICIAL_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_official_labels_postprocessed"

  if [[ "${MODALITY}" == "ct" && "${CT_AOPA_SOFT}" == "1" ]]; then
    AOPA_ARGS=()
    if [[ "${CT_AOPA_COMPONENT_SCORE}" == "1" ]]; then
      AOPA_ARGS+=(--component-score)
    fi
    if [[ "${CT_AOPA_ENABLE_CLOSING}" == "1" ]]; then
      AOPA_ARGS+=(--enable-closing)
    fi
    if [[ -n "${CT_AOPA_MAX_MEAN_ENTROPY}" ]]; then
      AOPA_ARGS+=(--max-mean-entropy "${CT_AOPA_MAX_MEAN_ENTROPY}")
    fi

    python "${TRACK4_ROOT}/scripts/postprocess_ct_aopa_soft.py" \
      --input-dir "${POSTPROCESSED_DIR}" \
      --probability-dir "${PRED_DIR}" \
      --output-dir "${CT_AOPA_POSTPROCESSED_DIR}" \
      --threshold-mode "${CT_AOPA_THRESHOLD_MODE}" \
      --p-class-threshold "${CT_AOPA_P_CLASS_THRESHOLD}" \
      --strong-other-threshold "${CT_AOPA_STRONG_OTHER_THRESHOLD}" \
      --bbox-margin-mm "${CT_AOPA_BBOX_MARGIN_MM}" \
      --max-distance-mm "${CT_AOPA_MAX_DISTANCE_MM}" \
      --min-voxels "${CT_AOPA_MIN_VOXELS}" \
      --min-mean-prob "${CT_AOPA_MIN_MEAN_PROB}" \
      --min-p10-prob "${CT_AOPA_MIN_P10_PROB}" \
      --closing-radius-voxel "${CT_AOPA_CLOSING_RADIUS_VOXEL}" \
      "${AOPA_ARGS[@]}"
    RESTORE_SOURCE_DIR="${CT_AOPA_POSTPROCESSED_DIR}"
    OFFICIAL_DIR="${TRACK4_ROOT}/outputs/${MODALITY}_val_official_labels_postprocessed_aopa_soft"
  fi
fi

python "${TRACK4_ROOT}/scripts/restore_label_values.py" \
  --pred-dir "${RESTORE_SOURCE_DIR}" \
  --mapping-json "${MAPPING_JSON}" \
  --output-dir "${OFFICIAL_DIR}"

if [[ "${SANITY_CHECK}" == "1" ]]; then
  python "${TRACK4_ROOT}/scripts/check_prediction_sanity.py" \
    --input-dir "${INPUT_DIR}" \
    --pred-dir "${OFFICIAL_DIR}" \
    --mapping-json "${MAPPING_JSON}" \
    --label-space official
fi

echo "Done. Official label predictions: ${OFFICIAL_DIR}"
