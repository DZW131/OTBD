#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  bash track4_wholeheart/scripts/predict_mr_aomyo_roi.sh

Submission-oriented MR inference pipeline:
  baseline nnU-Net 5-fold prediction
    -> class-aware-hd postprocess
    -> AO ROI add-only probability paste
    -> Myo ROI add-only probability paste
    -> official label restoration
    -> optional sanity check

Common environment overrides:
  MR_ROI_PRESET=score|safe       score: AO d3 + Myo d2, safe: AO d2 + Myo d2
  RUN_NAME=<name>                output folder under track4_wholeheart/outputs
  FOLDS="0 1 2 3 4"             baseline and ROI ensemble folds
  BASE_TRAINER=nnUNetTrainer     first-stage MR trainer
  ROI_TRAINER=nnUNetTrainerWholeHeartAug
  BASE_CHECKPOINT=checkpoint_final.pth
  ROI_CHECKPOINT=checkpoint_best.pth
  SANITY_CHECK=1                 check restored official-label outputs
  SKIP_BASE_PREDICT=1            reuse BASE_POSTPROCESSED_DIR instead of rerunning baseline
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DATASET_ID="${DATASET_ID:-402}"
DATASET_NAME="${DATASET_NAME:-Dataset402_CARE2026_WholeHeart_MR}"
AO_ROI_DATASET_ID="${AO_ROI_DATASET_ID:-452}"
MYO_ROI_DATASET_ID="${MYO_ROI_DATASET_ID:-453}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
BASE_TRAINER="${BASE_TRAINER:-nnUNetTrainer}"
ROI_TRAINER="${ROI_TRAINER:-nnUNetTrainerWholeHeartAug}"
FOLDS="${FOLDS:-0 1 2 3 4}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-checkpoint_final.pth}"
ROI_CHECKPOINT="${ROI_CHECKPOINT:-checkpoint_best.pth}"
MR_ROI_PRESET="${MR_ROI_PRESET:-score}"
RUN_NAME="${RUN_NAME:-mr_aomyo_roi_${MR_ROI_PRESET}}"
SANITY_CHECK="${SANITY_CHECK:-0}"
SKIP_BASE_PREDICT="${SKIP_BASE_PREDICT:-0}"
OVERWRITE="${OVERWRITE:-1}"

INPUT_DIR="${INPUT_DIR:-${nnUNet_raw}/${DATASET_NAME}/imagesTs}"
MAPPING_JSON="${MAPPING_JSON:-${nnUNet_raw}/${DATASET_NAME}/conversion_mapping.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRACK4_ROOT}/outputs/${RUN_NAME}}"
BASE_RAW_DIR="${BASE_RAW_DIR:-${OUTPUT_ROOT}/base_raw}"
BASE_POSTPROCESSED_DIR="${BASE_POSTPROCESSED_DIR:-${OUTPUT_ROOT}/base_class_aware_hd}"
AO_WORK_DIR="${AO_WORK_DIR:-${OUTPUT_ROOT}/ao_roi}"
MYO_WORK_DIR="${MYO_WORK_DIR:-${OUTPUT_ROOT}/myo_roi}"
AO_PASTED_DIR="${AO_PASTED_DIR:-${OUTPUT_ROOT}/ao_pasted_train_labels}"
FINAL_TRAIN_LABEL_DIR="${FINAL_TRAIN_LABEL_DIR:-${OUTPUT_ROOT}/train_labels}"
OFFICIAL_DIR="${OFFICIAL_DIR:-${OUTPUT_ROOT}/official_labels}"

AO_PROB_THRESHOLD="${AO_PROB_THRESHOLD:-0.90}"
MYO_PROB_THRESHOLD="${MYO_PROB_THRESHOLD:-0.90}"
MYO_DISTANCE_MM="${MYO_DISTANCE_MM:-2}"
ROI_MARGIN_MM="${ROI_MARGIN_MM:-20}"
ROI_MIN_MARGIN_VOXELS="${ROI_MIN_MARGIN_VOXELS:-8}"

case "${MR_ROI_PRESET}" in
  score)
    AO_DISTANCE_MM="${AO_DISTANCE_MM:-3}"
    ;;
  safe)
    AO_DISTANCE_MM="${AO_DISTANCE_MM:-2}"
    ;;
  *)
    echo "ERROR: MR_ROI_PRESET must be score or safe, got ${MR_ROI_PRESET}" >&2
    exit 1
    ;;
esac

NORMALIZED_FOLDS="$(echo "${FOLDS}" | xargs || true)"
if [[ "${NORMALIZED_FOLDS}" != "0 1 2 3 4" ]]; then
  echo "WARNING: MR AO/Myo ROI pipeline is submission-oriented with full 5-fold ensemble; current FOLDS=\"${NORMALIZED_FOLDS}\"." >&2
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "ERROR: INPUT_DIR does not exist: ${INPUT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${MAPPING_JSON}" ]]; then
  echo "ERROR: MAPPING_JSON does not exist: ${MAPPING_JSON}" >&2
  exit 1
fi

if [[ "${BASE_TRAINER}" == nnUNetTrainerWholeHeart* || "${ROI_TRAINER}" == nnUNetTrainerWholeHeart* ]]; then
  python "${TRACK4_ROOT}/scripts/install_wholeheart_trainer.py"
fi

if [[ "${OVERWRITE}" == "1" && "${SKIP_BASE_PREDICT}" == "1" ]]; then
  rm -rf "${AO_WORK_DIR}" "${MYO_WORK_DIR}" "${AO_PASTED_DIR}" "${FINAL_TRAIN_LABEL_DIR}" "${OFFICIAL_DIR}"
elif [[ "${OVERWRITE}" == "1" ]]; then
  rm -rf "${OUTPUT_ROOT}"
fi
mkdir -p "${OUTPUT_ROOT}" "${AO_WORK_DIR}" "${MYO_WORK_DIR}"

cat <<EOF
MR AO/Myo ROI inference
input_dir=${INPUT_DIR}
output_root=${OUTPUT_ROOT}
preset=${MR_ROI_PRESET}
folds=${NORMALIZED_FOLDS}
tta=enabled
baseline_trainer=${BASE_TRAINER}
baseline_checkpoint=${BASE_CHECKPOINT}
roi_trainer=${ROI_TRAINER}
roi_checkpoint=${ROI_CHECKPOINT}
ao_dataset=${AO_ROI_DATASET_ID}
myo_dataset=${MYO_ROI_DATASET_ID}
ao_prob_threshold=${AO_PROB_THRESHOLD}
ao_max_add_distance_mm=${AO_DISTANCE_MM}
myo_prob_threshold=${MYO_PROB_THRESHOLD}
myo_max_add_distance_mm=${MYO_DISTANCE_MM}
sanity_check=${SANITY_CHECK}
skip_base_predict=${SKIP_BASE_PREDICT}
EOF

if [[ "${SKIP_BASE_PREDICT}" != "1" ]]; then
  mkdir -p "${BASE_RAW_DIR}" "${BASE_POSTPROCESSED_DIR}"
  # shellcheck disable=SC2086
  nnUNetv2_predict \
    -i "${INPUT_DIR}" \
    -o "${BASE_RAW_DIR}" \
    -d "${DATASET_ID}" \
    -c "${CONFIGURATION}" \
    -tr "${BASE_TRAINER}" \
    -f ${FOLDS} \
    -chk "${BASE_CHECKPOINT}" \
    --save_probabilities

  python "${TRACK4_ROOT}/scripts/postprocess_predictions.py" \
    --input-dir "${BASE_RAW_DIR}" \
    --output-dir "${BASE_POSTPROCESSED_DIR}" \
    --label-space train \
    --preset class-aware-hd
else
  if [[ ! -d "${BASE_POSTPROCESSED_DIR}" ]]; then
    echo "ERROR: SKIP_BASE_PREDICT=1 requires BASE_POSTPROCESSED_DIR to exist: ${BASE_POSTPROCESSED_DIR}" >&2
    exit 1
  fi
fi

python "${TRACK4_ROOT}/scripts/mr_roi_refine.py" crop-inference \
  --images-dir "${INPUT_DIR}" \
  --pred-dir "${BASE_POSTPROCESSED_DIR}" \
  --output-images-ts-dir "${AO_WORK_DIR}/imagesTs" \
  --metadata-json "${AO_WORK_DIR}/metadata.json" \
  --target-label 6 \
  --target-name AO \
  --margin-mm "${ROI_MARGIN_MM}" \
  --min-margin-voxels "${ROI_MIN_MARGIN_VOXELS}" \
  --overwrite

# shellcheck disable=SC2086
nnUNetv2_predict \
  -i "${AO_WORK_DIR}/imagesTs" \
  -o "${AO_WORK_DIR}/pred_probs" \
  -d "${AO_ROI_DATASET_ID}" \
  -c "${CONFIGURATION}" \
  -tr "${ROI_TRAINER}" \
  -f ${FOLDS} \
  -chk "${ROI_CHECKPOINT}" \
  --save_probabilities

python "${TRACK4_ROOT}/scripts/mr_roi_refine.py" paste \
  --base-pred-dir "${BASE_POSTPROCESSED_DIR}" \
  --roi-pred-dir "${AO_WORK_DIR}/pred_probs" \
  --roi-prob-dir "${AO_WORK_DIR}/pred_probs" \
  --prob-threshold "${AO_PROB_THRESHOLD}" \
  --max-add-distance-mm "${AO_DISTANCE_MM}" \
  --metadata-json "${AO_WORK_DIR}/metadata.json" \
  --output-dir "${AO_PASTED_DIR}" \
  --target-label 6 \
  --roi-label 1 \
  --protect-labels 1,2,3,4,5,7 \
  --merge-mode add-only \
  --postprocess class-aware-hd

python "${TRACK4_ROOT}/scripts/mr_roi_refine.py" crop-inference \
  --images-dir "${INPUT_DIR}" \
  --pred-dir "${BASE_POSTPROCESSED_DIR}" \
  --output-images-ts-dir "${MYO_WORK_DIR}/imagesTs" \
  --metadata-json "${MYO_WORK_DIR}/metadata.json" \
  --target-label 5 \
  --target-name Myo \
  --margin-mm "${ROI_MARGIN_MM}" \
  --min-margin-voxels "${ROI_MIN_MARGIN_VOXELS}" \
  --overwrite

# shellcheck disable=SC2086
nnUNetv2_predict \
  -i "${MYO_WORK_DIR}/imagesTs" \
  -o "${MYO_WORK_DIR}/pred_probs" \
  -d "${MYO_ROI_DATASET_ID}" \
  -c "${CONFIGURATION}" \
  -tr "${ROI_TRAINER}" \
  -f ${FOLDS} \
  -chk "${ROI_CHECKPOINT}" \
  --save_probabilities

python "${TRACK4_ROOT}/scripts/mr_roi_refine.py" paste \
  --base-pred-dir "${AO_PASTED_DIR}" \
  --roi-pred-dir "${MYO_WORK_DIR}/pred_probs" \
  --roi-prob-dir "${MYO_WORK_DIR}/pred_probs" \
  --prob-threshold "${MYO_PROB_THRESHOLD}" \
  --max-add-distance-mm "${MYO_DISTANCE_MM}" \
  --metadata-json "${MYO_WORK_DIR}/metadata.json" \
  --output-dir "${FINAL_TRAIN_LABEL_DIR}" \
  --target-label 5 \
  --roi-label 1 \
  --protect-labels 1,2,3,4,6,7 \
  --merge-mode add-only \
  --postprocess class-aware-hd

python "${TRACK4_ROOT}/scripts/restore_label_values.py" \
  --pred-dir "${FINAL_TRAIN_LABEL_DIR}" \
  --mapping-json "${MAPPING_JSON}" \
  --output-dir "${OFFICIAL_DIR}"

if [[ "${SANITY_CHECK}" == "1" ]]; then
  python "${TRACK4_ROOT}/scripts/check_prediction_sanity.py" \
    --input-dir "${INPUT_DIR}" \
    --pred-dir "${OFFICIAL_DIR}" \
    --mapping-json "${MAPPING_JSON}" \
    --label-space official
fi

echo "Done."
echo "Train-label predictions: ${FINAL_TRAIN_LABEL_DIR}"
echo "Official-label predictions: ${OFFICIAL_DIR}"
