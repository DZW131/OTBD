#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
Usage:
  bash track4_wholeheart/scripts/predict_mr_final_roi_mae_grata.sh

Submission-oriented MR final pipeline:
  tuned AO/Myo ROI prediction
    -> MAE ResEncL prediction with conservative GraTa-lite TTA
    -> add-only import of MAE LV/RV/Myo/PA into tuned ROI
    -> protect LA/RA/AO from MAE overwrite
    -> class-aware-hd postprocess
    -> restore official CARE label values
    -> optional sanity check

Common environment overrides:
  RUN_NAME=mr_final_roi_mae_grata
  INPUT_DIR=<imagesTs dir>
  MAPPING_JSON=<conversion_mapping.json>
  FOLDS="0 1 2 3 4"
  GPU=0
  SKIP_TUNED_ROI=1              reuse BASE_TRAIN_LABEL_DIR
  BASE_TRAIN_LABEL_DIR=<dir>    tuned ROI train-label predictions
  SKIP_MAE_GRATA=1              reuse MAE_GRATA_POST_DIR
  MAE_GRATA_POST_DIR=<dir>      MAE+GraTa postprocessed train-label predictions
  SANITY_CHECK=1

GraTa-lite defaults:
  GRATA_STEPS=1
  GRATA_LR=1e-5
  GRATA_PARAM_SCOPE=norm-affine
  GRATA_ALIGNMENT_THRESHOLD=0.0

Fallback:
  Set SKIP_MAE_GRATA=1 and MAE_GRATA_POST_DIR to a non-GraTa MAE prediction
  directory to reproduce tuned ROI + MAE without GraTa.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

DATASET_ID="${DATASET_ID:-402}"
DATASET_NAME="${DATASET_NAME:-Dataset402_CARE2026_WholeHeart_MR}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLDS="${FOLDS:-0 1 2 3 4}"
GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-mr_final_roi_mae_grata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${TRACK4_ROOT}/outputs/${RUN_NAME}}"
INPUT_DIR="${INPUT_DIR:-${nnUNet_raw}/${DATASET_NAME}/imagesTs}"
MAPPING_JSON="${MAPPING_JSON:-${nnUNet_raw}/${DATASET_NAME}/conversion_mapping.json}"
SANITY_CHECK="${SANITY_CHECK:-0}"
OVERWRITE="${OVERWRITE:-1}"
PYTHON="${PYTHON:-python}"

SKIP_TUNED_ROI="${SKIP_TUNED_ROI:-0}"
TUNED_ROI_ROOT="${TUNED_ROI_ROOT:-${OUTPUT_ROOT}/tuned_roi}"
BASE_TRAIN_LABEL_DIR="${BASE_TRAIN_LABEL_DIR:-${TUNED_ROI_ROOT}/train_labels}"

SKIP_MAE_GRATA="${SKIP_MAE_GRATA:-0}"
MAE_GRATA_RAW_DIR="${MAE_GRATA_RAW_DIR:-${OUTPUT_ROOT}/mae_grata_raw}"
MAE_GRATA_POST_DIR="${MAE_GRATA_POST_DIR:-${OUTPUT_ROOT}/mae_grata_classaware}"
FUSION_ROOT="${FUSION_ROOT:-${OUTPUT_ROOT}/roi_mae_grata_fusion}"
FINAL_TRAIN_LABEL_DIR="${FINAL_TRAIN_LABEL_DIR:-${FUSION_ROOT}/fused_classaware}"
OFFICIAL_DIR="${OFFICIAL_DIR:-${OUTPUT_ROOT}/official_labels}"

MAE_FOLDS="${MAE_FOLDS:-${FOLDS}}"
MAE_CHECKPOINT_NAME="${MAE_CHECKPOINT_NAME:-checkpoint_final.pth}"
GRATA_STEPS="${GRATA_STEPS:-1}"
GRATA_LR="${GRATA_LR:-1e-5}"
GRATA_PARAM_SCOPE="${GRATA_PARAM_SCOPE:-norm-affine}"
GRATA_ENTROPY_WEIGHT="${GRATA_ENTROPY_WEIGHT:-1.0}"
GRATA_CONSISTENCY_WEIGHT="${GRATA_CONSISTENCY_WEIGHT:-1.0}"
GRATA_ALIGNMENT_THRESHOLD="${GRATA_ALIGNMENT_THRESHOLD:-0.0}"
GRATA_NOISE_STD="${GRATA_NOISE_STD:-0.01}"
GRATA_BRIGHTNESS_STD="${GRATA_BRIGHTNESS_STD:-0.02}"
GRATA_MAX_GRAD_NORM="${GRATA_MAX_GRAD_NORM:-1.0}"
TILE_STEP_SIZE="${TILE_STEP_SIZE:-0.5}"
DISABLE_MIRRORING="${DISABLE_MIRRORING:-0}"

safe_rm_output_dir() {
  local target="$1"
  if [[ "${OVERWRITE}" != "1" || -z "${target}" ]]; then
    return 0
  fi
  case "${target}" in
    "${TRACK4_ROOT}/outputs/"*)
      rm -rf "${target}"
      ;;
    *)
      echo "ERROR: refusing to remove non-output directory: ${target}" >&2
      exit 1
      ;;
  esac
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: missing directory: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: missing file: ${path}" >&2
    exit 1
  fi
}

require_dir "${INPUT_DIR}"
require_file "${MAPPING_JSON}"
mkdir -p "${OUTPUT_ROOT}"

cat <<EOF
MR final ROI + MAE + GraTa-lite pipeline
output_root=${OUTPUT_ROOT}
input_dir=${INPUT_DIR}
mapping_json=${MAPPING_JSON}
folds=${FOLDS}
gpu=${GPU}
skip_tuned_roi=${SKIP_TUNED_ROI}
base_train_label_dir=${BASE_TRAIN_LABEL_DIR}
skip_mae_grata=${SKIP_MAE_GRATA}
mae_grata_post_dir=${MAE_GRATA_POST_DIR}
fusion_root=${FUSION_ROOT}
official_dir=${OFFICIAL_DIR}
grata_steps=${GRATA_STEPS}
grata_lr=${GRATA_LR}
grata_param_scope=${GRATA_PARAM_SCOPE}
tta=$([[ "${DISABLE_MIRRORING}" == "1" ]] && echo disabled || echo enabled)
sanity_check=${SANITY_CHECK}
EOF

if [[ "${SKIP_TUNED_ROI}" != "1" ]]; then
  safe_rm_output_dir "${TUNED_ROI_ROOT}"
  MR_ROI_PRESET="${MR_ROI_PRESET:-tuned}" \
  OUTPUT_ROOT="${TUNED_ROI_ROOT}" \
  INPUT_DIR="${INPUT_DIR}" \
  MAPPING_JSON="${MAPPING_JSON}" \
  FOLDS="${FOLDS}" \
  SANITY_CHECK=0 \
  OVERWRITE=1 \
    bash "${TRACK4_ROOT}/scripts/predict_mr_aomyo_roi.sh"
else
  require_dir "${BASE_TRAIN_LABEL_DIR}"
fi

if [[ "${SKIP_MAE_GRATA}" != "1" ]]; then
  safe_rm_output_dir "${MAE_GRATA_RAW_DIR}"
  safe_rm_output_dir "${MAE_GRATA_POST_DIR}"
  (
    source "${TRACK4_ROOT}/scripts/env_mae.sh"
    MAE_MODEL_DIR="${MAE_MODEL_DIR:-${nnUNet_results}/${DATASET_NAME}/PretrainedTrainer__ptPlans__ResEncL_OpenMind_MAE____Spacing__1.00_0.89_0.89___Norm__Z__3d_fullres}"
    require_dir "${MAE_MODEL_DIR}"
    EXTRA_ARGS=()
    if [[ "${DISABLE_MIRRORING}" == "1" ]]; then
      EXTRA_ARGS+=(--disable-mirroring)
    fi
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="${GPU}" "${MAE_BIN_DIR}/python" "${TRACK4_ROOT}/scripts/predict_grata_lite.py" \
      --model-dir "${MAE_MODEL_DIR}" \
      --input-dir "${INPUT_DIR}" \
      --output-dir "${MAE_GRATA_RAW_DIR}" \
      --folds ${MAE_FOLDS} \
      --checkpoint "${MAE_CHECKPOINT_NAME}" \
      --device cuda \
      --tile-step-size "${TILE_STEP_SIZE}" \
      --overwrite \
      --steps "${GRATA_STEPS}" \
      --lr "${GRATA_LR}" \
      --param-scope "${GRATA_PARAM_SCOPE}" \
      --entropy-weight "${GRATA_ENTROPY_WEIGHT}" \
      --consistency-weight "${GRATA_CONSISTENCY_WEIGHT}" \
      --alignment-threshold "${GRATA_ALIGNMENT_THRESHOLD}" \
      --noise-std "${GRATA_NOISE_STD}" \
      --brightness-std "${GRATA_BRIGHTNESS_STD}" \
      --max-grad-norm "${GRATA_MAX_GRAD_NORM}" \
      "${EXTRA_ARGS[@]}"
  )

  "${PYTHON}" "${TRACK4_ROOT}/scripts/postprocess_predictions.py" \
    --input-dir "${MAE_GRATA_RAW_DIR}" \
    --output-dir "${MAE_GRATA_POST_DIR}" \
    --label-space train \
    --preset class-aware-hd
else
  require_dir "${MAE_GRATA_POST_DIR}"
fi

BASE_ROOT="${BASE_TRAIN_LABEL_DIR}" \
DONOR_ROOT="${MAE_GRATA_POST_DIR}" \
OUTPUT_ROOT="${FUSION_ROOT}" \
PYTHON="${PYTHON}" \
POSTPROCESS=1 \
POSTPROCESS_PRESET=class-aware-hd \
OVERWRITE=1 \
  bash "${TRACK4_ROOT}/scripts/fuse_mr_roi_mae_labels.sh"

"${PYTHON}" "${TRACK4_ROOT}/scripts/restore_label_values.py" \
  --pred-dir "${FINAL_TRAIN_LABEL_DIR}" \
  --mapping-json "${MAPPING_JSON}" \
  --output-dir "${OFFICIAL_DIR}"

if [[ "${SANITY_CHECK}" == "1" ]]; then
  "${PYTHON}" "${TRACK4_ROOT}/scripts/check_prediction_sanity.py" \
    --input-dir "${INPUT_DIR}" \
    --pred-dir "${OFFICIAL_DIR}" \
    --mapping-json "${MAPPING_JSON}" \
    --label-space official
fi

echo "Done."
echo "Final train-label predictions: ${FINAL_TRAIN_LABEL_DIR}"
echo "Official-label predictions: ${OFFICIAL_DIR}"
