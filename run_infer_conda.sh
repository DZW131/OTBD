#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
INPUT_DIR="${INPUT_DIR:-${PROJECT_ROOT}/input}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/output}"

export PROJECT_ROOT
export INPUT_DIR
export OUTPUT_DIR
export nnUNet_raw="${nnUNet_raw:-${PROJECT_ROOT}/DATASET/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJECT_ROOT}/DATASET/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJECT_ROOT}/DATASET/nnUNet_result}"
export TEST_NAME_PRED_MAPPING="${TEST_NAME_PRED_MAPPING:-${nnUNet_raw}/Dataset078_LIVER/Test_name_pred_mapping.json}"

IMAGES_TS="${nnUNet_raw}/Dataset078_LIVER/imagesTs"
IMAGES_TS_GENERATE="${nnUNet_raw}/Dataset078_LIVER/imagesTs_generate"
IMAGES_TS_GENERATE_PP="${nnUNet_raw}/Dataset078_LIVER/imagesTs_generate_PP"
POSTPROCESS_DIR="${nnUNet_results}/Dataset078_LIVER/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4"

export NNUNET_IMAGES_TS="${IMAGES_TS}"
export NNUNET_IMAGES_TS_GENERATE_PP="${IMAGES_TS_GENERATE_PP}"

mkdir -p "${INPUT_DIR}" "${OUTPUT_DIR}" "${IMAGES_TS}" "${IMAGES_TS_GENERATE}" "${IMAGES_TS_GENERATE_PP}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "INPUT_DIR=${INPUT_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"

echo "Running input name conversion"
python "${PROJECT_ROOT}/DATASET/test_template_2_nnunet_template.py"

echo "Running nnUNetv2_predict"
nnUNetv2_predict \
  -i "${IMAGES_TS}" \
  -o "${IMAGES_TS_GENERATE}" \
  -d 78 -c 3d_fullres --save_probabilities

echo "Running nnUNetv2_apply_postprocessing"
nnUNetv2_apply_postprocessing \
  -i "${IMAGES_TS_GENERATE}" \
  -o "${IMAGES_TS_GENERATE_PP}" \
  -pp_pkl_file "${POSTPROCESS_DIR}/postprocessing.pkl" \
  -np 8 \
  -plans_json "${POSTPROCESS_DIR}/plans.json"

echo "Running output name conversion"
python "${PROJECT_ROOT}/DATASET/nnunet_template_2_test_template.py"

echo "Done. Outputs are under ${OUTPUT_DIR}/LiSeg_pred"
