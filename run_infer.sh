#!/bin/bash
set -e  # 脚本出错立即退出

# 第一次运行脚本
echo "✅ 运行第一次模板处理脚本"
python /workspace/DATASET/test_template_2_nnunet_template.py

# 执行 nnUNet 推理
echo "✅ 运行 nnUNetv2_predict"
nnUNetv2_predict \
  -i /workspace/DATASET/nnUNet_raw/Dataset078_LIVER/imagesTs \
  -o /workspace/DATASET/nnUNet_raw/Dataset078_LIVER/imagesTs_generate \
  -d 78 -c 3d_fullres --save_probabilities


# 后处理
echo "✅ 运行 nnUNetv2_postprocessing"
nnUNetv2_apply_postprocessing \
  -i /workspace/DATASET/nnUNet_raw/Dataset078_LIVER/imagesTs_generate \
  -o /workspace/DATASET/nnUNet_raw/Dataset078_LIVER/imagesTs_generate_PP \
  -pp_pkl_file /workspace/DATASET/nnUNet_result/Dataset078_LIVER/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
  -np 8 \
  -plans_json /workspace/DATASET/nnUNet_result/Dataset078_LIVER/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json

# 第二次运行脚本
echo "✅ 运行第二次模板处理脚本"
python /workspace/DATASET/nnunet_template_2_test_template.py

echo "🎉 所有任务已完成"