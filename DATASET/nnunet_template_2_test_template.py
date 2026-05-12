### 根据json转换pred到正常格式的名字
import os
import shutil
import json

# 配置路径
mapping_json = '/workspace/DATASET/nnUNet_raw/Dataset078_LIVER/Test_name_pred_mapping.json'  # 映射文件路径
pred_DIR = '/workspace/DATASET/nnUNet_raw/Dataset078_LIVER/imagesTs_generate_PP'
target_dir = '/output/LiSeg_pred'                        # 复制目标目录

# 创建目标文件夹
os.makedirs(target_dir, exist_ok=True)

# 加载映射字典
with open(mapping_json, 'r') as f:
    name_map = json.load(f)

# 执行复制并重命名
for new_name, ori_path in name_map.items():
    dst_path = os.path.join(target_dir, ori_path)
    # print(dst_path)
    pred_path = os.path.join(pred_DIR, new_name)
    # print(pred_path)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(pred_path, dst_path)
    print(f"Copied: {pred_path} \n--> {dst_path}")

print(f"\n✅ All files copied to: {target_dir}")
