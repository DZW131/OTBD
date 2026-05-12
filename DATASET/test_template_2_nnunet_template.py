#### 把给定的 test 数据转换成 nnunet 可以接受的样子
import os
import glob
import json
import shutil

# 输入输出路径
project_root = os.environ.get('PROJECT_ROOT', '/workspace')
input_root = os.environ.get('INPUT_DIR', '/input')  # 修改为你的路径
output_dir = os.environ.get(
    'NNUNET_IMAGES_TS',
    os.path.join(project_root, 'DATASET/nnUNet_raw/Dataset078_LIVER/imagesTs')
)        # 修改为保存路径
mapping_file = os.environ.get(
    'TEST_NAME_PRED_MAPPING',
    os.path.join(project_root, 'DATASET/nnUNet_raw/Dataset078_LIVER/Test_name_pred_mapping.json')
)  # 保存映射字典的文件

# 创建输出目录（如不存在）
os.makedirs(output_dir, exist_ok=True)

# 初始化命名编号与映射字典
index = 1000
name_map = {}

GED4_list = sorted(glob.glob(os.path.join(input_root, "**", "GED4.nii.gz"), recursive=True))
# 遍历每个子文件夹
for GED4_path in GED4_list:
    
    index += 1
    nnunet_name = f"LIVER_{index:04d}_0000.nii.gz"
    pred_name = f"LIVER_{index:04d}.nii.gz"

    shutil.copy(GED4_path, os.path.join(output_dir, nnunet_name))

    ori_name = GED4_path.split("/")[-2] + "/" + "GED4_pred.nii.gz"
    name_map[pred_name] = ori_name
    print(f"Copied: {ori_name} \n--> {pred_name}")
    
with open(mapping_file, 'w') as f:
    json.dump(name_map, f, indent=4)

