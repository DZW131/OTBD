# Conda Setup for Servers Without Docker

Use this when the server cannot run Docker but has Conda and an NVIDIA GPU.

## 1. Upload the runnable project

The GitHub repository does not contain `checkpoint_*.pth` files because they are too large for regular GitHub storage.

Use one of these:

- Upload `test_nnunetv2_tta_server.tar.gz`, which includes `checkpoint_final.pth`.
- Or clone the GitHub repository and manually copy the five `checkpoint_final.pth` files back into:

```text
DATASET/nnUNet_result/Dataset078_LIVER/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
DATASET/nnUNet_result/Dataset078_LIVER/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_1/checkpoint_final.pth
DATASET/nnUNet_result/Dataset078_LIVER/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_2/checkpoint_final.pth
DATASET/nnUNet_result/Dataset078_LIVER/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth
DATASET/nnUNet_result/Dataset078_LIVER/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_4/checkpoint_final.pth
```

## 2. Create the Conda environment

On the server:

```bash
cd /path/to/test_nnunetv2_tta
bash setup_conda_env.sh
```

By default this creates an environment named `nnunet_tta` and installs PyTorch from the CUDA 12.6 wheel index.

If the server driver does not support CUDA 12.6, choose another PyTorch wheel index before running setup, for example:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash setup_conda_env.sh
```

## 3. Run inference

Input should contain cases with `GED4.nii.gz`:

```text
input/
  Case001/GED4.nii.gz
  Case002/GED4.nii.gz
```

Run:

```bash
conda activate nnunet_tta
INPUT_DIR=/path/to/input OUTPUT_DIR=/path/to/output ./run_infer_conda.sh
```

Output:

```text
/path/to/output/LiSeg_pred/<case>/GED4_pred.nii.gz
/path/to/output/log_infer.txt
```

## 4. Useful checks

```bash
nvidia-smi
conda activate nnunet_tta
python - <<'PY'
import torch, nnunetv2
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("nnunetv2:", nnunetv2.__path__)
PY
```

If `nnUNetv2_predict` cannot find the custom CoTTA code, rerun:

```bash
conda activate nnunet_tta
python scripts/install_custom_nnunet_files.py
```
