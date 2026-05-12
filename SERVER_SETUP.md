# Server Setup for `test_nnunetv2_tta`

This package is the 2025 CARE liver nnU-Netv2 inference container. It expects challenge-style input at `/input` and writes predictions to `/output/LiSeg_pred`.

## 1. Server prerequisites

- Linux server with NVIDIA GPU
- Docker installed
- NVIDIA Container Toolkit installed
- Enough disk for Docker image build and checkpoints

Quick checks:

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

## 2. Upload the project

Upload this folder to the server:

```text
test_nnunetv2_tta/
  Dockerfile
  run_infer.sh
  DATASET/
```

Do not rely on git for this directory unless it is in a clean standalone repository. A direct `scp`, `rsync`, or `.tar.gz` upload is safer.

Example from Windows PowerShell:

```powershell
cd D:\work\test_nnunetv2_tta\test_nnunetv2_tta
.\pack_for_server.ps1
```

This creates `D:\work\test_nnunetv2_tta_server.tar.gz` and skips files that are not needed by the default inference path.

On Linux, unpack it with:

```bash
tar -xzf test_nnunetv2_tta_server.tar.gz
```

## 3. Build Docker image

On the server:

```bash
cd /path/to/test_nnunetv2_tta
docker build -t care-liver-nnunetv2-tta:2025 .
```

The `.dockerignore` intentionally skips `checkpoint_best.pth`, training logs, progress images, and Python caches. The default inference script uses `checkpoint_final.pth`.

## 4. Expected input layout

The script scans:

```text
/input/**/GED4.nii.gz
```

For example:

```text
input/
  Case001/
    GED4.nii.gz
  Case002/
    GED4.nii.gz
```

## 5. Run inference

```bash
mkdir -p /path/to/input /path/to/output

docker run --rm --gpus all \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  care-liver-nnunetv2-tta:2025
```

Expected output:

```text
/path/to/output/LiSeg_pred/<case>/GED4_pred.nii.gz
/path/to/output/log_infer.txt
```

## 6. Smoke test after build

Use one small real case if available:

```bash
find /path/to/input -name GED4.nii.gz | head
docker run --rm --gpus all \
  -v /path/to/input:/input \
  -v /path/to/output:/output \
  care-liver-nnunetv2-tta:2025
find /path/to/output -type f | head
```

## 7. Known risks

- The Dockerfile clones the current `MIC-DKFZ/nnUNet` default branch during build. If upstream nnU-Net changes, rebuilds may break. For a competition submission, pin a tested nnU-Net commit.
- The inference code enables CoTTA based on whether the original mapped case path contains `C`. This was specific to the 2025 liver setup.
- This package is for the old liver task, not the 2026 Track 4 whole-heart task.
