# OTBD / test_nnunetv2_tta

This repository contains the CARE liver nnU-Netv2 TTA inference package and deployment scripts.

Large model checkpoint files (`checkpoint_*.pth`) are intentionally not tracked by Git because each file is larger than GitHub's regular file size limit. Use the server package workflow in `SERVER_SETUP.md` when a runnable bundle with checkpoints is needed.

## Server package

From Windows PowerShell:

```powershell
.\pack_for_server.ps1
```

Then upload the generated `test_nnunetv2_tta_server.tar.gz` to the server and follow `SERVER_SETUP.md`.

## Conda-only server

If Docker is unavailable on the server, upload the same server package and follow `CONDA_SETUP.md`:

```bash
tar -xzf test_nnunetv2_tta_server.tar.gz
cd test_nnunetv2_tta
bash setup_conda_env.sh
conda activate nnunet_tta
INPUT_DIR=/path/to/input OUTPUT_DIR=/path/to/output ./run_infer_conda.sh
```
