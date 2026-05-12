#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-track4_wholeheart}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Load your conda module or install Miniconda first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.11 -y
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip setuptools wheel

python -m pip install --no-cache-dir torch torchvision --index-url "${TORCH_INDEX_URL}"
python -m pip install --no-cache-dir nnunetv2 torchio nibabel SimpleITK scikit-image pandas

echo
echo "Conda environment '${ENV_NAME}' is ready."
echo "Next:"
echo "  conda activate ${ENV_NAME}"
echo "  python track4_wholeheart/scripts/convert_to_nnunet.py --train-root /path/to/train --val-root /path/to/val --task both"
