#!/usr/bin/env bash
# Prepare the conda env, install dependencies, convert data, and register datasets.
# This script intentionally does not start training.
set -euo pipefail

cd "$(dirname "$0")/../.."

bash scripts/train/00_create_conda_env.sh
bash scripts/train/01_install_llamafactory.sh
bash scripts/train/10_prepare_dataset.sh
conda run -n "${ENV_NAME:-llm-rec2026}" python scripts/train/11_register_dataset.py

echo "[ok] setup complete. Start training with:"
echo "bash scripts/train/20_train_full_sft.sh"
