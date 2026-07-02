#!/usr/bin/env bash
# Create or update the conda environment used for LLM-Rec SFT.
set -euo pipefail

cd "$(dirname "$0")/../.."

ENV_NAME="${ENV_NAME:-llm-rec2026}"
ENV_FILE="${ENV_FILE:-envs/llm-rec2026.yml}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda is not available. Install Miniconda/Anaconda on the server first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[run] update conda env: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "[run] create conda env: $ENV_NAME"
  conda env create -f "$ENV_FILE"
fi

echo "[ok] conda env ready: $ENV_NAME"
echo "[hint] activate with: conda activate $ENV_NAME"
