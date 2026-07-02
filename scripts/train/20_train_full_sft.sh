#!/usr/bin/env bash
# Start full-parameter SFT with LLaMA-Factory.
set -euo pipefail

cd "$(dirname "$0")/../.."

ENV_NAME="${ENV_NAME:-llm-rec2026}"
CONFIG="${CONFIG:-configs/train/onereason_full_sft.yaml}"
OUT_DIR="${OUT_DIR:-train_output/onereason_0.8b_full_sft}"
LOG="$OUT_DIR/train.log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ ! -f "$CONFIG" ]; then
  echo "[ERROR] config not found: $CONFIG" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED="${WANDB_DISABLED:-1}"

echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] llamafactory-cli train $CONFIG"
echo "[run] log -> $LOG"
conda run -n "$ENV_NAME" llamafactory-cli train "$CONFIG" 2>&1 | tee "$LOG"
