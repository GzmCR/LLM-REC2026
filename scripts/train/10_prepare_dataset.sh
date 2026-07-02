#!/usr/bin/env bash
# Convert official dataset/ and optional generated_dataset/ to Alpaca JSONL.
set -euo pipefail

cd "$(dirname "$0")/../.."

ENV_NAME="${ENV_NAME:-llm-rec2026}"
SEED="${SEED:-2026}"
OFFICIAL_DIR="${OFFICIAL_DIR:-dataset}"
AUG_DIR="${AUG_DIR:-generated_dataset}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-train_data}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-Explorer_LLM_Rec_Competition/demo/convert_jsonl.py}"

OFFICIAL_OUT="$TRAIN_DATA_DIR/data_official.jsonl"
AUG_OUT="$TRAIN_DATA_DIR/data_augmented.jsonl"
MIXED_OUT="$TRAIN_DATA_DIR/data_mixed.jsonl"
MIXED_TMP="$TRAIN_DATA_DIR/data_mixed.tmp.jsonl"

if [ ! -d "$OFFICIAL_DIR" ]; then
  echo "[ERROR] official dataset dir not found: $OFFICIAL_DIR" >&2
  exit 1
fi
if [ ! -f "$CONVERT_SCRIPT" ]; then
  echo "[ERROR] convert script not found: $CONVERT_SCRIPT" >&2
  exit 1
fi

mkdir -p "$TRAIN_DATA_DIR"

echo "[run] convert official SFT JSONL -> $OFFICIAL_OUT"
conda run -n "$ENV_NAME" python "$CONVERT_SCRIPT" \
  --input "$OFFICIAL_DIR" \
  --output "$OFFICIAL_OUT" \
  --shuffle \
  --shuffle-seed "$SEED"

USE_AUG=0
if [ -d "$AUG_DIR" ] && find "$AUG_DIR" -name "*.jsonl" -type f -size +0c | grep -q .; then
  USE_AUG=1
fi

if [ "$USE_AUG" -eq 1 ]; then
  echo "[run] convert augmented SFT JSONL -> $AUG_OUT"
  conda run -n "$ENV_NAME" python "$CONVERT_SCRIPT" \
    --input "$AUG_DIR" \
    --output "$AUG_OUT" \
    --shuffle \
    --shuffle-seed "$SEED"
  cat "$OFFICIAL_OUT" "$AUG_OUT" > "$MIXED_TMP"
else
  echo "[skip] no non-empty augmented JSONL found under $AUG_DIR"
  cp "$OFFICIAL_OUT" "$MIXED_TMP"
fi

echo "[run] shuffle mixed data -> $MIXED_OUT"
conda run -n "$ENV_NAME" python - "$MIXED_TMP" "$MIXED_OUT" "$SEED" <<'PY'
import random
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
seed = int(sys.argv[3])

lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
rng = random.Random(seed)
rng.shuffle(lines)
dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
print(f"[OK] wrote {len(lines)} lines to {dst}")
PY

rm -f "$MIXED_TMP"

echo "[ok] official lines: $(wc -l < "$OFFICIAL_OUT")"
if [ -s "$AUG_OUT" ]; then
  echo "[ok] augmented lines: $(wc -l < "$AUG_OUT")"
fi
echo "[ok] mixed lines: $(wc -l < "$MIXED_OUT")"
