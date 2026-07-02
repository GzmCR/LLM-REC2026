#!/usr/bin/env bash
# Install LLaMA-Factory and GPU acceleration packages in the conda env.
set -euo pipefail

cd "$(dirname "$0")/../.."

ENV_NAME="${ENV_NAME:-llm-rec2026}"
LF_DIR="${LLAMA_FACTORY_DIR:-third_party/LLaMA-Factory}"
LF_REPO="${LLAMA_FACTORY_REPO:-https://github.com/hiyouga/LLaMA-Factory.git}"
LF_REF="${LLAMA_FACTORY_REF:-v0.9.6.dev0}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1+cu126}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1+cu126}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.1+cu126}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu126}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}"
LIGER_VERSION="${LIGER_VERSION:-0.8.0}"

mkdir -p "$(dirname "$LF_DIR")"

if [ -d "$LF_DIR/.git" ]; then
  echo "[skip] $LF_DIR already exists"
else
  echo "[run] clone LLaMA-Factory -> $LF_DIR"
  git clone "$LF_REPO" "$LF_DIR"
fi

echo "[run] checkout LLaMA-Factory ref: $LF_REF"
if git -C "$LF_DIR" rev-parse --verify --quiet "$LF_REF" >/dev/null; then
  git -C "$LF_DIR" checkout "$LF_REF"
else
  git -C "$LF_DIR" fetch --tags origin
  if git -C "$LF_DIR" rev-parse --verify --quiet "$LF_REF" >/dev/null; then
    git -C "$LF_DIR" checkout "$LF_REF"
  else
    echo "[WARN] ref '$LF_REF' not found; keeping current LLaMA-Factory checkout" >&2
  fi
fi

echo "[run] install LLaMA-Factory editable"
conda run -n "$ENV_NAME" python -m pip install -U pip setuptools wheel
conda run -n "$ENV_NAME" python -m pip install -e "$LF_DIR"
if [ -f "$LF_DIR/requirements/metrics.txt" ]; then
  conda run -n "$ENV_NAME" python -m pip install -r "$LF_DIR/requirements/metrics.txt"
fi

echo "[run] install PyTorch CUDA wheels"
conda run -n "$ENV_NAME" python -m pip uninstall -y torch torchvision torchaudio sympy >/dev/null 2>&1 || true
conda run -n "$ENV_NAME" python -m pip install --no-deps \
  --index-url "$CUDA_WHEEL_INDEX" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION"
conda run -n "$ENV_NAME" python -m pip install --force-reinstall --no-deps "sympy==1.13.3"

echo "[run] install Liger Kernel and FlashAttention"
conda run -n "$ENV_NAME" python -m pip install --no-deps "liger-kernel==$LIGER_VERSION"
conda run -n "$ENV_NAME" python -m pip install "$FLASH_ATTN_WHEEL"
conda run -n "$ENV_NAME" python -m pip install tensorboard

echo "[run] optional flash_attention None-guard patch"
SITE_PACKAGES="$(conda run -n "$ENV_NAME" python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
FA_PY="$SITE_PACKAGES/transformers/integrations/flash_attention.py"
if [ -f "$FA_PY" ] && grep -q "s_aux=s_aux.to(query.dtype)," "$FA_PY"; then
  python - "$FA_PY" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
text = text.replace(
    "s_aux=s_aux.to(query.dtype),",
    "s_aux=s_aux.to(query.dtype) if s_aux is not None else None,",
)
path.write_text(text)
print(f"[ok] patched {path}")
PY
else
  echo "[skip] flash_attention patch not needed or file not found"
fi

echo "[verify] training environment"
conda run -n "$ENV_NAME" python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
try:
    import flash_attn
    print("flash_attn:", flash_attn.__version__)
except Exception as exc:
    print("[WARN] flash_attn import failed:", repr(exc))
for package in ["llamafactory", "liger-kernel", "transformers"]:
    try:
        print(f"{package}:", version(package))
    except PackageNotFoundError:
        print(f"[WARN] {package} package metadata not found")
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    print("liger qwen3 hook: OK")
except Exception as exc:
    print("[WARN] liger qwen3 hook import failed:", repr(exc))
PY

if conda run -n "$ENV_NAME" llamafactory-cli version >/tmp/llamafactory_version.txt 2>&1; then
  cat /tmp/llamafactory_version.txt
else
  echo "[WARN] llamafactory-cli version failed; inspect installation before training" >&2
fi

echo "[ok] install finished"
