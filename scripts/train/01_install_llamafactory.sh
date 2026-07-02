#!/usr/bin/env bash
# Install LLaMA-Factory and GPU acceleration packages in the conda env.
set -euo pipefail

cd "$(dirname "$0")/../.."

ENV_NAME="${ENV_NAME:-llm-rec2026}"
LF_DIR="${LLAMA_FACTORY_DIR:-third_party/LLaMA-Factory}"
LF_REPO="${LLAMA_FACTORY_REPO:-https://github.com/hiyouga/LLaMA-Factory.git}"
LF_REF="${LLAMA_FACTORY_REF:-v0.9.6.dev0}"
LF_CLONE_RETRIES="${LLAMA_FACTORY_CLONE_RETRIES:-3}"
LF_CLONE_DEPTH="${LLAMA_FACTORY_CLONE_DEPTH:-1}"
LF_ARCHIVE="${LLAMA_FACTORY_ARCHIVE:-}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1+cu126}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.1+cu126}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.1+cu126}"
CUDA_WHEEL_INDEX="${CUDA_WHEEL_INDEX:-https://download.pytorch.org/whl/cu126}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}"
LIGER_VERSION="${LIGER_VERSION:-0.8.0}"

is_llama_factory_source() {
  [ -f "$1/pyproject.toml" ] || [ -f "$1/setup.py" ]
}

extract_llama_factory_archive() {
  if [ -z "$LF_ARCHIVE" ]; then
    return 1
  fi
  if [ ! -f "$LF_ARCHIVE" ]; then
    echo "[ERROR] LLAMA_FACTORY_ARCHIVE not found: $LF_ARCHIVE" >&2
    return 1
  fi

  echo "[run] extract LLaMA-Factory archive: $LF_ARCHIVE -> $LF_DIR"
  python - "$LF_ARCHIVE" "$LF_DIR" <<'PY'
import shutil
import sys
import tempfile
from pathlib import Path

archive = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()
parent = dst.parent
parent.mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory(prefix="llamafactory_extract_") as tmp:
    tmp_path = Path(tmp)
    shutil.unpack_archive(str(archive), str(tmp_path))
    candidates = [p for p in tmp_path.rglob("*") if p.is_dir() and ((p / "pyproject.toml").exists() or (p / "setup.py").exists())]
    if not candidates:
        raise SystemExit(f"no Python project found in archive: {archive}")
    source = sorted(candidates, key=lambda p: len(p.parts))[0]
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(source, dst)
    print(f"[OK] extracted source tree: {source} -> {dst}")
PY
}

clone_llama_factory() {
  local attempt
  for attempt in $(seq 1 "$LF_CLONE_RETRIES"); do
    echo "[run] clone attempt $attempt/$LF_CLONE_RETRIES: $LF_REPO -> $LF_DIR"
    if git clone --depth "$LF_CLONE_DEPTH" --branch "$LF_REF" "$LF_REPO" "$LF_DIR"; then
      return 0
    fi
    if [ "$attempt" -lt "$LF_CLONE_RETRIES" ]; then
      echo "[WARN] clone failed; retrying in $((attempt * 5)) seconds..." >&2
      rm -rf "$LF_DIR"
      sleep $((attempt * 5))
    fi
  done

  echo "[WARN] shallow clone for ref '$LF_REF' failed; trying default branch clone..." >&2
  rm -rf "$LF_DIR"
  for attempt in $(seq 1 "$LF_CLONE_RETRIES"); do
    echo "[run] fallback clone attempt $attempt/$LF_CLONE_RETRIES: $LF_REPO -> $LF_DIR"
    if git clone --depth "$LF_CLONE_DEPTH" "$LF_REPO" "$LF_DIR"; then
      return 0
    fi
    if [ "$attempt" -lt "$LF_CLONE_RETRIES" ]; then
      echo "[WARN] fallback clone failed; retrying in $((attempt * 5)) seconds..." >&2
      rm -rf "$LF_DIR"
      sleep $((attempt * 5))
    fi
  done

  cat >&2 <<EOF
[ERROR] failed to clone LLaMA-Factory from: $LF_REPO

This is usually a GitHub/network TLS problem, not a training-script problem.
Try one of these on the server:

  # Use SSH if your server has a GitHub SSH key configured
  LLAMA_FACTORY_REPO=git@github.com:hiyouga/LLaMA-Factory.git bash scripts/train/01_install_llamafactory.sh

  # Or use your own reachable mirror/proxy URL
  LLAMA_FACTORY_REPO=<reachable_git_url> bash scripts/train/01_install_llamafactory.sh

  # Or pre-clone manually, then rerun this script
  mkdir -p third_party
  git clone --depth $LF_CLONE_DEPTH --branch $LF_REF $LF_REPO $LF_DIR
  bash scripts/train/01_install_llamafactory.sh

  # Or download the release/source zip on another machine, upload it, then run
  LLAMA_FACTORY_ARCHIVE=/path/to/LLaMA-Factory-$LF_REF.zip bash scripts/train/01_install_llamafactory.sh
EOF
  return 1
}

mkdir -p "$(dirname "$LF_DIR")"

if [ -d "$LF_DIR/.git" ]; then
  echo "[skip] $LF_DIR already exists"
elif [ -d "$LF_DIR" ] && is_llama_factory_source "$LF_DIR"; then
  echo "[skip] using existing LLaMA-Factory source tree: $LF_DIR"
elif [ -n "$LF_ARCHIVE" ]; then
  extract_llama_factory_archive
else
  if [ -e "$LF_DIR" ]; then
    echo "[WARN] removing incomplete non-git directory: $LF_DIR" >&2
    rm -rf "$LF_DIR"
  fi
  clone_llama_factory
fi

if [ -d "$LF_DIR/.git" ]; then
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
else
  echo "[skip] non-git LLaMA-Factory source tree; checkout skipped"
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
