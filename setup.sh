#!/usr/bin/env bash
# ============================================================================
# setup.sh — Why5 Agentic Model: universal environment setup
#
# Sets up the full development/training environment for the Why5 project:
#   • synth_factory (synthetic data generation)
#   • distill       (SFT → DPO distillation pipeline)
#   • benchmark     (evaluation harness)
#
# Usage:
#   bash setup.sh                          # full setup (auto-detects GPU)
#   bash setup.sh --cpu                    # CPU-only (no CUDA deps)
#   bash setup.sh --skip-models            # skip model downloads
#   bash setup.sh --skip-flashattn         # skip flash-attn (use SDPA)
#   bash setup.sh --skip-docker            # skip Docker SDK
#   bash setup.sh --skip-llamacpp          # skip llama.cpp build
#   bash setup.sh --minimal                # core deps only (no training stack)
#
# Override defaults via env:
#   TEACHER_MODEL=Qwen/Qwen3-4B    bash setup.sh
#   STUDENT_MODEL=Qwen/Qwen3-1.7B  bash setup.sh
#   HF_TOKEN=hf_xxx                bash setup.sh   # for gated models
#   WORK_DIR=/custom/path          bash setup.sh
#   CUDA_VERSION=12.8              bash setup.sh
#
# Idempotent: safe to re-run; each stage skips work already done.
# ============================================================================
set -euo pipefail

# ──────────────────────────────────────────────────── configurable defaults ──
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
VLLM_VERSION="${VLLM_VERSION:-0.9.2}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
CUDA_VERSION="${CUDA_VERSION:-12.8}"
WORK_DIR="${WORK_DIR:-$HOME/why5}"
DATA_ROOT="${DATA_ROOT:-$(ls -d /data 2>/dev/null || echo "$HOME")}"
HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
VENV_DIR="$WORK_DIR/venv"

PY_MIN_MAJOR=3
PY_MIN_MINOR=10

# ────────────────────────────────────────────────────────────── CLI flags ──
CPU_ONLY=0
SKIP_MODELS=0
WITH_FLASHATTN=1
WITH_DOCKER=1
WITH_LLAMACPP=1
WITH_TRAIN=1        # full training stack (torch, transformers, trl, etc.)
WITH_SYNTH=1        # synth_factory deps
WITH_BENCH=1        # benchmark deps
WITH_SMOKE=1

for arg in "$@"; do
  case "$arg" in
    --cpu)              CPU_ONLY=1 ;;
    --skip-models)      SKIP_MODELS=1 ;;
    --skip-flashattn)   WITH_FLASHATTN=0 ;;
    --skip-docker)      WITH_DOCKER=0 ;;
    --skip-llamacpp)    WITH_LLAMACPP=0 ;;
    --skip-smoke)       WITH_SMOKE=0 ;;
    --minimal)          WITH_TRAIN=0; WITH_SYNTH=0; WITH_BENCH=0; WITH_FLASHATTN=0; WITH_LLAMACPP=0; WITH_DOCKER=0 ;;
    -h|--help)          head -30 "$0" | grep '^#' | sed 's/^# \{0,2\}//'; exit 0 ;;
    *)                  echo "ERROR: unknown arg: $arg (see --help)" >&2; exit 1 ;;
  esac
done

# ──────────────────────────────────────────────────────────────── helpers ──
LOG()  { printf '\033[1;36m[setup %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
WARN() { printf '\033[1;33m[setup WARN]\033[0m %s\n' "$*"; }
DIE()  { printf '\033[1;31m[setup ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null; then SUDO="sudo"; fi

# ────────────────────────────────────────────────────────── GPU detection ──
detect_gpu() {
  if [ "$CPU_ONLY" = "1" ] || ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "none"; return 0
  fi
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee /dev/stderr
  nvidia-smi --query-gpu=name --format=csv,noheader | head -1
}

GPU_NAME="$(detect_gpu)"

case "$GPU_NAME" in
  *5090*|*B200*|*BLACKWELL*)
    GPU_ARCH="120"; GPU_FAMILY="Blackwell"; MIN_DRIVER=570; FLASHATTN_PREBUILT=0 ;;
  *4090*|*4080*|*3090*|*4060*|*L40*|*A40*|*A100*|*H100*)
    GPU_ARCH="89"; GPU_FAMILY="Ada/Ampere"; MIN_DRIVER=525; FLASHATTN_PREBUILT=1 ;;
  none)
    GPU_ARCH=""; GPU_FAMILY="CPU-only"; MIN_DRIVER=0; FLASHATTN_PREBUILT=0; WITH_FLASHATTN=0 ;;
  *)
    GPU_ARCH="89"; GPU_FAMILY="detected:$GPU_NAME (defaulting to sm_89)"; MIN_DRIVER=525; FLASHATTN_PREBUILT=1 ;;
esac

if [ "$GPU_FAMILY" != "CPU-only" ]; then
  DRIVER_MAJ="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
  [ "$DRIVER_MAJ" -ge "$MIN_DRIVER" ] || DIE \
    "driver < $MIN_DRIVER (found $DRIVER_MAJ) for $GPU_FAMILY — upgrade NVIDIA driver."
  LOG "GPU: $GPU_NAME ($GPU_FAMILY) → sm_$GPU_ARCH, driver $DRIVER_MAJ"
  [ "$GPU_ARCH" = "120" ] && export TORCH_CUDA_ARCH_LIST="12.0" || export TORCH_CUDA_ARCH_LIST="8.9"
else
  LOG "CPU-only mode — skipping CUDA-specific components"
fi

# ──────────────────────────────────────────────────── directories & space ──
mkdir -p "$WORK_DIR" "$HF_HOME"
avail_gb=$(df -BG --output=avail "$DATA_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9' || echo "0")
[ "${avail_gb:-0}" -ge 50 ] || WARN "only ${avail_gb}GB free on $DATA_ROOT (recommend ≥ 50GB for models + deps)"

# ──────────────────────────────────────────────────── Python ≥ 3.10 + venv ──
find_python() {
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null; then
      if "$cand" -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
        echo "$cand"; return 0
      fi
    fi
  done
  return 1
}

if ! PYBIN=$(find_python); then
  LOG "no Python ≥ 3.10 found — installing Miniforge"
  MINIFORGE="$WORK_DIR/miniforge.sh"
  curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -o "$MINIFORGE"
  bash "$MINIFORGE" -b -p "$WORK_DIR/miniforge3"
  "$WORK_DIR/miniforge3/bin/conda" create -y -p "$VENV_DIR" python=3.12
  PYBIN="$VENV_DIR/bin/python"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYBIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install -U pip wheel setuptools
LOG "Python: $(python --version) at $(which python)"

PIP="$VENV_DIR/bin/pip"

# ──────────────────────────────────────────────── CUDA toolkit (nvcc) ──
NVCC_BIN=""
if [ "$GPU_FAMILY" != "CPU-only" ] && { [ "$WITH_FLASHATTN" -eq 1 ] || [ "$WITH_LLAMACPP" -eq 1 ]; }; then
  if command -v nvcc >/dev/null; then NVCC_BIN="$(command -v nvcc)";
  elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC_BIN=/usr/local/cuda/bin/nvcc; fi

  nvcc_ok=0
  if [ -n "$NVCC_BIN" ]; then
    nvcc_ver="$("$NVCC_BIN" --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
    [ "$(python3 - "$nvcc_ver" <<'EOF'
import sys; print(1 if tuple(map(int, sys.argv[1].split('.')[:2])) >= (12, 8) else 0)
EOF
)" = "1" ] && nvcc_ok=1
  fi

  if [ "$nvcc_ok" != "1" ]; then
    LOG "installing CUDA ${CUDA_VERSION} toolkit (nvcc)"
    if [ -n "$SUDO" ]; then
      $SUDO apt-get update -y
      $SUDO apt-get install -y ca-certificates gnupg build-essential cmake ninja-build git curl
      $SUDO install -m 0755 -d /etc/apt/keyrings
      $SUDO test -f /etc/apt/keyrings/nvidia.gpg || \
        curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/$( . /etc/os-release; echo "${ID}${VERSION_ID/./}" )/$(uname -m)/31986faa-keyring.gpg" | $SUDO tee /etc/apt/keyrings/nvidia.gpg >/dev/null
      echo "deb [signed-by=/etc/apt/keyrings/nvidia.gpg] https://developer.download.nvidia.com/compute/cuda/repos/$( . /etc/os-release; echo "${ID}${VERSION_ID/./}" )/$(uname -m)/ /" | $SUDO tee /etc/apt/sources.list.d/cuda.list >/dev/null
      $SUDO apt-get update -y
      $SUDO apt-get install -y "cuda-toolkit-${CUDA_VERSION/./-}"
      NVCC_BIN=/usr/local/cuda-"$CUDA_VERSION"/bin/nvcc
    else
      $PIP install -q nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12
      NVCC_BIN="$(python -c 'import nvidia.cuda_nvcc, pathlib; print(pathlib.Path(nvidia.cuda_nvcc.__file__).parent/"bin"/"nvcc")')"
      LOG "using pip-provided nvcc at $NVCC_BIN"
    fi
  fi
  [ -x "$NVCC_BIN" ] || DIE "nvcc not found at '$NVCC_BIN'"
  LOG "nvcc: $("$NVCC_BIN" --version | tail -1)"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DEPENDENCY INSTALLATION — grouped by component
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ─────────────────────────────────────── 1. Core / shared dependencies ──
LOG "installing core dependencies"
$PIP install -q \
  "openai>=1.30" \
  "httpx>=0.24" \
  "tenacity>=8.2" \
  "tiktoken>=0.5" \
  "tqdm>=4.64" \
  "pyyaml>=6.0" \
  "numpy>=1.24" \
  "huggingface_hub[cli,hf_transfer]"

# ─────────────────────────────────── 2. PyTorch (CUDA or CPU) ──
if [ "$GPU_FAMILY" != "CPU-only" ]; then
  if python -c "import torch" 2>/dev/null && \
     python -c "import torch; assert torch.cuda.is_available() and tuple(map(int, torch.version.cuda.split('.')[:2])) >= (12, 8)" 2>/dev/null; then
    LOG "torch $(python -c 'import torch; print(torch.__version__)') already cu128 — skipping"
  else
    LOG "installing PyTorch ${TORCH_VERSION}+cu128"
    $PIP install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128
  fi
else
  if ! python -c "import torch" 2>/dev/null; then
    LOG "installing PyTorch (CPU)"
    $PIP install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
  fi
fi

# ──────────────────────────────── 3. synth_factory dependencies ──
if [ "$WITH_SYNTH" -eq 1 ]; then
  LOG "installing synth_factory dependencies"
  $PIP install -q \
    "jsonschema>=4.17"

  if [ "$WITH_DOCKER" -eq 1 ]; then
    $PIP install -q "docker>=6.0" \
      || WARN "docker SDK install failed (optional — sandbox falls back to subprocess)"
  fi
fi

# ─────────────────────── 4. Distillation / training stack ──
if [ "$WITH_TRAIN" -eq 1 ]; then
  LOG "installing training stack (transformers, trl, peft, datasets, accelerate)"
  # transformers pinned <4.53: vLLM 0.9.x custom config collision
  $PIP install -q \
    "transformers>=4.51.1,<4.53" \
    "trl>=0.17" \
    "datasets>=3.0" \
    "accelerate>=0.30" \
    "peft>=0.10" \
    "sentencepiece>=0.1.99" \
    "protobuf>=3.20" \
    "safetensors>=0.4" \
    "tensorboard>=2.16"

  if [ "$GPU_FAMILY" != "CPU-only" ]; then
    LOG "installing GPU training extras"
    $PIP install -q "bitsandbytes>=0.43" \
      || WARN "bitsandbytes install failed — 8-bit optimizer unavailable"

    $PIP install -q "deepspeed>=0.16" \
      || WARN "deepspeed install failed (optional — bitsandbytes 8-bit AdamW covers 24GB GPUs)"
  fi

  # wandb (optional — logs.py gracefully handles ImportError)
  $PIP install -q "wandb>=0.16" \
    || WARN "wandb install failed (optional — training will log to tensorboard only)"
fi

# ─────────────────────────────── 5. vLLM inference engine ──
if [ "$GPU_FAMILY" != "CPU-only" ]; then
  if python -c "import vllm" 2>/dev/null; then
    LOG "vLLM $(python -c 'import vllm; print(vllm.__version__)') present — skipping"
  else
    LOG "installing vLLM ${VLLM_VERSION}"
    $PIP install "vllm==${VLLM_VERSION}" \
      || WARN "vLLM install failed — teacher serving will need manual setup"
  fi
fi

# ──────────────────────────────── 6. flash-attn ──
if [ "$WITH_FLASHATTN" -eq 1 ] && [ "$GPU_FAMILY" != "CPU-only" ]; then
  if python -c "import flash_attn" 2>/dev/null; then
    LOG "flash-attn present — skipping"
  elif [ "$FLASHATTN_PREBUILT" -eq 1 ]; then
    LOG "installing prebuilt flash-attn (sm_$GPU_ARCH)"
    $PIP install -q ninja packaging psutil wheel
    $PIP install -q flash-attn --no-build-isolation \
      || WARN "prebuilt flash-attn failed — will use SDPA backend"
  else
    LOG "building flash-attn from source for sm_$GPU_ARCH (30-60 min)"
    $PIP install -q ninja packaging psutil wheel
    git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git "$WORK_DIR/flash-attention" \
      || git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$WORK_DIR/flash-attention"
    ( cd "$WORK_DIR/flash-attention"
      export CUDA_HOME="$(dirname "$(dirname "$NVCC_BIN")")"
      export PATH="$CUDA_HOME/bin:$PATH"
      export CUDA_ARCHES="${GPU_ARCH:0:2}.${GPU_ARCH:2:1}"
      export MAX_JOBS="${MAX_JOBS:-16}"
      $PIP install -v --no-build-isolation .
    ) || WARN "flash-attn source build failed — will use SDPA backend"
  fi
fi

# ─────────────────────────── 7. llama.cpp (optional) ──
if [ "$WITH_LLAMACPP" -eq 1 ] && [ "$GPU_FAMILY" != "CPU-only" ]; then
  if [ -x "$WORK_DIR/llama.cpp/build/bin/llama-server" ]; then
    LOG "llama.cpp present — skipping"
  else
    LOG "building llama.cpp with CUDA for sm_$GPU_ARCH"
    $PIP install -q cmake
    [ -n "$SUDO" ] && $SUDO apt-get install -y git build-essential cmake || true
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$WORK_DIR/llama.cpp"
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_BIN")")"
    export CUDACXX="$NVCC_BIN"
    cmake -S "$WORK_DIR/llama.cpp" -B "$WORK_DIR/llama.cpp/build" \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$GPU_ARCH" -DGGML_CUDA_F16=ON \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build "$WORK_DIR/llama.cpp/build" --config Release -j"${MAX_JOBS:-16}"
    LOG "llama.cpp → $WORK_DIR/llama.cpp/build/bin/"
  fi
fi

# ───────────────────────── 8. Benchmark dependencies ──
if [ "$WITH_BENCH" -eq 1 ]; then
  LOG "installing benchmark dependencies"
  $PIP install -q \
    "pandas>=2.0" \
    "scipy>=1.10" \
    || WARN "some benchmark deps failed (optional — core benchmark still works)"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL DOWNLOADS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$SKIP_MODELS" -eq 0 ]; then
  export HF_HOME
  export HF_HUB_ENABLE_HF_TRANSFER=1

  HFCLI="$( [ -x "$VENV_DIR/bin/huggingface-cli" ] && echo "$VENV_DIR/bin/huggingface-cli" || echo huggingface-cli )"
  [ -n "${HF_TOKEN:-}" ] && "$HFCLI" login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null \
    || LOG "HF_TOKEN not set — public models only"

  download_model() {
    LOG "downloading $1 → HF_HOME=$HF_HOME"
    if "$HFCLI" download "$1" --exclude "*.pth" "original/*" "*.gguf" "own_weights/*" "*.bin" 2>&1 | tail -2; then
      LOG "✓ $1"
    else
      DIE "failed to download '$1' — check model ID or HF_TOKEN"
    fi
  }

  download_model "$TEACHER_MODEL"
  if [ -n "$STUDENT_MODEL" ] && [ "$STUDENT_MODEL" != "$TEACHER_MODEL" ]; then
    download_model "$STUDENT_MODEL"
  fi
else
  LOG "skipping model downloads (--skip-models)"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VERIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOG "verifying installed packages"
"$VENV_DIR/bin/python" - <<'PYCHECK'
import importlib, sys

# (module, pip_name, component, required)
deps = [
    # ── core / shared ──
    ("openai",          "openai",          "core",         True),
    ("httpx",           "httpx",           "core",         True),
    ("tenacity",        "tenacity",        "core",         True),
    ("tiktoken",        "tiktoken",        "core",         True),
    ("tqdm",            "tqdm",            "core",         True),
    ("yaml",            "pyyaml",          "core",         True),
    ("numpy",           "numpy",           "core",         True),
    ("huggingface_hub", "huggingface_hub", "core",         True),
    # ── synth_factory ──
    ("jsonschema",      "jsonschema",      "synth",        False),
    ("docker",          "docker",          "synth",        False),
    # ── training ──
    ("torch",           "torch",           "training",     True),
    ("transformers",    "transformers",    "training",     False),
    ("trl",             "trl",             "training",     False),
    ("peft",            "peft",            "training",     False),
    ("datasets",        "datasets",        "training",     False),
    ("accelerate",      "accelerate",      "training",     False),
    ("safetensors",     "safetensors",     "training",     False),
    ("wandb",           "wandb",           "training",     False),
    # ── benchmark ──
    ("pandas",          "pandas",          "benchmark",    False),
    ("scipy",           "scipy",           "benchmark",    False),
]

missing_required = []
for mod, pip_name, component, required in deps:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", getattr(m, "VERSION", "?"))
        print(f"  ✓ {pip_name:20} {str(ver):12}  [{component}]")
    except ImportError:
        tag = "REQUIRED" if required else "optional"
        print(f"  ✗ {pip_name:20} {'—':12}  [{component}] ({tag})")
        if required:
            missing_required.append(pip_name)

if missing_required:
    print(f"\nERROR: missing required packages: {missing_required}")
    sys.exit(1)

# GPU check
try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        x = torch.randn(512, 512, device="cuda") @ torch.randn(512, 512, device="cuda")
        assert x.isfinite().all()
        print(f"\n  GPU: {name} sm_{cap[0]}{cap[1]} — CUDA matmul OK")
    else:
        print("\n  GPU: not available (CPU-only mode)")
except Exception as e:
    print(f"\n  GPU check: {e}")
PYCHECK

# ────────────────────────────────────── smoke test ──
if [ "$WITH_SMOKE" -eq 1 ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  LOG "smoke test: distillation pipeline (MOCK=1)"
  ( cd "$REPO_ROOT/distill" && env MOCK=1 N_D1=40 N_D5=8 N_ROLLOUTS=10 \
      ./run_distillation.sh >/dev/null 2>&1 ) \
    && LOG "✓ distill pipeline smoke test passed" \
    || WARN "distill pipeline smoke test failed — inspect distill/data/logs/"

  LOG "smoke test: benchmark harness (MOCK=1)"
  ( cd "$REPO_ROOT/benchmark" && env MOCK=1 N=20 ./run_benchmark.sh >/dev/null 2>&1 ) \
    && LOG "✓ benchmark smoke test passed" \
    || WARN "benchmark smoke test failed — inspect benchmark/logs/"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DONE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cat <<BANNER

============================================================
 ✓ Why5 environment setup complete
============================================================
 GPU:     ${GPU_NAME:-CPU-only} (${GPU_FAMILY})
 Python:  $(python --version 2>&1) at $VENV_DIR/bin/python
 HF cache: $HF_HOME

 Activate the environment:
   source $VENV_DIR/bin/activate

 Next steps:
   1. Generate synthetic data:
      cd synth_factory && python run_pipeline.py

   2. Run distillation:
      cd distill && bash run_distillation.sh

   3. Run benchmarks:
      cd benchmark && bash run_benchmark.sh

 Teacher serving (required for real distillation):
   vllm serve "$TEACHER_MODEL" --port 8000 --max-model-len 8192
============================================================
BANNER
LOG "all done"
