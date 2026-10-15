#!/usr/bin/env bash
# ============================================================================
# setup_5090_novita.sh — Why5 distillation/training environment on Novita GPUs
# (auto-detects RTX 5090 / Blackwell sm_120 and RTX 4090 / Ada sm_89)
#
# Per-GPU consequences that drive every pin below:
#   RTX 5090 (Blackwell, sm_120):
#     - NVIDIA driver >= 570.26 (CUDA 12.8) REQUIRED on the host.
#     - PyTorch must be a cu128 build (torch >= 2.7); cu124/cu126 wheels have
#       no sm_120 kernels and fail with "no kernel image available".
#     - vLLM needs >= 0.9.2 for official Blackwell wheels.
#     - flash-attn has NO prebuilt sm_120 wheel: source build with
#       CUDA_ARCHES="12.0" and nvcc >= 12.8 (~30-60 min). Skippable: PyTorch
#       SDPA and vLLM's TRITON/FlashInfer backends work on 5090.
#   RTX 4090 (Ada, sm_89):
#     - driver >= 525 (CUDA 12 minor-version compatibility) is enough.
#     - Same cu128 torch works (cu128 wheels include sm_89 kernels).
#     - vLLM >= 0.9.2 works; attention backends are fully supported.
#     - flash-attn: official PREBUILT wheels cover sm_89 — pip install, no
#       source build (~1 min instead of 30-60 min).
#   llama.cpp: built with -DCMAKE_CUDA_ARCHITECTURES=<detected arch>.
#
# Usage (on a fresh Novita 5090 or 4090 instance, Ubuntu 22.04/24.04):
#   bash setup_5090_novita.sh                 # full setup (auto-detects GPU)
#   bash setup_5090_novita.sh --models-only   # just download models
#   bash setup_5090_novita.sh --skip-flashattn --skip-llamacpp
#
# Override models / versions via env:
#   TEACHER_MODEL=Qwen/Qwen3.5-4B bash setup_5090_novita.sh
#   HF_TOKEN=hf_xxx bash setup_5090_novita.sh   # if gated models
#
# 4090-specific training notes (24GB vs 32GB):
#   - 2B student SFT fits with deepspeed ZeRO-2 CPU offload or 8-bit AdamW;
#     the setup installs deepspeed by default for exactly this case.
#   - Serve the 4B teacher at --gpu-memory-utilization 0.85 on 24GB.
#
# Idempotent: safe to re-run; each stage skips itself when already done.
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------- defaults --
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-4B}"
# Student 2B-class base checkpoint for SFT. Adjust to the chosen base;
# per Distillation Plan Option L1 the Qwen tokenizer is preferred.
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
VLLM_VERSION="${VLLM_VERSION:-0.9.2}"
TORCH_VERSION="${TORCH_VERSION:-2.7.1}"
CUDA_VERSION="${CUDA_VERSION:-12.8}"          # toolkit for nvcc builds
PY_MIN_MAJOR=3 PY_MIN_MINOR=10
WORK_DIR="${WORK_DIR:-$HOME/why5}"
# Novita mounts a data disk; keep HF cache there if present
DATA_ROOT="${DATA_ROOT:-$(ls -d /data 2>/dev/null || echo $HOME)}"
HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"
VENV_DIR="$WORK_DIR/venv"
LOG() { printf '\033[1;36m[setup %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
DIE() { printf '\033[1;31m[setup ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# NOTE: if Qwen/Qwen3.5-4B is not published under that id, pass a valid id:
#   TEACHER_MODEL=Qwen/Qwen3-4B-Instruct-2507 bash setup_5090_novita.sh

MODELS_ONLY=0; WITH_FLASHATTN=1; WITH_LLMACPP=1; WITH_VLLM=1; WITH_TRAIN_DEPS=1; WITH_SMOKE=1
for arg in "$@"; do
  case "$arg" in
    --models-only)      MODELS_ONLY=1 ;;
    --skip-flashattn)   WITH_FLASHATTN=0 ;;
    --skip-llamacpp)    WITH_LLMACPP=0 ;;
    --skip-vllm)        WITH_VLLM=0 ;;
    --skip-train-deps)  WITH_TRAIN_DEPS=0 ;;
    --skip-smoke)       WITH_SMOKE=0 ;;
    -h|--help)          grep '^#' "$0" | sed 's/^# \{0,2\}//'; exit 0 ;;
    *)                  DIE "unknown arg: $arg (see --help)" ;;
  esac
done

# Pip launcher bound to the environment we are building (venv if it exists,
# else the resolved system python) — used by EVERY pip-install stage below.
resolve_pip() {
  if [ -x "$VENV_DIR/bin/pip" ]; then echo "$VENV_DIR/bin/pip"; else
    echo "$(find_python) -m pip"
  fi
}

# ------------------------------------------------------------ preflight ----
# GPU auto-detection: every downstream pin (driver bar, torch arch list,
# flash-attn build strategy, llama.cpp arch) derives from this one decision.
command -v nvidia-smi >/dev/null || DIE "nvidia-smi not found — is this a GPU instance?"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee /dev/stderr
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
DRIVER_MAJ="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"

case "$GPU_NAME" in
  *5090*|*B200*|*RTX\s*50*|*BLACKWELL*)
    GPU_ARCH="120"            # Blackwell consumer: sm_120
    GPU_FAMILY="Blackwell"
    MIN_DRIVER=570            # CUDA 12.8 for sm_120 kernels
    FLASHATTN_PREBUILT=0      # no prebuilt sm_120 wheel -> source build
    ;;
  *4090*|*4080*|*3090*|*4060*|*L40*|*A40*|*RTX\s*40*)
    GPU_ARCH="89"             # Ada (4090) or Ampere (3090/A40): sm_89
    GPU_FAMILY="Ada/Ampere"
    MIN_DRIVER=525            # CUDA 12.x minor-version compatibility floor
    FLASHATTN_PREBUILT=1      # prebuilt sm_89 wheels exist -> pip, no build
    ;;
  *)
    GPU_ARCH="89"
    GPU_FAMILY="detected:$GPU_NAME (defaulting to sm_89 pins)"
    MIN_DRIVER=525
    FLASHATTN_PREBUILT=1
    ;;
esac
[ "$DRIVER_MAJ" -ge "$MIN_DRIVER" ] || DIE \
  "driver < $MIN_DRIVER (found $DRIVER_MAJ) for $GPU_FAMILY — re-provision the Novita instance with a matching image."
LOG "GPU: $GPU_NAME ($GPU_FAMILY) -> sm_$GPU_ARCH pins, min driver $MIN_DRIVER"

# unify arch pins for torch extensions, flash-attn build, llama.cpp
if [ "$GPU_ARCH" = "120" ]; then export TORCH_CUDA_ARCH_LIST="12.0"; else export TORCH_CUDA_ARCH_LIST="8.9"; fi
export FLASHATTN_PREBUILT
LLAMA_CUDA_ARCH="$GPU_ARCH"

mkdir -p "$WORK_DIR" "$HF_HOME"
avail_gb=$(df -BG --output=avail "$DATA_ROOT" | tail -1 | tr -dc '0-9')
[ "${avail_gb:-0}" -ge 100 ] || LOG "warning: only ${avail_gb}GB free on $DATA_ROOT (want >= 100GB for models+wheels+builds)"

# If we must sudo for build deps, do it once now (apt is only needed for
# compiling; pure-python stages don't require root).
SUDO=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null; then SUDO="sudo"; fi

# ------------------------------------------------------- python + venv -----
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
if [ "$MODELS_ONLY" -eq 0 ]; then
  if ! PYBIN=$(find_python); then
    LOG "no python >= 3.10 found — installing Miniforge (no sudo needed)"
    MINIFORGE="$WORK_DIR/miniforge.sh"
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh -o "$MINIFORGE"
    bash "$MINIFORGE" -b -p "$WORK_DIR/miniforge3"
    "$WORK_DIR/miniforge3/bin/conda" create -y -p "$VENV_DIR" python=3.12
    PYBIN="$VENV_DIR/bin/python"
  fi
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    "$PYBIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install -U pip wheel
  LOG "python: $(python --version) at $(which python)"
fi

# Pip launcher for every install stage below (venv pip if venv exists).
PIP="$(resolve_pip)"

# ------------------------------------------------------- cuda toolkit ------
# nvcc >= 12.8 is needed only to COMPILE flash-attn / llama.cpp.
# Order: existing nvcc -> NVIDIA apt repo (if sudo) -> conda toolkit (no sudo).
NVCC_BIN=""
if [ "$MODELS_ONLY" -eq 0 ] && { [ "$WITH_FLASHATTN" -eq 1 ] || [ "$WITH_LLMACPP" -eq 1 ]; }; then
  if command -v nvcc >/dev/null; then NVCC_BIN="$(command -v nvcc)";
  elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC_BIN=/usr/local/cuda/bin/nvcc; fi
  nvcc_ok=0
  if [ -n "$NVCC_BIN" ]; then
    nvcc_ver="$("$NVCC_BIN" --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
    [ "$(python3 - "$nvcc_ver" <<'EOF'
import sys
print(1 if tuple(map(int, sys.argv[1].split('.')[:2])) >= (12, 8) else 0)
EOF
)" = "1" ] && nvcc_ok=1
  fi
  if [ "$nvcc_ok" != "1" ]; then
    LOG "installing CUDA ${CUDA_VERSION} toolkit (nvcc) for sm_120 builds"
    if [ -n "$SUDO" ]; then
      $SUDO apt-get update -y
      $SUDO apt-get install -y ca-certificates gnupg build-essential cmake ninja-build git curl
      $SUDO install -m 0755 -d /etc/apt/keyrings
      $SUDO test -f /etc/apt/keyrings/nvidia.gpg || \
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/$( . /etc/os-release; echo "${ID}${VERSION_ID/./}" )/$(uname -m)/31986faa-keyring.gpg | $SUDO tee /etc/apt/keyrings/nvidia.gpg >/dev/null
      echo "deb [signed-by=/etc/apt/keyrings/nvidia.gpg] https://developer.download.nvidia.com/compute/cuda/repos/$( . /etc/os-release; echo "${ID}${VERSION_ID/./}" )/$(uname -m)/ /" | $SUDO tee /etc/apt/sources.list.d/cuda.list >/dev/null
      $SUDO apt-get update -y
      $SUDO apt-get install -y "cuda-toolkit-${CUDA_VERSION/./-}"
      NVCC_BIN=/usr/local/cuda-"$CUDA_VERSION"/bin/nvcc
    else
      # no sudo: conda-forge nvcc into the venv prefix
      pip install -q nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12
      NVCC_BIN="$(python -c 'import nvidia.cuda_nvcc, pathlib; print(pathlib.Path(nvidia.cuda_nvcc.__file__).parent/"bin"/"nvcc")')"
      LOG "using pip-provided nvcc at $NVCC_BIN (no sudo available)"
    fi
  fi
  [ -x "$NVCC_BIN" ] || DIE "nvcc not found at '$NVCC_BIN'"
  LOG "nvcc: $("$NVCC_BIN" --version | tail -1)"
fi

# ------------------------------------------------------------- pytorch -----
if [ "$MODELS_ONLY" -eq 0 ]; then
  if python -c "import torch" 2>/dev/null && \
     python - <<'EOF'
import sys, torch
arch = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
sys.exit(0 if torch.version.cuda and tuple(map(int, torch.version.cuda.split('.')[:2])) >= (12, 8) and torch.cuda.is_available() else 1)
EOF
  then
    LOG "torch $(python -c 'import torch; print(torch.__version__)') already cu128 — skipping"
  else
    LOG "installing PyTorch ${TORCH_VERSION}+cu128 (sm_120 kernels)"
    pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128
  fi
fi

# ---------------------------------------------------------------- vllm -----
if [ "$MODELS_ONLY" -eq 0 ] && [ "$WITH_VLLM" -eq 1 ]; then
  if "$VENV_DIR/bin/python" -c "import vllm" 2>/dev/null; then
    LOG "vLLM $($VENV_DIR/bin/python -c 'import vllm; print(vllm.__version__)') present — skipping"
  else
    LOG "installing vLLM ${VLLM_VERSION} (Blackwell support)"
    $PIP install "vllm==${VLLM_VERSION}" "huggingface_hub[cli,hf_transfer]" openai
  fi
fi

# --------------------------------------------------------- flash-attn -----
if [ "$MODELS_ONLY" -eq 0 ] && [ "$WITH_FLASHATTN" -eq 1 ]; then
  if "$VENV_DIR/bin/python" -c "import flash_attn" 2>/dev/null; then
    LOG "flash-attn present — skipping"
  elif [ "$FLASHATTN_PREBUILT" -eq 1 ]; then
    # sm_89 (4090/3090): official prebuilt wheels cover this arch
    LOG "installing prebuilt flash-attn wheel (sm_$GPU_ARCH)"
    $PIP install -q ninja packaging psutil wheel
    $PIP install -q flash-attn --no-build-isolation \
      || LOG "warning: prebuilt wheel install failed — re-run with source build (set FLASHATTN_PREBUILT=0)"
  else
    # sm_120 (5090): no prebuilt wheel; build from source for the target arch
    LOG "building flash-attn from source for sm_120 (30-60 min; Ctrl-C and re-run with --skip-flashattn to use SDPA/FlashInfer instead)"
    $PIP install -q ninja packaging psutil wheel
    git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git "$WORK_DIR/flash-attention" \
      || git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git "$WORK_DIR/flash-attention"
    cd "$WORK_DIR/flash-attention"
    # sm_120 is consumer Blackwell: no FA3/FA4; FA2 kernels compiled for arch 12.0
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_BIN")")"
    export PATH="$CUDA_HOME/bin:$PATH"
    export CUDA_ARCHES="12.0"                 # flash-attn's setup.py reads this
    export MAX_JOBS="${MAX_JOBS:-16}"
    $PIP install -v --no-build-isolation .
    cd - >/dev/null
  fi
fi

# ------------------------------------------------------------ llama.cpp ----
if [ "$MODELS_ONLY" -eq 0 ] && [ "$WITH_LLMACPP" -eq 1 ]; then
  if command -v "$WORK_DIR/llama.cpp/build/bin/llama-server" >/dev/null 2>&1 || \
     [ -x "$WORK_DIR/llama.cpp/build/bin/llama-server" ]; then
    LOG "llama.cpp present — skipping"
  else
    LOG "building llama.cpp with CUDA backend for sm_120"
    $PIP install -q cmake
    [ -n "$SUDO" ] && $SUDO apt-get install -y git build-essential || true
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$WORK_DIR/llama.cpp"
    export CUDA_HOME="$(dirname "$(dirname "$NVCC_BIN")")"
    export CUDACXX="$NVCC_BIN"
    cmake -S "$WORK_DIR/llama.cpp" -B "$WORK_DIR/llama.cpp/build" \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$LLAMA_CUDA_ARCH" -DGGML_CUDA_F16=ON \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build "$WORK_DIR/llama.cpp/build" --config Release -j"${MAX_JOBS:-16}"
    LOG "llama.cpp binaries in $WORK_DIR/llama.cpp/build/bin"
  fi
fi

# ------------------------------------------------- training stack deps -----

# --- pipeline dependencies: required by distill/why5_distill in real mode ---
# (openai = teacher/student HTTP clients; why5_synth + benchmark = stdlib only)
# Installed UNCONDITIONALLY (never skipped) into the venv — this is the exact
# dependency that used to go missing when the vLLM stage was skipped.
if [ "$MODELS_ONLY" -eq 0 ]; then
  LOG "installing pipeline deps (openai, huggingface_hub, numpy) into venv"
  $PIP install -q "openai>=1.30" "huggingface_hub[cli,hf_transfer]" numpy

  if [ "$WITH_TRAIN_DEPS" -eq 1 ]; then
    LOG "installing training stack (torch deps: transformers/trl/datasets/accelerate)"
    # transformers pinned <4.53: vLLM 0.9.x registers custom configs (aimv2/ovis)
    # that newer transformers ships natively -> import vllm crashes with
    # "'aimv2' is already used by a Transformers config"
    $PIP install -q "transformers>=4.51.1,<4.53" "trl>=0.17" "datasets>=3.0" accelerate \
      sentencepiece tensorboard bitsandbytes
    # deepspeed is optional (ZeRO offload alternative to 8-bit AdamW for 24GB);
    # separate so a failed install never blocks the rest
    $PIP install -q "deepspeed>=0.16" || \
      LOG "warning: deepspeed install failed (optional) — bitsandbytes 8-bit AdamW covers 24GB GPUs"
  fi
fi

# ---------------------------------------------------------- model files ----
export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER=1
$PIP show -q huggingface_hub >/dev/null 2>&1 || \
  $PIP install -q "huggingface_hub[cli,hf_transfer]"
HFCLI="$( [ -x "$VENV_DIR/bin/huggingface-cli" ] && echo "$VENV_DIR/bin/huggingface-cli" || echo huggingface-cli )"
[ -n "${HF_TOKEN:-}" ] && "$HFCLI" login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || \
  LOG "HF_TOKEN not set — public models only"

download_model() {  # $1 = repo id
  LOG "downloading $1 -> HF_HOME=$HF_HOME"
  if "$HFCLI" download "$1" --exclude "*.pth" "original/*" "*.gguf" \
       "own_weights/*" "*.bin" 2>&1 | tail -2; then
    LOG "done: $1"
  else
    DIE "failed to download '$1'. If the id is wrong/unpublished, re-run with e.g. TEACHER_MODEL=Qwen/Qwen3-4B-Instruct-2507"
  fi
}
download_model "$TEACHER_MODEL"
if [ -n "$STUDENT_MODEL" ] && [ "$STUDENT_MODEL" != "$TEACHER_MODEL" ]; then
  download_model "$STUDENT_MODEL"
fi

# -------------------------------------------------------------- verify -----
if [ "$MODELS_ONLY" -eq 0 ]; then
  LOG "verification"
  "$VENV_DIR/bin/python" - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
name, cap = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
x = torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
assert x.isfinite().all()
print(f"  torch {torch.__version__} (cu{torch.version.cuda}) on {name} sm_{cap[0]}{cap[1]} — matmul OK")
assert cap in ((12, 0), (8, 9)), f"expected sm_120 or sm_89, got sm_{cap[0]}{cap[1]}"
try:
    import vllm; print(f"  vLLM {vllm.__version__} import OK")
except Exception as e:
    print(f"  vLLM import FAILED: {e}")
try:
    import flash_attn; print(f"  flash-attn {flash_attn.__version__} import OK")
    from flash_attn import flash_attn_func
    q = torch.randn(1, 128, 8, 64, dtype=torch.bfloat16, device="cuda")
    o = flash_attn_func(q, q, q, causal=True)
    assert o.isfinite().all(); print("  flash-attn sm_120 kernel OK")
except Exception as e:
    print(f"  flash-attn unusable ({type(e).__name__}: {e}) — use SDPA/FlashInfer backend")
EOF
  [ -x "$WORK_DIR/llama.cpp/build/bin/llama-server" ] && \
    "$WORK_DIR/llama.cpp/build/bin/llama-server --version" | head -1 || true

  # every import the project actually uses (grep-verified across the repo)
  LOG "import verification (project deps)"
  "$VENV_DIR/bin/python" - <<'EOF'
import importlib
deps = {"openai": "pipeline (teacher/student clients)",
        "torch": "SFT/DPO training",
        "transformers": "model loading + SFT",
        "trl": "DPO trainer",
        "datasets": "DPO dataset builder",
        "huggingface_hub": "model download",
        "numpy": "torch ecosystem"}
missing = []
for mod, why in deps.items():
    try:
        importlib.import_module(mod)
        print(f"  {mod:16} OK  ({why})")
    except Exception as e:
        missing.append(mod)
        print(f"  {mod:16} MISSING ({why}): {type(e).__name__}")
if missing:
    raise SystemExit(f"missing modules: {missing} — re-run this setup script")
EOF
fi

# ------------------------------------------------------------ smoke test ----
# End-to-end proof that the environment can run the project: the full mock
# distillation pipeline and the benchmark self-test, no GPU needed.
if [ "$MODELS_ONLY" -eq 0 ] && [ "$WITH_SMOKE" -eq 1 ]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  LOG "smoke test: distillation pipeline (MOCK=1)"
  ( cd "$REPO_ROOT/distill" && env MOCK=1 N_D1=40 N_D5=8 N_ROLLOUTS=10 \
      ./run_distillation.sh >/dev/null 2>&1 ) \
    && LOG "smoke OK: distill pipeline (corpus -> SFT data -> DPO pairs)" \
    || DIE "smoke test failed for distill pipeline — inspect: distill/data/logs/"
  LOG "smoke test: benchmark harness (MOCK=1)"
  ( cd "$REPO_ROOT/benchmark" && env MOCK=1 N=20 ./run_benchmark.sh >/dev/null 2>&1 ) \
    && LOG "smoke OK: benchmark (self-test + flawed-detection)" \
    || DIE "smoke test failed for benchmark — inspect: benchmark/logs/"
fi

# --------------------------------------------------------------- next ------
TEACHER_SNAP="$(python - <<EOF
from huggingface_hub import snapshot_download; print(snapshot_download("$TEACHER_MODEL"))
EOF
)" || TEACHER_SNAP="<hf cache>"
cat <<BANNER

============================================================
 Setup complete on ${GPU_NAME}

 Serve the teacher (distill/run_distillation.sh expects port 8000):
   $VENV_DIR/bin/vllm serve "$TEACHER_MODEL" \\
       --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.90
   # on 5090 (32GB): TRITON/FlashInfer attention is safest:
   #   VLLM_ATTENTION_BACKEND=FLASHINFER  (or TRITON)

 llama.cpp alternative (GGUF, for quick student evals):
   $WORK_DIR/llama.cpp/build/bin/llama-server \\
       -m <model>.gguf -ngl 999 --port 8000

 Then, from the repo:
   cd distill && ./run_distillation.sh        # real teacher run
   python3 why5_distill/train_sft.py --student <student-path> \\
       --teacher "$TEACHER_MODEL" --data data/distilled/<version>

 HF cache: $HF_HOME
 Weights : $TEACHER_SNAP
============================================================
BANNER
LOG "all done"
