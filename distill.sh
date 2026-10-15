#!/usr/bin/env bash
# ============================================================================
# distill.sh — Why5 full distillation pipeline
#
# Orchestrates the complete distillation workflow:
#   Stage 0: Environment validation & dependency check
#   Stage T: Teacher model download & server startup (auto)
#   Stage 1: Corpus generation (teacher → synthetic training data)
#   Stage 2: SFT data preparation (tokenization, formatting)
#   Stage 3: Supervised fine-tuning (SFT with optional LoRA + KD)
#   Stage 4: On-policy rollouts → DPO correction pairs
#   Stage 5: DPO alignment training
#   Stage 6: Benchmark evaluation (optional)
#
# Usage:
#   bash distill.sh                              # full pipeline (auto-starts teacher)
#   bash distill.sh --mock                       # offline dry-run (no GPU/models needed)
#   bash distill.sh --no-teacher                 # skip teacher startup (already running)
#   bash distill.sh --stage 3                    # resume from stage 3 (SFT)
#   bash distill.sh --skip-benchmark             # skip final evaluation
#
# Environment variables:
#   TEACHER_MODEL     Teacher model ID          (default: Qwen/Qwen3.5-4B)
#   STUDENT_MODEL     Student base model        (default: Qwen/Qwen3-1.7B)
#   TEACHER_URL       Teacher API endpoint      (default: http://localhost:8000)
#   WORK_DIR          Working directory          (default: ~/why5)
#   HF_HOME           Hugging Face cache         (default: $DATA_ROOT/hf_cache)
#   N_D1              Depth-1 samples            (default: 200)
#   N_D5              Depth-5 samples            (default: 50)
#   N_ROLLOUTS        On-policy rollouts         (default: 60)
#   KD                Knowledge distillation α   (default: 0.0)
#   WANDB_PROJECT     W&B project name           (default: why5-distill)
#   MOCK              Dry-run mode (0/1)         (default: 0)
#
# Prerequisites:
#   1. Run setup.sh first to install all dependencies
#   2. Ensure sufficient GPU memory (see GPU profiles below)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ──────────────────────────────────────────────────── configurable defaults ──
MOCK="${MOCK:-0}"
N_D1="${N_D1:-200}"
N_D5="${N_D5:-50}"
N_ROLLOUTS="${N_ROLLOUTS:-60}"
KD="${KD:-0.0}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
TEACHER_URL="${TEACHER_URL:-http://localhost:8000}"
WORK_DIR="${WORK_DIR:-${HOME:-/root}/why5}"
VENV_DIR="${VENV_DIR:-$WORK_DIR/venv}"
VLLM_VERSION="${VLLM_VERSION:-0.9.2}"
WANDB_PROJECT="${WANDB_PROJECT:-why5-distill}"
START_STAGE="${START_STAGE:-0}"
START_TEACHER="${START_TEACHER:-1}"
SKIP_BENCHMARK=0

DATA_ROOT="${DATA_ROOT:-$(ls -d /data 2>/dev/null || echo "${HOME:-/root}")}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"

DATASET_VERSION="${DATASET_VERSION:-v0.5.0-distill-pilot}"
DATA_DIR="data/distilled/$DATASET_VERSION"
STUDENT_CKPT="ckpt/why5-sft"
DPO_CKPT="ckpt/why5-dpo"

# ────────────────────────────────────────────────────────────── CLI flags ──
MOCK_FLAG=""
for arg in "$@"; do
  case "$arg" in
    --mock)             MOCK=1 ;;
    --no-teacher)       START_TEACHER=0 ;;
    --skip-benchmark)   SKIP_BENCHMARK=1 ;;
    --stage)            shift_next=1 ;;  # handled below
    -h|--help)          head -40 "$0" | grep '^#' | sed 's/^# \{0,2\}//'; exit 0 ;;
    *)
      if [ "${shift_next:-0}" = "1" ]; then
        START_STAGE="$arg"; shift_next=0
      else
        echo "ERROR: unknown arg: $arg (see --help)" >&2; exit 1
      fi ;;
  esac
done

if [ "$MOCK" = "1" ]; then
  MOCK_FLAG="--mock"
  STUDENT_CKPT="ckpt/mock-why5-sft"
  DPO_CKPT="ckpt/mock-why5-dpo"
fi

# ──────────────────────────────────────────────────────────────── helpers ──
LOG()  { printf '\033[1;36m[distill %s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
WARN() { printf '\033[1;33m[distill WARN]\033[0m %s\n' "$*"; }
DIE()  { printf '\033[1;31m[distill ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

stage_banner() {
  printf '\n\033[1;35m══════════════════════════════════════════════════════════\033[0m\n'
  printf '\033[1;35m  Stage %s: %s\033[0m\n' "$1" "$2"
  printf '\033[1;35m══════════════════════════════════════════════════════════\033[0m\n\n'
}

# ──────────────────────────────────────────────── GPU detection & profiles ──
detect_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "none"; return 0; }
  local n
  n="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  case "$n" in
    *5090*) echo "5090" ;;
    *4090*) echo "4090" ;;
    *A100*) echo "A100" ;;
    *H100*) echo "H100" ;;
    *)      echo "$n" ;;
  esac
}

GPU_ID="$(detect_gpu)"
case "$GPU_ID" in
  5090)
    VLLM_MEM_UTIL="0.90"; VLLM_BACKEND="FLASHINFER"
    TRAIN_BATCH=8; TRAIN_ACCUM=4; TRAIN_OPTIM="adamw"
    MEM_NOTE="32GB Blackwell — full AdamW fits" ;;
  4090)
    VLLM_MEM_UTIL="0.88"; VLLM_BACKEND=""
    TRAIN_BATCH=4; TRAIN_ACCUM=8; TRAIN_OPTIM="adamw8bit"
    MEM_NOTE="24GB Ada — 8-bit optimizer for VRAM savings" ;;
  A100|H100)
    VLLM_MEM_UTIL="0.90"; VLLM_BACKEND=""
    TRAIN_BATCH=16; TRAIN_ACCUM=2; TRAIN_OPTIM="adamw"
    MEM_NOTE="datacenter GPU — large batch training" ;;
  none)
    VLLM_MEM_UTIL="0.90"; VLLM_BACKEND=""
    TRAIN_BATCH=2; TRAIN_ACCUM=8; TRAIN_OPTIM="adamw"
    MEM_NOTE="no GPU — MOCK=1 recommended" ;;
  *)
    VLLM_MEM_UTIL="0.90"; VLLM_BACKEND=""
    TRAIN_BATCH=6; TRAIN_ACCUM=6; TRAIN_OPTIM="adamw"
    MEM_NOTE="unrecognized GPU '$GPU_ID' — conservative defaults" ;;
esac

LOG "GPU profile: ${GPU_ID:-unknown} ($MEM_NOTE)"
LOG "  training: batch=$TRAIN_BATCH accum=$TRAIN_ACCUM optim=$TRAIN_OPTIM"
LOG "  serving:  mem-util=$VLLM_MEM_UTIL backend=${VLLM_BACKEND:-default}"
[ -n "$VLLM_BACKEND" ] && export VLLM_ATTENTION_BACKEND="$VLLM_BACKEND"

# ──────────────────────────────────────────────────────── Python resolver ──
find_python() {
  for p in "${WHY5_VENV:-}" \
           "${VIRTUAL_ENV:-}/bin/python" \
           "${CONDA_PREFIX:-}/bin/python" \
           "$VENV_DIR/bin/python" \
           "$(pwd)/venv/bin/python" \
           "$(pwd)/.venv/bin/python" \
           "${WORK_DIR:-${HOME:-/root}/why5}/venv/bin/python" \
           "${HOME:-/root}/why5/venv/bin/python"; do
    [ -n "$p" ] && [ -x "$p" ] && { echo "$p"; return 0; }
  done
  if command -v python3 >/dev/null 2>&1; then command -v python3; return 0; fi
  if command -v python >/dev/null 2>&1; then command -v python; return 0; fi
  return 1
}

if ! PYBIN=$(find_python); then
  DIE "no Python found. Run setup.sh first:  bash setup.sh"
fi
LOG "using Python: $PYBIN"

# ──────────────────────────────── attention backend detection ──
ATTN_IMPL="sdpa"
if "$PYBIN" -c "import flash_attn" >/dev/null 2>&1; then
  ATTN_IMPL="flash_attention_2"
  LOG "attention: flash_attention_2"
else
  LOG "attention: sdpa (flash-attn not available)"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 0: Environment validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$START_STAGE" -le 0 ]; then
  stage_banner 0 "Environment validation"

  # Check all required Python packages
  LOG "checking dependencies..."
  "$PYBIN" - "$MOCK" <<'DEPCHECK'
import importlib, sys
is_mock = (sys.argv[1] == "1")

required = {
    "openai":       "teacher/student API client",
    "torch":        "model training",
    "transformers": "model loading & tokenizer",
    "trl":          "SFT & DPO trainers",
    "peft":         "LoRA adapters",
    "datasets":     "dataset handling",
    "accelerate":   "distributed training",
    "tiktoken":     "tokenization",
    "tqdm":         "progress bars",
    "yaml":         "config parsing (pyyaml)",
    "numpy":        "numerical ops",
    "huggingface_hub": "model download",
}

# Optional but useful
optional = {
    "wandb":         "experiment tracking",
    "bitsandbytes":  "8-bit optimizer",
    "deepspeed":     "ZeRO offload",
    "flash_attn":    "flash attention 2",
    "vllm":          "teacher serving",
}

missing = []
for mod, desc in required.items():
    try:
        importlib.import_module(mod)
        print(f"  ✓ {mod}")
    except ImportError:
        missing.append(mod)
        print(f"  ✗ {mod} — {desc}")

print()
for mod, desc in optional.items():
    try:
        importlib.import_module(mod)
        print(f"  ✓ {mod} (optional)")
    except ImportError:
        print(f"  ○ {mod} not installed — {desc} (optional)")

if missing:
    if is_mock:
        print(f"\n[INFO] Missing packages: {missing}")
        print("  Proceeding because --mock mode does not require full training packages.")
    else:
        print(f"\nERROR: missing required packages: {missing}")
        print("To fix, run:")
        print("  bash setup.sh")
        print(f"Or install directly into this environment:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)

if not missing:
    print("\n  All required dependencies OK")
DEPCHECK
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TEACHER STARTUP (real mode only — auto-starts unless --no-teacher)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$MOCK" != "1" ]; then
  stage_banner T "Teacher model setup"
  serve_flags="--max-model-len 8192 --gpu-memory-utilization $VLLM_MEM_UTIL"

  # Ensure teacher model weights are cached
  download_model() {
    LOG "ensuring teacher model is cached: $1"
    "$PYBIN" - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
try:
    snapshot_download(sys.argv[1], local_files_only=True)
    print(f"  model already cached: {sys.argv[1]}")
except Exception:
    print(f"  downloading: {sys.argv[1]}")
    snapshot_download(sys.argv[1],
                      ignore_patterns=["*.pth", "*.gguf", "*.bin", "original/*"])
    print(f"  ✓ downloaded: {sys.argv[1]}")
PY
  }

  start_teacher() {
    mkdir -p data/logs
    LOG "starting teacher server in background → data/logs/teacher_serve.log"
    local vllm_cmd
    if [ -x "$VENV_DIR/bin/vllm" ]; then
      vllm_cmd="$VENV_DIR/bin/vllm serve"
    else
      vllm_cmd="$PYBIN -m vllm.entrypoints.openai.api_server --model"
    fi
    nohup $vllm_cmd "$TEACHER_MODEL" \
        --port 8000 $serve_flags \
        > data/logs/teacher_serve.log 2>&1 &
    echo $! > data/logs/teacher_serve.pid
    LOG "waiting for teacher startup (up to 15 min)..."
    for i in $(seq 1 180); do
      sleep 5
      if curl -sf -o /dev/null --max-time 3 "$TEACHER_URL/v1/models"; then
        LOG "✓ teacher is UP at $TEACHER_URL"
        return 0
      fi
    done
    DIE "teacher did not start in 15 min — check: tail -f data/logs/teacher_serve.log"
  }

  if curl -sf -o /dev/null --max-time 5 "$TEACHER_URL/v1/models"; then
    LOG "✓ teacher already running at $TEACHER_URL — skipping startup"
  elif [ "$START_TEACHER" = "1" ]; then
    download_model "$TEACHER_MODEL"
    start_teacher
  else
    LOG "teacher not running at $TEACHER_URL (--no-teacher was set, assuming external server)"
    WARN "pipeline will fail if teacher is not reachable when needed"
  fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 1: Corpus generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$START_STAGE" -le 1 ]; then
  stage_banner 1 "Corpus generation (teacher → training data)"
  LOG "generating N_D1=$N_D1 depth-1 + N_D5=$N_D5 depth-5 samples"

  "$PYBIN" distill/why5_distill/gen_corpus.py \
    --n "$N_D1" --n-d5 "$N_D5" \
    --out distill/data/distilled \
    --dataset-version "$DATASET_VERSION" \
    $MOCK_FLAG
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 2: SFT data preparation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$START_STAGE" -le 2 ]; then
  stage_banner 2 "SFT data preparation"

  if [ -f "distill/why5_distill/sft_data.py" ]; then
    "$PYBIN" distill/why5_distill/sft_data.py \
      --data "distill/$DATA_DIR" \
      --out "distill/data/sft_ready" \
      $MOCK_FLAG \
      || WARN "sft_data.py failed or not applicable — SFT trainer will handle raw data"
  else
    LOG "no separate sft_data.py — SFT trainer handles data formatting"
  fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 3: Supervised fine-tuning (SFT)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$START_STAGE" -le 3 ]; then
  stage_banner 3 "Supervised fine-tuning (SFT)"
  LOG "batch=$TRAIN_BATCH accum=$TRAIN_ACCUM optim=$TRAIN_OPTIM attn=$ATTN_IMPL KD=$KD"

  "$PYBIN" distill/why5_distill/train_sft.py \
    --data "distill/$DATA_DIR" \
    --out "distill/$STUDENT_CKPT" \
    --kd "$KD" \
    --batch-size "$TRAIN_BATCH" \
    --grad-accum "$TRAIN_ACCUM" \
    --optim "$TRAIN_OPTIM" \
    --attn "$ATTN_IMPL" \
    $MOCK_FLAG
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 4: On-policy rollouts → DPO correction pairs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$START_STAGE" -le 4 ]; then
  stage_banner 4 "On-policy correction → DPO pairs"
  LOG "generating $N_ROLLOUTS rollouts from SFT checkpoint"

  "$PYBIN" distill/why5_distill/on_policy.py \
    --student "distill/$STUDENT_CKPT" \
    --n "$N_ROLLOUTS" \
    --out distill/data/dpo_pairs.jsonl \
    $MOCK_FLAG
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 5: DPO alignment training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$START_STAGE" -le 5 ]; then
  stage_banner 5 "DPO alignment training"

  "$PYBIN" distill/why5_distill/dpo_train.py \
    --pairs distill/data/dpo_pairs.jsonl \
    --model "distill/$STUDENT_CKPT" \
    --out "distill/$DPO_CKPT" \
    --attn "$ATTN_IMPL" \
    $MOCK_FLAG
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STAGE 6: Benchmark evaluation (optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ "$START_STAGE" -le 6 ] && [ "$SKIP_BENCHMARK" -eq 0 ]; then
  stage_banner 6 "Benchmark evaluation"

  if [ -f "benchmark/run_benchmark.sh" ]; then
    ( cd benchmark && bash run_benchmark.sh $MOCK_FLAG ) \
      || WARN "benchmark did not complete successfully"
  elif [ -f "benchmark/why5_bench.py" ]; then
    "$PYBIN" benchmark/why5_bench.py \
      --model "distill/$DPO_CKPT" \
      $MOCK_FLAG \
      || WARN "benchmark did not complete successfully"
  else
    WARN "no benchmark runner found — skipping evaluation"
  fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COMPLETE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cat <<BANNER

============================================================
 ✓ Distillation pipeline complete
============================================================
 Mode:        $([ "$MOCK" = "1" ] && echo "MOCK (dry-run)" || echo "REAL")
 GPU:         ${GPU_ID:-none} ($MEM_NOTE)
 Teacher:     $TEACHER_MODEL
 Student:     $STUDENT_MODEL
 SFT ckpt:    distill/$STUDENT_CKPT
 DPO ckpt:    distill/$DPO_CKPT
 Data:        distill/$DATA_DIR

 Next steps:
   • Evaluate: cd benchmark && bash run_benchmark.sh --model distill/$DPO_CKPT
   • Export:   huggingface-cli upload <your-repo> distill/$DPO_CKPT
   • Serve:    vllm serve distill/$DPO_CKPT --port 8001
============================================================
BANNER
LOG "all done"
