#!/usr/bin/env bash
# End-to-end distillation driver (DISTILLATION_PLAN_QWEN35_4B.md).
#
# Real run prerequisites (per-GPU serve flags):
#   # 5090 (32GB Blackwell):
#   vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 8192 \
#        --gpu-memory-utilization 0.90        # VLLM_ATTENTION_BACKEND=FLASHINFER
#   # 4090 (24GB Ada):
#   vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 8192 \
#        --gpu-memory-utilization 0.85
#   # Student after SFT (either GPU):
#   vllm serve ckpt/why5-sft --port 8001 --enable-auto-tool-choice \
#        --tool-call-parser hermes
#
# Add MOCK=1 to run the full pipeline offline (no models, no GPUs):
#   MOCK=1 ./run_distillation.sh
set -euo pipefail
cd "$(dirname "$0")"

MOCK="${MOCK:-0}"
N_D1="${N_D1:-200}"
N_D5="${N_D5:-50}"
N_ROLLOUTS="${N_ROLLOUTS:-60}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-4B}"
TEACHER_URL="${TEACHER_URL:-http://localhost:8000}"
VENV_DIR="${VENV_DIR:-${WORK_DIR:-${HOME:-/root}/why5}/venv}"   # setup's venv
STUDENT_CKPT="ckpt/why5-sft"

# HF cache: same location the setup script uses (set BEFORE any hf/vllm call
# so START_TEACHER never re-downloads the model into ~/.cache as a 2nd copy)
DATA_ROOT="${DATA_ROOT:-$(ls -d /data 2>/dev/null || echo "${HOME:-/root}")}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf_cache}"

MOCK_FLAG=""
if [ "$MOCK" = "1" ]; then
  MOCK_FLAG="--mock"
  STUDENT_CKPT="ckpt/mock-why5-sft"
fi

# ---- GPU detection & profiles -----------------------------------------------
# 5090 = 32GB Blackwell (sm_120): 2B SFT fits in VRAM; FlashInfer attention is
#        the safest vLLM backend on Blackwell.
# 4090 = 24GB Ada (sm_89): same vLLM; 2B SFT needs the 8-bit AdamW optimizer
#        (bitsandbytes) to fit Adam states in 24GB; smaller batch.
detect_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "none"; return 0; }
  local n
  n="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  case "$n" in
    *5090*) echo "5090" ;;
    *4090*) echo "4090" ;;
    *) echo "$n" ;;
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
    MEM_NOTE="24GB Ada — 8-bit optimizer keeps 2B SFT in VRAM" ;;
  none)
    VLLM_MEM_UTIL="0.90"; VLLM_BACKEND=""
    TRAIN_BATCH=2; TRAIN_ACCUM=8; TRAIN_OPTIM="adamw"
    MEM_NOTE="no GPU detected — CPU-only; MOCK=1 recommended" ;;
  *)
    VLLM_MEM_UTIL="0.90"; VLLM_BACKEND=""
    TRAIN_BATCH=6; TRAIN_ACCUM=6; TRAIN_OPTIM="adamw"
    MEM_NOTE="unrecognized GPU '$GPU_ID' — conservative defaults" ;;
esac
echo "GPU profile: ${GPU_ID:-unknown} ($MEM_NOTE)"
echo "  serving: mem-util=$VLLM_MEM_UTIL backend=${VLLM_BACKEND:-default} | training: batch=$TRAIN_BATCH accum=$TRAIN_ACCUM optim=$TRAIN_OPTIM"
[ -n "$VLLM_BACKEND" ] && export VLLM_ATTENTION_BACKEND="$VLLM_BACKEND"

# ---- resolve python: venv from setup_5090_novita.sh, then system python ----
find_python() {
  for p in "${WHY5_VENV:-}" \
           "${WORK_DIR:-${HOME:-/root}/why5}/venv/bin/python" \
           "${HOME:-/root}/why5/venv/bin/python"; do
    [ -n "$p" ] && [ -x "$p" ] && { echo "$p"; return 0; }
  done
  # PATH interpreter first (venv users often activate one); hard-coded
  # absolute paths only as last resort
  if command -v python3 >/dev/null 2>&1; then command -v python3; return 0; fi
  if command -v python >/dev/null 2>&1; then command -v python; return 0; fi
  for p in /usr/local/bin/python3 /usr/bin/python3; do
    [ -x "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}
if ! PYBIN=$(find_python); then
  cat >&2 <<EOF

ERROR: no python (>= 3.10) found on this machine.

Fix: run the environment setup first, then re-run this script (it will
find the venv automatically):
    bash ../setup/setup_5090_novita.sh          # installs miniforge + venv

Or point this script at an existing venv:
    WHY5_VENV=/path/to/venv ./run_distillation.sh
EOF
  exit 1
fi
echo "using python: $PYBIN"
case "$PYBIN" in
  */venv/bin/python) : ;;
  *) echo "note: using SYSTEM python (not the setup venv) — if imports fail," \
     "run: bash ../setup/setup_5090_novita.sh" ;;
esac

# ---- self-heal deps: real mode needs openai (mock mode needs nothing) -------
if [ "$MOCK" != "1" ]; then
  if ! "$PYBIN" -c "import openai" >/dev/null 2>&1; then
    echo "installing missing dependency: openai -> $("$PYBIN" --version 2>&1 | cut -d' ' -f2) env"
    "$PYBIN" -m pip install -q openai || {
      echo "ERROR: automatic install failed. run manually:" >&2
      echo "  $PYBIN -m pip install openai" >&2; exit 1
    }
    echo "openai installed OK"
  fi
fi

# ---- teacher model download: ensure weights are in HF_HOME before serving --
model_cached() {  # $1 = repo id -> 0 if already in HF cache (offline check)
  HF_HUB_OFFLINE=1 "$PYBIN" - "$1" <<'PY' >/dev/null 2>&1
import sys
try:
    from huggingface_hub import snapshot_download
    snapshot_download(sys.argv[1], local_files_only=True)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}
download_model() {  # $1 = repo id
  echo "downloading teacher model: $1 -> HF_HOME=$HF_HOME"
  if ! "$PYBIN" - "$1" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1],
                  ignore_patterns=["*.pth", "*.gguf", "*.bin", "original/*"])
PY
  then
    echo "ERROR: model download failed. Fix manually:" >&2
    echo "  $PYBIN -m pip install 'huggingface_hub[cli,hf_transfer]'" >&2
    echo "  $PYBIN -c \"from huggingface_hub import snapshot_download; snapshot_download('$1')\"" >&2
    exit 2
  fi
  echo "teacher model ready: $1"
}

# ---- teacher preflight: real mode needs the vLLM teacher on :8000 -----------
if [ "$MOCK" != "1" ]; then
  serve_flags="--max-model-len 8192 --gpu-memory-utilization $VLLM_MEM_UTIL"
  [ "$GPU_ID" = "5090" ] && export VLLM_ATTENTION_BACKEND="$VLLM_BACKEND"
  start_teacher() {
    LOG_TAG="teacher"
    mkdir -p data/logs
    echo "starting teacher in background: data/logs/teacher_serve.log"
    nohup "$VENV_DIR/bin/vllm" serve "$TEACHER_MODEL" \
        --port 8000 $serve_flags \
        > data/logs/teacher_serve.log 2>&1 &
    echo $! > data/logs/teacher_serve.pid
    echo "waiting for startup (up to 15 min; model download included)..."
    for i in $(seq 1 180); do
      sleep 5
      if curl -sf -o /dev/null --max-time 3 "$TEACHER_URL/v1/models"; then
        echo "teacher is UP at $TEACHER_URL"
        return 0
      fi
    done
    echo "teacher did not come up in 15 min — last log lines:" >&2
    tail -20 data/logs/teacher_serve.log >&2
    echo "check: tail -f data/logs/teacher_serve.log" >&2
    exit 2
  }

  if curl -sf -o /dev/null --max-time 5 "$TEACHER_URL/v1/models"; then
    echo "teacher OK at $TEACHER_URL"
  elif [ "${START_TEACHER:-0}" = "1" ]; then
    download_model "$TEACHER_MODEL"   # skipped if already in HF cache
    start_teacher
  else
    cat >&2 <<EOF

ERROR: teacher not reachable at $TEACHER_URL — start it first, then re-run.

  Easiest (this script starts and waits for it in the background):
    START_TEACHER=1 ./run_distillation.sh

  Or manually, in a separate terminal:
  # 5090 (32GB Blackwell):
  VLLM_ATTENTION_BACKEND=FLASHINFER "$VENV_DIR/bin/vllm" serve "$TEACHER_MODEL" \\
      --port 8000 $serve_flags
  # 4090 (24GB Ada):
  "$VENV_DIR/bin/vllm" serve "$TEACHER_MODEL" --port 8000 $serve_flags

Wait for "Application startup complete", then re-run this script.
(Offline? MOCK=1 ./run_distillation.sh needs no server.)
EOF
    exit 2
  fi
fi

echo "== Stage 0/1: corpus generation with Qwen 3.5 teacher =="
"$PYBIN" why5_distill/gen_corpus.py \
  --n "$N_D1" --n-d5 "$N_D5" \
  --out data/distilled --dataset-version v0.5.0-distill-pilot $MOCK_FLAG

echo "== Stage 3: SFT (mock = data-pipeline dry run only) =="
"$PYBIN" why5_distill/train_sft.py \
  --data data/distilled/v0.5.0-distill-pilot \
  --out "$STUDENT_CKPT" --kd "${KD:-0.0}" \
  --batch-size "$TRAIN_BATCH" --grad-accum "$TRAIN_ACCUM" --optim "$TRAIN_OPTIM" \
  $MOCK_FLAG

echo "== Stage 4: on-policy correction -> DPO pairs =="
"$PYBIN" why5_distill/on_policy.py \
  --student "$STUDENT_CKPT" --n "$N_ROLLOUTS" \
  --out data/dpo_pairs.jsonl $MOCK_FLAG

echo "== Stage 4b: DPO on correction pairs =="
"$PYBIN" why5_distill/dpo_train.py \
  --pairs data/dpo_pairs.jsonl --model "$STUDENT_CKPT" \
  --out ckpt/why5-dpo $MOCK_FLAG

echo "== distillation pipeline complete =="
