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
STUDENT_CKPT="ckpt/why5-sft"

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
  for p in "${WHY5_VENV:-}/bin/python" \
           "${WORK_DIR:-${HOME:-/root}/why5}/venv/bin/python" \
           "${HOME:-/root}/why5/venv/bin/python" \
           /usr/local/bin/python3 /usr/bin/python3; do
    [ -n "$p" ] && [ -x "$p" ] && { echo "$p"; return 0; }
  done
  command -v python3 >/dev/null 2>&1 && { command -v python3; return 0; }
  command -v python >/dev/null 2>&1 && { command -v python; return 0; }
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
