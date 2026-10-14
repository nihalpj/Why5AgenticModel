#!/usr/bin/env bash
# End-to-end distillation driver (DISTILLATION_PLAN_QWEN35_4B.md).
#
# Real run prerequisites:
#   # Teacher (Stage 0):
#   vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 8192
#   # Student after SFT (Stage 4):
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

echo "== Stage 0/1: corpus generation with Qwen 3.5 teacher =="
python3 why5_distill/gen_corpus.py \
  --n "$N_D1" --n-d5 "$N_D5" \
  --out data/distilled --dataset-version v0.5.0-distill-pilot $MOCK_FLAG

echo "== Stage 3: SFT (mock = data-pipeline dry run only) =="
python3 why5_distill/train_sft.py \
  --data data/distilled/v0.5.0-distill-pilot \
  --out "$STUDENT_CKPT" --kd "${KD:-0.0}" $MOCK_FLAG

echo "== Stage 4: on-policy correction -> DPO pairs =="
python3 why5_distill/on_policy.py \
  --student "$STUDENT_CKPT" --n "$N_ROLLOUTS" \
  --out data/dpo_pairs.jsonl $MOCK_FLAG

echo "== Stage 4b: DPO on correction pairs =="
python3 why5_distill/dpo_train.py \
  --pairs data/dpo_pairs.jsonl --model "$STUDENT_CKPT" \
  --out ckpt/why5-dpo $MOCK_FLAG

echo "== distillation pipeline complete =="
