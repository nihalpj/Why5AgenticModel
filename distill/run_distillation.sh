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

echo "== Stage 0/1: corpus generation with Qwen 3.5 teacher =="
"$PYBIN" why5_distill/gen_corpus.py \
  --n "$N_D1" --n-d5 "$N_D5" \
  --out data/distilled --dataset-version v0.5.0-distill-pilot $MOCK_FLAG

echo "== Stage 3: SFT (mock = data-pipeline dry run only) =="
"$PYBIN" why5_distill/train_sft.py \
  --data data/distilled/v0.5.0-distill-pilot \
  --out "$STUDENT_CKPT" --kd "${KD:-0.0}" $MOCK_FLAG

echo "== Stage 4: on-policy correction -> DPO pairs =="
"$PYBIN" why5_distill/on_policy.py \
  --student "$STUDENT_CKPT" --n "$N_ROLLOUTS" \
  --out data/dpo_pairs.jsonl $MOCK_FLAG

echo "== Stage 4b: DPO on correction pairs =="
"$PYBIN" why5_distill/dpo_train.py \
  --pairs data/dpo_pairs.jsonl --model "$STUDENT_CKPT" \
  --out ckpt/why5-dpo $MOCK_FLAG

echo "== distillation pipeline complete =="
