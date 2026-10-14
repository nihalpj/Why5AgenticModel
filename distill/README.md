# Why5 Distillation Scripts — Qwen 3.5 4B → Why5 2B

Implementation of `DISTILLATION_PLAN_QWEN35_4B.md`. Five stages, each runnable
standalone or via `./run_distillation.sh`.

## Layout

| File | Plan stage | Purpose |
|---|---|---|
| `why5_distill/teacher.py` | 0 | Qwen 3.5 4B adapter: seed → Hermes-style prompt → parse `<why>` blocks. Depth is forced by the rubric; teacher never routes. `--mock` = offline template teacher. |
| `why5_distill/gen_corpus.py` | 1 | Factory corpus generation with the teacher plugged into `TraceGenerator`. 4 candidates/seed, median-length valid one kept (FR-B2). Writes the §3.7 JSONL contract + manifest + splits. |
| `why5_distill/sft_data.py` | 3 | JSONL → tokenized SFT dataset. Loss masks: system/user/tool-result = −100; depth token, why-steps, plan, tool args, final = supervised. No-tool records upweighted ×1.5. |
| `why5_distill/train_sft.py` | 3 | SFT loop (weighted CE on masked positions) + optional forward-KL distillation (`--kd β`) against teacher logits. KL excludes protocol control tokens (depth contract stays rubric-pure). Requires Qwen tokenizer on the student (plan Option L1) if `--kd > 0`. |
| `why5_distill/on_policy.py` | 4 | Student rollouts on held-out seeds → environment-as-judge (parse/schema/sandbox; no teacher scoring) → teacher-corrected traces → DPO pairs JSONL. Warns if correction yield < 50% (plan gate). |
| `why5_distill/dpo_train.py` | 4b | DPO (`trl`) on the correction pairs; chosen = gate-clean teacher fix, rejected = the student's own failed rollout. |
| `run_distillation.sh` | all | Orchestrates 0→4b. `MOCK=1` runs everything offline (no models, no GPUs). |

## Quick start (offline, no GPU)

```bash
cd distill
MOCK=1 ./run_distillation.sh          # full pipeline with mock teacher/student
MOCK=1 N_D1=100 N_ROLLOUTS=30 ./run_distillation.sh
```

## Real run

```bash
# 1. serve the teacher (Stage 0)
vllm serve Qwen/Qwen3.5-4B --port 8000 --max-model-len 8192

# 2. generate the distilled corpus (Stage 1)
python3 why5_distill/gen_corpus.py --n 50000 --out data/distilled

# 3. SFT the student (Stage 3); --kd 0.5 to add logit distillation
python3 why5_distill/train_sft.py --student why5-2b-base \
    --teacher Qwen/Qwen3.5-4B --data data/distilled/<version> --kd 0.5

# 4. serve the student, run on-policy correction (Stage 4)
vllm serve ckpt/why5-sft --port 8001
python3 why5_distill/on_policy.py --student ckpt/why5-sft --n 5000

# 5. DPO (Stage 4b)
python3 why5_distill/dpo_train.py --pairs data/dpo_pairs.jsonl --model ckpt/why5-sft
```

## Guardrails encoded here (plan §5)

- Depth tokens: CE-supervised by rubric labels, excluded from KL — the teacher
  cannot soften the 3/4/5 contract.
- Judging is sandbox-only in Stage 4 — teacher agreement is never a reward.
- Malformed student rollouts are DPO-rejected as-is (`{MALFORMED}`), teaching
  away from format errors at the student's actual failure states.

## Verified

`MOCK=1 ./run_distillation.sh` (N_D1=100, N_ROLLOUTS=30): 100 D1 records at
the 40/40/20 tier quota + 20 D5 counterfactuals, all G1–G10 green; SFT
data pipeline 117 train examples with correct masks; 30 rollouts → 5 failures
judged → 5 DPO pairs with 100% correction yield.
