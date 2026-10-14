"""Why5 distillation package: Qwen 3.5 4B teacher -> Why5 2B student.

Implements DISTILLATION_PLAN_QWEN35_4B.md:
    teacher.py     Stage 0/1: Qwen teacher adapter (seed -> prompt -> parse)
    gen_corpus.py  Stage 1: factory generation with the Qwen teacher
    sft_data.py    Stage 3: JSONL -> tokenized SFT dataset with loss masks
    train_sft.py   Stage 3: SFT training (+ optional logit KD)
    on_policy.py   Stage 4: student rollouts, sandbox judging, teacher fixes
    dpo_train.py   Stage 4: DPO on correction pairs
"""

__version__ = "0.1.0"
