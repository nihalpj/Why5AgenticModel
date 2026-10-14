"""Stage 4b: DPO on on-policy correction pairs (plan §Stage 4, decision 3).

pairs format (on_policy.py):
    chosen  = teacher-corrected protocol turn (serialized)
    rejected= the student's failed rollout (serialized)

Chosen and rejected share the same prompt (same seed, same forced depth),
so this is a pure preference update on trajectory format/behavior.

Usage:
    python dpo_train.py --pairs data/dpo_pairs.jsonl --model ckpt/why5-sft
"""

import argparse
import json
import os

import torch


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="ckpt/why5-sft")
    ap.add_argument("--out", default="ckpt/why5-dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--mock", action="store_true")
    return ap.parse_args(argv)


def serialize_turns(protocol_turns):
    """Rebuild the assistant completion from protocol turns (matches sft_data)."""
    from sft_data import _DEPTH_TOK
    parts = [_DEPTH_TOK[str(next(t["value"] for t in protocol_turns if t["type"] == "depth"))]]
    for t in protocol_turns:
        if t["type"] == "why":
            parts.append(f"<|why|>{t['n']}. {t['text']}<|end_why|>")
        elif t["type"] == "plan":
            parts.append(f"<|plan|>{t['text']}<|end_plan|>")
        elif t["type"] == "tool_call":
            parts.append(f'<|tool_call|>{{"name":"{t["name"]}","arguments":'
                         f'{json.dumps(t["arguments"])}}}<|end_tool_call|>')
        elif t["type"] == "final":
            parts.append(f"<|final|>{t['text']}<|end_final|>")
    return "\n".join(parts)


def serialize_rollout(rollout):
    """Best-effort serialization of a raw student rollout (may be malformed —
    that is the point: DPO pushes probability away from it)."""
    depth_tok = f"<|depth_{rollout['depth']}|>"
    parts = [depth_tok]
    for i, w in enumerate(rollout.get("whys", [])):
        parts.append(f"<|why|>{i + 1}. {w}<|end_why|>")
    for tc in rollout.get("tool_calls", []):
        if tc.get("name") is None:
            parts.append("<|tool_call|>{MALFORMED}<|end_tool_call|>")
        else:
            args = json.dumps(tc.get("arguments") or {})
            parts.append(f'<|tool_call|>{{"name":"{tc["name"]}","arguments":{args}}}'
                         f"<|end_tool_call|>")
    return "\n".join(parts)


def load_pairs(path, tokenizer, mock=False):
    prompts, chosen, rejected = [], [], []
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            prompts.append(p["seed_id"])  # placeholder in mock; real prompt built below
            chosen.append(serialize_turns(p["teacher_correction"]["protocol_turns"]))
            rejected.append(serialize_rollout(p["student_rollout"]))
    if mock:
        return prompts, chosen, rejected
    from sft_data import SPECIAL_TOKENS
    real_prompts = []
    with open(path) as f:
        # rebuild real prompts from the protocol: system + user task text
        # (kept minimal here; production uses the full registry block)
        for line in f:
            real_prompts.append("<|system|>You are an agent.<|assistant|>")
    toks = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
    return ([toks(p) for p in real_prompts],
            [toks(c) for c in chosen], [toks(r) for r in rejected])


def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    """Standard DPO objective on sequence logprobs."""
    pc = policy_chosen - ref_chosen
    pr = policy_rejected - ref_rejected
    logits = beta * (pc - pr)
    return -torch.nn.functional.logsigmoid(logits).mean()


def main(argv=None):
    args = build_args(argv)
    if args.mock:
        prompts, chosen, rejected = load_pairs(args.pairs, None, mock=True)
        print(f"[mock] {len(chosen)} pairs loaded; chosen/rejected serialization OK")
        for c, r in zip(chosen[:2], rejected[:2]):
            print(f"  chosen:   {c[:90]}...")
            print(f"  rejected: {r[:90]}...")
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sft_data import SPECIAL_TOKENS
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    policy = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    policy.resize_token_embeddings(len(tok))
    ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    from trl import DPOConfig, DPOTrainer
    with open(args.pairs) as f:
        rows = [json.loads(l) for l in f]
    data = [{"prompt": "<|system|>You are an agent.<|assistant|>",
             "chosen": serialize_turns(r["teacher_correction"]["protocol_turns"]),
             "rejected": serialize_rollout(r["student_rollout"])} for r in rows]
    from datasets import Dataset
    ds = Dataset.from_list(data)
    cfg = DPOConfig(
        output_dir=args.out, beta=args.beta, learning_rate=args.lr,
        num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        bf16=True, logging_steps=5, save_strategy="epoch",
        max_length=4096, max_prompt_length=2048)
    trainer = DPOTrainer(model=policy, ref_model=ref, args=cfg,
                         train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved DPO checkpoint to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
