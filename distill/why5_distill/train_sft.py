"""Stage 3: SFT training of the Why5 2B student on the distilled corpus.

Loss: weighted CE on assistant spans (labels != -100), where per-example
weight implements the anti-imitation upweight for no-tool traces (plan
§Stage 3). Optional forward-KL distillation on the SAME masked positions
against the Qwen 3.5 4B teacher logits (plan §3.2, decision-gated).

Depth control tokens are CE-only (rubric supervision); they are excluded
from the KL term so the teacher can never soften the depth contract
(plan §3.2 "never KL on protocol control tokens").

Usage (student = any HF causal LM, teacher = Qwen 3.5 4B):
    python train_sft.py --student why5-2b-base --teacher Qwen/Qwen3.5-4B \
        --data ../distill/data/distilled --kd 0.5
    python train_sft.py --data ... --mock   # dry-run without GPU/models
"""

import argparse
import json
import math
import os
import sys


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="why5-2b-base")
    ap.add_argument("--teacher", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data", required=True, help="dataset root (from gen_corpus.py)")
    ap.add_argument("--out", default="ckpt/why5-sft")
    ap.add_argument("--kd", type=float, default=0.0,
                    help="KL weight; 0 disables teacher forward passes entirely")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw",
                    help="adamw8bit (bitsandbytes) keeps Adam states in VRAM — "
                         "required for 2B SFT on 24GB (RTX 4090), optional on 32GB")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true",
                    help="tokenizer/dataset dry-run; no models loaded, no training")
    return ap.parse_args(argv)


def load_models(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    student_tok = AutoTokenizer.from_pretrained(args.student)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher)
    for tok in (student_tok,):
        tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    student = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    student.resize_token_embeddings(len(student_tok))
    teacher = None
    if args.kd > 0:
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.bfloat16)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    return student, teacher, student_tok, teacher_tok


def kd_mask(labels, input_ids, student_tok):
    """Positions included in the KL term: supervised positions that are NOT
    depth control tokens and NOT a protocol special token boundary."""
    special_ids = set(student_tok.additional_special_tokens_ids)
    depth_ids = {student_tok.convert_tokens_to_ids(f"<|depth_{i}|>") for i in (3, 4, 5)}
    m = (labels != -100)
    # exclude teacher-uninterpretable protocol tokens by ID
    excl = special_ids
    depth_ids &= excl  # depth tokens are part of additional_special_tokens
    ids = input_ids
    keep = m & ~torch.isin(ids, torch.tensor(sorted(excl), device=ids.device))
    return keep


def train(args):
    from sft_data import SPECIAL_TOKENS, DistillSFTDataset, collate, list_shards
    shards, keep = list_shards(args.data, split="train")
    if not shards:
        sys.exit(f"no .jsonl shards under {args.data}")
    print(f"training on {len(shards)} shards from {args.data}")

    if args.mock:  # exercise data pipeline only — no torch required
        ds = DistillSFTDataset(shards, tokenizer=None, mock_tokenize=True, keep_ids=keep)
        n_train = sum(1 for ex in ds.examples if any(l != -100 for l in ex["labels"]))
        n_up = sum(1 for ex in ds.examples if ex["loss_weight"] > 1.0)
        print(f"[mock] {len(ds)} examples; {n_train} with supervised tokens; "
              f"{n_up} no-tool upweighted x1.5")
        sample = ds[0]
        n_sup = sum(1 for l in sample["labels"] if l != -100)
        print(f"[mock] example 0: {len(sample['input_ids'])} tokens, {n_sup} supervised")
        return

    import torch
    import torch.nn.functional as F
    torch.manual_seed(args.seed)

    student, teacher, s_tok, t_tok = load_models(args)
    ds = DistillSFTDataset(shards, tokenizer=s_tok, max_len=args.max_len, keep_ids=keep)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate(b, s_tok.pad_token_id or s_tok.eos_token_id))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    student.to(device)
    if teacher:
        teacher.to(device)
    if args.optim == "adamw8bit":
        try:
            import bitsandbytes as bnb  # pip install bitsandbytes
            opt = bnb.optim.AdamW8bit(student.parameters(), lr=args.lr,
                                      weight_decay=0.01)
            print(f"optimizer: AdamW8bit (bitsandbytes) — Adam states stay 8-bit "
                  f"({args.batch_size=}, {args.grad_accum=})")
        except ImportError:
            print("warning: bitsandbytes not installed — falling back to AdamW "
                  "(~12GB extra VRAM; may OOM on 24GB GPUs). "
                  "pip install bitsandbytes")
            opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                                    weight_decay=0.01)
    else:
        opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = math.ceil(len(dl) / args.grad_accum) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.03)

    student.train()
    step = 0
    for epoch in range(args.epochs):
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = student(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"])
            logits = out.logits[:, :-1]
            labels = batch["labels"][:, 1:]
            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1), ignore_index=-100, reduction="none")
            mask = (labels != -100).reshape(-1)
            w = batch["loss_weight"].unsqueeze(1).expand_as(labels).reshape(-1)[mask]
            loss = (ce[mask] * w).sum() / w.sum().clamp(min=1)

            if teacher is not None and args.kd > 0:
                with torch.no_grad():
                    t_logits = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"]).logits[:, :-1]
                # same token space required (plan Option L1: Qwen tokenizer)
                if t_logits.size(-1) != logits.size(-1):
                    sys.exit("teacher/student vocab mismatch: adopt the Qwen tokenizer "
                             "(plan §3.2 Option L1) or set --kd 0")
                keep = kd_mask(labels, batch["input_ids"][:, 1:], s_tok)
                t_lp = F.log_softmax(t_logits.float(), dim=-1)
                s_lp = F.log_softmax(logits.float(), dim=-1)
                kl = F.kl_div(s_lp, t_lp, log_target=True, reduction="none").sum(-1)
                denom = keep.sum().clamp(min=1)
                loss = loss + args.kd * (kl * keep).sum() / denom

            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}")

    os.makedirs(args.out, exist_ok=True)
    student.save_pretrained(args.out)
    s_tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_config.json"), "w") as f:
        json.dump({**vars(args), "total_steps": total_steps,
                   "kd_active": args.kd > 0}, f, indent=2)
    print(f"saved student checkpoint to {args.out}")


if __name__ == "__main__":
    train(build_args())
