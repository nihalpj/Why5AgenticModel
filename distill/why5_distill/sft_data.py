"""Stage 3: distilled JSONL -> tokenized SFT dataset with loss masks.

Masking policy (Main Plan §4.2):
    - train ONLY on assistant spans, split into segments:
        why-steps, plan, tool-call JSON arguments  -> labels kept
        depth token <|depth_N|>                    -> labels kept (rubric label)
        <|final|> answer text                      -> labels kept
    - system / user / tool-result spans            -> labels = -100
    - anti-imitation upweight (plan §Stage 3): refusal / no-tool traces
      get loss weight 1.5x via per-example `loss_weight`.
"""

import json
import os

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # mock/offline use: character-level stand-ins, no tensors
    torch = None
    class Dataset:  # minimal shim preserving __len__/__getitem__ protocol
        pass

# Control tokens emitted by the serialized protocol; the tokenizer must know
# them (add_special_tokens at model load time; see train_sft.py).
SPECIAL_TOKENS = [
    "<|system|>", "<|user|>", "<|assistant|>",
    "<|tools_begin|>", "<|tools_end|>",
    "<|tool_call|>", "<|end_tool_call|>",
    "<|tool_result|>", "<|end_tool_result|>",
    "<|depth_3|>", "<|depth_4|>", "<|depth_5|>",
    "<|why|>", "<|end_why|>",
    "<|plan|>", "<|end_plan|>",
    "<|final|>", "<|end_final|>",
    "<|retry|>",
]

_DEPTH_TOK = {"3": "<|depth_3|>", "4": "<|depth_4|>", "5": "<|depth_5|>"}


def assistant_text(rec):
    """Rebuild the assistant turn from protocol_turns (single source of truth)."""
    parts = []
    depth = next(t["value"] for t in rec["protocol_turns"] if t["type"] == "depth")
    parts.append(_DEPTH_TOK[str(depth)])
    for t in rec["protocol_turns"]:
        if t["type"] == "why":
            parts.append(f"<|why|>{t['n']}. {t['text']}<|end_why|>")
        elif t["type"] == "plan":
            parts.append(f"<|plan|>{t['text']}<|end_plan|>")
        elif t["type"] == "tool_call":
            args = json.dumps(t["arguments"])
            parts.append(f'<|tool_call|>{{"name":"{t["name"]}","arguments":{args}}}<|end_tool_call|>')
        elif t["type"] == "final":
            parts.append(f"<|final|>{t['text']}<|end_final|>")
    return "\n".join(parts)


def is_no_tool_record(rec):
    return rec["metrics"]["tool_calls"] == 0


class DistillSFTDataset(Dataset):
    """JSONL (factory contract) -> input_ids / labels / loss_weight."""

    def __init__(self, jsonl_paths, tokenizer, max_len=4096,
                 no_tool_weight=1.5, mock_tokenize=False, keep_ids=None):
        self.examples = []
        self.no_tool_weight = no_tool_weight
        self.tokenizer = tokenizer
        self.mock = mock_tokenize or tokenizer is None
        paths = [jsonl_paths] if isinstance(jsonl_paths, str) else jsonl_paths
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if keep_ids is not None and rec["record_id"] not in keep_ids:
                        continue
                    if not rec.get("validation", {}).get("passed", False):
                        raise ValueError(f"record {rec['record_id']} failed validation")
                    self.examples.append(self._build(rec))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        if self.mock:
            return ex  # already int lists (see _build)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in ex.items()}

    def _build(self, rec):
        registry = json.dumps(rec["tools_registry"])
        prompt = (
            "<|system|>You are an agent. Reason in why-steps, then call tools, then answer."
            f"<|tools_begin|>{registry}<|tools_end|>"
            f"<|user|>{rec['messages'][1]['content']}<|assistant|>"
        )
        completion = assistant_text(rec)
        if self.mock:
            # character-level stand-in so masking logic is testable without a tokenizer
            pid = list(prompt.encode("utf-8", "ignore"))
            cid = list(completion.encode("utf-8", "ignore")) + [0]
            return {"input_ids": pid + cid, "labels": [-100] * len(pid) + cid,
                    "loss_weight": self.no_tool_weight if is_no_tool_record(rec) else 1.0}
        pid = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        cid = self.tokenizer(completion, add_special_tokens=False)["input_ids"] + \
            [self.tokenizer.eos_token_id]
        return {"input_ids": pid + cid, "labels": [-100] * len(pid) + cid,
                "loss_weight": self.no_tool_weight if is_no_tool_record(rec) else 1.0}


def collate(batch, pad_id):
    """Pad to right; labels padded with -100. Returns tensors + loss_weight."""
    maxlen = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    w = torch.zeros(len(batch), dtype=torch.float)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        lab[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
        attn[i, :n] = 1
        w[i] = b["loss_weight"]
    return {"input_ids": ids, "labels": lab, "attention_mask": attn, "loss_weight": w}


def list_shards(root, split="train"):
    """Shards + record-id keep-set for a dataset root (honors splits/default.json)."""
    split_file = os.path.join(root, "splits", "default.json")
    keep = None
    if os.path.exists(split_file):
        keep = set(json.load(open(split_file)).get(split, []))
    shards = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn.endswith(".jsonl"):
                shards.append(os.path.join(dirpath, fn))
    return shards, keep
