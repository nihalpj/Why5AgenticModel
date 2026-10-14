"""JSONL storage: shard writer, manifest, and reference reader.

Implements SYNTHETIC_DATA_PLAN.md §3.7: one record per line, immutable
shards, SHA-256 manifest, train/val/holdout splits, and a pydantic-free
reference reader with schema checking (production ships pydantic models).
"""

import hashlib
import json
import os
import random


def _strip_internal(rec):
    out = {k: v for k, v in rec.items() if not k.startswith("_")}
    out["metrics"] = {
        "reasoning_tokens": sum(len(t.get("text", "").split()) for t in rec["protocol_turns"]
                                if t["type"] in ("why", "plan")) * 1,  # word-level pilot proxy
        "tool_calls": sum(1 for t in rec["protocol_turns"] if t["type"] == "tool_call"),
        "retries": rec.get("_retries", 0),
        "total_tokens": sum(len(json.dumps(m)) for m in rec["messages"]) // 4,  # chars/4 proxy
    }
    out["provenance"] = {
        "pipeline_version": "why5_synth@0.1.0",
        "timestamps": {"generated": None, "validated": None},
        "shard_id": None,
    }
    return out


class DatasetWriter:
    def __init__(self, out_dir, dataset_version="v0.4.0-pilot", shard_size=1000):
        self.root = os.path.join(out_dir, dataset_version)
        self.shard_size = shard_size
        self.manifest = {}

    def write_class(self, dataset_class, records):
        """Write validated records of one class into shards + manifest entry."""
        from why5_synth.validator import GATES  # local import to avoid cycle

        d = os.path.join(self.root, dataset_class.lower() + "_" +
                         {"D1": "why_traces", "D3": "recovery", "D5": "counterfactual",
                          "D4": "no_tool"}.get(dataset_class, dataset_class.lower()))
        os.makedirs(d, exist_ok=True)
        shards = []
        rng = random.Random(123)
        for i in range(0, len(records), self.shard_size):
            shard_id = f"{dataset_class.lower()}-shard-{i // self.shard_size:04d}"
            path = os.path.join(d, f"{shard_id}.jsonl")
            h = hashlib.sha256()
            with open(path, "w", encoding="utf-8") as f:
                chunk = records[i:i + self.shard_size]
                for rec in chunk:
                    clean = _strip_internal(rec)
                    clean["provenance"]["shard_id"] = shard_id
                    line = json.dumps(clean, ensure_ascii=False)
                    assert "\n" not in line and len(line.encode()) <= 65536
                    f.write(line + "\n")
                    h.update(line.encode() + b"\n")
            shards.append({"shard": shard_id, "file": os.path.relpath(path, self.root),
                           "records": len(chunk), "sha256": h.hexdigest()})
        self.manifest[dataset_class] = shards
        return shards

    def finalize(self, records, val_frac=0.02, holdout_frac=0.01):
        """Write manifest.json and train/val/holdout splits by record_id."""
        rng = random.Random(456)
        ids = [r["record_id"] for r in records]
        rng.shuffle(ids)
        n_val, n_hold = int(len(ids) * val_frac), int(len(ids) * holdout_frac)
        splits = {"holdout": sorted(ids[:n_hold]),
                  "val": sorted(ids[n_hold:n_hold + n_val]),
                  "train": sorted(ids[n_hold + n_val:])}
        with open(os.path.join(self.root, "manifest.json"), "w") as f:
            json.dump({"datasets": self.manifest,
                       "counts": {k: sum(s["records"] for s in v) for k, v in self.manifest.items()},
                       "total_records": len(ids)}, f, indent=2)
        os.makedirs(os.path.join(self.root, "splits"), exist_ok=True)
        with open(os.path.join(self.root, "splits", "default.json"), "w") as f:
            json.dump(splits, f)
        return self.root


def read_dataset(root, require_passed=True):
    """Reference reader: yield records from all shards under a dataset root."""
    manifest = json.load(open(os.path.join(root, "manifest.json")))
    for cls, shards in manifest["datasets"].items():
        for shard in shards:
            with open(os.path.join(root, shard["file"]), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if require_passed and not rec["validation"]["passed"]:
                        raise ValueError(f"record {rec['record_id']} failed validation")
                    yield rec
