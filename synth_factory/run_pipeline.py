"""Pilot batch driver: generate validated synthetic data per the plan.

Pilot targets (scaled down from production 2B tokens):
    D1  why-traces            ~400 records (3/4/5 depth mix 40/40/20)
    D3  recovery traces       ~80  records (fault injection)
    D5  counterfactual depth  ~200 records (forced wrong depths)

Usage: python -m run_pipeline [--n-d1 400 --n-d3 80 --n-d5 200]
"""

import argparse
import collections
import json
import sys

from why5_synth import ToolZoo, SeedGenerator, TraceGenerator, Validator, DatasetWriter
from why5_synth.seeds import counterfactual_depths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-d1", type=int, default=400)
    ap.add_argument("--n-d3", type=int, default=80)
    ap.add_argument("--n-d5", type=int, default=200)
    ap.add_argument("--out", default="output")
    args = ap.parse_args()

    zoo = ToolZoo(procedural_count=20)
    seeds = SeedGenerator(rng_seed=42)
    gen = TraceGenerator(zoo, rng_seed=13)
    val = Validator(zoo)

    accepted = collections.defaultdict(list)
    stats = collections.Counter()

    # ---- D1: why-traces at gold depth (40/40/20 tier quota per plan §4.2) --
    print(f"tool zoo: {len(zoo)} tools ({zoo.version})")
    quota = {3: int(args.n_d1 * 0.4), 4: int(args.n_d1 * 0.4), 5: args.n_d1 - 2 * int(args.n_d1 * 0.4)}
    have = collections.Counter()
    while len(accepted["D1"]) < args.n_d1:
        seed = seeds.next_seed(zoo)
        if have[seed["gold_depth"]] >= quota[seed["gold_depth"]]:
            stats["quota_skip"] += 1
            continue
        rec = gen.generate(seed, dataset_class="D1")
        stats["generated"] += 1
        if rec is None:
            stats["sandbox_fail"] += 1
            continue
        if val.validate(rec):
            accepted["D1"].append(rec)
            have[seed["gold_depth"]] += 1
            stats["accepted_D1"] += 1
        else:
            stats["rejected_D1"] += 1

    # ---- D3: recovery traces (fault injection on multi-call seeds) --------
    while len(accepted["D3"]) < args.n_d3:
        seed = seeds.next_seed(zoo)
        if len(seed["call_graph"]) < 2:
            continue
        fault_tool = seed["call_graph"][0]["tool"]
        rec = gen.generate(seed, dataset_class="D3", fault_tool=fault_tool)
        stats["generated"] += 1
        if rec is None:
            stats["sandbox_fail"] += 1
            continue
        if rec["_retries"] < 1:      # recovery traces must contain a retry
            continue
        if val.validate(rec):
            accepted["D3"].append(rec)
            stats["accepted_D3"] += 1
        else:
            stats["rejected_D3"] += 1

    # ---- D5: counterfactual depths ----------------------------------------
    while len(accepted["D5"]) < args.n_d5:
        seed = seeds.next_seed(zoo)
        for forced in counterfactual_depths(seed["gold_depth"]):
            rec = gen.generate(seed, dataset_class="D5", forced_depth=forced)
            stats["generated"] += 1
            if rec is None:
                stats["sandbox_fail"] += 1
                continue
            # counterfactual labeling: task still succeeds here (template gen
            # always executes the full graph) -> 'survived' or 'wasteful'
            if forced > seed["gold_depth"]:
                rec["counterfactual_outcome"] = "wasteful"
            else:
                rec["counterfactual_outcome"] = "degraded"
            if val.validate(rec):
                accepted["D5"].append(rec)
                stats["accepted_D5"] += 1
            else:
                stats["rejected_D5"] += 1
            if len(accepted["D5"]) >= args.n_d5:
                break

    # ---- write JSONL -------------------------------------------------------
    writer = DatasetWriter(args.out)
    all_recs = []
    for cls in ("D1", "D3", "D5"):
        shards = writer.write_class(cls, accepted[cls])
        all_recs += accepted[cls]
        print(f"{cls}: {len(accepted[cls])} records in {len(shards)} shards")
    root = writer.finalize(all_recs)
    print(f"dataset root: {root}")

    # ---- report ------------------------------------------------------------
    depth_dist = collections.Counter(r["gold_depth"] for r in accepted["D1"])
    print("\n=== pilot report ===")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  D1 gold-depth distribution: {dict(sorted(depth_dist.items()))}")
    acc_rate = stats["accepted_D1"] / max(1, stats["accepted_D1"] + stats["rejected_D1"] + stats["sandbox_fail"])
    print(f"  overall acceptance rate: {acc_rate:.1%}")
    with open(f"{root}/report.json", "w") as f:
        json.dump({"stats": dict(stats), "depth_distribution": dict(depth_dist),
                   "acceptance_rate": acc_rate}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
