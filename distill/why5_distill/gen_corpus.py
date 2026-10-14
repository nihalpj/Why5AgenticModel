"""Stage 1: corpus generation with the Qwen 3.5 teacher plugged into the
Factory (DISTILLATION_PLAN §Stage 1). Structure (depth, plan, calls,
sandbox, gates) comes from the Factory; the teacher supplies why-content.

Usage:
    python gen_corpus.py --n 200 --out data/distilled [--mock]
"""

import argparse
import collections
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "synth_factory"))

from why5_synth import ToolZoo, SeedGenerator, Validator, DatasetWriter
from why5_synth.generator import TraceGenerator

try:
    from why5_distill.teacher import QwenTeacher
    from why5_distill.logs import log, start_file_log
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from teacher import QwenTeacher
    from logs import log, start_file_log


class DistillTraceGenerator(TraceGenerator):
    """TraceGenerator whose why-steps come from the Qwen teacher.

    Plan FR-B2: sample n candidates, keep the median-length one that
    parses to exactly `depth` steps. Candidates that fail to parse count
    against the teacher's format-acceptance rate, not the record.
    """

    def __init__(self, zoo, teacher, rng_seed=13):
        super().__init__(zoo, rng_seed=rng_seed)
        self.teacher = teacher
        self.teacher_stats = collections.Counter()

    def _generate_whys(self, seed, depth):
        candidates = self.teacher.why_steps(seed, depth, self.zoo, n_candidates=4)
        valid = [c for c in candidates if c is not None]
        self.teacher_stats["sampled"] += len(candidates)
        self.teacher_stats["valid"] += len(valid)
        self.teacher_stats["format_reject"] += len(candidates) - len(valid)
        if not valid:
            raise RuntimeError(
                f"teacher produced no valid candidate for seed {seed['seed_id']} "
                f"at depth {depth} (all {len(candidates)} candidates malformed)")
        valid.sort(key=lambda steps: sum(len(s.split()) for s in steps))
        chosen = valid[len(valid) // 2]  # median length: no verbosity bias
        self.teacher_stats["accepted"] += 1
        return chosen

    def _record_teacher_meta(self, rec):
        rec["teacher"] = {**self.teacher.describe(), "candidates_sampled": 4}
        return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of D1 records")
    ap.add_argument("--n-d5", type=int, default=50, help="counterfactual records")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "distilled"))
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--mock", action="store_true", help="offline teacher, no server")
    ap.add_argument("--dataset-version", default="v0.5.0-distill-pilot")
    ap.add_argument("--log-every", type=int, default=25)
    args = ap.parse_args()

    start_file_log(os.path.join(
        args.out, "logs",
        f"gen_corpus_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    log("corpus", f"teacher={'mock' if args.mock else args.model} "
                  f"@ {args.base_url} | n={args.n} n_d5={args.n_d5} | out={args.out}")

    zoo = ToolZoo(procedural_count=20)
    teacher = QwenTeacher(base_url=args.base_url, model=args.model, mock=args.mock, seed=1)
    gen = DistillTraceGenerator(zoo, teacher, rng_seed=13)
    seeds = SeedGenerator(rng_seed=4242)  # fresh stream, disjoint from SFT-pilot seed space
    val = Validator(zoo)
    log("corpus", f"tool zoo: {len(zoo)} tools ({zoo.version})")

    quota = {3: int(args.n * 0.4), 4: int(args.n * 0.4), 5: args.n - 2 * int(args.n * 0.4)}
    have = collections.Counter()
    accepted = collections.defaultdict(list)
    stats = collections.Counter()

    # ---- D1: gold-depth traces --------------------------------------------
    while len(accepted["D1"]) < args.n:
        seed = seeds.next_seed(zoo)
        if have[seed["gold_depth"]] >= quota[seed["gold_depth"]]:
            continue
        stats["seeds"] += 1
        try:
            rec = gen.generate(seed, dataset_class="D1")
        except RuntimeError as e:
            stats["teacher_exhausted"] += 1
            continue
        if rec is None:
            stats["sandbox_fail"] += 1
            continue
        gen._record_teacher_meta(rec)
        if val.validate(rec):
            accepted["D1"].append(rec)
            have[seed["gold_depth"]] += 1
        else:
            stats["rejected"] += 1
        n_acc = len(accepted["D1"])
        if n_acc % args.log_every == 0 and n_acc:
            fmt = gen.teacher_stats
            tacc = fmt["valid"] / max(1, fmt["sampled"])
            log("corpus", f"D1 {n_acc}/{args.n} accepted | tiers "
                          f"{dict(sorted(have.items()))} | seeds={stats['seeds']} "
                          f"rej={stats['rejected']} sb_fail={stats['sandbox_fail']} "
                          f"teacher_fmt_acc={tacc:.0%}")

    # ---- D5: counterfactual depths (teacher content at wrong depths) -------
    log("corpus", f"D1 complete: {len(accepted['D1'])} records, tiers "
                  f"{dict(sorted(have.items()))}")
    from why5_synth.seeds import counterfactual_depths
    while len(accepted["D5"]) < args.n_d5:
        seed = seeds.next_seed(zoo)
        for forced in counterfactual_depths(seed["gold_depth"]):
            try:
                rec = gen.generate(seed, dataset_class="D5", forced_depth=forced)
            except RuntimeError:
                break
            if rec is None:
                break
            rec["counterfactual_outcome"] = "wasteful" if forced > seed["gold_depth"] else "degraded"
            gen._record_teacher_meta(rec)
            if val.validate(rec):
                accepted["D5"].append(rec)
            if len(accepted["D5"]) >= args.n_d5:
                break

    log("corpus", f"D5 complete: {len(accepted['D5'])} counterfactual records")

    # ---- write JSONL (same §3.7 contract as the Factory) --------------------
    log("corpus", "writing JSONL shards + manifest + splits")
    writer = DatasetWriter(args.out, dataset_version=args.dataset_version)
    all_recs = []
    for cls in ("D1", "D5"):
        writer.write_class(cls, accepted[cls])
        all_recs += accepted[cls]
    root = writer.finalize(all_recs)

    fmt = gen.teacher_stats
    teacher_acc = fmt["valid"] / max(1, fmt["sampled"])
    report = {
        "records": {c: len(v) for c, v in accepted.items()},
        "d1_depth_distribution": dict(sorted(have.items())),
        "teacher": {"format_acceptance": teacher_acc, **dict(fmt)},
        "stats": dict(stats),
    }
    with open(os.path.join(root, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log("corpus", f"DONE root={root} | D1={len(accepted['D1'])} D5={len(accepted['D5'])} "
                  f"| teacher_fmt_acc={teacher_acc:.0%}")
    print(json.dumps(report, indent=2))
    print(f"dataset root: {root}")


if __name__ == "__main__":
    main()
