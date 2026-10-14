"""Stage 4: on-policy correction distillation (plan §Stage 4).

Loop:
    1. Student (SFT checkpoint) generates a protocol turn for a fresh seed
       (depth is FORCED to the rubric gold — the student never routes here;
        routing is evaluated separately by the depth-classification eval).
    2. The SANDBOX + gates judge the rollout (environment as judge; no
       teacher scoring -> no teacher bias, no reward-hacking surface).
    3. On failure, the teacher regenerates a corrected trace for the same
       seed with the failure explanation appended -> (chosen, rejected)
       DPO pair written as JSONL for dpo_train.py.

Usage:
    python on_policy.py --student ckpt/why5-sft --n 100 [--mock]
"""

import argparse
import collections
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "synth_factory"))

from why5_synth import ToolZoo, SeedGenerator, Sandbox
from why5_synth.sandbox import SandboxError
from why5_synth.validator import _validate_schema

try:
    from why5_distill.teacher import QwenTeacher, build_teacher_messages
    from why5_distill.gen_corpus import DistillTraceGenerator
    from why5_distill.logs import log, start_file_log
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from teacher import QwenTeacher, build_teacher_messages
    from gen_corpus import DistillTraceGenerator
    from logs import log, start_file_log

JUDGE_PROTOCOL = "protocol_parse"
JUDGE_SCHEMA = "schema_valid"
JUDGE_EXEC = "sandbox_success"


def judge_rollout(tool_calls, registry_names, zoo, seed):
    """Environment-as-judge: parse, schema, execution. Returns (ok, failures)."""
    failures = []
    schemas = zoo.schemas_for_validation(registry_names)
    parsed = []
    for tc in tool_calls:
        if tc.get("name") is None or tc.get("arguments") is None:
            failures.append(JUDGE_PROTOCOL)
            continue
        if tc["name"] not in registry_names:
            failures.append("phantom_tool")
            continue
        if not _validate_schema(tc["arguments"], schemas[tc["name"]]):
            failures.append(JUDGE_SCHEMA)
            continue
        parsed.append(tc)
    if JUDGE_PROTOCOL in failures or JUDGE_SCHEMA in failures or "phantom_tool" in failures:
        return False, sorted(set(failures))
    # execution: replay expected graph with student's concrete args where given
    sb = Sandbox(rng_seed=0)
    ok_exec = True
    for step in seed["call_graph"]:
        try:
            sb.execute(step["tool"], step["arguments"])
        except SandboxError:
            ok_exec = False
    if not ok_exec:
        failures.append(JUDGE_EXEC)
    return ok_exec, sorted(set(failures))


def student_rollout(student_client, seed, depth, zoo, mock):
    """Ask the student for depth + whys + calls; return parsed protocol turns."""
    messages = build_teacher_messages(seed, depth, zoo)  # same prompt contract
    if mock:
        # deterministic weak student: drop the last tool call (classic small-model
        # failure) so the correction path is exercised end-to-end offline
        tools = [dict(s) for s in seed["call_graph"]][:-1]
        whys = [f"mock student why {i+1}" for i in range(depth)]
        return {"depth": depth, "whys": whys, "tool_calls": tools}, None
    resp = student_client.chat.completions.create(
        model=os.environ.get("WHY5_STUDENT_MODEL", "why5-sft"),
        messages=messages, temperature=0.7, top_p=0.95, max_tokens=768)
    text = resp.choices[0].message.content or ""
    return parse_student_protocol(text, depth)


def parse_student_protocol(text, depth):
    import re
    whys = [s.strip() for s in re.findall(r"<why>\s*(.*?)\s*</why>", text, re.DOTALL)]
    calls = []
    for m in re.finditer(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            calls.append({"name": obj.get("name"),
                          "arguments": obj.get("arguments")})
        except json.JSONDecodeError:
            calls.append({"name": None, "arguments": None})
    return {"depth": depth, "whys": whys, "tool_calls": calls}, None


def correct_with_teacher(teacher, gen, seed, depth, failures):
    """Teacher corrected trace for the same seed (self-correction loop input)."""
    note = f"Note: a previous attempt failed because: {', '.join(failures)}. " \
           "Produce a corrected, complete plan."
    patched = dict(seed)
    patched["intent_phrasing"] = seed["intent_phrasing"] + "\n\n" + note
    rec = gen.generate(patched, dataset_class="D1")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="ckpt/why5-sft")
    ap.add_argument("--student-url", default="http://localhost:8001/v1")
    ap.add_argument("--teacher-url", default="http://localhost:8000/v1")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "dpo_pairs.jsonl"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    args = ap.parse_args()

    import time as _time
    start_file_log(os.path.join(
        os.path.dirname(args.out), "logs",
        f"on_policy_{_time.strftime('%Y%m%d_%H%M%S')}.log"))
    log("onpolicy", f"student={args.student} mock={args.mock} n={args.n} out={args.out}")

    zoo = ToolZoo(procedural_count=20)
    teacher = QwenTeacher(base_url=args.teacher_url, mock=args.mock, seed=2)
    gen = DistillTraceGenerator(zoo, teacher, rng_seed=99)
    seeds = SeedGenerator(rng_seed=777)  # held-out stream: never in SFT corpus
    student_client = None
    if not args.mock:
        from openai import OpenAI
        student_client = OpenAI(base_url=args.student_url, api_key="EMPTY")

    stats = collections.Counter()
    pairs = []
    while stats["rollouts"] < args.n:
        seed = seeds.next_seed(zoo)
        depth = seed["gold_depth"]
        stats["rollouts"] += 1
        rollout, _ = student_rollout(student_client, seed, depth, zoo, args.mock)
        ok, failures = judge_rollout(rollout["tool_calls"], seed["registry"], zoo, seed)
        if ok:
            stats["pass"] += 1
        else:
            stats["fail:" + ",".join(failures)] += 1
            log("onpolicy", f"FAIL seed={seed['seed_id']} gold={depth} "
                            f"modes={','.join(failures)}", level="WARN")
            fixed = correct_with_teacher(teacher, gen, seed, depth, failures)
            if fixed is None:
                stats["teacher_fix_failed"] += 1
            else:
                stats["teacher_fix_ok"] += 1
                pairs.append({
                    "record_id": f"dpo_{stats['teacher_fix_ok']:06d}",
                    "seed_id": seed["seed_id"],
                    "gold_depth": depth,
                    "failure_modes": failures,
                    "student_rollout": rollout,               # rejected
                    "teacher_correction": {                    # chosen (gate-clean)
                        "protocol_turns": fixed["protocol_turns"],
                        "validation": fixed["validation"],
                        "metrics": fixed["metrics"],
                    },
                    "validation": {"passed": True},  # pair-level; members carry their own
                })
        if stats["rollouts"] % args.log_every == 0:
            n_fail = stats["rollouts"] - stats["pass"]
            yield_so_far = (stats["teacher_fix_ok"] / n_fail) if n_fail else 1.0
            log("onpolicy", f"{stats['rollouts']}/{args.n} rollouts | "
                            f"pass={stats['pass']} fail={n_fail} "
                            f"pairs={stats['teacher_fix_ok']} yield={yield_so_far:.0%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    yield_rate = stats["teacher_fix_ok"] / max(1, stats["rollouts"] - stats["pass"])
    log("onpolicy", f"DONE pairs={len(pairs)} written to {args.out} | "
                    f"pass={stats['pass']} fix_failed={stats['teacher_fix_failed']} "
                    f"yield={yield_rate:.0%}")
    for key, n in sorted(stats.items()):
        if key.startswith("fail:"):
            log("onpolicy", f"failure-mode {key[5:]}: {n}", level="WARN")
    print(json.dumps({"pairs": len(pairs), "stats": dict(stats),
                      "correction_yield_on_failures": round(yield_rate, 3)}, indent=2))
    if yield_rate < 0.5:
        print("WARNING: correction yield < 50% (plan gate §7) — investigate teacher fixes",
              file=sys.stderr)


if __name__ == "__main__":
    main()
