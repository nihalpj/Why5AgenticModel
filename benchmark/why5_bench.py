#!/usr/bin/env python3
"""Why5 benchmark harness — 5-why protocol + agentic tool-calling evaluation.

Implements the Why-specific metrics from PRD §7.2 / Main Plan §6.2 against any
OpenAI-compatible endpoint serving a Why5-protocol student:

    depth_accuracy        chosen tier == gold tier              (bar >= 0.80)
    calibration_within_1  |chosen - gold| <= 1                  (bar >= 0.95)
    step_count_exactness  why-count == declared depth           (bar >= 0.98)
    tool_order            calls match seed graph, in order      (bar >= 0.90)
    schema_valid          every call validates vs JSON Schema   (bar >= 0.90)
    exec_success          sandbox replays the student's calls   (bar >= 0.85)
    task_success          all of the above                      (bar >= 0.80)
    latency_p50 / p95     per-tier wall latency (informational)
    depth_response_curve  success at forced 3/4/5 per gold tier (R1 evidence)

The judge is the FACTORY (gates + sandbox), never a teacher model.

Mock mode (--mock) uses the factory's own generator as a perfect student and
--inject-flaws simulates a weak one, so the harness itself is testable offline:
a perfect model must score 100%, a flawed model must be caught.
"""

import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "synth_factory"))
sys.path.insert(0, os.path.join(_REPO, "distill"))

from why5_synth import ToolZoo, SeedGenerator, Sandbox                  # noqa: E402
from why5_synth.sandbox import SandboxError                             # noqa: E402
from why5_synth.validator import _validate_schema                       # noqa: E402
from why5_distill.teacher import QwenTeacher                            # noqa: E402
from why5_distill.gen_corpus import DistillTraceGenerator               # noqa: E402
from why5_distill.sft_data import assistant_text                        # noqa: E402
from why5_distill.logs import log, start_file_log                       # noqa: E402

BARS = {  # reference gates (PRD §7.2 / plan §6.2 release bar)
    "depth_accuracy": 0.80, "calibration_within_1": 0.95,
    "step_count_exactness": 0.98, "tool_order": 0.90,
    "schema_valid": 0.90, "exec_success": 0.85, "task_success": 0.80,
}

# ----------------------------------------------------------------- parsing --

_DEPTH_RE = re.compile(r"<\|depth_([345])\|>")
_WHY_RE = re.compile(r"<\|why\|>(.*?)<\|end_why\|>", re.DOTALL)
_CALL_RE = re.compile(r"<\|tool_call\|>(.*?)<\|end_tool_call\|>", re.DOTALL)


def parse_protocol(text):
    """Parse a student response into protocol fields. Never raises."""
    out = {"parse_ok": False, "depth": None, "whys": [], "calls": [],
           "malformed_calls": 0}
    m = _DEPTH_RE.search(text)
    if not m:
        return out
    out["depth"] = int(m.group(1))
    out["whys"] = [w.strip() for w in _WHY_RE.findall(text) if w.strip()]
    for raw in _CALL_RE.findall(text):
        try:
            obj = json.loads(raw)
            out["calls"].append({"name": obj.get("name"),
                                 "arguments": obj.get("arguments") or {}})
        except json.JSONDecodeError:
            out["malformed_calls"] += 1
            out["calls"].append({"name": None, "arguments": None})
    out["parse_ok"] = True
    return out


# ------------------------------------------------------------------ judge --

def judge(parsed, seed, zoo):
    """Score one rollout against the seed's ground truth (factory as judge)."""
    expected = [s["tool"] for s in seed["call_graph"]]
    got = [c["name"] for c in parsed["calls"]]
    checks = {
        "parse_ok": parsed["parse_ok"] and parsed["malformed_calls"] == 0,
        "step_exact": parsed["depth"] is not None and len(parsed["whys"]) == parsed["depth"],
        "tool_order": got == expected,
    }
    # schema validity of each emitted call
    schemas = zoo.schemas_for_validation(seed["registry"])
    checks["schema_valid"] = all(
        c["name"] in schemas and isinstance(c["arguments"], dict)
        and _validate_schema(c["arguments"], schemas[c["name"]])
        for c in parsed["calls"]) and len(parsed["calls"]) > 0
    # sandbox replay with the STUDENT's concrete args (refs must be resolved)
    sb = Sandbox(rng_seed=0)
    exec_ok = checks["schema_valid"]
    for c in parsed["calls"]:
        try:
            sb.execute(c["name"], c["arguments"])
        except (SandboxError, Exception):
            exec_ok = False
            break
    checks["exec_success"] = exec_ok
    checks["task_success"] = all(checks.values())
    checks["depth_correct"] = parsed["depth"] == seed["gold_depth"]
    return checks


# ------------------------------------------------------------------ clients

SYSTEM = ("You are an agent. First emit <|depth_3|>, <|depth_4|> or <|depth_5|> "
          "by task complexity, then EXACTLY that many <why>...</why> steps, a "
          "<|plan|>, your <|tool_call|> JSON calls, then <|final|>.")


class HTTPStudent:
    """Queries an OpenAI-compatible endpoint (vLLM serving the student)."""

    def __init__(self, url, model, force_param="force_depth"):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.force_param = force_param

    def generate(self, seed, zoo, forced=None):
        messages = [
            {"role": "system", "content": SYSTEM +
             (f" You MUST use depth {forced}." if forced else "")},
            {"role": "user", "content":
             f"Task: {seed['intent_phrasing']}\n\nAvailable tools:\n"
             + json.dumps([{k: t[k] for k in ("name", "description", "parameters")}
                           for t in zoo.registry(seed["registry"])])},
        ]
        payload = {"model": self.model, "messages": messages,
                   "temperature": 0.0, "max_tokens": 1024}
        if forced:
            payload[self.force_param] = forced  # vLLM extra param (graceful below)
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if forced and e.code == 400:  # server without force_depth support
                payload.pop(self.force_param, None)
                req = urllib.request.Request(
                    self.url, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = json.loads(r.read())
            else:
                raise
        latency = time.perf_counter() - t0
        text = body["choices"][0]["message"]["content"] or ""
        return text, latency, body.get("usage", {})


class MockStudent:
    """Perfect student = factory generator; --inject-flaws weakens it."""

    def __init__(self, zoo, rng_seed=5, inject_flaws=False):
        self.zoo = zoo
        self.rng = __import__("random").Random(rng_seed)
        self.inject = inject_flaws
        self.gen = DistillTraceGenerator(
            zoo, QwenTeacher(mock=True, seed=6), rng_seed=61)

    def generate(self, seed, zoo, forced=None):
        t0 = time.perf_counter()
        rec = self.gen.generate(seed, dataset_class="D1", forced_depth=forced)
        if rec is None:  # sandbox-failed generation -> empty response
            return "", time.perf_counter() - t0, {}
        turns = rec["protocol_turns"]
        if self.inject:
            r = self.rng.random()
            if r < 0.30:                                # drop last tool call
                calls = [i for i, t in enumerate(turns) if t["type"] == "tool_call"]
                if calls:
                    i = calls[-1]
                    turns = turns[:i] + turns[i + 1:]
            elif r < 0.50:                              # wrong depth declaration
                turns = [dict(t) for t in turns]
                wrong = {3: 5, 4: 3, 5: 4}[turns[0]["value"]]
                turns[0] = {"type": "depth", "value": wrong}
            elif r < 0.60:                              # malformed call JSON
                turns = [dict(t) for t in turns]
                idx = [i for i, t in enumerate(turns) if t["type"] == "tool_call"]
                if idx:
                    turns[idx[0]] = {"type": "tool_call", "name": "not_a_tool",
                                     "arguments": {"bogus": True}}
        text = assistant_text({"protocol_turns": turns})
        return text, time.perf_counter() - t0, {}


# ------------------------------------------------------------------- runs --

def run_suite(client, zoo, n, rng_seed=9999, forced=None, quota=None,
              log_every=10, tag="suite"):
    """One pass over a held-out seed stream (rng 9999 — disjoint from SFT/RL)."""
    seeds = SeedGenerator(rng_seed=rng_seed)
    rows = []
    quota = quota or {}
    have = collections.Counter()
    fail_examples = 0
    while len(rows) < n:
        seed = seeds.next_seed(zoo)
        if quota and have[seed["gold_depth"]] >= quota.get(seed["gold_depth"], 0):
            continue
        text, latency, usage = client.generate(seed, zoo, forced=forced)
        parsed = parse_protocol(text)
        checks = judge(parsed, seed, zoo)
        rows.append({
            "seed_id": seed["seed_id"], "gold_depth": seed["gold_depth"],
            "forced_depth": forced, "chosen_depth": parsed["depth"],
            "why_count": len(parsed["whys"]), "latency_s": round(latency, 3),
            "reasoning_tokens": usage.get("completion_tokens")
            or sum(len(w.split()) for w in parsed["whys"]),
            **checks,
        })
        have[seed["gold_depth"]] += 1
        if len(rows) % log_every == 0 or len(rows) == n:
            log(tag, f"{len(rows)}/{n} | ok={sum(r['task_success'] for r in rows)} "
                     f"| forced={forced or 'auto'} | tiers={dict(sorted(have.items()))}")
        if not checks["task_success"] and fail_examples < 3:
            fail_examples += 1
            failed = [k for k, v in checks.items() if not v]
            log(tag, f"example failure #{fail_examples} seed={seed['seed_id']} "
                     f"gold={seed['gold_depth']} chosen={parsed['depth']} "
                     f"why_n={len(parsed['whys'])} failed_checks={failed}",
                level="WARN")
    return rows


def aggregate(rows):
    n = len(rows)
    agg = {
        "n": n,
        "step_count_exactness": sum(r["step_exact"] for r in rows) / n,
        "tool_order": sum(r["tool_order"] for r in rows) / n,
        "schema_valid": sum(r["schema_valid"] for r in rows) / n,
        "exec_success": sum(r["exec_success"] for r in rows) / n,
        "task_success": sum(r["task_success"] for r in rows) / n,
    }
    auto = [r for r in rows if r["forced_depth"] is None]
    if auto:
        agg["depth_accuracy"] = sum(r["depth_correct"] for r in auto) / len(auto)
        agg["calibration_within_1"] = sum(
            r["chosen_depth"] is not None
            and abs(r["chosen_depth"] - r["gold_depth"]) <= 1
            for r in auto) / len(auto)
    for tier in (3, 4, 5):
        lat = sorted(r["latency_s"] for r in rows if r["chosen_depth"] == tier)
        if lat:
            agg[f"latency_p50_depth{tier}"] = round(lat[len(lat) // 2], 3)
    succ_tok = [r["task_success"] / max(1, r["reasoning_tokens"] / 1000)
                for r in rows]
    agg["success_per_1k_reasoning_tokens"] = round(
        sum(succ_tok) / len(succ_tok), 4)
    agg["gates"] = {k: {"value": round(v, 4) if isinstance(v, float) else v,
                        "bar": BARS.get(k), "pass": BARS.get(k) is None or v >= BARS[k]}
                    for k, v in agg.items() if k in BARS}
    agg["all_gates_pass"] = all(g["pass"] for g in agg["gates"].values())
    return agg


def depth_response_curve(client, zoo, per_cell):
    """Counterfactual sweep: success at forced 3/4/5 per gold tier (R1)."""
    curve = {}
    for gold in (3, 4, 5):
        quota = {gold: per_cell}
        for forced in (3, 4, 5):
            rows = run_suite(client, zoo, per_cell, rng_seed=4200 + gold,
                             forced=forced, quota=quota, log_every=per_cell,
                             tag="curve")
            rate = round(sum(r["task_success"] for r in rows) / len(rows), 3)
            curve[f"gold{gold}_forced{forced}"] = rate
            log("curve", f"gold={gold} forced={forced} success={rate:.0%}")
    gaps = {g: round(curve[f"gold{g}_forced{g}"] - curve[f"gold{g}_forced3"], 3)
            for g in (3, 4, 5)}
    return {"cells": curve, "success_at_depth_gap": gaps}


# ------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="why5-sft")
    ap.add_argument("--n", type=int, default=60, help="auto-depth suite size")
    ap.add_argument("--curve-n", type=int, default=10,
                    help="samples per (gold,forced) cell; 0 disables curve")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--inject-flaws", action="store_true",
                    help="mock only: simulate a weak student")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "benchmark_report.json"))
    args = ap.parse_args()

    start_file_log(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs",
        f"bench_{time.strftime('%Y%m%d_%H%M%S')}.log"))
    log("bench", f"mode={'mock' if args.mock else 'live'} "
                 f"endpoint={None if args.mock else args.url} "
                 f"model={None if args.mock else args.model} n={args.n} "
                 f"curve_n={args.curve_n}"
                 + (" inject_flaws=1" if args.inject_flaws else ""))

    zoo = ToolZoo(procedural_count=20)
    client = (MockStudent(zoo, inject_flaws=args.inject_flaws)
              if args.mock else HTTPStudent(args.url, args.model))

    quota = {3: int(args.n * .4), 4: int(args.n * .4), 5: args.n - 2 * int(args.n * .4)}
    rows = run_suite(client, zoo, args.n, quota=quota, tag="suite")
    report = {"mode": "mock-flawed" if (args.mock and args.inject_flaws)
              else "mock" if args.mock else "live",
              "endpoint": None if args.mock else args.url,
              "model": None if args.mock else args.model,
              "suite": aggregate(rows), "rows": rows}
    if args.curve_n > 0:
        report["depth_response_curve"] = depth_response_curve(client, zoo,
                                                              args.curve_n)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)

    s = report["suite"]
    print(f"\n=== why5-bench [{report['mode']}] n={s['n']} ===")
    for k in ("step_count_exactness", "depth_accuracy", "calibration_within_1",
              "tool_order", "schema_valid", "exec_success", "task_success"):
        if k in s:
            bar = BARS.get(k)
            mark = "" if bar is None else ("  PASS" if s[k] >= bar else "  FAIL")
            print(f"  {k:26} {s[k]:7.1%}{mark}")
    for k in sorted(k for k in s if k.startswith("latency_")):
        print(f"  {k:26} {s[k]}s")
    if "depth_response_curve" in report:
        print("  depth-response curve (success @ forced depth):")
        for cell, v in report["depth_response_curve"]["cells"].items():
            print(f"    {cell:18} {v:.0%}")
        print(f"    success@depth gaps: {report['depth_response_curve']['success_at_depth_gap']}")
    print(f"  all gates pass: {s['all_gates_pass']}")
    print(f"  report: {args.out}")
    log("bench", f"DONE all_gates_pass={s['all_gates_pass']} "
                 f"task_success={s['task_success']:.1%} report={args.out}",
        level="INFO" if s["all_gates_pass"] else "ERROR")
    return 0 if s["all_gates_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
