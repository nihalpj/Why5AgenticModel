"""Trace generation: why-traces in the token protocol (D1/D3/D4/D5).

The production factory samples a 70B-class teacher (SYNTHETIC_DATA_PLAN.md
§3.2); this pilot implements the same output contract with template
families — phrasing is drawn from 6 families to avoid template collapse,
and the why-step content is derived from seed structure so gates G2/G4/G5
hold exactly. The TraceGenerator API is teacher-pluggable: swap
`_generate_whys` with a vLLM call and everything downstream is unchanged.
"""

import json
import random
import uuid

from .sandbox import Sandbox

PHRASING_FAMILIES = [
    ("question",  "{n}. {q}"),
    ("because",   "{n}. {q} Because {reason}, {answer_lower}."),
    ("stepping",  "{n}. Stepping back: {q_lower} {answer}."),
    ("formal",    "{n}. Determine: {q_lower} Conclusion: {answer}."),
    ("terse",     "{n}. {q_short} -> {answer_short}"),
    ("checklist", "{n}. Check: {q_lower} Verified: {answer}."),
]

# Tier curriculum (SYNTHETIC_DATA_PLAN.md §3.2): step k asks a fixed class
# of question; tier-3 stops at feasibility, tier-4 adds contingency, tier-5
# adds hypothesis revision.
STEP_QUESTIONS = {
    1: [("What is the user actually asking for?", "the core intent and deliverable"),
        ("What outcome does this request need?", "a completed task with a verifiable result")],
    2: [("What information is missing before I can act?", "the concrete data obtainable from tools"),
        ("Which inputs do I not yet have?", "entity identifiers and state only tools can provide")],
    3: [("Which tool provides it, and is this feasible?", "a direct mapping from gaps to registry tools"),
        ("Can the registry satisfy this in one pass?", "yes/no plus the required call sequence")],
    4: [("What could fail, and in what order must calls run?", "dependencies and contingency handling"),
        ("Is the plan sufficient and correctly ordered?", "call ordering and branch conditions")],
    5: [("What could my earlier reasoning have gotten wrong?", "an explicit revision or confirmation of steps 1-4"),
        ("Which assumption should I re-verify before acting?", "the load-bearing assumption and its check")],
}


def _why_text(family, n, question, answer):
    q, q_lower = question, question[0].lower() + question[1:]
    q_short = question.split(",")[0]
    answer_short = answer.split(",")[0]
    tmpl = family[1]
    return tmpl.format(n=n, q=q, q_lower=q_lower,
                       q_short=q_short, answer=answer, answer_short=answer_short,
                       answer_lower=answer, reason=answer)


class TraceGenerator:
    def __init__(self, zoo, rng_seed=13, teacher=None):
        self.zoo = zoo
        self.rng = random.Random(rng_seed)
        self.teacher = teacher  # future: vLLM teacher callable

    # -- public API ---------------------------------------------------------

    def generate(self, seed, dataset_class="D1", forced_depth=None, fault_tool=None):
        """Build a full record for a seed. Returns dict or None on failure."""
        depth = forced_depth or seed["gold_depth"]
        sb = Sandbox(rng_seed=self.rng.randint(0, 10**9))
        faults = [fault_tool] if fault_tool else []

        # hypothesis revision (tier-5 step 5) must reference prior steps
        whys = self._generate_whys(seed, depth)
        plan = self._plan_text(seed)

        # execute calls (with retry semantics for faults); the assistant's
        # tool calls carry CONCRETE resolved arguments — refs like
        # "<get_order.total>" are a seed-graph notation only
        tool_calls, retries, executed, ok = [], 0, [], True
        for step in seed["call_graph"]:
            attempt, max_retries = 0, 3
            while True:
                try:
                    res = sb.execute(step["tool"], step["arguments"], faults=faults)
                    resolved = sb.executed[-1]["arguments"]
                    tool_calls.append({"tool": step["tool"], "arguments": resolved,
                                       "result": res})
                    executed.append({"tool": step["tool"], "arguments": resolved})
                    break
                except Exception as e:
                    if attempt >= max_retries:
                        ok = False
                        break
                    attempt += 1
                    retries += 1
                    faults = []  # transient fault succeeds on retry
                    tool_calls.append({"tool": step["tool"], "arguments": step["arguments"],
                                       "retry_of": tool_calls[-1]["tool"] if tool_calls else None,
                                       "result": None})
                # unreachable
            if not ok:
                break

        if not ok:
            return None

        record = {
            "record_id": str(uuid.uuid4()),
            "dataset_class": dataset_class,
            "seed_id": seed["seed_id"],
            "recipe": {"id": f"{dataset_class.lower()}_pilot_v1", "version": 1},
            "toolzoo_version": self.zoo.version,
            "teacher": {"model_id": "template-gen@pilot" if self.teacher is None else "teacher-70b",
                        "temperature": 0.7, "top_p": 0.95, "candidates_sampled": 1},
            "gold_depth": seed["gold_depth"],
            "forced_depth": forced_depth,
            "counterfactual_outcome": None,
            "tools_registry": self.zoo.registry(seed["registry"]),
            "messages": self._messages(seed, depth, whys, plan, tool_calls),
            "protocol_turns": ([{"type": "depth", "value": depth}] +
                               [{"type": "why", "n": i + 1, "text": w} for i, w in enumerate(whys)] +
                               [{"type": "plan", "text": plan}] +
                               [{"type": "tool_call", "name": tc["tool"], "arguments": tc["arguments"]}
                                for tc in tool_calls] +
                               [{"type": "final", "text": self._final_text(seed, tool_calls)}]),
            "call_graph_expected": seed["call_graph"],
            "call_graph_executed": executed,
            "validation": {},
            "metrics": {},
            "provenance": {},
            # pilot-internal fields used by the validator, stripped at write time
            "_seed": seed,
            "_retries": retries,
            "_registry_names": seed["registry"],
        }
        return record

    # -- internals ----------------------------------------------------------

    def _generate_whys(self, seed, depth):
        family = self.rng.choice(PHRASING_FAMILIES)
        whys = []
        gaps = self._gap_answers(seed)
        for i in range(depth):
            q_bank = STEP_QUESTIONS[i + 1]
            question = self.rng.choice(q_bank)[0]
            answer = gaps[min(i, len(gaps) - 1)]
            whys.append(_why_text(family, i + 1, question, answer))
        return whys

    def _gap_answers(self, seed):
        tools = [s["tool"] for s in seed["call_graph"]]
        answers = [
            f"the user needs: {seed['intent_phrasing'].strip().rstrip('.')}",
            f"missing inputs are {', '.join(t for t in tools)} lookups",
            f"{tools[0]} provides the entry point",
        ]
        if len(tools) > 1:
            answers.append(f"calls must run in order {', '.join(tools)} because later calls consume earlier outputs")
        answers.append("the dependency assumption from earlier steps holds after checking argument bindings")
        return answers

    def _plan_text(self, seed):
        return " -> ".join(s["tool"] for s in seed["call_graph"])

    def _messages(self, seed, depth, whys, plan, tool_calls):
        assistant = [f"<|depth_{depth}|>"]
        for i, w in enumerate(whys):
            assistant.append(f"<|why|>{w}<|end_why|>")
        assistant.append(f"<|plan|>{plan}<|end_plan|>")
        for tc in tool_calls:
            if tc.get("result") is None:
                continue
            assistant.append(
                f'<|tool_call|>{{"name":"{tc["tool"]}","arguments":{json.dumps(tc["arguments"])}}}<|end_tool_call|>')
        assistant.append(f"<|final|>{self._final_text(seed, tool_calls)}<|end_final|>")
        msgs = [
            {"role": "system",
             "content": "You are an agent. Decide depth (3/4/5), reason exactly that many why-steps, "
                        "plan, call tools, then answer."},
            {"role": "user", "content": seed["intent_phrasing"]},
            {"role": "assistant", "content": "\n".join(assistant)},
        ]
        # surface tool results as tool-role turns (eval-harness replay view)
        for tc in tool_calls:
            if tc.get("result") is not None:
                msgs.append({"role": "tool", "tool_name": tc["tool"],
                             "content": json.dumps(tc["result"])})
        return msgs

    def _final_text(self, seed, tool_calls):
        if seed["final_kind"] == "refund_report":
            refund = next((tc for tc in tool_calls if tc["tool"] == "create_refund" and tc.get("result")), None)
            root = ("the card had insufficient funds on the first two attempts "
                    "before the third succeeded")
            if refund:
                return (f"Root cause: {root}. Refund {refund['result']['refund_id']} "
                        f"for {refund['result']['amount']} is {refund['result']['status']}.")
            return f"Root cause: {root}. Refund was not eligible under policy."
        if seed["final_kind"] == "ticket":
            t = next(tc for tc in tool_calls if tc["tool"] == "create_ticket")
            return f"Ticket {t['result']['ticket_id']} opened with priority {t['result']['priority']}."
        last = tool_calls[-1]["result"] if tool_calls else {}
        if "content" in last:
            return f"Summary: {last['content'][:200]}"
        return f"Done. Result: {str(last)[:200]}"
