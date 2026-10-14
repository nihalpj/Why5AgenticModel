"""Stage 0/1: Qwen 3.5 4B teacher adapter.

Bridge between the Factory seed format and a Qwen 3.5 teacher served on
vLLM (OpenAI-compatible). The teacher NEVER decides depth: the requested
step count comes from the rubric gold label embedded in the prompt, and
outputs are parsed back into exactly N why-steps or rejected (G1/G2).

A --mock mode reimplements the API surface locally so the entire
distillation pipeline can be exercised without GPUs (CI / laptop runs);
mock responses use the pilot's template families.
"""

import json
import random
import re

SYSTEM_CONTRACT = (
    "You are a careful reasoning assistant that powers an agent. "
    "You MUST respond with exactly {n} why-steps inside <why>...</why> tags "
    "(numbered 1..{n}), each one level deeper than the previous. "
    "Do not add extra why-steps. Do not solve the task in the why-steps; "
    "only decompose it."
)

USER_TEMPLATE = """Task: {task}

Available tools:
{tools}

Reason about this task in exactly {n} why-steps (root-cause style: each
step asks a deeper question than the last). Step focus:
{focus}
"""


def build_teacher_messages(seed, depth, zoo):
    """Factory seed -> Hermes-style chat for the teacher."""
    focus = [
        "1: what the user actually needs",
        "2: what information is missing",
        "3: which tools provide it and feasibility",
    ]
    if depth >= 4:
        focus.append("4: ordering, dependencies, what could fail")
    if depth >= 5:
        focus.append("5: which earlier assumption to revise or re-verify")
    registry = zoo.registry(seed["registry"])
    return [
        {"role": "system", "content": SYSTEM_CONTRACT.format(n=depth)},
        {"role": "user", "content": USER_TEMPLATE.format(
            task=seed["intent_phrasing"],
            tools=json.dumps(
                [{k: t[k] for k in ("name", "description", "parameters")} for t in registry],
                indent=1),
            n=depth,
            focus="\n".join(focus),
        )},
    ]


_WHY_RE = re.compile(r"<why>\s*(.*?)\s*</why>", re.DOTALL)


def parse_teacher_whys(text, depth):
    """Parse teacher output; returns list of exactly `depth` step strings or None."""
    steps = [s.strip() for s in _WHY_RE.findall(text) if s.strip()]
    steps = [re.sub(r"^\d+[.:)\-]\s*", "", s) for s in steps]  # strip leading numbers
    if len(steps) != depth:
        return None
    return steps


# ---------------------------------------------------------------------------
# Teacher client: OpenAI-compatible (vLLM serve) with offline mock fallback
# ---------------------------------------------------------------------------

class TeacherError(Exception):
    pass


class QwenTeacher:
    """Qwen 3.5 4B teacher behind an OpenAI-compatible endpoint.

    Set `mock=True` (or pass --mock in CLIs) to run without a server.
    """

    def __init__(self, base_url="http://localhost:8000/v1", model="Qwen/Qwen3.5-4B",
                 temperature=0.7, top_p=0.95, max_tokens=512, mock=False,
                 api_key="EMPTY", seed=0):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.mock = mock
        self.rng = random.Random(seed)
        if not mock:
            from openai import OpenAI  # deferred: not needed in mock mode
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    # -- public API ----------------------------------------------------------

    def why_steps(self, seed, depth, zoo, n_candidates=4):
        """Return `n_candidates` lists of why-step strings (plan FR-B2).

        Each candidate is either a parsed list of length `depth`, or None
        (format rejection -> counts against teacher acceptance).
        """
        messages = build_teacher_messages(seed, depth, zoo)
        return [self._one_candidate(messages, depth, seed)
                for _ in range(n_candidates)]

    def describe(self):
        return {"model_id": "mock-template-gen" if self.mock else self.model,
                "temperature": self.temperature, "top_p": self.top_p}

    # -- internals ------------------------------------------------------------

    def _one_candidate(self, messages, depth, seed):
        if self.mock:
            return self._mock_completion(messages, depth, seed)
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages,
            temperature=self.temperature, top_p=self.top_p,
            max_tokens=self.max_tokens)
        return parse_teacher_whys(resp.choices[0].message.content or "", depth)

    def _mock_completion(self, messages, depth, seed):
        """Deterministic-ish offline stand-in: template families from the pilot."""
        families = [
            "{q}",
            "{q} Because {a}.",
            "Stepping back: {q_lower} {a}",
            "Check: {q_lower} Verified: {a}.",
        ]
        tools = [s["tool"] for s in seed["call_graph"]] or ["(no tools)"]
        banks = [
            ("What is the user actually asking for?",
             f"the user needs: {seed['intent_phrasing'].rstrip('.')}"),
            ("What information is missing?",
             f"missing inputs are obtainable via {', '.join(tools)}"),
            ("Which tool provides it, and is this feasible?",
             f"{tools[0]} provides the entry point"),
            ("What could fail, and in what order must calls run?",
             f"calls run in order {', '.join(tools)}"),
            ("Which assumption should I re-verify before acting?",
             "the dependency assumption holds after checking argument bindings"),
        ]
        fam = self.rng.choice(families)
        steps = []
        for i in range(depth):
            q, a = banks[min(i, len(banks) - 1)]
            q_lower = q[0].lower() + q[1:]
            steps.append(fam.format(q=q, q_lower=q_lower, a=a))
        return steps
