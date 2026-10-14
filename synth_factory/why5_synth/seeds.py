"""Seed generation: structured task specs with exact gold-depth labels.

Seeds are pure structured data (SYNTHETIC_DATA_PLAN.md §3.1): the gold tier is
a deterministic function of seed features via GoldDepthRubric, never
model-labeled. Ambiguous seeds (overlapping tier features) are discarded
instead of soft-labeled.
"""

import hashlib
import random
from dataclasses import dataclass, field


class GoldDepthRubric:
    """Deterministic tier assignment (SYNTHETIC_DATA_PLAN.md §3.1).

    LOW  (3): 1 tool call, no dependencies, no branch, clear intent
    HIGH (4): 2-5 sequential dependent calls, OR one conditional branch,
              OR mild ambiguity
    MAX  (5): >=5 calls or mixed parallel/sequential, hypothesis revision
              required, or retry-reasoning required
    """

    @staticmethod
    def score(seed):
        n_calls = len(seed["call_graph"])
        has_deps = seed["complexity_features"].get("cross_tool_state", False)
        has_branch = seed["complexity_features"].get("conditional_branch", False)
        has_parallel = seed["complexity_features"].get("parallel_calls", False)
        has_revision = seed["complexity_features"].get("hypothesis_revision", False)
        has_retry = seed["complexity_features"].get("retry_reasoning", False)
        ambiguous = seed["complexity_features"].get("ambiguous_intent", False)

        if n_calls == 1 and not (has_deps or has_branch or ambiguous or has_revision or has_retry):
            return 3
        if has_parallel or has_revision or has_retry or n_calls >= 5:
            return 5
        if 2 <= n_calls <= 4 and (has_deps or has_branch or ambiguous):
            return 4
        return None  # ambiguous -> discard


# ---------------------------------------------------------------------------
# Intent templates. Each intent declares the domain tools it needs and a
# parameterized user phrasing; call graphs are derived from the template.
# ---------------------------------------------------------------------------

@dataclass
class Intent:
    domain: str
    task_type: str
    phrasings: list
    tools: list                       # tools the correct solution uses, in order
    features: dict = field(default_factory=dict)
    final_kind: str = "answer"        # answer | ticket | email | refund_report ...


INTENTS = [
    Intent("commerce", "lookup",
           ["What is the status of order {oid}?",
            "Look up order {oid} for me."],
           ["get_order"],
           final_kind="answer"),
    Intent("commerce", "weather_lookup",
           ["What's the weather in {city} right now?",
            "Current conditions in {city}, please."],
           ["get_weather"],
           final_kind="answer"),
    Intent("kb", "search",
           ["Search the knowledge base for how to handle {topic}.",
            "Find KB articles about {topic}."],
           ["search_knowledge_base"],
           final_kind="answer"),
    Intent("commerce", "diagnose_refund",
           ["Refund order {oid} and tell me why its payment failed twice.",
            "Order {oid} failed to pay twice — diagnose it and issue a refund if allowed."],
           ["get_order", "get_payment_log", "check_refund_eligibility", "create_refund"],
           features={"cross_tool_state": True, "conditional_branch": True, "ambiguous_intent": True},
           final_kind="refund_report"),
    Intent("commerce", "customer_history",
           ["List the last orders of customer {cid} that are still open, then summarize.",
            "Show me customer {cid}'s open orders."],
           ["list_customer_orders"],
           final_kind="answer"),
    Intent("itsm", "file_ticket",
           ["Open a {priority} ticket titled '{title}' describing: {desc}",
            "Create a {priority} ITSM ticket for '{title}' — {desc}"],
           ["create_ticket"],
           final_kind="ticket"),
    Intent("analytics", "sql_lookup",
           ["Run this against the warehouse: {sql}",
            "Execute this read-only query: {sql}"],
           ["query_sql"],
           final_kind="answer"),
    Intent("workspace", "read_doc",
           ["Read {path} and summarize the first section.",
            "What does {path} contain?"],
           ["read_file"],
           final_kind="answer"),
    Intent("workspace", "explore",
           ["List {path}, then read the most relevant file and summarize it.",
            "Explore {path} and tell me what's there."],
           ["list_directory", "read_file"],
           features={"cross_tool_state": True, "ambiguous_intent": True},
           final_kind="answer"),
    Intent("commerce", "full_audit",
           ["Audit customer {cid}: list their orders, check payment logs for the newest one, "
            "verify refund eligibility, process the refund, and email a summary to {email}.",
            "Do a full account review of {cid} ending with a refund if warranted and an email to {email}."],
           ["list_customer_orders", "get_payment_log", "check_refund_eligibility",
            "create_refund", "send_email"],
           features={"cross_tool_state": True, "conditional_branch": True, "retry_reasoning": True},
           final_kind="answer"),
]

_ENTITY_POOLS = {
    "oid": ["4451", "8830", "10294", "55219", "77301"],
    "cid": ["cus_7f2a", "cus_91bc", "cus_0d44"],
    "city": ["Osaka", "Lyon", "Porto", "Kyoto", "Turin"],
    "topic": ["failed recurring payments", "order address changes", "exporting invoices"],
    "priority": ["high", "medium", "urgent"],
    "title": ["Payment gateway timeouts", "Cannot access billing portal", "Nightly sync failing"],
    "desc": [" intermittent 502s from the gateway since Monday",
             " users report login loops after the SSO change",
             " the 03:00 UTC job has not run for two days"],
    "sql": ["SELECT count(*) FROM payments WHERE status='failed'",
            "SELECT month, sum(total) FROM orders GROUP BY month ORDER BY month"],
    "path": ["reports/q3.md", "config/deploy.yaml", "notes/handover.txt"],
    "email": ["ops@example.com", "finance@example.com"],
}


class SeedGenerator:
    """Deterministic seed sampler: same seed -> identical seed stream."""

    def __init__(self, rng_seed=42, max_registry=24):
        self.rng = random.Random(rng_seed)
        self.max_registry = max_registry
        self._counter = 0

    def _sid(self, material):
        h = hashlib.sha256(f"{material}-{self._counter}".encode()).hexdigest()[:12]
        return f"seed_{h}"

    def _entities(self, intent):
        ents, used = {}, set(intent.phrasings[0].split())
        for key, pool in _ENTITY_POOLS.items():
            if "{%s}" % key in intent.phrasings[0]:
                ents[key] = self.rng.choice(pool)
        return ents

    def _call_graph(self, intent, ents):
        graph = []
        # simple parameter binding: later calls reference earlier outputs
        bindings = {}
        for i, tool in enumerate(intent.tools):
            args = {}
            if tool == "get_order":
                args["order_id"] = ents.get("oid", "4451")
            elif tool == "get_payment_log":
                args["order_id"] = "<get_order.order_id>" if "get_order" in intent.tools else "4451"
            elif tool == "check_refund_eligibility":
                args = ({"order_total": "<get_order.total>", "days_since_purchase": "<get_order.days_since>"}
                        if "get_order" in intent.tools else
                        {"order_total": 129.99, "days_since_purchase": 3})
            elif tool == "create_refund":
                args = ({"order_id": "<get_order.order_id>", "amount": "<get_order.total>",
                         "reason": "requested_by_customer"}
                        if "get_order" in intent.tools else
                        {"order_id": "4451", "amount": 129.99, "reason": "requested_by_customer"})
            elif tool == "list_customer_orders":
                args = {"customer_id": ents.get("cid", "cus_7f2a"), "status": "paid"}
            elif tool == "get_weather":
                args = {"city": ents.get("city", "Osaka")}
            elif tool == "search_knowledge_base":
                args = {"query": ents.get("topic", "failed recurring payments")}
            elif tool == "send_email":
                args = {"to": ents.get("email", "ops@example.com"),
                        "subject": "Account audit summary",
                        "body": "<summary of prior tool results>"}
            elif tool == "create_ticket":
                args = {"title": ents.get("title", "Payment gateway timeouts"),
                        "priority": ents.get("priority", "high"),
                        "description": ents.get("desc", "intermittent failures")}
            elif tool == "query_sql":
                args = {"sql": ents.get("sql", "SELECT 1")}
            elif tool == "read_file":
                args = {"path": ents.get("path", "reports/q3.md")}
            elif tool == "list_directory":
                args = {"path": "reports"}
            else:
                args = {"filter": "active"}
            graph.append({"tool": tool, "arguments": args})
        return graph

    def next_seed(self, zoo):
        """Sample one seed; discards ambiguous rubric outcomes internally."""
        for _ in range(50):
            intent = self.rng.choice(INTENTS)
            ents = self._entities(intent)
            seed = {
                "seed_id": None,
                "task_type": intent.task_type,
                "domain": intent.domain,
                "intent_phrasing": self.rng.choice(intent.phrasings).format(**ents),
                "entities": ents,
                "tools_available": intent.tools,
                "registry_size": 0,
                "call_graph": self._call_graph(intent, ents),
                "complexity_features": dict(intent.features),
                "faults": [],
                "final_kind": intent.final_kind,
            }
            seed["seed_id"] = self._sid(seed["intent_phrasing"])
            self._counter += 1
            tier = GoldDepthRubric.score(seed)
            if tier is None:
                continue
            seed["gold_depth"] = tier
            # registry: solution tools + distractors up to max_registry
            distractors = [n for n in zoo.names() if n not in intent.tools]
            self.rng.shuffle(distractors)
            n_dist = self.rng.choice([0, 3, 8, 15])
            seed["registry"] = intent.tools + distractors[:n_dist]
            seed["registry_size"] = len(seed["registry"])
            return seed
        raise RuntimeError("could not sample an unambiguous seed")


def counterfactual_depths(gold_depth):
    """Forced depths != gold (SYNTHETIC_DATA_PLAN.md §3.4)."""
    return [d for d in (3, 4, 5) if d != gold_depth]
