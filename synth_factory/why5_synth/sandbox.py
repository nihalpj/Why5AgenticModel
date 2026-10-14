"""Sandbox: simulates tool execution for seed call graphs (pilot).

Production uses SQLite/shell/REST-sim containers (SYNTHETIC_DATA_PLAN.md
§3.5); this pilot sandbox emulates the same contract: execute(name, args)
-> result dict, with fault injection support for D3 recovery traces.
"""

import random
import re

_REF = re.compile(r"^<[a-z_0-9]+\.[a-z_0-9]+>$")


class SandboxError(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind  # "http_500" | "schema_mismatch" | "empty_result"


class Sandbox:
    """Executes calls in seed call-graph order, threading outputs forward.

    Arguments containing "<tool.key>" references are resolved from prior
    results; unresolved references raise SandboxError("schema_mismatch"),
    which the generator/validator treats as a failed trace.
    """

    def __init__(self, rng_seed=7):
        self.rng = random.Random(rng_seed)
        self.state = {}
        self.executed = []

    def _resolve(self, value):
        if isinstance(value, str) and _REF.match(value):
            ref = value[1:-1].split(".", 1)
            if len(ref) != 2:
                raise SandboxError("schema_mismatch", f"malformed reference {value!r}")
            tool, key = ref
            prior = self.state.get(tool)
            if prior is None or key not in prior:
                raise SandboxError("schema_mismatch", f"unresolved reference {value!r}")
            return prior[key]
        return value

    def _tool_result(self, name, args):
        if name == "get_order":
            return {"order_id": args["order_id"], "status": "paid", "total": 129.99,
                    "days_since": 3, "customer_id": "cus_7f2a"}
        if name == "get_payment_log":
            return {"attempts": [
                {"id": 1, "status": "failed", "code": "insufficient_funds"},
                {"id": 2, "status": "failed", "code": "insufficient_funds"},
                {"id": 3, "status": "succeeded", "code": None}]}
        if name == "check_refund_eligibility":
            eligible = args["days_since_purchase"] <= 30 and args["order_total"] <= 10000
            return {"allowed": eligible, "policy": "30-day full refund window"}
        if name == "create_refund":
            return {"refund_id": f"re_{self.rng.randint(10**5, 10**6)}",
                    "amount": args["amount"], "status": "pending"}
        if name == "list_customer_orders":
            return {"orders": [{"order_id": "4451", "status": "paid", "total": 129.99},
                               {"order_id": "8830", "status": "refunded", "total": 45.0}]}
        if name == "get_weather":
            return {"city": args["city"], "temp_c": 18, "conditions": "cloudy"}
        if name == "search_knowledge_base":
            return {"articles": [{"id": "kb_12", "title": f"Handling {args['query']}"}]}
        if name == "send_email":
            return {"message_id": f"msg_{self.rng.randint(10**4, 10**5)}", "queued": True}
        if name == "create_ticket":
            return {"ticket_id": f"tkt_{self.rng.randint(10**3, 10**4)}",
                    "priority": args["priority"], "state": "open"}
        if name == "query_sql":
            return {"rows": [{"count": 4}], "rowcount": 1}
        if name == "read_file":
            return {"path": args["path"],
                    "content": "# Report\n\nQ3 revenue was up 12% driven by renewals."}
        if name == "list_directory":
            return {"entries": [{"name": "q3.md", "type": "file"}, {"name": "raw", "type": "dir"}]}
        # procedural tools
        return {"matched": True, "tool": name, "filter": args.get("filter")}

    def execute(self, name, args, faults=None):
        faults = faults or []
        resolved = {k: self._resolve(v) if isinstance(v, str) else v for k, v in args.items()}
        if name in faults:
            raise SandboxError("http_500", f"simulated server error calling {name}")
        result = self._tool_result(name, resolved)
        self.state[name] = result
        self.executed.append({"tool": name, "arguments": resolved})
        return result

    def run_graph(self, graph, faults=None):
        """Execute the full graph; raises SandboxError on failure."""
        for step in graph:
            self.execute(step["tool"], step["arguments"], faults=faults)
        return self.executed
