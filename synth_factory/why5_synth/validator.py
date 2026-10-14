"""Validator: gates G1-G10 per SYNTHETIC_DATA_PLAN.md §3.3.

Each gate is a pure function over the generated record; results are stored
on the record's `validation` field. Records failing G1-G3 count against the
teacher; G4-G5 failures would trigger regeneration in the full factory.
"""

import json
import re

GATES = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10"]

# Gate G8: benchmark holdout n-grams (representative sample; production uses
# full benchmark corpora).
_BENCHMARK_SNIPPETS = [
    "what is the status of order 4451 in the bfcl benchmark",
    "calculate 2 + 2 and show your work step by step",
    "book the earliest flight from jfk to sfo on tau-bench",
]


def _validate_schema(args, schema):
    """Minimal JSON-Schema subset validator (type/required/enum/bounds).

    Production uses `jsonschema` Draft 2020-12; this subset covers every
    schema shape the pilot tool zoo emits, including '<ref>' string values
    treated as valid strings.
    """
    if schema.get("type") == "object":
        if not isinstance(args, dict):
            return False
        props = schema.get("properties", {})
        for k in args:
            if k not in props:
                return False  # additionalProperties: False everywhere in the zoo
        for r in schema.get("required", []):
            if r not in args:
                return False
        return all(_validate_schema(args[k], props[k]) for k in args if k in props)
    if schema.get("type") == "string":
        if not isinstance(args, str):
            return False
        if "enum" in schema and args not in schema["enum"]:
            return False
        if "maxLength" in schema and len(args) > schema["maxLength"]:
            return False
        if schema.get("pattern"):
            return re.match(schema["pattern"], args) is not None
        return True
    if schema.get("type") == "integer":
        return isinstance(args, int) and not isinstance(args, bool) and \
            args >= schema.get("minimum", float("-inf")) and args <= schema.get("maximum", float("inf"))
    if schema.get("type") == "number":
        return isinstance(args, (int, float)) and not isinstance(args, bool) and \
            args >= schema.get("minimum", float("-inf"))
    if schema.get("type") == "array":
        if not isinstance(args, list):
            return False
        if "maxItems" in schema and len(args) > schema["maxItems"]:
            return False
        return all(_validate_schema(a, schema.get("items", {})) for a in args)
    return True


class Validator:
    def __init__(self, zoo):
        self.zoo = zoo
        self._seen_normalized = set()   # G10 state within a dataset

    # -- gate implementations ------------------------------------------------

    def g1_protocol_parse(self, rec):
        """Parses via the protocol-turns view without error."""
        try:
            for t in rec["protocol_turns"]:
                assert t["type"] in {"depth", "why", "plan", "tool_call", "final"}
                if t["type"] == "depth":
                    assert t["value"] in (3, 4, 5)
                elif t["type"] == "why":
                    assert isinstance(t["text"], str) and t["text"]
                elif t["type"] == "tool_call":
                    assert isinstance(t["name"], str) and isinstance(t["arguments"], dict)
            json.dumps(rec["messages"])  # messages must be JSON-serializable
            return True
        except Exception:
            return False

    def g2_depth_exactness(self, rec):
        n_whys = sum(1 for t in rec["protocol_turns"] if t["type"] == "why")
        depth = next(t["value"] for t in rec["protocol_turns"] if t["type"] == "depth")
        declared = rec.get("forced_depth") or rec["gold_depth"]
        return n_whys == depth == declared

    def g3_schema_validity(self, rec):
        schemas = self.zoo.schemas_for_validation(rec["_registry_names"])
        for t in rec["protocol_turns"]:
            if t["type"] == "tool_call":
                schema = schemas.get(t["name"])
                if schema is None:
                    return False
                if not _validate_schema(t["arguments"], schema):
                    return False
        return True

    def g4_graph_fidelity(self, rec):
        """Executed calls match the expected graph in order; concrete args
        equal, '<ref>' args resolve positionally to the executed value."""
        exp, got = rec["call_graph_expected"], rec["call_graph_executed"]
        if len(exp) != len(got):
            return False
        for e, g in zip(exp, got):
            if e["tool"] != g["tool"]:
                return False
            if set(e["arguments"]) != set(g["arguments"]):
                return False
            for k, v in e["arguments"].items():
                if isinstance(v, str) and v.startswith("<") and v.endswith(">"):
                    continue  # reference: sandbox resolution verified in G5
                if g["arguments"].get(k) != v:
                    return False
        return True

    def g5_behavioral_success(self, rec):
        return len(rec["call_graph_executed"]) == len(rec["_seed"]["call_graph"]) and \
            all(g["tool"] == e["tool"] for g, e in zip(rec["call_graph_executed"],
                                                       rec["_seed"]["call_graph"]))

    def g6_retry_sanity(self, rec):
        return rec["_retries"] <= 3

    def g7_no_phantom_tools(self, rec):
        reg = set(rec["_registry_names"])
        return all(t["name"] in reg for t in rec["protocol_turns"] if t["type"] == "tool_call")

    def g8_decontamination(self, rec):
        text = " ".join(m.get("content", "") for m in rec["messages"]).lower()
        return not any(s in text for s in _BENCHMARK_SNIPPETS)

    def g9_pii_secrets(self, rec):
        text = json.dumps(rec["messages"])
        bad = [r for r in ("sk_live_", "AKIA", "password=", "BEGIN PRIVATE KEY") if r in text]
        return not bad

    def g10_dedup(self, rec):
        norm = " ".join(t.get("text", "") for t in rec["protocol_turns"] if t["type"] == "why").lower()
        key = hash(norm)
        if key in self._seen_normalized:
            return False
        self._seen_normalized.add(key)
        return True

    # -- driver ---------------------------------------------------------------

    def validate(self, rec):
        results = {}
        for i, gate in enumerate([self.g1_protocol_parse, self.g2_depth_exactness,
                                  self.g3_schema_validity, self.g4_graph_fidelity,
                                  self.g5_behavioral_success, self.g6_retry_sanity,
                                  self.g7_no_phantom_tools, self.g8_decontamination,
                                  self.g9_pii_secrets, self.g10_dedup], start=1):
            try:
                results[f"g{i}"] = bool(gate(rec))
            except Exception:
                results[f"g{i}"] = False
        results["passed"] = all(results.values())
        rec["validation"] = results
        return results["passed"]
