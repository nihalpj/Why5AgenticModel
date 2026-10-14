"""Tool zoo: versioned registry of JSON-Schema tool definitions.

200 real-style + 200 procedural tools is the production target; this pilot
builds a representative subset of each with the same schema discipline
(SYNTHETIC_DATA_PLAN.md §2.3). Every tool must describe parameters with a
proper JSON Schema (Draft 2020-12 style) so gate G3 can validate arguments.
"""

TOOLZOO_VERSION = "v0.4.0-pilot"


def _id(name):
    return {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,32}$", "description": f"Identifier for the {name}."}


def _enum(values, desc):
    return {"type": "string", "enum": list(values), "description": desc}


def _int(lo, hi, desc):
    return {"type": "integer", "minimum": lo, "maximum": hi, "description": desc}


REAL_TOOLS = [
    {
        "name": "get_order",
        "description": "Fetch an order by ID, including line items, totals, and payment status.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": _id("order"),
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_payment_log",
        "description": "Fetch payment attempts (authorization, capture, failure codes) for an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": _id("order"),
                "limit": _int(1, 50, "Maximum number of attempts to return."),
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "check_refund_eligibility",
        "description": "Check refund policy eligibility given order total and days since purchase.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_total": {"type": "number", "minimum": 0, "description": "Order total in account currency."},
                "days_since_purchase": _int(0, 3650, "Days elapsed since the order was placed."),
            },
            "required": ["order_total", "days_since_purchase"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_refund",
        "description": "Create a refund against a paid order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": _id("order"),
                "amount": {"type": "number", "minimum": 0.01, "description": "Amount to refund."},
                "reason": _enum(
                    ["duplicate", "fraudulent", "requested_by_customer", "quality_issue"],
                    "Canonical refund reason code.",
                ),
            },
            "required": ["order_id", "amount", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_customer_orders",
        "description": "List orders for a customer, newest first.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": _id("customer"),
                "status": _enum(["open", "paid", "refunded", "canceled"], "Filter by order status."),
                "limit": _int(1, 100, "Maximum number of orders to return."),
            },
            "required": ["customer_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_knowledge_base",
        "description": "Full-text search over the support knowledge base.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": 200, "description": "Free-text search query."},
                "category": _enum(["billing", "shipping", "returns", "technical"], "Restrict search to a category."),
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "maxLength": 64, "description": "City name."},
                "units": _enum(["celsius", "fahrenheit"], "Temperature units."),
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    {
        "name": "send_email",
        "description": "Send an email on behalf of the authenticated account.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "format": "email", "description": "Recipient address."},
                "subject": {"type": "string", "maxLength": 120, "description": "Email subject."},
                "body": {"type": "string", "maxLength": 8000, "description": "Plain-text body."},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_ticket",
        "description": "Open a support ticket in the ITSM system.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "maxLength": 120, "description": "Ticket title."},
                "priority": _enum(["low", "medium", "high", "urgent"], "Ticket priority."),
                "description": {"type": "string", "maxLength": 4000, "description": "Problem description."},
            },
            "required": ["title", "priority", "description"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_sql",
        "description": "Run a read-only SQL query against the analytics warehouse.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "maxLength": 4000, "description": "SELECT-only SQL statement."},
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "maxLength": 256, "description": "Relative file path."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_directory",
        "description": "List entries of a workspace directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "maxLength": 256, "description": "Relative directory path."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
]

# ---------------------------------------------------------------------------
# Procedural tools: adversarial schema shapes (SYNTHETIC_DATA_PLAN.md §2.3)
# ---------------------------------------------------------------------------

_ADJ = ["inventory", "shipment", "invoice", "payroll", "reservation", "subscription"]
_NOUN = ["record", "entry", "item", "summary", "status", "history"]
_PARAM_SHAPES = [
    {"type": "string", "enum": ["active", "inactive", "archived"], "description": "Lifecycle state filter."},
    {"type": "integer", "minimum": 0, "maximum": 500, "description": "Numeric lookup key."},
    {"type": "object", "properties": {"year": _int(2000, 2100, "Year."), "quarter": _int(1, 4, "Quarter.")},
     "required": ["year", "quarter"], "additionalProperties": False,
     "description": "Fiscal period selector."},
    {"type": "array", "items": {"type": "string", "maxLength": 24}, "maxItems": 8,
     "description": "List of tag filters."},
]


def _build_procedural_tools(count):
    import itertools

    tools, combos = [], itertools.product(_ADJ, _NOUN)
    for i, (adj, noun) in enumerate(combos):
        if i >= count:
            break
        n1, n2 = f"{adj}_{noun}", f"get_{noun}_by_id"
        param_name = "filter"
        tools.append({
            "name": n1,
            "description": f"Look up {adj} {noun} information by filter. Related tools with different schemas exist; choose by description.",
            "parameters": {
                "type": "object",
                "properties": {param_name: _PARAM_SHAPES[i % len(_PARAM_SHAPES)]},
                "required": [param_name],
                "additionalProperties": False,
            },
        })
        # ambiguous-name distractor sharing style with real tools
        tools.append({
            "name": n2,
            "description": f"Fetch a single {noun} by its identifier. Not scoped to {adj}.",
            "parameters": {
                "type": "object",
                "properties": {f"{noun}_id": _id(noun)},
                "required": [f"{noun}_id"],
                "additionalProperties": False,
            },
        })
    return tools


class ToolZoo:
    """Versioned, read-only registry of tool definitions."""

    def __init__(self, procedural_count=20):
        self.version = TOOLZOO_VERSION
        self._tools = {t["name"]: t for t in REAL_TOOLS}
        for t in _build_procedural_tools(procedural_count):
            self._tools.setdefault(t["name"], t)

    def __len__(self):
        return len(self._tools)

    def names(self):
        return sorted(self._tools)

    def get(self, name):
        return self._tools.get(name)

    def registry(self, names):
        """Return schema blocks for the given tool names (registry view)."""
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"tools not in zoo: {missing}")
        return [self._tools[n] for n in names]

    def schemas_for_validation(self, names):
        """name -> parameters-schema mapping used by gate G3."""
        return {n: self._tools[n]["parameters"] for n in names}
