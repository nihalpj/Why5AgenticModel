# Product Requirements Document (PRD)

## Why5 — A 2B-Parameter Adaptive-Reasoning Agentic Model with Toyota 5-Why Depth Control

| Field | Value |
|---|---|
| Product Name | Why5 Agentic Model (working title) |
| Version | 0.1 (Draft) |
| Date | 2026-10-14 |
| Status | For Review |
| Owner | Founding Team / Model Engineering |

---

## 1. Summary

Why5 is a ~2 billion parameter language model purpose-built for **agentic workloads** — tool calling, multi-step task execution, and orchestration. Its differentiator is **adaptive reasoning depth** inspired by Toyota's 5-Why root-cause analysis: the model dynamically selects how many "why" reasoning iterations to perform before answering, based on task complexity:

- **LOW complexity → 3 whys** (fast, cheap path for simple queries)
- **HIGH complexity → 4 whys** (moderate deliberation)
- **MAX complexity → 5 whys** (full root-cause depth for hard, multi-step problems)

This avoids the fixed-cost overhead of large always-on reasoning models while preserving depth where it matters — critical for latency-sensitive, cost-sensitive agentic deployments (edge, on-prem, high-volume SaaS).

---

## 2. Background & Problem Statement

### 2.1 Problem
Current small models (1B–7B) used in agentic pipelines suffer from two opposing failures:

1. **Under-reasoning**: they call tools with malformed arguments or skip necessary verification steps, causing cascading agent-loop failures.
2. **Over-reasoning**: when wrapped in forced chain-of-thought or reasoning-mode prompts, they burn tokens and latency on trivial requests.

### 2.2 Insight
Toyota's 5-Why method demonstrates that problem decomposition has a *natural depth*: stop when the root cause is reached. We operationalize this as a **trainable depth controller**: the model learns to classify the required reasoning depth (3/4/5 why-levels) and generates exactly that many structured self-questioning steps before emitting tool calls or final answers.

### 2.3 Why 2B parameters?
- Fits on a single consumer/edge GPU (quantized) and in CPU-bound inference for embedded agentic runtimes.
- Cheap enough for high-volume tool-calling loops where the model is invoked dozens of times per task.
- Large enough to achieve reliable structured output (JSON tool schemas) and multi-turn state tracking.

---

## 3. Goals & Non-Goals

### Goals
1. Train a 2B-parameter model with best-in-class **function/tool calling** accuracy at its size (target: ≥ 90% on BFCL v3 and τ-bench-lite style evals within its parameter class).
2. Implement **Why-depth controller**: classifier + constrained generation guaranteeing exactly 3, 4, or 5 reasoning steps, selected per query.
3. Deliver **agentic workflow competence**: multi-turn planning, tool-result ingestion, error recovery/retries, and termination decisions.
4. Provide production serving: structured output (JSON Schema-constrained), < 400ms first-token latency on a single A10G/L4 class GPU.
5. Release with an eval harness that scores both task success rate and **reasoning-efficiency** (success per reasoning token).

### Non-Goals (v1)
- Multimodal input (vision/audio).
- General chat excellence / leaderboard chasing on open-ended benchmarks.
- On-device fine-tuning tooling.
- Code-from-scratch generation beyond tool-argument construction.
- > 512K context window (v1 targets 256K).

---

## 4. Target Users & Use Cases

| User | Use Case |
|---|---|
| Agent-framework developers (LangGraph, AutoGen, in-house orchestrators) | Drop-in tool-calling brain for multi-agent systems |
| SaaS/platform teams | Embedded autonomous workflow execution (CRM ops, IT automation, data retrieval agents) |
| Edge/IoT product teams | Offline agentic assistant on-device |
| RPA vendors | LLM-powered retry/recovery reasoning inside automation pipelines |

### Canonical usage example
```
User: "Refund order #4451 and tell me why it failed twice."

[Depth classifier: complexity=HIGH → 4 whys]
Why-1: What does the user want? → refund + diagnosis
Why-2: What do I need first? → order status via get_order(4451)
Why-3: Why did it fail before? → inspect payment_attempts via get_payment_log
Why-4: Is refund allowed given policy state? → check refund_policy(order_total, days_since_purchase)
→ tool calls: get_order → get_payment_log → check_refund_eligibility → create_refund
→ Final answer with root-cause explanation.
```

---

## 5. Functional Requirements

### 5.1 Why-Depth Controller (core differentiator)
| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Model must classify each incoming query into complexity tier: LOW / HIGH / MAX | P0 |
| FR-2 | LOW → exactly 3 why-steps; HIGH → exactly 4; MAX → exactly 5, enforced via constrained decoding or structured tags | P0 |
| FR-3 | Each why-step must be a discrete, inspectable reasoning unit (`<why n="1">…</why>`) for observability | P0 |
| FR-4 | Depth must be overridable by the orchestrator (force tier via system prompt or API param) | P1 |
| FR-5 | Depth classification must be tunable post-training via a lightweight calibration head (no full fine-tune needed to adjust thresholds) | P1 |
| FR-6 | Distribution of tier selection reported per request in response metadata (`depth_used: 3|4|5`) | P0 |

### 5.2 Tool Calling
| ID | Requirement | Priority |
|---|---|---|
| FR-10 | Accept tool definitions as JSON Schema; emit valid tool calls with arguments conforming to schema (constrained decoding) | P0 |
| FR-11 | Support parallel tool calls and sequential dependent calls in one turn | P0 |
| FR-12 | Ingest tool results mid-conversation and decide: call again, adjust args, or answer | P0 |
| FR-13 | Graceful error handling: on tool error, produce corrected retry or explicit failure report, max N retries (configurable, default 3) | P0 |
| FR-14 | Null/empty tool registry → model behaves as plain instruction model, no hallucinated calls | P0 |
| FR-15 | Support ≥ 64 registered tools per session with accurate selection | P1 |

### 5.3 Agentic Workflow
| ID | Requirement | Priority |
|---|---|---|
| FR-20 | Multi-turn state tracking over ≥ 20 turns without tool-call argument drift | P0 |
| FR-21 | Plan–act–observe loop compatibility (ReAct / plan-and-execute styles) | P0 |
| FR-22 | Termination judgment: model must emit final answer when task complete, not loop | P0 |
| FR-23 | Context: 256K tokens (v1), with summarization-aware training and long-context tool-result handling (position interpolation / long-rope fine-tune) | P0 |
| FR-24 | Optional scratchpad isolation: why-steps kept out of final user-facing answer | P1 |

---

## 6. Model & Training Specification

### 6.1 Architecture (indicative)
- Dense transformer, ~2.0B params: ~32 layers, hidden ~2048–2304, GQA attention, RoPE, SwiGLU, RMSNorm.
- Tokenizer: 128K BPE (reuse of an existing open tokenizer acceptable to shorten training).
- Native special tokens for the why-protocol and tool protocol: `<|tools|>`, `<|tool_call|>`, `<|tool_result|>`, `<|why|>`, `<|depth:3|>` etc.

### 6.2 Training pipeline
1. **Base pretraining** (~2T tokens): curated code + web + docs + API-documentation-heavy corpus (tool schemas, README, OpenAPI specs) — critical for tool-argument fluency.
2. **Why-protocol SFT**: synthesized traces where every example contains a gold depth label and exactly N why-steps followed by tool calls/answer. Depth labels derived from complexity heuristics + teacher-model labeling.
3. **Tool-calling SFT**: BFCL-format, ToolBench/xLAM-style, plus internal synthetic multi-turn agent trajectories.
4. **Agentic RL (GRPO/PPO-style)**: rollout-based training on sandboxed environments (code execution, DB queries, web-sim APIs); reward = task success + schema validity − reasoning-token penalty − retry count.
5. **Depth-calibration stage**: train the tier classifier / calibration head with a cost-aware objective (accuracy vs. latency budget), enabling FR-5.

### 6.3 Data requirements (order of magnitude)
| Stage | Examples | Notes |
|---|---|---|
| Pretraining | ~2T tokens | API-doc-heavy mixture |
| Why-SFT | ~500K traces | balanced 3/4/5 split (~40/40/20) |
| Tool SFT | ~800K traces | incl. failure/recovery traces |
| RL rollouts | ~100K episodes | sandboxed tool environments |

---

## 7. Evaluation Plan

### 7.1 Quality benchmarks
- **Tool calling**: BFCL v3, API-Bank, ToolBench, τ-bench (retail/airline simplified).
- **Agentic**: SWE-bench-Lite-style mini-tasks (stretch), WebArena-lite simulation (stretch), internal multi-turn agent suite.
- **Reasoning sanity**: GSM8K, ARC-Challenge (guard against general regression).

### 7.2 Why-specific metrics (new)
| Metric | Definition | Target (v1) |
|---|---|---|
| Depth Accuracy | chosen tier matches gold tier | ≥ 80% |
| Depth Calibration Error | E[chosen − gold] within ±1 | ≥ 95% |
| Step-Count Exactness | generated why-steps == declared depth | ≥ 98% (constrained: 100%) |
| Reasoning Efficiency | task success per 1K reasoning tokens | ≥ 1.3× baseline fixed-CoT 2B model |
| Success @ depth | task success at tier-3 for gold-LOW tasks vs. running everything at tier-5 | ≤ 2% gap |

### 7.3 Latency targets (single L4/A10G, FP8/INT8)
| Tier | P50 time-to-final-token (short task) |
|---|---|
| 3 whys | ≤ 1.2s |
| 4 whys | ≤ 1.6s |
| 5 whys | ≤ 2.2s |

---

## 8. Technical Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Depth classifier misroutes hard queries to 3 whys | Task failure | Constrained generation only guarantees step count; allow orchestrator override + conservative default-to-4 in ambiguous cases |
| Why-format training data is synthetic → brittle phrasing | Format drift in production | Diversify templates; add format-robustness RL; strict decoding constraints |
| 2B ceiling on multi-tool composition (>32 tools) | Selection errors | Retrieval-based tool filtering upstream; train with noisy tool lists |
| Looping / refusal to terminate | Agent hangs | Termination-focused RL reward; hard turn cap in harness |
| Reward hacking in RL (fake success) | Silent quality loss | Held-out sandbox envs; human audit sampling ≥ 1% rollouts |

---

## 9. Milestones

| Phase | Duration | Deliverable | Exit criteria |
|---|---|---|---|
| M0: Design & data infra | 4 wks | Data pipeline, why-trace synthesis, eval harness v0 | 50K validated why-traces; eval harness runs end-to-end |
| M1: Base model | 8 wks | 2B base checkpoint | Perplexity on par with open 2B baselines |
| M2: SFT (why + tools) | 4 wks | SFT checkpoint | BFCL ≥ 85% (class); Step-Count Exactness ≥ 98% |
| M3: Agentic RL | 6 wks | RL checkpoint | Internal agent suite success ≥ 75%; termination violations < 1% |
| M4: Calibration + serving | 3 wks | Quantized builds (INT8/4-bit), OpenAI-compatible server with `depth` param | Latency targets met; Depth Accuracy ≥ 80% |
| M5: Beta release | 2 wks | Weights + eval report + integration guide | All P0 FRs verified |

---

## 10. Open Questions

1. License: fully open weights vs. source-available?
2. Should tier classification run as a separate fast-head pass (adds one forward pass) or inline at generation start (cheaper, slightly less accurate)?
3. Priority of a fine-tuning kit so enterprises can add proprietary tools — v1 or v2?
4. Do we expose intermediate why-steps to end users (transparency) or only to the orchestrator (cleaner UX)? Default: orchestrator-only.

---

## 11. Appendix

- **A. Why-trace format spec** (draft): system → depth token → N× `<why>` blocks → `<plan>` → tool calls → `<answer>`.
- **B. Reward function spec**: R = 1.0·success + 0.2·schema_valid − 0.05·reasoning_tokens/100 − 0.3·retries.
- **C. Serving contract**: OpenAI Chat Completions-compatible + extension fields `depth_used`, `why_trace` (optional), `tool_retries`.
