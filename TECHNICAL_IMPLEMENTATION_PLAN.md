# Technical Implementation Plan

## Why5 — 2B-Parameter Adaptive-Reasoning Agentic Model (256K Context)

| Field | Value |
|---|---|
| Document Version | 1.0 |
| Date | 2026-10-14 |
| Depends On | PRD v0.1 (PRD_Adaptive_Why_Agentic_Model.md) |
| Status | Ready for Engineering Review |

---

## 1. Overview

This plan translates the PRD into an executable engineering program: model architecture, data pipeline, four training stages, the Why-depth controller design, serving stack, evaluation harness, and a 27-week schedule with staffing and compute budgets.

**Program flow:**

```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐
│ M0: Data│──▶│ M1: Base │──▶│ M2: SFT │──▶│ M3: Agentic│──▶│ M4: Calib │──▶│ M5: Beta│
│ & Infra │   │ Pretrain │   │ (why+tool)│  │ RL        │   │ + Serving │   │ Release │
└─────────┘   └──────────┘   └─────────┘   └──────────┘   └──────────┘   └─────────┘
```

---

## 2. Model Architecture Specification

### 2.1 Final architecture (frozen at M0 exit)

| Component | Choice | Rationale |
|---|---|---|
| Params | ~2.05B total (~1.8B non-embedding) | Fits INT4 on 8GB VRAM; FP8 single L4 |
| Layers | 30 | Depth/width balance for structured output reliability |
| Hidden dim | 2304 | Standard efficient ratio |
| Attention | GQA: 32 query heads, 8 KV heads | KV-cache reduction critical at 256K |
| Head dim | 64 | — |
| FFN | SwiGLU, intermediate 6144 | — |
| Norm | RMSNorm (pre-norm), eps 1e-5 | — |
| Position | RoPE, base 500K, θ scaling for 256K | Long-context without NTK hacks at serve time |
| Vocab | 131,072 tokens (reuse Llama-3-style tokenizer) | Familiar token efficiency on code/JSON |
| Activation | No QK-norm in v1; revisit if training instability | Reduce moving parts |
| Tied embeddings | Yes (tie input/output) | Param efficiency at this scale |
| Max positions | 262,144 (256K) | FR-23 |

### 2.2 Special tokens & serialized protocols

Reserve 32 special-token IDs. Core set:

```
<|system|> <|user|> <|assistant|>          # chat framing
<|tools_begin|> <|tools_end|>              # tool registry block
<|tool_call|> <|end_tool_call|>            # each call: {"name": ..., "arguments": {...}}
<|tool_result|> <|end_tool_result|>        # wrapped result payload
<|depth_3|> <|depth_4|> <|depth_5|>        # emitted BEFORE first <why> step
<|why|> <|end_why|>                        # one per reasoning step
<|plan|> <|end_plan|>                      # tool sequence plan after whys
<|final|> <|end_final|>                    # user-facing answer boundary
<|retry|>                                  # explicit retry marker
```

**Canonical turn structure (training target format):**

```
<|assistant|>
<|depth_4|>
<|why|>1. What is the user actually asking for? …<|end_why|>
<|why|>2. What information is missing? …<|end_why|>
<|why|>3. Which tool provides it, and with what constraints? …<|end_why|>
<|why|>4. Is the plan sufficient, and what could fail? …<|end_why|>
<|plan|>get_order(4451) → get_payment_log(order) → check_refund_eligibility(...)<|end_plan|>
<|tool_call|>{"name":"get_order","arguments":{"order_id":4451}}<|end_tool_call|>
```

Tool results return as `<|tool_result|>…json…<|end_tool_result|>` user-role turns; the model's *subsequent* turns skip the why-block (depth is decided once per task, not per turn) and continue with plan/act steps.

### 2.3 Why-depth controller — two mechanisms, one decision at M0

| Option | Design | Pros | Cons |
|---|---|---|---|
| **A. Inline (chosen default)** | First generated token is `<|depth_N|>`; constrained decoding then forces exactly N `<why>` blocks | Zero extra latency, single model | Classification and generation entangled |
| **B. Calibration head** | Small classifier head on final-layer CLS-style pooled state at prompt end; trained post-hoc (FR-5) | Separately tunable without touching generation; enables threshold tuning | Extra forward pass (cheap: prompt-only, no decode) |

**Decision:** Ship A for generation, and *also* train head B on frozen checkpoints as the FR-5 calibration mechanism (used for offline threshold analysis and optional server-side pre-routing). Both share the training signal, so B is nearly free.

**Constrained decoding contract (server-enforced, FR-2):**
1. Logit mask admits only `<|depth_3|>`, `<|depth_4|>`, `<|depth_5|>` as first assistant token.
2. After depth token, a grammar/FSM permits exactly N repetitions of the `<why>…</why>` nonterminal, then `<plan>`, then zero or more tool calls or `<final>`.
3. Override param `force_depth: 3|4|5` from the API replaces step 1 (FR-4).
4. Metadata `depth_used` returned in every response (FR-6).

---

## 3. Data Pipeline (M0)

### 3.1 Infrastructure

- **Storage:** S3-compatible object store; ~180TB raw, ~45TB post-dedup.
- **Processing:** Ray-based distributed dedup (MinHash), language ID, quality classifiers (small LM scorers), PII scrubbing, secrets scanning (critical — we train on API docs).
- **Synthesis cluster:** vLLM serving a 70B-class teacher model + tool-sandbox fleet.

### 3.2 Pretraining mixture (~2T tokens)

| Source | % | Tokens | Notes |
|---|---|---|---|
| Filtered web (English-heavy) | 38% | 760B | Quality-scored |
| Code (permissive licenses) | 25% | 500B | JSON/YAML heavy sampling — tool-schema fluency |
| API docs / OpenAPI specs / README / man pages | 12% | 240B | **Differentiator corpus**; upsample ×3 |
| Technical Q&A (StackOverflow-class, filtered) | 10% | 200B | — |
| Math & logic | 8% | 160B | Reasoning substrate |
| Multi-lingual | 5% | 100B | Top 15 languages |
| Agent-trajectory text (synthetic) | 2% | 40B | Early exposure to tool-call format |

### 3.3 Why-trace synthesis (500K traces)

Teacher generates traces; a rule-based + model-based complexity tierer assigns gold labels:

**Gold-tier rubric (deterministic first, model-assisted second):**
- **LOW (3):** single tool, single hop, no dependencies between calls, no error branches, unambiguous intent. (~40% of data)
- **HIGH (4):** 2–5 sequential calls, conditional branching, or ambiguous intent needing disambiguation. (~40%)
- **MAX (5):** ≥5 calls or parallel+sequential mix, requires hypothesis revision (initial why-answer proven wrong by later evidence), cross-tool state carryover, or retry-after-error reasoning. (~20%)

**Quality gates:** every trace schema-validated (tool args vs. registry JSON Schemas), executed in sandbox to verify success, and step-count exactness = N verified by parser. Rejection-sample until gates pass; target teacher rejection rate < 60% (else improve prompts).

**Counterfactual augmentation (key for depth calibration):** for ~100K tasks, generate traces at *wrong* depths (e.g., a gold-MAX task run at 3 whys) and record failure/quality loss → trains the classifier's cost-aware signal (§6.3).

### 3.4 Tool-calling SFT (800K traces)

- 400K from public sets reformatted (BFCL, ToolBench, API-Bank, xLAM) into our token protocol.
- 400K synthetic across a **tool zoo of 400+ tools**: 200 real-schema tools (Stripe, GitHub, Slack, SQL, shell, HTTP), 200 procedurally generated schemas (adversarial: optional params, enums, nested objects, shared-name tools).
- Explicit coverage quotas: parallel calls 15%, dependent chains 40%, error→retry 15%, no-tool-needed 10%, ≥32 tool registry 20%.

### 3.5 RL environments (100K+ episodes)

Sandboxed, verifiable environments:
1. **SQL console** over synthetic DBs (order/payment analytics) — reward = correct final answer via gold query.
2. **File/shell sandbox** — reward = filesystem end-state match.
3. **Simulated REST world** — 60 services with fault injection (429s, 500s, schema drift) for retry training.
4. **Multi-agent message board** — model orchestrates worker calls; reward = task graph completion.

Each episode logs: depth chosen, why-step count, tool calls, retries, success, token counts.

---

## 4. Training Execution

### 4.0 Stack

- Framework: Megatron-LM + custom data loader for interleaved protocols; RL with verl (GRPO).
- Precision: BF16; FP8 for ≥8K seq len if stability allows.
- Hardware assumption: 64× H100 node-cluster rental (M1), 16× H100 (M2–M4).
- Parallelism (M1): TP=8 within node is unnecessary at 2B — use **ZeRO-2 / FSDP, seq-parallel for long sequences**, global batch ~4M tokens, seq packing to 8K during main phase.

### 4.1 M1: Base pretraining (weeks 5–12)

| Phase | Tokens | Seq len | Notes |
|---|---|---|---|
| 1. Main | 1.7T | 8K | LR 3e-4 cosine → 3e-5, warmup 2K steps |
| 2. Long-context anneal | 240B | 8K→64K→256K curriculum | RoPE base already 500K; fine-tune attention at length |
| 3. Tool-protocol anneal | 60B | mixed | Raw `<tool_call>` format exposure so SFT isn't starting cold |

**Exit gates:** perplexity within 3% of best open 2B baseline on our eval mix; 256K passkey retrieval ≥ 99% at 256K; loss curves clean.

### 4.2 M2: SFT — why + tools (weeks 13–16)

- Data: 500K why-traces + 800K tool-traces + 200K general-instruction (prevent catastrophic narrowing; no why-block on these — model must learn when *not* to reason, FR-14 includes "no registry → plain answer").
- 3 epochs, LR 1e-5 → 5e-7, packed 16K sequences.
- **Loss masking:** why-steps and plan trained; tool-call JSON arguments trained; tool results masked.
- Include 15% "no-tool" turns mid-trajectory to train answer-without-call.
- **Exit gates:** BFCL ≥ 85% class-parity; Step-Count Exactness ≥ 98%; format-parse errors < 0.1%; depth accuracy ≥ 70% (pre-calibration).

### 4.3 M3: Agentic RL (weeks 17–22)

- Algorithm: GRPO on verl; group size 8 rollouts/prompt; KL anchor to M2 checkpoint (β tuned 0.01–0.05).
- Reward (PRD App. B):
  `R = 1.0·success + 0.2·schema_valid − 0.05·(reasoning_tokens/100) − 0.3·retries + bonuses`
  - +0.1 termination correctness (no premature final, no infinite loop)
  - +0.05·depth match to gold tier (only when counterfactual data indicates depth mattered)
  - −1.0 hallucinated tool call (hard penalty)
- Curriculum: env 1→2→3→4; fault-injection rate 0%→30% over stage.
- Turn cap 12 per episode (hard env limit), reward-zero beyond cap.
- **Anti-reward-hacking:** frozen holdout env variants (renamed tools, permuted schemas) evaluated weekly; ≥1% human audit of rollouts.
- **Exit gates:** internal suite ≥ 75%; termination violations < 1%; retry abuse (max retries hit) < 3%; holdout gap < 5 pts.

### 4.4 M4: Calibration + serving readiness (weeks 23–25)

1. Train depth-calibration head (Option B) on frozen M3 with cost-aware objective:
   `minimize E[cost(tier) · 1{tier≠gold}]` where cost(5) > cost(4) > cost(3) — biases toward under-selection exactly as much as efficiency allows; sweep operating points, pick threshold meeting "Success@depth ≤ 2% gap" (PRD §7.2).
2. If inline classifier (Option A) shows < 78% depth accuracy, distill calibrated routing into generation via brief preference stage (choose depth token per DPO pairs from counterfactual traces).
3. Quantization: GPTQ/AWQ INT4 + FP8; verify constrained-decoding compatibility.
4. Long-context KV optimizations: paged attention, optional KV cache quantization INT8.

---

## 5. Serving Stack

### 5.1 Server

- vLLM fork with three additions:
  1. **FSM grammar engine** for the why/tool protocol (logit masking per §2.3).
  2. Response extensions: `depth_used`, `why_trace` (optional include), `tool_retries`.
  3. `force_depth` request parameter.
- OpenAI Chat Completions + Responses-API compatible; tools passed as OpenAI `tools=[…]` JSON Schemas.
- Scratchpad stripping: `why_trace` excluded from conversation history echo but *retained in server-side KV across turns within a task* (FR-24) — why context persists for the task's duration even though users don't see it.

### 5.2 Latency engineering (targets: PRD §7.3)

| Tier | Budget breakdown (L4, FP8, ~1K-token prompt) |
|---|---|
| 3 whys | ~250 why-tokens + 1 call ≈ 1.2s TTFT-to-call |
| 5 whys | ~450 why-tokens + calls ≈ 2.2s |

Speculative decoding with a 350M draft model (same tokenizer) if tier-3 misses budget.

### 5.3 Deployment artifacts

- GGUF (Q4_K_M / Q5) for on-device; vLLM container (FP8/INT8) for server; ONNX export deferred to v2.

---

## 6. Evaluation Harness

### 6.1 Harness design

Single repo, three modes: `eval/quality` (static benchmarks), `eval/agent` (sandboxed rollouts), `eval/depth` (counterfactual depth studies). Every run emits a signed JSON report; CI gates milestones on these reports.

### 6.2 Gates per milestone

| Gate | M2 | M3 | M4 (release bar) |
|---|---|---|---|
| BFCL v3 | ≥ 85% | ≥ 88% | ≥ 90% |
| API-Bank L3 | ≥ 70% | ≥ 75% | ≥ 78% |
| Internal agent suite | — | ≥ 75% | ≥ 78% |
| Step-Count Exactness | ≥ 98% | ≥ 98% | 100% (constrained) |
| Depth Accuracy | ≥ 70% | ≥ 76% | ≥ 80% |
| Success@depth gap (tier-3 vs tier-5 on gold-LOW) | — | ≤ 4% | ≤ 2% |
| Termination violations | — | < 1% | < 0.5% |
| GSM8K (regression guard) | ≥ baseline − 2 pts | hold | hold |
| 256K passkey / needle | ≥ 95% | ≥ 98% | ≥ 99% |
| P50 tier-3 latency (L4 FP8) | — | — | ≤ 1.2s |

### 6.3 Depth counterfactual study (new eval, run at M3 & M4)

For a 500-task stratified set: run every task at depths 3, 4, 5 (forced). Produces the **depth-response curve** — the empirical justification for the whole product thesis. Deliverable: public chart in the model card. If curves are flat (depth doesn't matter), escalate to steering committee — this is the program's biggest scientific risk.

---

## 7. Schedule & Staffing

### 7.1 Timeline (27 weeks)

| Weeks | Milestone | Key deliverables |
|---|---|---|
| 1–4 | M0 | Data infra live; 50K validated why-traces; arch frozen; eval harness v0; teacher prompts tuned |
| 5–12 | M1 | Base 2B checkpoint; 256K long-context anneal; exit evals pass |
| 13–16 | M2 | SFT checkpoint; BFCL ≥ 85%; protocol FSM drafted |
| 17–22 | M3 | RL checkpoint; 4 sandbox envs; depth counterfactual v1 |
| 23–25 | M4 | Calibration head; quantized builds; vLLM fork feature-complete; latency targets met |
| 26–27 | M5 | Beta: weights, model card with depth-response curves, integration guide, eval reports |

**Parallel tracks (weeks 1–25):** eval harness maturation; tool-zoo expansion; red-team on prompt injection via tool results.

### 7.2 Staffing

| Role | FTE | Phase |
|---|---|---|
| Research scientist (training lead) | 1 | Full |
| ML engineers (data/training/RL) | 3 | Full |
| Infra engineer (cluster, storage, CI) | 1 | Full |
| Eval engineer | 1 | Weeks 1–27 |
| Serving engineer (vLLM fork) | 1 | Weeks 8–27 |
| PM / part-time security review | 0.5 | Full |

### 7.3 Compute budget (rental, indicative)

| Stage | Config | Duration | Est. cost |
|---|---|---|---|
| M1 pretrain | 64× H100 | 8 wks | $1.1–1.4M |
| M2 SFT | 16× H100 | 4 wks | $80–120K |
| M3 RL | 16–32× H100 + CPU sandboxes | 6 wks | $200–300K |
| M4 + eval + contingency | 8× H100 | 5 wks | $60–90K |
| **Total** | | | **~$1.5–1.9M** |

---

## 8. Risk Register (engineering view)

| # | Risk | Trigger/leading indicator | Mitigation owner |
|---|---|---|---|
| R1 | Depth-response curves flat → thesis weak | M3 counterfactual interim results | Training lead; pivot to fixed-4 default + expose as API knob |
| R2 | Teacher why-traces too templated | Format-drift eval drop on held-out phrasings | Data lead; template diversification quota |
| R3 | 256K training instability / memory blowup | Loss spikes in long-context anneal | Infra; seq-parallel, chunked attention fallback |
| R4 | RL reward hacking | Holdout-env gap > 5 pts | RL engineer; reward shaping, audit sampling |
| R5 | Constrained decoding breaks quantized kernels | M4 integration tests | Serving engineer; grammar on dequantized logits path |
| R6 | Tool-result prompt injection | Red-team findings | Security review; tool-result sandbox framing training data |
| R7 | 2B ceiling on 64-tool selection | Registry-scale eval misses | Retrieval-based tool filtering recipe in integration guide |

---

## 9. Definition of Done (Beta, M5)

- All P0 FRs verified by eval harness reports committed to repo.
- Release bar (§6.2, M4 column) fully met.
- Public artifacts: weights (license per PRD open question), model card with depth-response curves, vLLM server image, quickstart (LangGraph + OpenAI-compat examples), eval reproduction commands.
- Go/no-go review with steering committee; sign-off on open questions from PRD §10.
