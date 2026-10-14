# Distillation Plan

## Qwen 3.5 4B Teacher → Why5 2B Adaptive-Depth Agentic Model

| Field | Value |
|---|---|
| Document Version | 1.0 |
| Date | 2026-10-14 |
| Depends On | PRD v0.1; TECHNICAL_IMPLEMENTATION_PLAN.md; SYNTHETIC_DATA_PLAN.md; DATA_PRD_2B_Token_Requirement.md |
| Teacher | Qwen 3.5 4B (dense), served via vLLM |
| Student | Why5 2B (Main Plan §2.1) |
| Status | Ready for Engineering Review |

---

## 1. What "distilling the 5-Why principle" means here

The 5-why behavior decomposes into three separable learning targets, and they distill differently:

| Target | Source of truth | Distilled from teacher? |
|---|---|---|
| **Depth routing** (3/4/5 tier choice) | Factory rubric (deterministic, exact) | **No** — labels come from seeds (FR-A2). The teacher's own routing is only used as a weak signal/filter. |
| **Why-step content** (the reasoning itself) | Teacher generations, gate-filtered | **Yes** — response-based sequence distillation |
| **Tool competence** (schema-valid calls, ordering, retry judgment) | Teacher + sandbox verification | **Yes** — teacher proposes, sandbox disposes (G3–G5) |

**Core principle: the student never learns step counts from the teacher.** Counts are structural ground truth from the rubric; the teacher supplies the *content* that fills the 3/4/5 skeleton. This is what makes the depth controller exact rather than imitated.

---

## 2. Teacher roles (three jobs, one model + one escalation path)

| Role | Where used | Notes |
|---|---|---|
| **R1. Bulk trace teacher** | D1/D2/D3/D4/D6 generation in the Factory | Qwen 3.5 4B replaces the 70B-class default for the bulk of the 2B tokens. Teacher-pluggable `TraceGenerator` (already built) takes it as the `teacher` callable. |
| **R2. Escalation teacher** | Gold-MAX (tier-5) traces, D5 label verification | 4B ceiling risk on hardest tier → route tier-5 seeds (and D5 dual-teacher checks, FR-B4) to a larger model (Qwen 3.5 14B+ or the original 70B plan). Est. ~20% of teacher spend. |
| **R3. On-policy corrector** | Stage 4 below | Reranks/corrects *student* rollouts to build SFT/DPO pairs — the single highest-leverage distillation technique for agentic behavior. |

**Why this teacher is attractive:** 4B on vLLM serves ~15–20× the throughput of a 70B on the same fleet. Bulk-teacher cost drops from ~$200K to ~$15–25K; total D5/escalation spend keeps the corpus program well under the original NFR-2 ($275K) ceiling. It also leaves headroom to spend more on RL rollouts (M3), which are the real bottleneck.

---

## 3. Distillation strategy: response-based first, logit-based optional

### 3.1 Primary: response-based (sequence-level) KD — recommended

Student SFTs on gate-validated teacher traces in the token protocol. This is already the M2 plan; nothing new to build.

- **Pros:** tokenizer-agnostic, works with the existing Factory + gates, verified-correct by construction (we only train on sandbox-passing traces — the teacher's errors never enter the corpus).
- **Cons:** throws away teacher confidence/dark knowledge.

### 3.2 Optional stage: on-policy logit KD — only if M2 gates miss

White-box token-level KL on teacher logits during SFT. Prerequisites and caveats:

- **Vocab alignment:** Qwen tokenizer (~151K) vs. student vocab (131K, Main Plan §2.1). Two options:
  - **Option L1 (recommended if logit KD is pursued):** switch the student to the Qwen tokenizer at M0. Same family as teacher → exact token-ID alignment, no projection layer, and better pretraining mixture compatibility with Qwen-style corpora. Requires updating Main Plan §2.1 (one-line change at M0, architecture not yet frozen).
  - **Option L2:** token-mapping matrix / vocabulary projection head; fiddly, only worth it if tokenizer must stay Llama-3-style.
- **What to distill:** KL on the *constrained* positions only (tool-call argument tokens, why-step text). Never KL on protocol control tokens (`<|depth_N|>` — those are rubric-supervised).
- **Cost:** one extra teacher forward pass per SFT sequence; at 4B that's cheap even at 2B-token scale.

**Decision rule:** run M2 with response-based KD against the §6.2 gates (BFCL ≥ 85%, Step-Count Exactness ≥ 98%, depth accuracy ≥ 70%). Add logit KD only for the specific failing axis.

---

## 4. Pipeline (five stages)

### Stage 0 — Teacher preparation (week 1–2, parallel with Factory infra)

1. Serve Qwen 3.5 4B on 1–2× H100 with vLLM (fits easily; target ≥ 8K tok/s aggregate).
2. **Format adapter:** build the prompt-side contract that maps Factory seed specs + tool registry into the teacher's native tool-call template (Qwen uses Hermes-style tool templates), and parse its outputs back into the Factory's `protocol_turns` structure. This is the only new glue code.
3. **Teacher calibration run:** 2K seeds across tiers → measure per-tier gate pass rates. Exit criteria: ≥ 60% acceptance on tiers 3–4; if tier-5 acceptance < 40%, activate escalation teacher for tier 5 permanently (R2).

### Stage 1 — Bulk trace distillation (weeks 3–8)

Factory as built, with Qwen 3.5 4B as R1:

- Seeds → 4 teacher candidates → gates G1–G10 → accepted traces (plan §3.2–3.3 unchanged).
- **Teacher-routing filter (new, cheap):** log the teacher's *own* depth choice before revealing the rubric label; traces where teacher and rubric disagree by > 1 tier are flagged and 100% human-audited (sample) rather than rejected — disagreement data feeds the depth-calibration analysis.
- Tier quotas (40/40/20) enforced at storage as in the pilot.

### Stage 2 — Escalation for hard traces (weeks 5–8, overlapping)

- All gold-MAX seeds and all D5 verification passes go to the escalation teacher (70B or Qwen 3.5 14B+).
- Budget guard: escalation share ≤ 20% of teacher tokens; if it exceeds, revisit the tier-5 rubric (are we over-injecting `retry_reasoning`?).

### Stage 3 — Student SFT on distilled corpus (weeks 13–16 = M2)

Unchanged from Main Plan §4.2 (loss masking, no-tool mix, 3 epochs), with two additions:

- **Anti-imitation weighting:** upweight the 15% no-tool/refusal traces by 1.5× — small teachers under-generate refusal, and the student will inherit that bias if uncorrected (FR-14).
- Length control: keep the median-length candidate selection (plan FR-B2) so the 2B student doesn't inherit 4B verbosity.

### Stage 4 — On-policy correction distillation (weeks 13–18, overlapping M2/M3)

The step that matters most for agentic behavior:

1. Student (M2 checkpoint) rolls out on Factory seeds it has **never** seen in training (fresh seed stream, same rubric).
2. Rollouts scored by gates G1–G10 + sandbox execution (no teacher needed for scoring — the environment is the judge).
3. For failed rollouts: teacher generates a corrected trace for the same seed → **(failed student, corrected teacher) preference pairs**.
4. Pairs feed a DPO stage before M3 RL, and the same pairs seed GRPO prompts in M3.

This fixes the classic small-model failure mode (malformed args, skipped verification steps) at exactly the states where the student actually errs, rather than where the teacher happens to sample.

### Stage 5 — RL with teacher-as-verifier bootstrap (weeks 17–22 = M3)

- GRPO reward stays environment-grounded (Main Plan §4.3) — **never** a raw teacher score (reward hacking surface).
- Limited teacher use: early RL, when sandbox reward is sparse, teacher agreement is a small auxiliary term (+0.05) decayed to zero by 30% of training.

---

## 5. What is *not* distillable (guardrails)

1. **Step counts** — rubric-only, FSM-enforced at serve time (§2.3 of Main Plan). A teacher that "usually does 4 whys" is irrelevant.
2. **Termination judgment** — RL on environment signal; teacher imitation here produced loops in pilot-scale probes and in published agentic-distillation reports. Treat teacher termination behavior as noise.
3. **Injection resistance (D8)** — train from template-labeled correct behavior, never from teacher completions: a 4B teacher is itself injection-susceptible, and distilling its compliance would train the vulnerability in.

---

## 6. Compute & cost delta vs. the 70B plan

| Item | 70B plan | Qwen 3.5 4B plan | Delta |
|---|---|---|---|
| Bulk teacher fleet | 8× H100, 10 wks, ~$210K | 2× H100, 6–8 wks, ~$20–30K | **−$180K** |
| Escalation teacher (tier-5 + D5 verify) | included | 4× H100 shared, ~$40–60K | new but bounded |
| On-policy correction (Stage 4 rollouts + corrections) | not planned | ~$15–25K | new |
| RL compute freed | — | reinvest ~$100–150K into M3 rollouts | reallocation |
| **Corpus program total** | ~$210–275K | **~$180–240K** | roughly neutral, higher quality ceiling |

---

## 7. Evaluation gates (unchanged targets, added teacher diagnostics)

All Main Plan §6.2 gates apply unchanged to the student. Added teacher-side dashboards:

| Diagnostic | Target |
|---|---|
| Teacher acceptance per tier (Stage 0 calibration) | ≥ 60% tiers 3–4; ≥ 40% tier 5 else escalate |
| Teacher–rubric routing agreement | tracked per tier; disagreement > 25% on any tier triggers rubric re-audit |
| On-policy correction yield | ≥ 50% of failed student rollouts yield a valid teacher correction |
| Post-DPO repeat-failure rate | student fails the same seed ≤ 20% of pre-DPO rate |

---

## 8. Risks

| # | Risk | Mitigation |
|---|---|---|
| DS-1 | 4B teacher ceiling on tier-5 reasoning quality | Escalation teacher R2; Stage 0 calibration decides the split empirically |
| DS-2 | Student inherits teacher's tool-selection biases | Sandbox-verified filtering (G4/G5); on-policy correction targets student's own error states |
| DS-3 | Format drift between Qwen tool templates and our protocol | Stage 0 format adapter + G1 parse gate on every trace |
| DS-4 | Teacher-generated why-steps shorter/shallower than 70B output | Template entropy + human audit score gates (§7 of Data PRD); escalation for shallow tier-5 |
| DS-5 | Tokenizer switch (L1) discovered late to break something | Decide at M0 freeze; fallback is Option L2 or pure response-based KD |
| DS-6 | Reward hacking via teacher auxiliary term in RL | Decay to zero by 30% of M3; environment reward always dominant |

---

## 9. Decisions needed at M0

1. **Tokenizer:** adopt Qwen tokenizer for the student (enables Option L1 logit KD + ecosystem alignment) — recommended yes; revisit Main Plan §2.1.
2. **Escalation teacher identity:** 70B-class (original plan) vs. Qwen 3.5 14B+ (same family, simpler parsing) — recommendation: same-family escalation.
3. Whether Stage 4 DPO runs before M3 RL (recommended) or is folded into GRPO as initialization pairs.
