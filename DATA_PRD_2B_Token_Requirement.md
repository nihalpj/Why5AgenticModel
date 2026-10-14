# Product Requirements Document (PRD)

## 2-Billion-Token Synthetic Data Requirement — Why5 Training Corpus

| Field | Value |
|---|---|
| Product Name | Why5 Synthetic Data Factory (the "Factory") |
| Version | 1.0 |
| Date | 2026-10-14 |
| Status | Approved for Build (pilot validated) |
| Depends On | PRD_Adaptive_Why_Agentic_Model.md; TECHNICAL_IMPLEMENTATION_PLAN.md §3–4; SYNTHETIC_DATA_PLAN.md (implementation) |
| Consumers | M1 long-context anneal, M2 SFT, M3 RL prompt seeds, M4 calibration head |

---

## 1. Summary

This PRD formalizes the requirement to produce **2,000,000,000 (2B) tokens** of synthetic, machine-validated training data for the Why5 model. The data must be delivered as **JSONL shards with per-record validation evidence**, in the model's native token protocol, across **eight dataset classes (D1–D8)** with exact distribution quotas, full provenance, and bit-for-bit reproducibility. Every token shipped to training must have passed all ten validation gates (G1–G10).

**One-line requirement:** *2B tokens, zero unvalidated, fully reproducible, on schedule for the M2 SFT freeze (week 12).*

---

## 2. Background & Problem Statement

Why5's differentiators — the 3/4/5 why-depth controller, tool-calling reliability, and termination judgment — cannot be learned from public corpora: no dataset pairs queries with gold depth labels and exact-step reasoning traces. The training program (M1–M4) is therefore **blocked on a purpose-built corpus** with properties that only controlled synthesis can guarantee:

1. **Exact supervision**: gold depth (3/4/5), call graphs, and end-states known before generation, so labels are derived, not annotated.
2. **Executable verification**: every trajectory replayable in a sandbox to prove the tool calls actually succeed.
3. **Counterfactual depth signal**: tasks regenerated at wrong depths, because the calibration head and depth-response eval have no other training/eval signal.
4. **Scale within budget**: 2B tokens is the size at which M2 SFT reaches its BFCL ≥ 85% gate without repeating epochs into memorization, at ≤ $275K teacher compute.

A prior pilot build (see §11) validated the factory architecture end-to-end at 680 records with a 54% acceptance rate.

---

## 3. Goals & Non-Goals

### Goals
1. Deliver **2.0B tokens** across D1–D8, frozen and immutable, by **week 12** (M2 SFT start).
2. **100% of records pass gates G1–G10** before entering a shard; no exceptions, no manual waivers.
3. Gold depth labels **exact by construction** (deterministic rubric over seed features), with ambiguous seeds discarded (< 10% discard rate).
4. Distribution quotas enforced at storage time: 40/40/20 tier split (D1), registry-size buckets (D2), domain caps ≤ 15%.
5. Full reproducibility: recipe registry + seeds → identical dataset SHA-256 hashes.
6. Deliver D5 counterfactual labels (degraded/survived/wasteful) with ≥ 90% dual-teacher agreement on audit samples.

### Non-Goals
- Real-web pretraining data curation (covered by Main Plan §3.2).
- RL environment engineering (Main Plan §3.5) — Factory only supplies RL prompt seeds.
- Human preference data for DPO beyond the counterfactual pairs (D5).
- Multilingual synthesis beyond English in v1 (matches model v1 scope).

---

## 4. The 2B-Token Requirement (normative)

### 4.1 Token budget by dataset class

| # | Class | Tokens (min) | Records (≈) | Avg tokens/record | Consumer | Format |
|---|---|---|---|---|---|---|
| D1 | Why-traces at gold depth (3/4/5) | 450M | 500K | ~900 | M2 SFT | JSONL |
| D2 | Tool-calling trajectories | 500M | 800K | ~625 | M2 SFT | JSONL |
| D3 | Error → recovery traces | 100M | 150K | ~670 | M2 SFT | JSONL |
| D4 | No-tool / plain instruction | 100M | 200K | ~500 | M2 SFT | JSONL |
| D5 | Counterfactual depth (task × wrong depth) | 150M | 300K | ~500 | M3 reward, M4 calibration | JSONL |
| D6 | Multi-turn agent episodes (10–25 turns) | 200M | 40K | ~5,000 | M2 SFT + M3 prompts | JSONL |
| D7 | Long-context 64K–256K sessions | 400M | ~8K sessions | ~50,000 | M1 anneal + M2 | JSONL |
| D8 | Prompt-injection defense traces | 100M | 150K | ~670 | M2 SFT | JSONL |
| | **Total** | **2,000M** | **~2.15M** | | | |

### 4.2 Hard constraints on the budget
- **R-1**: No class may ship under budget by more than 5% without steering-committee sign-off; surplus classes may not substitute for deficit classes (mix design is normative).
- **R-2**: Token counts are measured by the production tokenizer (131K vocab), not character proxies, at dataset freeze.
- **R-3**: ≤ 8% near-duplicate content per class (G10 measured); ≤ 0.5% benchmark contamination (G8).
- **R-4**: All 2B tokens must originate from records with `validation.passed = true`; loaders must reject anything else (§3.7.4 of the Data Plan).

---

## 5. Functional Requirements

### 5.1 Seed & label requirements
| ID | Requirement | Priority |
|---|---|---|
| FR-A1 | Seeds are structured task specs (intent × call graph × faults × distractors), never free text | P0 |
| FR-A2 | Gold depth = deterministic function of seed features (rubric); ambiguous seeds discarded, not soft-labeled | P0 |
| FR-A3 | Seed space large enough that G10 dedup rejection stays < 8% at full scale (≥ 2,000 intents × parameterized entities) | P0 |
| FR-A4 | Tool zoo ≥ 400 schemas (200 real-style + 200 procedural), versioned; each schema passes the self-describability test ≥ 95% | P0 |
| FR-A5 | Every dataset version pins its toolzoo version | P0 |

### 5.2 Generation requirements
| ID | Requirement | Priority |
|---|---|---|
| FR-B1 | Why-step phrasing drawn from ≥ 6 families; template entropy ≥ 0.55 (anti-collapse) | P0 |
| FR-B2 | 4 candidates sampled per seed; median-length passing candidate kept (verbosity-bias control) | P1 |
| FR-B3 | G4–G5 failures get one self-correction regeneration with the validation error appended, then seed re-qualification | P1 |
| FR-B4 | D5: forced depths = gold ±1/−2; outcome labels (degraded/survived/wasteful) verified by second teacher pass ≥ 90% agreement | P0 |
| FR-B5 | D7: critical facts/args uniformly distributed across 0–256K context positions (no positional shortcut) | P0 |
| FR-B6 | D8: ≥ 500 distinct injection patterns; correct behavior (treat tool output as data) is the only accepted label | P0 |
| FR-B7 | Teacher-pluggable generation API: template pilot and 70B teacher interchangeable without downstream changes | P1 |

### 5.3 Validation & storage requirements
| ID | Requirement | Priority |
|---|---|---|
| FR-C1 | Gates G1–G10 executed per record; all must pass; results stored on the record | P0 |
| FR-C2 | Output = JSONL, one record/line, UTF-8, ≤ 64KB/line, shard-hashed manifest (SHA-256), train/val/holdout splits by record ID | P0 |
| FR-C3 | Records carry full provenance: seed_id, recipe id+version, teacher id+params, toolzoo version, gate results, timestamps, shard_id | P0 |
| FR-C4 | Shards immutable after freeze; corrections produce a new dataset version, never an edit | P0 |
| FR-C5 | Reference reader validates schema and rejects `validation.passed != true` | P0 |
| FR-C6 | Class-specific gate relaxation only as specified (D4 skips G3–G5 + adds refusal check; D5 accepts labeled failure; D8 requires injection resistance) | P1 |

---

## 6. Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-1 | Throughput | ≥ 250K validated records/week sustained (weeks 5–10) |
| NFR-2 | Cost ceiling | ≤ $275K total (teacher + sandboxes + orchestration + annotation) |
| NFR-3 | Reproducibility | Rerun of any recipe → identical shard hashes |
| NFR-4 | Decontamination | 8-gram MinHash + embedding double-check vs. all benchmark holds |
| NFR-5 | Human audit | 200 samples/week, quality ≥ 4.0/5, annotator κ ≥ 0.7 |
| NFR-6 | Observability | Daily dashboard: acceptance rate, gate failure breakdown, $/accepted-token per recipe |
| NFR-7 | Security | No real PII/secrets (G9); injection corpus reviewed by security before training use |

---

## 7. Acceptance Criteria (dataset freeze, week 12)

The 2B-token corpus is **accepted** when all of the following hold:

1. [ ] Token totals per §4.1 verified with production tokenizer (R-2), per class, from manifest counts.
2. [ ] 100% of shipped records have `validation.passed = true` (FR-C1/C5).
3. [ ] D1 tier split = 40/40/20 ± 2 pts; D2 registry buckets 25/35/25/15 ± 3 pts; no domain > 15% anywhere.
4. [ ] G10 dup rate < 8%, G8 contamination < 0.5%, template entropy ≥ 0.55.
5. [ ] D5 label agreement ≥ 90% on audited sample; depth-outcome distribution reported (no class empty).
6. [ ] Manifest + splits published; spot re-run of 3 recipes reproduces shard hashes exactly (NFR-3).
7. [ ] Human audit ≥ 4.0/5 with κ ≥ 0.7 for two consecutive weeks.
8. [ ] Pilot-to-production deltas documented (acceptance rate, gate failure mix).

---

## 8. Delivery Schedule (aligned to Main Plan)

| Weeks | Requirement milestone | Exit |
|---|---|---|
| 1–2 | Factory infra + tool zoo v1.0 | 5K validated records/day |
| 3–4 | **50K validated D1 traces (M0 exit)** | Gates live, dashboard live |
| 5–7 | D1+D2 full rate; D3/D4 streams | 700M tokens accepted |
| 8–9 | D5 counterfactuals; D6 episodes; D7 long-context | 1.5B tokens accepted |
| 10–12 | D8; final dedup/decont; **freeze for M2** | All §7 criteria pass |
| 13+ | Maintenance: D5/D8 refresh for M3/M4 | On demand |

---

## 9. Dependencies & Stakeholders

| Dependency | Owner | Needed by |
|---|---|---|
| 70B-class teacher fleet (8× H100) | Infra | Week 3 |
| Tool zoo normalization (200 real schemas) | Data Eng | Week 2 |
| Benchmark holdout register (for G8) | Eval Eng | Week 1 |
| Sandbox fleet (SQLite/shell/REST-sim) | Infra | Week 3 |
| Security review of D8 corpus | Security | Week 10 |
| Consumers: M2 SFT loader, M4 calibration trainer | Training | Week 12 |

---

## 10. Risks

| # | Risk | Requirement impact | Mitigation |
|---|---|---|---|
| PR-1 | Template collapse → training on boilerplate | Violates FR-B1; M2 format-drift | Entropy metric gates freeze (§7.4); family rotation |
| PR-2 | Seed-space exhaustion at scale | G10 > 8% blocks freeze (§7.3) | Intent × entity expansion; procedural tools |
| PR-3 | Teacher systematic depth bias | D5 labels unreliable (FR-B4) | Dual-teacher cross-check; rubric re-audit |
| PR-4 | Benchmark leakage | NFR-4 breach; eval integrity | Double-check decontamination; holds never enter factory |
| PR-5 | Cost overrun | NFR-2 breach | $/accepted-token weekly review; fail-fast seed rejection |
| PR-6 | Synthetic–real distribution gap | M2 eval gap | 15% real-data SFT mix (Main Plan); per-source eval deltas |

---

## 11. Current Status (pilot evidence)

The Factory pilot (this repo, `synth_factory/`) validated the full architecture:

- **680 records accepted** (D1 400 @ 40/40/20 quota · D3 80 · D5 200), 100% passing G1–G10, JSONL + manifest + splits exactly per FR-C2/C3.
- **Acceptance rate 54%** (plan band: 40–75%); dominant rejection = G10 dedup from the small 10-intent pool → confirms FR-A3 scale requirement.
- Sandbox ref-parsing bug found and fixed during pilot (free-text placeholders misread as argument references) — argues for FR-C1's strict per-record gates.
- Known pilot deltas vs. this PRD: template generation instead of 70B teacher (FR-B7 allows swap); classes D2/D4/D6/D7/D8 not yet implemented; token metrics use a chars/4 proxy (R-2 applies at freeze).

---

## 12. Open Questions

1. Exact dual-teacher configuration for FR-B4 verification (same model, different seed vs. different model family)?
2. D7 storage: one JSONL line per 256K session approaches the 64KB line limit — per-session sidecar files vs. raised limit vs. compressed multi-line convention?
3. Should 5% of D2 registries include *duplicated-name* tools (same name, different schema) in v1, or defer to v2?
4. Charge-back model: does the Factory budget absorb M3 prompt-seed regeneration during RL (weeks 17–22)?
