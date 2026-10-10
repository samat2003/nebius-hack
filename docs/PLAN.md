# Alienese Implementation Plan

This plan intentionally starts with **no new model training**.

The goal is to build a reliable, observable runtime first, collect real decision traces, and only then decide how mini-Jev-SWE should be fine-tuned.

## Phase 0 — Repository and contracts

### Deliverables

- repository engineering rules;
- architecture document;
- public license;
- environment contract;
- package skeleton;
- typed core contracts;
- fake providers;
- CI foundation.

### Quality gate

No live API key is required to run the test suite.

---

## Phase 1 — Protocol, events, and replayable state

Implement:

- OpenAI-compatible request/response subset;
- event normalization;
- WorkingState reconstruction;
- typed errors;
- request/trace ids;
- idempotency;
- deterministic replay fixtures;
- structured logging;
- OpenTelemetry hooks.

### Quality gate

The same normalized event history reconstructs the same WorkingState deterministically.

Restarting the service must not lose correctness.

---

## Phase 2 — Provider runtime (Completed)

Implement a shared remote-provider execution layer (`src/alienese/providers/runtime/`) with:

- end-to-end turn deadlines (`RequestContext.deadline_monotonic` + `DeadlineBudget`);
- safe-to-retry vs. ambiguous-completion bounded retry (`RetryConfig`);
- per-attempt circuit breaker with single-probe `HALF_OPEN` admission (`ProviderCircuitBreaker`);
- per-provider concurrency and waiter queue limits (`ProviderConcurrencyLimiter`);
- pre-allocation outbound JSON depth/byte caps and streaming response byte caps (`ProviderHttpClient`);
- origin-scoped HTTPS credentials (`follow_redirects=False`, `trust_env=False`);
- normalized errors and truthful nullable usage/cost extraction (`ProviderCallTelemetry`);
- tracing.

Add provider adapters for:

- NVIDIA Nemotron 3 Super through NVIDIA API Catalog (`NvidiaBuildGenerator`, `LIVE_VERIFIED`);
- NVIDIA Nemotron through Nebius Token Factory (`NebiusTokenFactoryGenerator`, `PENDING_CREDENTIALS` / `MOCK_VERIFIED`);
- EmbeddingGemma 2 (`EmbeddingGemmaRetriever`, `BLOCKED_HOSTING` / `MOCK_VERIFIED`);
- `samatv256/mini-Jev` pinned at `step-010626` (`MiniJevController`, `BLOCKED_HOSTING` / `MOCK_VERIFIED`).

### Quality gate

Fault tests cover timeout, deadline exhaustion, cancellation, rate limit (`429` + `Retry-After`), `5xx`, circuit-breaker single-probe `HALF_OPEN` recovery, oversized payloads, redirect rejection, malformed responses, and behavioral parity between `NvidiaBuildGenerator` and `NebiusTokenFactoryGenerator`. Default test suite runs 100% offline.

---

## Phase 3 — Grounding and canonical capabilities (Completed)

Normalize harness-specific tools into canonical capabilities and extract grounded evidence from history and tool results.

Implemented deliverables:
- `src/alienese/grounding/evidence.py`: Immutable typed evidence records (`GroundingEvidence`) across 10 categories (`FILE_PATH`, `SYMBOL`, `TEST_TARGET`, `TEST_COMMAND`, `SEARCH_PATTERN`, `STACK_FRAME`, `FAILURE_MESSAGE`, `EXIT_STATUS`, `MUTATION_TARGET`, `VERIFICATION_RESULT`).
- `src/alienese/grounding/normalization.py`: Sanitization and bounds enforcement for paths, symbols, test targets, and deduplication with trust hierarchy (`SYSTEM_PROMPT` > `USER_DIRECTIVE` > `AGENT_COMMITTED` > `UNTRUSTED_EXTERNAL`).
- `src/alienese/grounding/extractors/`: Modular deterministic extractors with hard resource bounding (32KB content cap, 50 event scan cap, 100 evidence record cap).
- `src/alienese/grounding/argument_resolution.py`: `GroundedArgumentResolver` mapping JSON schemas to evidence parameters without fabricating paths or arguments.
- `src/alienese/grounding/ranking.py`: Deterministic scoring and candidate bounding ($K=8$), prioritizing user-directed requests, failures, and unverified mutations.
- `src/alienese/grounding/policy.py`: Strict enforcement of `tool_choice` policies (`none`, named tool, `required` restricted to low-risk tools, `auto`).
- `src/alienese/grounding/eval.py`: Offline evaluation CLI benchmarking candidate sets against golden decision points.
- `tests/fixtures/grounding/decision_points.json`: Curated dataset with 35 labeled coding decision points across train (18), dev (7), and held-out test (10) splits.

### Primary metric

**Oracle Candidate Recall@K**

### Measured results

| Metric | Phase 1 Baseline | Phase 3 Engine | Lift |
| --- | --- | --- | --- |
| **Held-Out Test Oracle Recall@1** | 10.00% | **90.00%** | **+80.00%** |
| **Held-Out Test Oracle Recall@4** | 10.00% | **90.00%** | **+80.00%** |
| **Held-Out Test Oracle Recall@8** | 10.00% | **90.00%** | **+80.00%** |
| **Held-Out Test Arg Completeness** | 100.00% | **100.00%** | +0.00% |
| **Held-Out Test Executable Validity** | 100.00% | **100.00%** | 0.0% |
| **Held-Out Test Fabrication Count** | 0 | **0** | 0 |
| **Overall (N=35) Oracle Recall@1** | 17.14% | **82.86%** | **+65.72%** |
| **Overall (N=35) Oracle Recall@4/8**| 17.14% | **85.71%** | **+68.57%** |

### Quality gate

Passed: On the 35-point curated decision set, Oracle Recall@1 reached 90.00% on held-out test (82.86% overall) with zero fabrications and 100% executable validity before mini-Jev is allowed to control production decisions.

---

## Phase 4 — mini-Jev shadow mode

mini-Jev receives real WorkingState + CandidateActions but does not control the output.

Record:

- candidate scores;
- selected action;
- baseline action;
- eventual task outcome;
- score margin / entropy;
- disagreement categories.

### Quality gate

We understand where the current checkpoint succeeds and fails on real coding decisions.

No fine-tuning yet.

---

## Phase 5 — Guarded policy control

Enable mini-Jev only for decisions that satisfy configured safety/uncertainty rules.

Add:

- deterministic bypasses;
- loop detection;
- mutation -> verification obligation;
- EXPAND_SEARCH / DEFER meta-actions;
- controller fallback;
- budget enforcement.

### Quality gate

Guarded mode never silently forces a low-confidence candidate when the action space is inadequate.

---

## Phase 6 — Context engine and EmbeddingGemma

Begin in shadow mode.

Measure:

- required-evidence recall@K;
- context token reduction;
- candidate survival rate;
- latency and API cost.

Only activate pruning after demonstrating that mandatory/relevant evidence survives.

### Quality gate

Retrieval improves cost/context size without materially hurting end-to-end correctness.

EmbeddingGemma remains optional if measurements do not justify it.

---

## Phase 7 — Typed Nemotron generation

Implement typed GenerationJobs:

- EXPLAIN_FAILURE;
- GENERATE_PATCH;
- WRITE_TEST;
- SYNTHESIZE_SEARCH;
- ANSWER.

Use narrow context packets and structural output validation.

Do not use Nemotron as a generic argument filler when an action can be grounded.

### Quality gate

Generation jobs are independently testable and trace exactly why the generator was invoked.

---

## Phase 8 — End-to-end benchmark

Freeze:

- task corpus;
- harness;
- tool set;
- timeout/max-turn policy;
- system version.

Run ablations:

1. generator-only baseline;
2. generator + deterministic context/grounding;
3. + mini-Jev;
4. + EmbeddingGemma.

Later add external decision-model baselines if useful.

Measure:

- solve rate;
- tests passing;
- cost per solved task;
- generator tokens per solved task;
- all-provider cost;
- wall-clock time;
- provider/model calls;
- tool calls;
- repeated actions;
- fallback rate;
- context reduction.

### Quality gate

Claims in README/demo must be generated from reproducible benchmark data.

---

## Phase 9 — mini-Jev-SWE training

Only now classify observed policy failures, for example:

- premature generation;
- unnecessary re-read;
- wrong test selection;
- weak recovery after failed patch;
- premature finish;
- failure to expand search;
- poor context-retention decisions.

Create training/evaluation data from real Alienese decision points.

First validate the new checkpoint through offline replay, then rerun end-to-end tasks.

### Quality gate

mini-Jev-SWE must improve replay and/or end-to-end metrics without hiding increased fallback or compute costs.

---

## Phase 10 — Hackathon hardening and demo

Prepare:

- deterministic setup instructions;
- public hosted endpoint/demo;
- reproducible benchmark report;
- trace visualization;
- failure/recovery demo;
- architecture diagram;
- model/provider attribution;
- Nebius/Nemotron integration explanation;
- explicit description of work created during the hackathon;
- feedback on Nebius/NVIDIA tooling.

The demo should show the system making and recovering from real decisions, not only a successful final patch.
