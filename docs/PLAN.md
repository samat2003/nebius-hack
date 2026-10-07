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

## Phase 2 — Provider runtime

Implement a shared remote-provider execution layer with:

- deadlines;
- bounded retry;
- circuit breaker;
- per-provider concurrency limits;
- normalized errors;
- usage/cost extraction;
- tracing.

Add provider adapters for:

- EmbeddingGemma 2;
- mini-Jev;
- NVIDIA Nemotron through Nebius Token Factory.

### Quality gate

Fault tests cover timeout, rate limit, 5xx, malformed responses, and unavailable optional providers.

---

## Phase 3 — Grounding and canonical capabilities

Normalize harness-specific tools into canonical capabilities.

Extract from history/tool results:

- file paths;
- symbols;
- test names;
- commands;
- stack traces;
- failures;
- mutation/verification state.

Construct complete grounded CandidateActions where possible.

### Primary metric

**Oracle Candidate Recall@K**

### Quality gate

On a manually labeled decision set, the candidate engine retains an acceptable next action at high recall before mini-Jev is allowed to control production decisions.

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
