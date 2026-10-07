# Alienese Architecture

## 1. Purpose

Alienese is a unified inference runtime exposed as a single OpenAI-compatible API.

Its purpose is to decompose a coding agent's model-facing workload into specialized responsibilities:

- retrieve and reduce context;
- decide the next action;
- generate or reason only when open-ended synthesis is necessary.

The initial system uses three remote model providers behind API keys:

```text
Coding harness
      |
      | OpenAI-compatible request
      v
+--------------------------------------+
|               Alienese               |
|                                      |
| protocol -> events -> working state  |
|               |                      |
|        grounding / context           |
|               |                      |
|        candidate construction        |
|               |                      |
|          policy / guards             |
+----------+-----------+---------------+
           |           |
           v           v
     Retriever      Controller
  EmbeddingGemma      mini-Jev
           \           /
            \         /
             v       v
             Generator
              Nemotron
          Nebius Token Factory
                 |
                 v
      OpenAI-compatible response
```

Not every turn calls every provider.

## 2. External contract

Alienese should initially expose:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Later, after the turn engine is stable:

- streaming;
- `/v1/responses`;
- additional compatibility surfaces.

The caller remains responsible for executing tool calls and returning tool results.

## 3. Source of truth

The request/event history is the correctness source of truth.

Alienese may maintain:

- prefix state checkpoints;
- embedding caches;
- parsed entity caches;
- idempotency records;
- trace indexes.

Loss of those caches may reduce performance but must not make state unrecoverable.

## 4. Turn lifecycle

```text
1. Validate request
2. Assign request_id / trace_id / deadline
3. Normalize messages and tool activity into typed events
4. Reconstruct WorkingState
5. Extract deterministic evidence and grounded entities
6. Construct raw CandidateActions
7. Optionally rank/prune context or candidates
8. Apply deterministic policy guards
9. If ambiguous, call controller
10. If selected action requires synthesis, build a typed GenerationJob
11. Call generator at most within the configured budget
12. Validate artifact structurally
13. Return exactly one external action or assistant response
14. Emit trace/metrics asynchronously or best-effort
```

## 5. Core typed objects

### Event

A normalized immutable observation derived from protocol input.

Examples:

- USER_GOAL
- ASSISTANT_MESSAGE
- TOOL_CALL
- TOOL_RESULT
- TEST_RESULT
- FAILURE
- FILE_OBSERVATION
- MUTATION
- VERIFICATION

Events preserve ordering and provenance.

### WorkingState

A deterministic projection over events.

It may contain:

- objective;
- explicit constraints;
- known files/symbols/tests/commands;
- latest failure;
- current diff/mutation state;
- recent action signatures;
- latest verification;
- available canonical capabilities;
- context references.

### CandidateAction

A possible next action.

Required properties should include:

- stable internal id;
- canonical intent;
- external tool binding, if any;
- complete or partial arguments;
- evidence/provenance;
- whether generation is required;
- risk/cost class.

### DecisionResult

Contains:

- selected candidate id;
- candidate scores;
- controller identity/revision;
- latency;
- guard/fallback metadata.

### GenerationJob

A narrow synthesis request with a typed purpose, for example:

- PATCH
- EXPLAIN_FAILURE
- WRITE_TEST
- SYNTHESIZE_SEARCH
- ANSWER

A GenerationJob contains only the context needed for that job.

## 6. Grounding

Grounding converts observed evidence into concrete action possibilities.

Evidence sources include:

1. exact/direct references;
2. stack traces and failures;
3. structural/lexical matches;
4. recently modified or observed entities;
5. semantic relevance.

Direct evidence outranks embedding similarity.

The system should prefer:

```text
READ src/auth/session.py
RUN pytest tests/test_session.py
SEARCH invalidate_session
```

over:

```text
READ
TEST
SEARCH
```

when arguments can be grounded from known state.

## 7. Candidate policy

The controller must receive a bounded candidate set.

The candidate engine should optimize **oracle action recall**, not merely semantic similarity.

When the correct action may not be represented, include escape actions such as:

- EXPAND_SEARCH;
- REQUEST_EVIDENCE;
- SYNTHESIZE_SEARCH;
- DEFER_TO_GENERATOR.

Candidate construction is a critical subsystem. A controller cannot recover from a missing correct action.

## 8. Retrieval

EmbeddingGemma 2 is optional on any particular turn.

Use retrieval only when candidate/context scale justifies it.

Mandatory context is preserved by deterministic policy. Semantic retrieval operates on optional context.

Candidate/context ranking may combine:

- exact-evidence score;
- lexical score;
- recency score;
- semantic score;
- diversity bonus.

Embedding vectors should be cached with a versioned content hash including model/revision/instruction/dimension.

## 9. Controller

mini-Jev is a finite-choice policy.

It should:

- receive compact state;
- receive a bounded candidate list;
- return scores and a selected candidate.

It should not:

- invent arbitrary paths or shell commands when grounding is possible;
- generate patches;
- summarize the entire history;
- orchestrate provider calls.

Initially support modes:

- shadow;
- guarded;
- full.

Guarded mode uses score margin, entropy, risk, loop state, and deterministic policy—not raw top-1 score alone.

## 10. Generator

Nemotron through Nebius Token Factory handles open-ended synthesis.

It receives typed GenerationJobs rather than a generic autonomous-agent prompt.

Examples:

- explain a failure from selected evidence;
- create a minimal patch;
- generate a targeted test;
- synthesize a new search hypothesis;
- produce the final user-facing answer.

Generator output is structurally validated. Semantic correctness is ultimately established through subsequent harness execution and verification.

## 11. Deterministic guards

Examples:

- FINISH may be ineligible after mutation until verification occurs when verification tools exist;
- identical repeated actions with no new evidence are penalized or removed;
- exact stack-trace references survive semantic pruning;
- malformed external schemas cause typed compatibility errors;
- budgets and deadlines are enforced in code;
- retries are bounded.

## 12. Reliability

Each remote provider is wrapped by a shared provider runtime with:

- timeout;
- one bounded retry where appropriate;
- circuit breaker;
- concurrency semaphore;
- usage extraction;
- normalized errors;
- tracing.

Expected degradation:

| Failure | Behavior |
| --- | --- |
| Retriever unavailable | deterministic lexical/recency fallback |
| Controller unavailable | configured generator/monolithic fallback |
| Controller uncertain | guarded fallback |
| Generator unavailable | continue non-generative investigation if useful, otherwise explicit failure |
| Cache lost | reconstruct/recompute |
| Observability unavailable | inference continues |

## 13. Idempotency

Support an idempotency key or deterministic request fingerprint.

A retried logical request must return the same completed response rather than making a new policy decision.

## 14. Trust and security

Repository contents, tool output, generated text, webpages, and shell output are untrusted data.

System policy and user instructions must remain distinguishable from untrusted context.

Secrets must never be logged or intentionally passed into model context.

## 15. Observability

Use three layers:

### Metrics

Examples:

- request latency;
- provider latency/errors;
- raw vs selected context;
- raw vs final candidate count;
- retriever/controller/generator bypass rates;
- fallback rate;
- generator tokens;
- cost per turn / solved task.

### Structured logs

Operational events with request/trace ids and no secrets.

### Decision traces

Replayable artifacts containing normalized state, candidates, scores, selected action, provider revisions, policy version, and fallback metadata.

## 16. Evaluation

Evaluate subsystems separately before end-to-end claims:

| Subsystem | Metric |
| --- | --- |
| State reconstruction | deterministic equivalence |
| Grounding | entity extraction accuracy |
| Candidate engine | oracle action recall@K |
| Retriever | required-evidence recall@budget |
| Controller | choice accuracy/regret/calibration |
| Guards | bad-decision escape rate |
| Generator | artifact/test success |
| Full system | solved tasks |
| Efficiency | total cost/tokens/latency per solved task |

Final benchmarks must report all provider costs and latency, not generator tokens alone.
