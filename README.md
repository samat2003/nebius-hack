# Alienese

Alienese is a unified inference API for coding agents that composes specialized models instead of relying on one large language model for every step.

It exposes a single OpenAI-compatible API endpoint that can be plugged into an existing coding harness, agent runtime, IDE backend, or evaluation framework.

## Core idea

Alienese separates three responsibilities:

- **Retrieval / context reduction** — EmbeddingGemma 2 reduces large context and candidate spaces into a smaller relevant working set.
- **Decision / control** — mini-Jev, later fine-tuned as **mini-Jev-SWE**, selects the next action from a bounded set of grounded candidates.
- **Generation / reasoning** — NVIDIA Nemotron (`nvidia/nemotron-3-super-120b-a12b` live-verified via NVIDIA API Catalog and Nebius Token Factory) handles open-ended reasoning and code generation when needed.

The runtime—not the models—owns orchestration, state reconstruction, validation, fallbacks, budgets, and observability.

## Initial architecture

The first implementation uses existing model endpoints with **no new training**. The goal is to validate the system architecture before fine-tuning mini-Jev-SWE.

```text
Coding harness
      |
      | OpenAI-compatible request
      v
+-----------------------------+
|          Alienese           |
|                             |
| protocol + state            |
| grounding + candidates      |
| context selection           |
| policy + validation         |
+-----+-----------+-----------+
      |           |
      |           |
      v           v
 Embedding     mini-Jev
 Gemma 2       controller
      \           /
       \         /
        v       v
       Nemotron
 (NVIDIA API Catalog /
  Nebius Token Factory)
        |
        v
OpenAI-compatible response
```

Not every request calls every model. Deterministic logic should bypass model calls whenever possible.

## Engineering principles

1. Alienese owns the state machine. Models never orchestrate one another.
2. One external request returns at most one externally visible action.
3. Correctness must not depend on opaque server-side session state.
4. Candidate actions should be grounded before they reach the controller.
5. Nemotron is a generator/reasoner, not a generic tool-argument filler.
6. Retrieval may reduce optional context but must never remove correctness-critical state.
7. Remote calls are bounded by deadlines, retries, budgets, and circuit breakers.
8. Every important decision is traceable and replayable.
9. Provider-specific SDKs stay outside the core runtime.
10. Training begins only after the architecture produces useful traces and measurable failure modes.

See [AGENTS.md](AGENTS.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/PLAN.md](docs/PLAN.md), and [docs/providers/compatibility-matrix.md](docs/providers/compatibility-matrix.md).

## Status

Phase 0, Phase 1, Phase 2, and Phase 3 implemented:

- **Phase 0/1 Runtime Foundation**:
  - typed core domain contracts (`NormalizedEvent`, `WorkingState`, `CandidateAction`, `DecisionResult`, `GenerationJob`, `ReplayArtifact`);
  - 4-tier trust model (`SYSTEM_TRUSTED`, `USER`, `MODEL_GENERATED`, `UNTRUSTED_EXTERNAL`);
  - deterministic OpenAI protocol normalization and `WorkingState` reconstruction with canonical `state_digest`;
  - strict `Retriever`, `Controller`, and `Generator` protocols with deterministic offline fake providers (`FakeRetriever`, `FakeController`, `FakeGenerator`) and fault injection;
  - single-turn `TurnEngine` enforcing one external action per turn and pre-serialization tool schema validation;
  - idempotent request handling (`Idempotency-Key`) separating HTTP attempt `request_id` from logical `operation_id`;
  - central secret redaction, structured logging (`structlog`), and OpenTelemetry tracing hooks.
- **Phase 2 Provider Runtime & NVIDIA API Catalog Integration**:
  - shared remote-provider execution layer (`src/alienese/providers/runtime/`) with end-to-end turn deadlines (`RequestContext.deadline_monotonic` + `DeadlineBudget`), safe-to-retry vs. ambiguous-completion retry policy, per-attempt circuit breaker with single-probe `HALF_OPEN` admission, bounded concurrency + waiter queue admission, pre-allocation JSON byte/depth caps, streaming response byte caps, and origin-scoped HTTPS credentials (`follow_redirects=False`, `trust_env=False`);
  - live-verified `NvidiaBuildGenerator` (`nvidia/nemotron-3-super-120b-a12b` at `https://integrate.api.nvidia.com/v1`) with reasoning suppression (`enable_thinking=False`), verbatim `content` preservation, reasoning-only output rejection, and HTTP `202` pending-invocation fail-fast handling;
  - live-verified `NebiusTokenFactoryGenerator` (`LIVE_VERIFIED`, `nvidia/nemotron-3-super-120b-a12b` at verified regional endpoint `https://api.tokenfactory.us-central1.nebius.com/v1`) with reasoning suppression (`chat_template_kwargs={"enable_thinking": False}`), vLLM `message.reasoning` normalization, strict bidirectional credential isolation, and explicit origin allowlists;
  - typed `EmbeddingGemmaRetriever` (`google/embeddinggemma-300m`, Matryoshka truncation + L2 renormalization, mandatory item preservation) and `MiniJevController` (`samatv256/mini-Jev` pinned at `step-010626` / `3a1d1d19d85e9863146307fe4b769e8bbe242c4e`, single-candidate bypass, disposition preservation, simplex probability verification), both `MOCK_VERIFIED` (`BLOCKED_HOSTING`);
  - truthful nullable token and cost telemetry (`ProviderCallTelemetry` and `ChatCompletionResponse.usage`).
- **Phase 3 Deterministic Grounding and Candidate Construction (Completed)**:
  - deterministic typed evidence extraction (`GroundingEvidence`: file paths, symbols, test targets/commands, search patterns, stack frames, failure messages, exit statuses, mutation targets, verification results);
  - strict repository path containment rejecting out-of-workspace, Windows drive, UNC, URI, and traversal escape paths (with schema defaults boundary validation ensuring uncontained paths cannot leak from tool schemas; physical symlink containment remains the coding harness runtime sandbox responsibility);
  - validated verification-command form (`is_safe_verification_command`) enforcing conservative per-executable option allowlists (`pytest`, `python -m unittest`, `ruff`, `mypy`), explicitly rejecting plugin loading (e.g., `pytest -p`), config overrides (`--override-ini`), module-loading flags, shell pipelines, chaining, and redirections;
  - positive mutation confirmation (`classify_mutation_outcome`) requiring structured success output or specific target-file matching text, strictly rejecting empty, unrelated, wrong-file, or failed results, and distinguishing attempted mutations (`OBSERVED`) from confirmed applied mutations (`CONFIRMED`);
  - explicit verification outcome classifier (`PASSED` [positive confirmation only], `FAILED`, `UNKNOWN`);
  - strict 4-tier trust boundaries (`SYSTEM_PROMPT` > `USER_DIRECTIVE` > `AGENT_COMMITTED` > `UNTRUSTED_EXTERNAL`);
  - zero-hallucination argument resolution (`GroundedArgumentResolver`) binding JSON schema parameters to extracted evidence without inventing paths, tests, or commands, with evidence taking precedence over schema defaults;
  - tool-choice eligibility enforcement before candidate truncation and multi-factor deterministic candidate ranking/bounding (default $K=8$);
  - golden evaluation dataset (`tests/fixtures/grounding/decision_points.json`) with 35 curated coding decision points across train, dev, and held-out test splits;
  - offline evaluation CLI (`python -m alienese.grounding.eval`) with evidence-support proxy metric (checking exact arguments against evidence and validated tool defaults without unhashable-object crashes) demonstrating:
    - Held-out test split (Action N=9, Abstain N=1): Oracle Recall@1 **8/9 (88.9%)** (vs Phase 1 baseline **3/9 [33.3%]**, **+55.6% lift**), Recall@4/8 **8/9 (88.9%)**, Exact Arg Accuracy **8/9 (88.9%)**, Arg Presence **8/8 (100.0%)**, Abstention Correctness **1/1 (100.0%)**, Executable Validity **10/10 (100.0%)**, Evidence Support (Proxy) **10/10 (100.0%)**, Unsupported Args **0/10**;
    - Overall corpus (Total N=35, Action N=29, Abstain N=6): Oracle Recall@1 **23/29 (79.3%)** (vs Phase 1 baseline **6/29 [20.7%]**, **+58.6% lift**), Recall@4/8 **24/29 (82.8%)**, Exact Arg Accuracy **24/29 (82.8%)**, Arg Presence Rate **22/26 (84.6%)**, Abstention Correctness **6/6 (100.0%)**, Executable Validity **27/27 (100.0%)**, Evidence Support (Proxy) **27/27 (100.0%)**, Unsupported Args **0/27**.

Default `pytest` execution remains 100% offline and deterministic (`ALIENESE_PROVIDER_MODE=fake` by default, even when `.env` contains live API keys).

## Developer setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run static checks and offline test suite (no API keys or network calls used)
ruff check .
ruff format --check .
mypy src tests
pytest

# Run offline grounding candidate evaluation against golden dataset
python -m alienese.grounding.eval

# Optional: run bounded live NVIDIA API Catalog integration test (requires GENERATOR_API_KEY in .env)
RUN_LIVE_NVIDIA_TESTS=1 pytest -m live_nvidia -v

# Optional: run bounded live Nebius Token Factory integration test (requires NEBIUS_TOKEN_FACTORY_KEY in .env)
RUN_LIVE_NEBIUS_TESTS=1 pytest -m live_nebius -v

# Start the development server in fake-provider mode (default)
uvicorn alienese.api.app:create_app --factory --host 127.0.0.1 --port 8000

# Start the development server in hybrid mode with NVIDIA API Catalog generator
ALIENESE_PROVIDER_MODE=hybrid GENERATOR_PROVIDER=nvidia_build \
  uvicorn alienese.api.app:create_app --factory --host 127.0.0.1 --port 8000

# Start the development server in hybrid mode with Nebius Token Factory generator
ALIENESE_PROVIDER_MODE=hybrid GENERATOR_PROVIDER=nebius_token_factory \
  uvicorn alienese.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

## Provider Mode Matrix (Phase 2)

| `ALIENESE_PROVIDER_MODE` | `RETRIEVER_PROVIDER` | `CONTROLLER_PROVIDER` | `GENERATOR_PROVIDER` | Behavior |
| --- | --- | --- | --- | --- |
| `fake` (default) | `fake` | `fake` | `fake` (or ignored `.env` override) | All 3 providers run deterministically in-process (`FakeRetriever`, `FakeController`, `FakeGenerator`); zero network calls. |
| `hybrid` | `fake` | `fake` | `nvidia_build` | `FakeRetriever` + `FakeController` + live `NvidiaBuildGenerator` (`https://integrate.api.nvidia.com/v1`). |
| `hybrid` | `fake` | `fake` | `nebius_token_factory` | `FakeRetriever` + `FakeController` + live-verified `NebiusTokenFactoryGenerator` (`https://api.tokenfactory.us-central1.nebius.com/v1`; requires `NEBIUS_TOKEN_FACTORY_KEY`; defaults to `nvidia/nemotron-3-super-120b-a12b`). |
| `remote` | Any | Any | Any | Rejected with `CompatibilityError` (`unsupported_provider_mode`) until `mini-Jev` and `EmbeddingGemma 2` hosting are enabled. |

See [docs/providers/compatibility-matrix.md](docs/providers/compatibility-matrix.md) and [docs/adr/0002-remote-provider-runtime-and-telemetry.md](docs/adr/0002-remote-provider-runtime-and-telemetry.md).

## Supported external API subset (Phase 1/2)

### Endpoints

- `GET /health` — runtime liveness, version, and active provider mode (`fake` or `hybrid`).
- `GET /v1/models` — exposes `alienese-default` as the single supported public model.
- `POST /v1/chat/completions` — single-turn non-streaming completion returning either one assistant text response or one validated tool call.

### `POST /v1/chat/completions` compatibility contract

| Parameter | Status | Behavior |
| --- | --- | --- |
| `model` | Supported | Must be `alienese-default`; unknown model IDs rejected with `compatibility_error` (`model_not_supported`) |
| `messages` | Supported | `system`, `developer`, `user`, `assistant`, and `tool` roles normalized into typed events |
| `tools` | Supported | Function tool schemas bound to `ExternalToolBinding`; unknown tools preserved as `CUSTOM_TOOL` |
| `tool_choice` | Supported | `"auto"`, `"none"`, `"required"`, or named function object (`{"type": "function", "function": {"name": "..."}}`). Never fabricates missing required tool arguments |
| `temperature` | Supported | Validated (`[0.0, 2.0]`) and forwarded to `GenerationJob` |
| `max_tokens` / `max_completion_tokens` | Supported | Validated (`>= 1`) and forwarded to `GenerationJob`; conflicting values rejected with `CompatibilityError` |
| `user`, `metadata` | Accepted (no-op) | Validated and redacted in logs; currently have no policy effect |
| `stream=true` | Rejected (400) | Fails with `compatibility_error` (`streaming_not_supported`) |
| `n != 1` | Rejected (400) | Fails with `compatibility_error` (`multiple_choices_not_supported`) |
| `parallel_tool_calls=true` | Rejected (400) | Fails with `compatibility_error` (`parallel_tool_calls_not_supported`) |
| `functions` / `function_call` | Rejected (400) | Fails with `compatibility_error` (`legacy_functions_not_supported`) |
| Other OpenAI fields / `/v1/responses` | Rejected (400) | Rejected explicitly with `compatibility_error` or `invalid_request_error` |

### Idempotency and correlation headers

- `Idempotency-Key`: Maps a canonical request fingerprint to a stable logical `operation_id` and completed response. Retried HTTP requests with the same key and identical body receive a fresh per-attempt `X-Request-ID` in logs and headers while returning the exact previously completed logical response (`X-Idempotent-Replay: true` and original `X-Operation-ID`). Reusing a key with a different request payload fails with HTTP 409 (`idempotency_conflict`). In Phase 1/2, idempotency state is stored in memory (`InMemoryIdempotencyStore`); guarantees are bounded by the configured entry capacity (`ALIENESE_IDEMPOTENCY_MAX_ENTRIES`), TTL (`ALIENESE_IDEMPOTENCY_TTL_SECONDS`), and single-process lifetime.
- `X-Request-ID`: Per-HTTP-attempt correlation identifier (preserved if valid or generated).
- `X-Trace-ID`: Application-level correlation identifier (kept distinct from OpenTelemetry internal trace IDs).
- `traceparent`: Optional W3C Trace Context header used for OpenTelemetry distributed span propagation.

## Repository structure

```text
src/alienese/
├── api/
│   ├── app.py
│   ├── chat_completions.py
│   ├── errors.py
│   └── models.py
├── contracts/
│   ├── candidates.py
│   ├── context.py
│   ├── decisions.py
│   ├── events.py
│   ├── generation.py
│   ├── providers.py
│   ├── state.py
│   └── traces.py
├── engine/
│   ├── idempotency.py
│   ├── normalize.py
│   ├── reconstruct.py
│   └── turn.py
├── grounding/
│   ├── extractors/
│   │   ├── base.py
│   │   ├── commands.py
│   │   ├── failures.py
│   │   ├── mutation.py
│   │   ├── paths.py
│   │   ├── symbols.py
│   │   └── tests.py
│   ├── argument_resolution.py
│   ├── candidate_builder.py
│   ├── eval.py
│   ├── evidence.py
│   ├── normalization.py
│   ├── policy.py
│   └── ranking.py
├── observability/
│   ├── logging.py
│   ├── redaction.py
│   └── tracing.py
├── providers/
│   ├── controller/
│   │   └── mini_jev.py
│   ├── generator/
│   │   ├── nebius_token_factory.py
│   │   └── nvidia_build.py
│   ├── retriever/
│   │   └── embeddinggemma.py
│   ├── runtime/
│   │   ├── circuit_breaker.py
│   │   ├── client.py
│   │   ├── concurrency.py
│   │   ├── deadlines.py
│   │   ├── errors.py
│   │   ├── retry.py
│   │   └── telemetry.py
│   ├── base.py
│   └── fake.py
├── storage/
│   ├── idempotency.py
│   └── traces.py
└── config.py

tests/
├── contracts/
├── fixtures/
├── integration/
├── replay/
└── unit/
```
