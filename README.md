# Alienese

Alienese is a unified inference API for coding agents that composes specialized models instead of relying on one large language model for every step.

It exposes a single OpenAI-compatible API endpoint that can be plugged into an existing coding harness, agent runtime, IDE backend, or evaluation framework.

## Core idea

Alienese separates three responsibilities:

- **Retrieval / context reduction** — EmbeddingGemma 2 reduces large context and candidate spaces into a smaller relevant working set.
- **Decision / control** — mini-Jev, later fine-tuned as **mini-Jev-SWE**, selects the next action from a bounded set of grounded candidates.
- **Generation / reasoning** — NVIDIA Nemotron on Nebius Token Factory handles open-ended reasoning and code generation when needed.

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
   Nebius Token Factory
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

See [AGENTS.md](AGENTS.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/PLAN.md](docs/PLAN.md).

## Status

Phase 0 and Phase 1 runtime foundation implemented:

- typed core domain contracts (`NormalizedEvent`, `WorkingState`, `CandidateAction`, `DecisionResult`, `GenerationJob`, `ReplayArtifact`);
- 4-tier trust model (`SYSTEM_TRUSTED`, `USER`, `MODEL_GENERATED`, `UNTRUSTED_EXTERNAL`);
- deterministic OpenAI protocol normalization and `WorkingState` reconstruction with canonical `state_digest`;
- strict `Retriever`, `Controller`, and `Generator` protocols with deterministic offline fake providers (`FakeRetriever`, `FakeController`, `FakeGenerator`) and fault injection;
- single-turn `TurnEngine` enforcing one external action per turn and pre-serialization tool schema validation;
- idempotent request handling (`Idempotency-Key`) separating HTTP attempt `request_id` from logical `operation_id`;
- central secret redaction, structured logging (`structlog`), and OpenTelemetry tracing hooks.

No external services, model APIs, GPUs, model downloads, or network access are required for development tests and fake-provider runtime mode. Real model provider adapters are scheduled for Phase 2.

## Developer setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run static checks and offline test suite
ruff check .
ruff format --check .
mypy src
pytest

# Start the development server in fake-provider mode
uvicorn alienese.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

## Supported external API subset (Phase 1)

### Endpoints

- `GET /health` — runtime liveness, version, and provider mode (`fake`).
- `GET /v1/models` — available logical model identifiers.
- `POST /v1/chat/completions` — single-turn non-streaming completion returning either one assistant text response or one validated tool call.

### `POST /v1/chat/completions` compatibility contract

| Parameter | Status | Behavior |
| --- | --- | --- |
| `model` | Supported | Validated non-empty string; echoed in response |
| `messages` | Supported | `system`, `developer`, `user`, `assistant`, and `tool` roles normalized into typed events |
| `tools` | Supported | Function tool schemas bound to `ExternalToolBinding`; unknown tools preserved as `CUSTOM_TOOL` |
| `tool_choice` | Supported | `"auto"`, `"none"`, `"required"`, or named function object (`{"type": "function", "function": {"name": "..."}}`). Never fabricates missing required tool arguments |
| `temperature` | Supported | Validated (`[0.0, 2.0]`) and forwarded to `GenerationJob` |
| `max_tokens` / `max_completion_tokens` | Supported | Validated (`>= 1`) and forwarded to `GenerationJob`; conflicting values rejected with `CompatibilityError` |
| `user`, `metadata` | Accepted (no-op) | Validated and redacted in logs; currently have no policy effect in Phase 1 |
| `stream=true` | Rejected (400) | Fails with `compatibility_error` (`streaming_not_supported`) |
| `n != 1` | Rejected (400) | Fails with `compatibility_error` (`multiple_choices_not_supported`) |
| `parallel_tool_calls=true` | Rejected (400) | Fails with `compatibility_error` (`parallel_tool_calls_not_supported`) |
| `functions` / `function_call` | Rejected (400) | Fails with `compatibility_error` (`legacy_functions_not_supported`) |
| Other OpenAI fields / `/v1/responses` | Rejected (400) | Rejected explicitly with `compatibility_error` or `invalid_request_error` |

### Idempotency and correlation headers

- `Idempotency-Key`: Maps a canonical request fingerprint to a stable logical `operation_id` and completed response. Retried HTTP requests with the same key and identical body receive a fresh per-attempt `X-Request-ID` in logs and headers while returning the exact previously completed logical response (`X-Idempotent-Replay: true` and original `X-Operation-ID`). Reusing a key with a different request payload fails with HTTP 409 (`idempotency_conflict`).
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
├── observability/
│   ├── logging.py
│   ├── redaction.py
│   └── tracing.py
├── providers/
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
