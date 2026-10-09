# ADR 0002: Remote Provider Runtime, End-to-End Deadlines, and Truthful Telemetry

- **Status**: Accepted
- **Date**: 2026-10-08
- **Phase**: Phase 2 (Provider Runtime & NVIDIA API Catalog Integration)

## Context

Phase 2 introduces Alienese's shared remote-provider execution layer (`src/alienese/providers/runtime/`) and integrates NVIDIA Nemotron 3 Super (`nvidia/nemotron-3-super-120b-a12b`) through NVIDIA API Catalog (`https://integrate.api.nvidia.com/v1`), alongside portability and boundary adapters for Nebius Token Factory (`NebiusTokenFactoryGenerator`), EmbeddingGemma 2 (`EmbeddingGemmaRetriever`), and `samatv256/mini-Jev` (`MiniJevController`).

To satisfy Invariants 7, 8, 9, and 14 in `AGENTS.md` without altering the core Phase 0/1 orchestration architecture, three narrow contract refinements were required:

1. **End-to-End Turn Deadline Propagation**: Provider timeouts, concurrency queue admission waits, and retry backoff sleeps must be bounded by the remaining turn deadline rather than independent per-provider timers.
2. **Truthful Nullable Token and Cost Telemetry**: Phase 1 defaulted `ProviderCallTelemetry` token and cost fields to `0` / `0.0`. However, fake providers and some upstream error or non-standard responses do not report token usage, and live per-token USD billing rates are not verified across all providers in Phase 2. Reporting `0` or `0.0` conflates "zero tokens/cost" with "unknown/unmeasured".
3. **Strict Retry & Circuit-Breaker Separation**: Retrying an ambiguous post-transmission failure (e.g., `httpx.ReadTimeout` after the request body was sent) risks duplicate billing and latency blowup, and counting local concurrency saturation (`httpx.PoolTimeout`) or HTTP `429` rate limits as upstream outages would falsely trip the provider circuit breaker.

## Decisions

### 1. Three-Tier Deadline Contract & Operational `deadline_monotonic` on `RequestContext`

- `RequestContext` includes an optional `deadline_monotonic: float | None = Field(default=None, exclude=True)`.
- Three distinct deadline scopes are enforced:
  1. **Incoming HTTP request deadline** (`request_deadline_seconds`): established in `POST /v1/chat/completions` before reading the incremental ASGI request body stream (`request.stream()`) and before `IdempotencyCoordinator.execute()`.
  2. **Duplicate idempotent waiter deadline** (`idempotency_wait_timeout_seconds` bounded by the incoming HTTP request deadline): bounds how long a concurrent duplicate request waits on the per-key lock. If exhausted, the duplicate waiter fails with `ProviderTimeout(code="idempotency_wait_timeout", status_code=504)` without cancelling the in-flight original leader operation.
  3. **Logical turn execution deadline** (`turn_timeout_seconds` bounded by the leader's incoming deadline): `TurnEngine.execute_turn()` caps `ctx.deadline_monotonic` to `min(ctx.deadline_monotonic, now + turn_timeout_seconds)`. `DeadlineBudget` enforces this budget across concurrency slot acquisition, HTTP connection/read timeouts, and retry backoff scheduling.
- Because `RequestContext` is operational metadata (and `deadline_monotonic` is excluded from serialization), `WorkingState.state_digest` and `ReplaySemantics` equality remain strictly deterministic and unaffected by wall-clock or monotonic deadlines.

### 2. Nullable Token/Cost Fields and `serving_fingerprint` vs. `model_revision`

- `ProviderCallTelemetry` fields `prompt_tokens`, `completion_tokens`, `total_tokens`, `retried_prompt_tokens`, `retried_completion_tokens`, `estimated_cost_usd`, and `serving_fingerprint` default to `None` rather than `0` / `0.0`.
- `ProviderCallTelemetry` records `attempt_count`, `failed_attempt_count`, `upstream_request_id` (`nvcf-reqid` / `x-request-id`), and `serving_fingerprint` (extracted from OpenAI-compatible `system_fingerprint`).
- `GenerationResult.model_revision` and `RetrievalResult.model_revision` are populated only when an explicit `model_revision` is reported by the provider; otherwise they remain `"unknown"` (never conflated with `system_fingerprint`).
- `ChatCompletionResponse.usage` is `UsageInfo | None = None`: it is populated when the generator reports valid integer token counts and `None` when token usage is unknown.

### 3. Per-Attempt Concurrency Permits, Raw Byte Limits, and Circuit-Breaker Accounting

- `ProviderConcurrencyLimiter.acquire(deadline)` wraps each individual upstream HTTP attempt rather than the outer retry loop, ensuring retry backoff (`Retry-After` or exponential backoff) never holds an active provider concurrency slot.
- `ProviderHttpClient` sends `Accept-Encoding: identity`, rejects unsolicited compressed `Content-Encoding` headers (`code="unsupported_content_encoding"`), and reads response streams via `response.aiter_raw()` so `max_response_bytes` is enforced before any decompression expansion.
- `ProviderHttpClient` classifies failures into `SAFE_RETRYABLE_TRANSPORT`, `SAFE_RETRYABLE_UPSTREAM`, `RATE_LIMITED` (`429`), `AMBIGUOUS_COMPLETION` (`ReadTimeout`, `WriteTimeout`, `RemoteProtocolError`), `LOCAL_SATURATION` (`PoolTimeout`, local concurrency queue timeout), and `NON_RETRYABLE_CONTRACT`. By default (`retry_ambiguous_failures=False`), `AMBIGUOUS_COMPLETION` errors fail fast without automatic retry.
- `ProviderCircuitBreaker` checks admission before every attempt (including retries), counts each failed upstream outage attempt, excludes `LOCAL_SATURATION` and `RATE_LIMITED` (`429`), and admits at most one concurrent probe (`half_open_max_probes=1`) in `HALF_OPEN` state.

### 4. Explicit Provider Mode Matrix, Isolated Credentials, & Strict Origin Allowlists

- `ALIENESE_PROVIDER_MODE` defaults to `"fake"`. Even when `.env` contains live `GENERATOR_API_KEY` and `NEBIUS_TOKEN_FACTORY_KEY` credentials, `fake` mode keeps all providers in-process (`FakeRetriever`, `FakeController`, `FakeGenerator`) and makes zero network calls.
- `ALIENESE_PROVIDER_MODE="hybrid"` enables the configured remote generator while enforcing `RETRIEVER_PROVIDER="fake"` and `CONTROLLER_PROVIDER="fake"`:
  - `GENERATOR_PROVIDER="nvidia_build"` resolves `GENERATOR_API_KEY`, `GENERATOR_BASE_URL` (default `https://integrate.api.nvidia.com/v1`), and `GENERATOR_MODEL` (default `nvidia/nemotron-3-super-120b-a12b`), restricted to the explicit origin allowlist `{"integrate.api.nvidia.com"}`.
  - `GENERATOR_PROVIDER="nebius_token_factory"` resolves `NEBIUS_TOKEN_FACTORY_KEY`, `NEBIUS_TOKEN_FACTORY_BASE_URL` (default `https://api.tokenfactory.us-central1.nebius.com/v1`), and `NEBIUS_TOKEN_FACTORY_MODEL` (default `nvidia/nemotron-3-super-120b-a12b`), restricted to the explicit origin allowlist `{"api.tokenfactory.us-central1.nebius.com", "api.tokenfactory.nebius.com"}`.
  - Cross-provider credential reuse (`nvapi-*` on Nebius or Nebius keys on NVIDIA) and unallowlisted wildcard subdomains are rejected before serving (`CompatibilityError`).
- `ALIENESE_PROVIDER_MODE="remote"` is explicitly rejected with `CompatibilityError` until `mini-Jev` and `EmbeddingGemma 2` hosting endpoints are integrated in subsequent phases.

## Consequences

- Default `pytest` execution remains 100% offline and deterministic even on developer workstations with `.env` populated (`-m 'not live_nvidia and not live_nebius'`).
- Switching between `nvidia_build` and `nebius_token_factory` requires only setting `GENERATOR_PROVIDER` with zero changes to `TurnEngine`, `WorkingState`, or the public OpenAI-compatible API, while keeping NVIDIA and Nebius credentials strictly isolated.
- Telemetry artifacts never fabricate zero token usage, zero USD cost, or fake model revisions when measurements are unavailable.
