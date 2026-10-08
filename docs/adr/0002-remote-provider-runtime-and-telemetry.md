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

### 1. Operational `deadline_monotonic` on `RequestContext`

- `RequestContext` includes an optional `deadline_monotonic: float | None = Field(default=None, exclude=True)`.
- `TurnEngine.execute_turn()` initializes `deadline_monotonic` at turn start when not already set by the caller.
- `DeadlineBudget` computes remaining budget from `ctx.deadline_monotonic` across concurrency slot acquisition, HTTP connection/read timeouts, and retry backoff scheduling. No new call or retry may begin once the deadline is exhausted.
- Because `RequestContext` is operational metadata (and `deadline_monotonic` is excluded from serialization), `WorkingState.state_digest` and `ReplaySemantics` equality remain strictly deterministic and unaffected by wall-clock or monotonic deadlines.

### 2. Nullable Token and Cost Fields in `ProviderCallTelemetry` and `ChatCompletionResponse`

- `ProviderCallTelemetry` fields `prompt_tokens`, `completion_tokens`, `total_tokens`, `retried_prompt_tokens`, `retried_completion_tokens`, and `estimated_cost_usd` default to `None` rather than `0` / `0.0`.
- `ProviderCallTelemetry` records `attempt_count`, `failed_attempt_count`, and `upstream_request_id` (`nvcf-reqid` / `x-request-id`).
- `ChatCompletionResponse.usage` is `UsageInfo | None = None`: it is populated when the generator reports valid integer token counts and `None` when token usage is unknown (such as in `fake` provider mode or when an upstream response omits `usage`).

### 3. Safe-to-Retry vs. Ambiguous-Completion Classification & Circuit-Breaker Accounting

- `ProviderHttpClient` classifies failures into `SAFE_RETRYABLE_TRANSPORT` (`ConnectError`, `ConnectTimeout`), `SAFE_RETRYABLE_UPSTREAM` (HTTP `408`, `500`, `502`, `503`, `504`), `RATE_LIMITED` (HTTP `429`), `AMBIGUOUS_COMPLETION` (`ReadTimeout`, `WriteTimeout`, `RemoteProtocolError`), `LOCAL_SATURATION` (`PoolTimeout`, local concurrency queue timeout), and `NON_RETRYABLE_CONTRACT` (`400`, `401`, `403`, `404`, `422`, malformed JSON/schema).
- By default (`retry_ambiguous_failures=False`), `AMBIGUOUS_COMPLETION` errors fail fast without automatic retry.
- `ProviderCircuitBreaker` counts each failed upstream outage attempt (`SAFE_RETRYABLE_UPSTREAM` and `AMBIGUOUS_COMPLETION`), excludes `LOCAL_SATURATION` and `RATE_LIMITED` (`429`), and admits at most one concurrent probe (`half_open_max_probes=1`) in `HALF_OPEN` state.

### 4. Explicit Provider Mode Matrix

- `ALIENESE_PROVIDER_MODE` defaults to `"fake"`. Even when `.env` contains live `GENERATOR_API_KEY` credentials, `fake` mode keeps all providers in-process (`FakeRetriever`, `FakeController`, `FakeGenerator`) and makes zero network calls.
- `ALIENESE_PROVIDER_MODE="hybrid"` enables the configured remote generator (`GENERATOR_PROVIDER="nvidia_build"` or `"nebius_token_factory"`) while enforcing `RETRIEVER_PROVIDER="fake"` and `CONTROLLER_PROVIDER="fake"`.
- `ALIENESE_PROVIDER_MODE="remote"` is explicitly rejected with `CompatibilityError` until `mini-Jev` and `EmbeddingGemma 2` hosting endpoints are integrated in subsequent phases.

## Consequences

- Default `pytest` execution remains 100% offline and deterministic even on developer workstations with `.env` populated.
- Switching between `nvidia_build` and `nebius_token_factory` requires only configuration changes (`GENERATOR_PROVIDER`, `GENERATOR_API_KEY`, `GENERATOR_BASE_URL`, `GENERATOR_MODEL`) with zero changes to `TurnEngine`, `WorkingState`, or the public OpenAI-compatible API.
- Telemetry artifacts never fabricate zero token usage or zero USD cost when measurements are unavailable.
