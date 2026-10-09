# Alienese Provider Compatibility Matrix (Phase 2)

This document records the verification status, capability coverage, and measured live probe evidence for each provider adapter in the Alienese runtime.

## 1. Status Vocabulary

| Status | Meaning |
| --- | --- |
| `LIVE_VERIFIED` | Verified against a live remote endpoint with a bounded, non-sensitive synthetic probe in addition to offline mock-transport tests. |
| `MOCK_VERIFIED` | Verified offline via `httpx.MockTransport` unit and contract tests covering success, retry, circuit-breaker, deadline, and malformed-response cases. |
| `PENDING_CREDENTIALS` | Adapter and behavioral parity tests are implemented (`MOCK_VERIFIED`), awaiting target provider credentials/credits for live verification. |
| `BLOCKED_HOSTING` | Adapter contract and offline validation tests are implemented (`MOCK_VERIFIED`), awaiting remote hosting endpoint deployment. |
| `DOCUMENTED` | Contract and specification documented in repository ADRs/architecture docs. |
| `UNSUPPORTED` | Explicitly rejected with a typed `CompatibilityError` (`HTTP 400`). |

---

## 2. Provider Verification Matrix

| Role | Provider Identifier | Default Model / Checkpoint | Endpoint Origin | Phase 2 Status | Active in Runtime Modes |
| --- | --- | --- | --- | --- | --- |
| Generator | `fake` | `nvidia/nemotron` | In-process | `MOCK_VERIFIED` | `fake` (default) |
| Generator | `nvidia_build` | `nvidia/nemotron-3-super-120b-a12b` | `https://integrate.api.nvidia.com/v1` | `LIVE_VERIFIED` | `hybrid` (`GENERATOR_PROVIDER=nvidia_build`) |
| Generator | `nebius_token_factory` | `nvidia/nemotron-3-super-120b-a12b` | `https://api.tokenfactory.us-central1.nebius.com/v1` | `LIVE_VERIFIED` | `hybrid` (`GENERATOR_PROVIDER=nebius_token_factory`) |
| Retriever | `fake` | `google/embeddinggemma-2` | In-process | `MOCK_VERIFIED` | `fake`, `hybrid` |
| Retriever | `embeddinggemma` (`EmbeddingGemmaRetriever`) | `google/embeddinggemma-2` | Configured HTTPS origin | `BLOCKED_HOSTING` (`MOCK_VERIFIED`) | Direct adapter unit tests (Phase 6 runtime enablement) |
| Controller | `fake` | `samatv256/mini-Jev` | In-process | `MOCK_VERIFIED` | `fake`, `hybrid` |
| Controller | `mini_jev` (`MiniJevController`) | `samatv256/mini-Jev` (`step-010626` / `3a1d1d19d85e9863146307fe4b769e8bbe242c4e`) | Configured HTTPS origin | `BLOCKED_HOSTING` (`MOCK_VERIFIED`) | Direct adapter unit tests (Phase 4/5 runtime enablement) |

---

## 3. Capability & Protocol Coverage Matrix

| Capability / Feature | `nvidia_build` (`NvidiaBuildGenerator`) | `nebius_token_factory` (`NebiusTokenFactoryGenerator`) | `embeddinggemma` (`EmbeddingGemmaRetriever`) | `mini_jev` (`MiniJevController`) |
| --- | --- | --- | --- | --- |
| Non-streaming JSON request/response | `LIVE_VERIFIED` | `LIVE_VERIFIED` | `MOCK_VERIFIED` (`BLOCKED_HOSTING`) | `MOCK_VERIFIED` (`BLOCKED_HOSTING`) |
| Streaming (`stream=true`) | `UNSUPPORTED` | `UNSUPPORTED` | `UNSUPPORTED` | `UNSUPPORTED` |
| End-to-end `RequestContext.deadline_monotonic` enforcement | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Safe-to-retry vs. ambiguous-failure retry separation | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Per-attempt circuit breaker (`CLOSED` / `OPEN` / `HALF_OPEN` single-probe) | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Bounded per-attempt concurrency (permit released during retry backoff) | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Pre-allocation JSON depth, raw byte caps (`max_request_bytes`, `max_response_bytes`), & `Accept-Encoding: identity` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Origin-scoped auth (`follow_redirects=False`, `trust_env=False`, HTTPS required) | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Reasoning suppression (`chat_template_kwargs={"enable_thinking": false}`) | `LIVE_VERIFIED` | `LIVE_VERIFIED` | N/A | N/A |
| Reasoning-only / empty `content` rejection (`InvalidProviderResponse`) | `MOCK_VERIFIED` | `MOCK_VERIFIED` | N/A | N/A |
| HTTP `202 Accepted` pending invocation fail-fast (`ProviderUnavailable`) | `MOCK_VERIFIED` | `MOCK_VERIFIED` | N/A | N/A |
| Matryoshka truncation (`768`/`512`/`256`/`128`) + L2 renormalization | N/A | N/A | `MOCK_VERIFIED` | N/A |
| Mandatory item preservation (`mandatory=True`) | N/A | N/A | `MOCK_VERIFIED` | N/A |
| Single-candidate deterministic bypass (0 network calls) | N/A | N/A | N/A | `MOCK_VERIFIED` |
| Probability simplex & `selected_id` argmax verification | N/A | N/A | N/A | `MOCK_VERIFIED` |
| Nullable token/cost telemetry (`estimated_cost_usd=None` when unverified) & `serving_fingerprint` vs. `model_revision="unknown"` | `LIVE_VERIFIED` | `LIVE_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |

---

## 4. Measured Live NVIDIA API Catalog Evidence (Sanitized)

### 4.1 Gate A — Direct `POST /v1/chat/completions` Contract Probe

- **Timestamp (UTC)**: `2026-10-08T15:10:03.614180+00:00`
- **Endpoint Hostname**: `integrate.api.nvidia.com` (`POST /v1/chat/completions`)
- **Requested & Reported Model**: `nvidia/nemotron-3-super-120b-a12b` (`model_revision="unknown"`, `system_fingerprint` recorded as `ProviderCallTelemetry.serving_fingerprint`)
- **Request Parameters**: `max_tokens=64`, `temperature=1.0`, `top_p=0.95`, `stream=False`, `chat_template_kwargs={"enable_thinking": False}`
- **HTTP Status**: `200 OK` (`nvcf-status: fulfilled`)
- **Upstream Request ID Header**: `nvcf-reqid: 0684d371-d4c8-42ad-90cf-6c1fee135bd6`
- **Observed Top-Level JSON Keys**: `["choices", "created", "id", "model", "object", "service_tier", "system_fingerprint", "usage"]`
- **Observed Choice Structure**:
  - `choices_count`: `1`
  - `choice_keys`: `["finish_reason", "index", "logprobs", "message"]`
  - `message_keys`: `["content", "reasoning_content", "role"]`
  - `message.role`: `"assistant"`
  - `message.content`: non-empty `str` (`147` chars)
  - `message.reasoning_content`: `null` (confirmed suppressed when `enable_thinking=False`)
  - `finish_reason`: `"stop"`
- **Observed Usage Block**:
  - `prompt_tokens`: `27`
  - `completion_tokens`: `24`
  - `total_tokens`: `51`
  - `completion_tokens_details.reasoning_tokens`: `0`
- **Measured Latency**: `704.12 ms`
- **Verification Outcome**: `LIVE_VERIFIED`

### 4.2 Gate D — Opt-In End-to-End Hybrid Gateway Turn (`tests/integration/test_live_nvidia.py`)

- **Command**: `RUN_LIVE_NVIDIA_TESTS=1 pytest -m live_nvidia -v`
- **Provider Mode**: `ALIENESE_PROVIDER_MODE=hybrid`, `GENERATOR_PROVIDER=nvidia_build`, `RETRIEVER_PROVIDER=fake`, `CONTROLLER_PROVIDER=fake`
- **Budget Governance**: `MAX_LIVE_REQUESTS=2` (executed `1`), `MAX_OUTPUT_TOKENS_PER_REQUEST=64`, `MAX_ATTEMPTS_PER_REQUEST=1`, `MAX_TEST_DURATION_SECONDS=30.0`
- **Result**: `1 passed` (`HTTP 200`, `model="alienese-default"`, `finish_reason="stop"`, non-empty `choices[0].message.content`, populated `usage`, `TraceMode.METADATA_ONLY` artifact verified free of secrets).

---

## 5. Measured Live Nebius Token Factory Evidence (Sanitized)

### 5.1 Model Catalog Verification (`GET /v1/models`)

- **Endpoint Hostname**: `api.tokenfactory.us-central1.nebius.com` (`GET /v1/models`)
- **HTTP Status**: `200 OK`
- **Upstream Request ID Header**: `x-request-id: 8f150dffacc7709287b148d5c2502a89`
- **Measured Latency**: `262.48 ms`
- **Catalog Size**: `18` models
- **Confirmed Nemotron 3 Super Entry**: `{"id": "nvidia/nemotron-3-super-120b-a12b", "object": "model", "owned_by": "system"}`

### 5.2 Direct `POST /v1/chat/completions` Contract Probe

- **Timestamp (UTC)**: `2026-10-08T20:43:51.663635+00:00`
- **Endpoint Hostname**: `api.tokenfactory.us-central1.nebius.com` (`POST /v1/chat/completions`)
- **Requested & Reported Model**: `nvidia/nemotron-3-super-120b-a12b` (`model_revision="unknown"`, `system_fingerprint="vllm-0.1.dev1+g5001743e3-dp2-9bbaa064"` recorded as `ProviderCallTelemetry.serving_fingerprint`)
- **Request Parameters**: `max_tokens=64`, `temperature=0.7`, `top_p=0.95`, `stream=False`, `chat_template_kwargs={"enable_thinking": False}`
- **HTTP Status**: `200 OK`
- **Upstream Request ID Header**: `x-request-id: ac1a8bc8932fd09dd55ad462798a3560`
- **Observed Top-Level JSON Keys**: `["choices", "created", "ec_transfer_params", "id", "kv_transfer_params", "metrics", "model", "moderation", "object", "prompt_logprobs", "prompt_text", "prompt_token_ids", "service_tier", "system_fingerprint", "usage"]`
- **Observed Choice Structure**:
  - `choices_count`: `1`
  - `choice_keys`: `["finish_reason", "index", "logprobs", "message", "routed_experts", "stop_reason", "token_ids"]`
  - `message_keys`: `["annotations", "audio", "content", "function_call", "reasoning", "refusal", "role", "tool_calls"]`
  - `message.role`: `"assistant"`
  - `message.content`: non-empty `str` (`198` chars)
  - `message.reasoning`: `null` (confirmed suppressed when `chat_template_kwargs={"enable_thinking": False}`)
  - `finish_reason`: `"stop"`
- **Observed Usage Block**:
  - `prompt_tokens`: `37`
  - `completion_tokens`: `36`
  - `total_tokens`: `73`
  - `completion_tokens_details.reasoning_tokens`: `0`
- **Measured Latency**: `522.34 ms`
- **Verification Outcome**: `LIVE_VERIFIED`

### 5.3 Opt-In End-to-End Nebius Hybrid Gateway Turn (`tests/integration/test_live_nebius.py`)

- **Command**: `RUN_LIVE_NEBIUS_TESTS=1 pytest -m live_nebius -v`
- **Provider Mode**: `ALIENESE_PROVIDER_MODE=hybrid`, `GENERATOR_PROVIDER=nebius_token_factory`, `RETRIEVER_PROVIDER=fake`, `CONTROLLER_PROVIDER=fake`
- **Credential Isolation**: `NEBIUS_TOKEN_FACTORY_KEY` resolved exclusively (`generator_api_key=None`)
- **Budget Governance**: `MAX_LIVE_REQUESTS=1` (executed `1` + `1` idempotent cache replay), `MAX_OUTPUT_TOKENS_PER_REQUEST=64`, `MAX_ATTEMPTS_PER_REQUEST=1`, `MAX_TEST_DURATION_SECONDS=30.0`
- **Result**: `1 passed` (`HTTP 200`, `model="alienese-default"`, `finish_reason="stop"`, `prompt_tokens=34`, `completion_tokens=35`, `total_tokens=69`, `X-Idempotent-Replay` verified, `TraceMode.METADATA_ONLY` artifact verified free of secrets and raw prompts).


