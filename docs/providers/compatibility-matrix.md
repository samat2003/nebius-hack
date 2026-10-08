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
| Generator | `nvidia_build` | `nvidia/nemotron-3-super-120b-a12b` | `https://integrate.api.nvidia.com/v1` | `LIVE_VERIFIED` | `hybrid` |
| Generator | `nebius_token_factory` | Explicit `GENERATOR_MODEL` required | `https://api.tokenfactory.nebius.com/v1` | `PENDING_CREDENTIALS` (`MOCK_VERIFIED`) | `hybrid` (when Nebius credentials supplied) |
| Retriever | `fake` | `google/embeddinggemma-2` | In-process | `MOCK_VERIFIED` | `fake`, `hybrid` |
| Retriever | `embeddinggemma` (`EmbeddingGemmaRetriever`) | `google/embeddinggemma-300m` | Configured HTTPS origin | `BLOCKED_HOSTING` (`MOCK_VERIFIED`) | Direct adapter unit tests (Phase 6 runtime enablement) |
| Controller | `fake` | `samatv256/mini-Jev` | In-process | `MOCK_VERIFIED` | `fake`, `hybrid` |
| Controller | `mini_jev` (`MiniJevController`) | `samatv256/mini-Jev` (`step-010626` / `3a1d1d19d85e9863146307fe4b769e8bbe242c4e`) | Configured HTTPS origin | `BLOCKED_HOSTING` (`MOCK_VERIFIED`) | Direct adapter unit tests (Phase 4/5 runtime enablement) |

---

## 3. Capability & Protocol Coverage Matrix

| Capability / Feature | `nvidia_build` (`NvidiaBuildGenerator`) | `nebius_token_factory` (`NebiusTokenFactoryGenerator`) | `embeddinggemma` (`EmbeddingGemmaRetriever`) | `mini_jev` (`MiniJevController`) |
| --- | --- | --- | --- | --- |
| Non-streaming JSON request/response | `LIVE_VERIFIED` | `MOCK_VERIFIED` (`PENDING_CREDENTIALS`) | `MOCK_VERIFIED` (`BLOCKED_HOSTING`) | `MOCK_VERIFIED` (`BLOCKED_HOSTING`) |
| Streaming (`stream=true`) | `UNSUPPORTED` | `UNSUPPORTED` | `UNSUPPORTED` | `UNSUPPORTED` |
| End-to-end `RequestContext.deadline_monotonic` enforcement | `LIVE_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Safe-to-retry vs. ambiguous-failure retry separation | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Per-attempt circuit breaker (`CLOSED` / `OPEN` / `HALF_OPEN` single-probe) | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Bounded concurrency + waiter queue admission | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Pre-allocation JSON depth & byte caps (`max_request_bytes`, `max_response_bytes`) | `LIVE_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Origin-scoped auth (`follow_redirects=False`, `trust_env=False`, HTTPS required) | `LIVE_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |
| Reasoning suppression (`chat_template_kwargs={"enable_thinking": false}`) | `LIVE_VERIFIED` | N/A | N/A | N/A |
| Reasoning-only / empty `content` rejection (`InvalidProviderResponse`) | `MOCK_VERIFIED` | `MOCK_VERIFIED` | N/A | N/A |
| HTTP `202 Accepted` pending invocation fail-fast (`ProviderTimeout`) | `MOCK_VERIFIED` | N/A | N/A | N/A |
| Matryoshka truncation (`768`/`512`/`256`/`128`) + L2 renormalization | N/A | N/A | `MOCK_VERIFIED` | N/A |
| Mandatory item preservation (`mandatory=True`) | N/A | N/A | `MOCK_VERIFIED` | N/A |
| Single-candidate deterministic bypass (0 network calls) | N/A | N/A | N/A | `MOCK_VERIFIED` |
| Probability simplex & `selected_id` argmax verification | N/A | N/A | N/A | `MOCK_VERIFIED` |
| Nullable token/cost telemetry (`estimated_cost_usd=None` when unverified) | `LIVE_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` | `MOCK_VERIFIED` |

---

## 4. Measured Live NVIDIA API Catalog Evidence (Sanitized)

### 4.1 Gate A — Direct `POST /v1/chat/completions` Contract Probe

- **Timestamp (UTC)**: `2026-10-08T15:10:03.614180+00:00`
- **Endpoint Hostname**: `integrate.api.nvidia.com` (`POST /v1/chat/completions`)
- **Requested & Reported Model**: `nvidia/nemotron-3-super-120b-a12b`
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
