"""Hardened, origin-bound pooled HTTP client for remote Alienese providers.

Enforces:
- Origin-scoped bearer authentication (`scheme`, `host`, `port`, and base path prefix).
- Strict HTTPS requirement in deployment (`allow_insecure_http_for_testing=False` by default).
- Redirect rejection (`follow_redirects=False`) and environment proxy isolation (`trust_env=False`).
- Pre-allocation outbound JSON depth and byte-size validation before network transmission.
- Incremental streaming response consumption with cumulative byte cap before JSON parsing.
- End-to-end turn deadline enforcement across concurrency admission, HTTP attempts, and retries.
- Safe-vs-ambiguous retry policy and per-attempt circuit-breaker accounting.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import SecretStr

from alienese.api.errors import (
    CompatibilityError,
    InvalidProviderResponse,
    ProviderError,
)
from alienese.contracts.context import RequestContext
from alienese.contracts.decisions import ProviderCallTelemetry
from alienese.observability.logging import get_request_logger
from alienese.providers.runtime.circuit_breaker import CircuitState, ProviderCircuitBreaker
from alienese.providers.runtime.concurrency import ProviderConcurrencyLimiter
from alienese.providers.runtime.deadlines import DEFAULT_TURN_TIMEOUT_SECONDS, DeadlineBudget
from alienese.providers.runtime.errors import (
    FailureCategory,
    classify_http_status,
    classify_transport_exception,
    map_http_status_to_error,
    map_transport_exception_to_error,
)
from alienese.providers.runtime.retry import (
    RetryConfig,
    compute_retry_delay_seconds,
    should_retry_failure,
)
from alienese.providers.runtime.telemetry import (
    build_provider_telemetry,
    extract_serving_fingerprint,
    extract_upstream_request_id,
    parse_usage_dict,
)

DEFAULT_MAX_REQUEST_BYTES = 262_144  # 256 KiB
DEFAULT_MAX_RESPONSE_BYTES = 524_288  # 512 KiB
DEFAULT_MAX_JSON_DEPTH = 24
_ALLOWED_TEST_HTTP_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "testserver"})
_ALLOWED_CONTENT_ENCODINGS = frozenset({"", "identity"})


def _is_identity_content_encoding(raw_header: str) -> bool:
    """Return True only if Content-Encoding is absent or strictly 'identity'."""
    encodings = [part.strip().lower() for part in raw_header.split(",") if part.strip()]
    if not encodings:
        return True
    return all(enc in _ALLOWED_CONTENT_ENCODINGS for enc in encodings)


def _effective_port(scheme: str, port: int | None) -> int:
    if port is not None:
        return port
    return 443 if scheme == "https" else 80


def _validate_json_depth(value: Any, *, max_depth: int, current_depth: int = 0) -> None:
    if current_depth > max_depth:
        raise ValueError(f"JSON structure exceeds maximum nesting depth of {max_depth}.")
    if isinstance(value, Mapping):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError("JSON object keys must be strings.")
            _validate_json_depth(v, max_depth=max_depth, current_depth=current_depth + 1)
    elif isinstance(value, (list, tuple)):
        for elem in value:
            _validate_json_depth(elem, max_depth=max_depth, current_depth=current_depth + 1)


@dataclass(frozen=True)
class ProviderHttpResponse:
    """Validated JSON response and operational telemetry from a remote provider call."""

    data: dict[str, Any]
    headers: dict[str, str]
    status_code: int
    telemetry: ProviderCallTelemetry


class ProviderHttpClient:
    """Pooled, origin-bound HTTP client with deadline, retry, circuit-breaker, and byte caps."""

    def __init__(
        self,
        *,
        provider_name: str,
        base_url: str,
        api_key: SecretStr | str,
        default_timeout_seconds: float = DEFAULT_TURN_TIMEOUT_SECONDS,
        attempt_timeout_seconds: float | None = None,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        max_json_depth: int = DEFAULT_MAX_JSON_DEPTH,
        max_connections: int = 20,
        max_keepalive_connections: int = 10,
        retry_config: RetryConfig | None = None,
        circuit_breaker: ProviderCircuitBreaker | None = None,
        concurrency_limiter: ProviderConcurrencyLimiter | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        allow_insecure_http_for_testing: bool = False,
        clock: Callable[[], float] = time.monotonic,
        epoch_clock: Callable[[], float] = time.time,
        sleep_func: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._provider_name = provider_name
        self._allow_insecure_http_for_testing = allow_insecure_http_for_testing
        self._clock = clock
        self._epoch_clock = epoch_clock
        self._sleep = sleep_func

        secret_val = (
            api_key.get_secret_value().strip()
            if isinstance(api_key, SecretStr)
            else str(api_key).strip()
        )
        if not secret_val:
            raise CompatibilityError(
                f"Provider '{provider_name}' requires a non-empty API key.",
                param="api_key",
                code="missing_provider_api_key",
            )
        self._api_key = SecretStr(secret_val)

        cleaned_url = base_url.strip().rstrip("/")
        parsed = urlparse(cleaned_url)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        if not scheme or not hostname:
            raise CompatibilityError(
                f"Provider '{provider_name}' base_url must be a valid absolute URL.",
                param="base_url",
                code="invalid_provider_base_url",
            )
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise CompatibilityError(
                f"Provider '{provider_name}' base_url must not contain userinfo, query, "
                "or fragment components.",
                param="base_url",
                code="invalid_provider_base_url",
            )
        if scheme != "https" and not (
            allow_insecure_http_for_testing
            and scheme == "http"
            and hostname in _ALLOWED_TEST_HTTP_HOSTS
        ):
            raise CompatibilityError(
                f"Provider '{provider_name}' base_url must use HTTPS.",
                param="base_url",
                code="insecure_provider_base_url",
            )

        eff_port = _effective_port(scheme, parsed.port)
        base_path = parsed.path.rstrip("/") or ""
        if ".." in base_path.split("/"):
            raise CompatibilityError(
                f"Provider '{provider_name}' base_url contains invalid path segments.",
                param="base_url",
                code="invalid_provider_base_url",
            )

        self._base_url = cleaned_url
        self._expected_origin = (scheme, hostname, eff_port)
        self._base_path_prefix = base_path

        self._default_timeout_seconds = default_timeout_seconds
        self._attempt_timeout_seconds = attempt_timeout_seconds
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes
        self._max_json_depth = max_json_depth

        self._retry_config = retry_config or RetryConfig()
        self._circuit_breaker = circuit_breaker or ProviderCircuitBreaker(
            provider_name=provider_name,
            clock=clock,
        )
        self._concurrency = concurrency_limiter or ProviderConcurrencyLimiter(
            provider_name=provider_name,
        )

        self._client = httpx.AsyncClient(
            headers={"Accept-Encoding": "identity"},
            timeout=httpx.Timeout(default_timeout_seconds),
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
            follow_redirects=False,
            trust_env=False,
            verify=True,
            transport=transport,
        )
        self._closed = False

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def expected_origin(self) -> tuple[str, str, int]:
        return self._expected_origin

    @property
    def circuit_breaker(self) -> ProviderCircuitBreaker:
        return self._circuit_breaker

    @property
    def concurrency_limiter(self) -> ProviderConcurrencyLimiter:
        return self._concurrency

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def aclose(self) -> None:
        """Close the underlying pooled HTTP client."""
        if not self._closed:
            self._closed = True
            await self._client.aclose()

    def _resolve_and_verify_url(self, endpoint_path: str) -> str:
        """Resolve `endpoint_path` against `base_url` and enforce strict origin/path policy."""
        cleaned_path = endpoint_path.strip()
        if (
            not cleaned_path.startswith("/")
            or "://" in cleaned_path
            or "?" in cleaned_path
            or "#" in cleaned_path
            or ".." in cleaned_path.split("/")
        ):
            raise ProviderError(
                f"Provider '{self._provider_name}' rejected invalid endpoint path.",
                code="invalid_provider_endpoint_path",
            )

        target_url = f"{self._base_url}{cleaned_path}"
        parsed = urlparse(target_url)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower()
        eff_port = _effective_port(scheme, parsed.port)

        if (scheme, hostname, eff_port) != self._expected_origin:
            raise ProviderError(
                f"Provider '{self._provider_name}' target URL origin mismatch; "
                "refusing to transmit credentials.",
                code="provider_origin_mismatch",
            )

        if self._base_path_prefix and not parsed.path.startswith(self._base_path_prefix + "/"):
            raise ProviderError(
                f"Provider '{self._provider_name}' target URL path escaped configured prefix.",
                code="provider_origin_mismatch",
            )

        return target_url

    def _encode_and_validate_request_body(self, payload: Mapping[str, Any]) -> bytes:
        """Validate JSON depth and byte size before transmitting over the network."""
        try:
            _validate_json_depth(payload, max_depth=self._max_json_depth)
            body_str = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise ProviderError(
                f"Provider '{self._provider_name}' outbound request payload failed JSON "
                f"validation: {exc}",
                code="invalid_outbound_payload",
            ) from exc

        body_bytes = body_str.encode("utf-8")
        if len(body_bytes) > self._max_request_bytes:
            raise ProviderError(
                f"Provider '{self._provider_name}' outbound request size ({len(body_bytes)} bytes) "
                f"exceeds maximum limit of {self._max_request_bytes} bytes.",
                code="request_payload_too_large",
                status_code=400,
            )
        return body_bytes

    async def _read_bounded_stream(
        self,
        response: httpx.Response,
        deadline: DeadlineBudget,
    ) -> bytes:
        """Read raw identity response stream enforcing `max_response_bytes` before decompression."""
        content_length_hdr = response.headers.get("content-length")
        if content_length_hdr is not None:
            try:
                declared_len = int(content_length_hdr)
            except ValueError:
                declared_len = -1
            if declared_len > self._max_response_bytes:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' declared Content-Length "
                    f"({declared_len} bytes) exceeds maximum limit of "
                    f"{self._max_response_bytes} bytes.",
                    code="response_payload_too_large",
                )

        buffer = bytearray()
        chunk_iter = (
            response.aiter_raw() if not response.is_stream_consumed else response.aiter_bytes()
        )
        async for chunk in chunk_iter:
            deadline.require_remaining(
                provider_name=self._provider_name,
                phase="response_stream_read",
            )
            if not chunk:
                continue
            if len(buffer) + len(chunk) > self._max_response_bytes:
                raise InvalidProviderResponse(
                    f"Provider '{self._provider_name}' response body exceeded maximum limit "
                    f"of {self._max_response_bytes} bytes.",
                    code="response_payload_too_large",
                )
            buffer.extend(chunk)
        return bytes(buffer)

    async def post_json(
        self,
        ctx: RequestContext,
        endpoint_path: str,
        payload: Mapping[str, Any],
    ) -> ProviderHttpResponse:
        """Execute a bounded, origin-verified POST JSON call with deadline and retry governance."""
        if self._closed:
            raise ProviderError(
                f"ProviderHttpClient for '{self._provider_name}' has already been closed.",
                code="provider_client_closed",
            )

        logger = get_request_logger(ctx, f"provider.{self._provider_name}")
        deadline = DeadlineBudget.from_context(
            ctx,
            default_timeout_seconds=self._default_timeout_seconds,
            clock=self._clock,
        )
        deadline.require_remaining(
            provider_name=self._provider_name,
            phase="pre_request",
        )

        target_url = self._resolve_and_verify_url(endpoint_path)
        body_bytes = self._encode_and_validate_request_body(payload)

        start_monotonic = self._clock()
        retried_prompt_tokens = 0
        retried_completion_tokens = 0
        has_retried_usage = False

        for attempt_no in range(1, self._retry_config.max_attempts + 1):
            deadline.require_remaining(
                provider_name=self._provider_name,
                phase=f"attempt_{attempt_no}_admission",
            )
            # Fail fast before waiting in concurrency queue if circuit is already OPEN
            if self._circuit_breaker.state == CircuitState.OPEN:
                self._circuit_breaker.before_attempt()

            retry_delay: float | None = None

            async with self._concurrency.acquire(deadline):
                attempt_timeout = deadline.attempt_timeout_seconds(
                    self._attempt_timeout_seconds,
                    provider_name=self._provider_name,
                    phase=f"attempt_{attempt_no}",
                )
                was_half_open_probe = self._circuit_breaker.before_attempt()

                headers = {
                    "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "X-Client-Request-ID": ctx.request_id,
                }
                req = self._client.build_request(
                    "POST",
                    target_url,
                    headers=headers,
                    content=body_bytes,
                    timeout=httpx.Timeout(attempt_timeout),
                )

                response: httpx.Response | None = None
                probe_accounted = False
                try:
                    response = await self._client.send(req, stream=True)
                    status_code = response.status_code
                    resp_headers = {k.lower(): v for k, v in response.headers.items()}
                    upstream_req_id = extract_upstream_request_id(resp_headers)

                    content_encoding = resp_headers.get("content-encoding", "").strip()
                    if not _is_identity_content_encoding(content_encoding):
                        self._circuit_breaker.record_failure(
                            FailureCategory.NON_RETRYABLE_CONTRACT,
                            was_half_open_probe=was_half_open_probe,
                        )
                        probe_accounted = True
                        raise InvalidProviderResponse(
                            f"Provider '{self._provider_name}' returned unsolicited or "
                            f"unsupported Content-Encoding '{content_encoding.lower()}'; "
                            "only 'identity' is permitted.",
                            code="unsupported_content_encoding",
                        )

                    raw_body = await self._read_bounded_stream(response, deadline)

                    if status_code != 200:
                        # Check if failed response reported any usage tokens
                        try:
                            err_json = json.loads(raw_body.decode("utf-8"))
                            if isinstance(err_json, Mapping):
                                p_t, c_t, _ = parse_usage_dict(err_json.get("usage"))
                                if p_t is not None or c_t is not None:
                                    has_retried_usage = True
                                    retried_prompt_tokens += p_t or 0
                                    retried_completion_tokens += c_t or 0
                        except Exception:
                            pass

                        category = classify_http_status(status_code)
                        self._circuit_breaker.record_failure(
                            category,
                            was_half_open_probe=was_half_open_probe,
                        )
                        probe_accounted = True

                        if should_retry_failure(
                            category,
                            attempt_no=attempt_no,
                            config=self._retry_config,
                        ):
                            retry_delay = compute_retry_delay_seconds(
                                attempt_no=attempt_no,
                                config=self._retry_config,
                                retry_after_header=resp_headers.get("retry-after"),
                                deadline=deadline,
                                provider_name=self._provider_name,
                                now_epoch=self._epoch_clock(),
                            )
                            logger.warning(
                                "provider_http_retry_scheduled",
                                provider=self._provider_name,
                                attempt_no=attempt_no,
                                upstream_status=status_code,
                                delay_seconds=round(retry_delay, 3),
                            )
                        else:
                            raise map_http_status_to_error(
                                provider_name=self._provider_name,
                                status_code=status_code,
                                upstream_request_id=upstream_req_id,
                            )
                    else:
                        # HTTP 200 validation
                        content_type = resp_headers.get("content-type", "").strip().lower()
                        media_type = content_type.split(";", 1)[0].strip()
                        if media_type != "application/json":
                            self._circuit_breaker.record_failure(
                                FailureCategory.NON_RETRYABLE_CONTRACT,
                                was_half_open_probe=was_half_open_probe,
                            )
                            probe_accounted = True
                            raise InvalidProviderResponse(
                                f"Provider '{self._provider_name}' returned unsupported "
                                f"Content-Type '{media_type or 'missing'}'; "
                                "expected 'application/json'.",
                                code="unsupported_media_type",
                            )

                        try:
                            parsed_json = json.loads(raw_body.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            self._circuit_breaker.record_failure(
                                FailureCategory.NON_RETRYABLE_CONTRACT,
                                was_half_open_probe=was_half_open_probe,
                            )
                            probe_accounted = True
                            raise InvalidProviderResponse(
                                f"Provider '{self._provider_name}' returned malformed "
                                "JSON response.",
                                code="malformed_provider_json",
                            ) from exc

                        if not isinstance(parsed_json, dict):
                            self._circuit_breaker.record_failure(
                                FailureCategory.NON_RETRYABLE_CONTRACT,
                                was_half_open_probe=was_half_open_probe,
                            )
                            probe_accounted = True
                            raise InvalidProviderResponse(
                                f"Provider '{self._provider_name}' response root must be "
                                "a JSON object.",
                                code="invalid_provider_response",
                            )

                        try:
                            _validate_json_depth(parsed_json, max_depth=self._max_json_depth)
                        except ValueError as exc:
                            self._circuit_breaker.record_failure(
                                FailureCategory.NON_RETRYABLE_CONTRACT,
                                was_half_open_probe=was_half_open_probe,
                            )
                            probe_accounted = True
                            raise InvalidProviderResponse(
                                f"Provider '{self._provider_name}' response exceeded maximum "
                                "JSON nesting depth.",
                                code="response_json_depth_exceeded",
                            ) from exc

                        self._circuit_breaker.record_success(
                            was_half_open_probe=was_half_open_probe,
                        )
                        probe_accounted = True

                        elapsed_ms = (self._clock() - start_monotonic) * 1000.0
                        prompt_tok, completion_tok, total_tok = parse_usage_dict(
                            parsed_json.get("usage")
                        )
                        telemetry = build_provider_telemetry(
                            latency_ms=elapsed_ms,
                            request_attempt_id=ctx.request_id,
                            upstream_request_id=upstream_req_id,
                            serving_fingerprint=extract_serving_fingerprint(parsed_json),
                            attempt_count=attempt_no,
                            failed_attempt_count=attempt_no - 1,
                            prompt_tokens=prompt_tok,
                            completion_tokens=completion_tok,
                            total_tokens=total_tok,
                            retried_prompt_tokens=(
                                retried_prompt_tokens if has_retried_usage else None
                            ),
                            retried_completion_tokens=(
                                retried_completion_tokens if has_retried_usage else None
                            ),
                            estimated_cost_usd=None,
                        )
                        return ProviderHttpResponse(
                            data=parsed_json,
                            headers=resp_headers,
                            status_code=status_code,
                            telemetry=telemetry,
                        )

                except httpx.HTTPError as exc:
                    category = classify_transport_exception(exc)
                    if not probe_accounted:
                        self._circuit_breaker.record_failure(
                            category,
                            was_half_open_probe=was_half_open_probe,
                        )
                        probe_accounted = True

                    if should_retry_failure(
                        category,
                        attempt_no=attempt_no,
                        config=self._retry_config,
                    ):
                        retry_delay = compute_retry_delay_seconds(
                            attempt_no=attempt_no,
                            config=self._retry_config,
                            retry_after_header=None,
                            deadline=deadline,
                            provider_name=self._provider_name,
                            now_epoch=self._epoch_clock(),
                        )
                        logger.warning(
                            "provider_transport_retry_scheduled",
                            provider=self._provider_name,
                            attempt_no=attempt_no,
                            error_type=type(exc).__name__,
                            delay_seconds=round(retry_delay, 3),
                        )
                    else:
                        raise map_transport_exception_to_error(
                            provider_name=self._provider_name,
                            exc=exc,
                        ) from exc
                finally:
                    if not probe_accounted:
                        self._circuit_breaker.release_aborted_probe(
                            was_half_open_probe=was_half_open_probe,
                        )
                    if response is not None:
                        await response.aclose()

            # Concurrency permit is released before sleeping for retry backoff!
            if retry_delay is not None:
                await self._sleep(retry_delay)

        raise ProviderError(
            f"Provider '{self._provider_name}' exhausted all configured attempts.",
            code="provider_attempts_exhausted",
        )
