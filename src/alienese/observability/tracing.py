"""OpenTelemetry tracing hooks for TurnEngine phases.

Invariants enforced:
- Uses W3C `traceparent` propagation when provided; never forces arbitrary
  application correlation IDs (`X-Trace-ID`) into OpenTelemetry internal trace IDs.
- Records only safe metadata attributes (IDs, counts, digests) and never raw source
  code, prompts, or tool output.
- Exporter or telemetry errors are caught and isolated so observability never fails inference.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from alienese.contracts.context import RequestContext, is_valid_traceparent
from alienese.observability.redaction import is_sensitive_key, redact_string

_PROPAGATOR = TraceContextTextMapPropagator()

# Forbidden attribute keys that might carry raw prompt/source/tool content
_FORBIDDEN_CONTENT_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "content",
        "prompt",
        "messages",
        "tool_output",
        "source_code",
        "raw_body",
    }
)


def sanitize_span_attributes(
    attributes: dict[str, str | int | float | bool] | None,
) -> dict[str, str | int | float | bool]:
    """Filter and redact span attributes so raw content and secrets never enter spans."""
    if not attributes:
        return {}
    clean: dict[str, str | int | float | bool] = {}
    for key, val in attributes.items():
        norm_key = key.strip().lower()
        if norm_key in _FORBIDDEN_CONTENT_ATTRIBUTES or is_sensitive_key(norm_key):
            continue
        if isinstance(val, str):
            clean[key] = redact_string(val)
        elif isinstance(val, (int, float, bool)):
            clean[key] = val
    return clean


class RuntimeTracer:
    """Best-effort OpenTelemetry tracer wrapper for the Alienese TurnEngine."""

    def __init__(
        self,
        *,
        provider: TracerProvider | None = None,
        exporter: SpanExporter | None = None,
    ) -> None:
        if provider is not None:
            self._provider = provider
        else:
            self._provider = TracerProvider()
            if exporter is not None:
                self._provider.add_span_processor(SimpleSpanProcessor(exporter))
        self._tracer = self._provider.get_tracer("alienese.runtime", "0.1.0")

    @contextmanager
    def start_turn_span(
        self,
        ctx: RequestContext,
        attributes: dict[str, str | int | float | bool] | None = None,
    ) -> Iterator[Any]:
        """Start the root `alienese.turn` span with optional W3C traceparent context."""
        parent_ctx: Any = None
        try:
            if ctx.traceparent and is_valid_traceparent(ctx.traceparent):
                parent_ctx = _PROPAGATOR.extract({"traceparent": ctx.traceparent})
        except Exception:
            parent_ctx = None

        merged_attrs: dict[str, str | int | float | bool] = {
            "alienese.request_id": ctx.request_id,
            "alienese.operation_id": ctx.operation_id,
            "alienese.correlation_trace_id": ctx.correlation_trace_id,
        }
        if attributes:
            merged_attrs.update(attributes)

        with self.span("alienese.turn", attributes=merged_attrs, parent_context=parent_ctx) as span:
            yield span

    @contextmanager
    def span(
        self,
        name: str,
        *,
        attributes: dict[str, str | int | float | bool] | None = None,
        parent_context: otel_context.Context | None = None,
    ) -> Iterator[Any]:
        """Create a child span whose telemetry errors never propagate to inference."""
        safe_attrs = sanitize_span_attributes(attributes)
        cm: Any = None
        span_obj: Any = None
        try:
            cm = self._tracer.start_as_current_span(
                name,
                context=parent_context,
                attributes=safe_attrs,
            )
            span_obj = cm.__enter__()
        except Exception:
            cm = None
            span_obj = _NoOpSpan()

        try:
            yield span_obj
        finally:
            if cm is not None:
                with suppress(Exception):
                    cm.__exit__(None, None, None)

    @staticmethod
    def set_attributes_safe(
        span: Any,
        attributes: dict[str, str | int | float | bool],
    ) -> None:
        """Safely attach sanitized attributes to an active span."""
        if span is None:
            return
        with suppress(Exception):
            safe_attrs = sanitize_span_attributes(attributes)
            for key, val in safe_attrs.items():
                span.set_attribute(key, val)


class _NoOpSpan:
    """Fallback span used if OpenTelemetry initialization or span creation fails."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None
