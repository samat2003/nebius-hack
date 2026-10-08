"""FastAPI application factory for the Alienese runtime service.

Exposes:
- GET /health
- GET /v1/models
- POST /v1/chat/completions
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from alienese import __version__
from alienese.api.chat_completions import build_request_context, router
from alienese.api.errors import AlieneseError, ProtocolError
from alienese.api.models import HealthResponse, ModelInfo, ModelListResponse
from alienese.config import Settings
from alienese.contracts.context import RequestContext
from alienese.contracts.providers import Controller, Generator, Retriever
from alienese.engine.idempotency import IdempotencyCoordinator
from alienese.engine.turn import TurnEngine
from alienese.observability.logging import configure_logging, get_request_logger
from alienese.observability.tracing import RuntimeTracer
from alienese.providers.fake import FakeController, FakeGenerator, FakeRetriever
from alienese.storage.idempotency import IdempotencyStore, InMemoryIdempotencyStore
from alienese.storage.traces import InMemoryTraceStore, TraceStore


def _extract_request_context(request: Request) -> RequestContext:
    existing = getattr(request.state, "request_context", None)
    if isinstance(existing, RequestContext):
        return existing
    return build_request_context(
        x_request_id=request.headers.get("X-Request-ID"),
        x_trace_id=request.headers.get("X-Trace-ID"),
        traceparent=request.headers.get("traceparent"),
        idempotency_key=request.headers.get("Idempotency-Key"),
    )


def create_app(
    *,
    settings: Settings | None = None,
    retriever: Retriever | None = None,
    controller: Controller | None = None,
    generator: Generator | None = None,
    idempotency_store: IdempotencyStore | None = None,
    trace_store: TraceStore | None = None,
    tracer: RuntimeTracer | None = None,
) -> FastAPI:
    """Create and configure the Alienese FastAPI application."""
    cfg = settings or Settings()
    configure_logging(cfg.alienese_log_level)

    eff_retriever = retriever or FakeRetriever(model_id=cfg.retriever_model)
    eff_controller = controller or FakeController(model_id=cfg.controller_model)
    eff_generator = generator or FakeGenerator(model_id=cfg.generator_model)
    eff_idem_store = idempotency_store or InMemoryIdempotencyStore()
    eff_trace_store = trace_store if trace_store is not None else InMemoryTraceStore()
    eff_tracer = tracer or RuntimeTracer()

    app = FastAPI(
        title="Alienese Runtime",
        version=__version__,
        docs_url=None,
        redoc_url=None,
    )

    app.state.settings = cfg
    app.state.trace_store = eff_trace_store
    app.state.idempotency_store = eff_idem_store
    app.state.idempotency = IdempotencyCoordinator(eff_idem_store)
    app.state.turn_engine = TurnEngine(
        retriever=eff_retriever,
        controller=eff_controller,
        generator=eff_generator,
        trace_store=eff_trace_store,
        tracer=eff_tracer,
    )

    @app.exception_handler(AlieneseError)
    async def _handle_alienese_error(request: Request, exc: AlieneseError) -> JSONResponse:
        ctx = _extract_request_context(request)
        logger = get_request_logger(ctx, "api.errors")
        logger.warning(
            "api_error_raised",
            error_type=exc.error_type,
            error_code=exc.code,
            status_code=exc.status_code,
        )
        envelope = exc.to_envelope(
            request_id=ctx.request_id,
            operation_id=ctx.operation_id,
            correlation_trace_id=ctx.correlation_trace_id,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=envelope.model_dump(mode="json"),
            headers={
                "X-Request-ID": ctx.request_id,
                "X-Operation-ID": ctx.operation_id,
                "X-Trace-ID": ctx.correlation_trace_id,
            },
        )

    @app.exception_handler(ValidationError)
    async def _handle_pydantic_validation_error(
        request: Request,
        exc: ValidationError,
    ) -> JSONResponse:
        ctx = _extract_request_context(request)
        errs = exc.errors()
        if errs:
            first_err = errs[0]
            loc = ".".join(str(part) for part in first_err.get("loc", ())) or None
            msg = str(first_err.get("msg", "Invalid request payload."))
        else:
            loc = None
            msg = "Invalid request payload."
        proto_err = ProtocolError(msg, param=loc, code="validation_error")
        envelope = proto_err.to_envelope(
            request_id=ctx.request_id,
            operation_id=ctx.operation_id,
            correlation_trace_id=ctx.correlation_trace_id,
        )
        return JSONResponse(
            status_code=proto_err.status_code,
            content=envelope.model_dump(mode="json"),
            headers={
                "X-Request-ID": ctx.request_id,
                "X-Operation-ID": ctx.operation_id,
                "X-Trace-ID": ctx.correlation_trace_id,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        ctx = _extract_request_context(request)
        errs = exc.errors()
        if errs:
            first_err = errs[0]
            loc = ".".join(str(part) for part in first_err.get("loc", ())) or None
            msg = str(first_err.get("msg", "Malformed request payload."))
        else:
            loc = None
            msg = "Malformed request payload."
        proto_err = ProtocolError(msg, param=loc, code="validation_error")
        envelope = proto_err.to_envelope(
            request_id=ctx.request_id,
            operation_id=ctx.operation_id,
            correlation_trace_id=ctx.correlation_trace_id,
        )
        return JSONResponse(
            status_code=proto_err.status_code,
            content=envelope.model_dump(mode="json"),
            headers={
                "X-Request-ID": ctx.request_id,
                "X-Operation-ID": ctx.operation_id,
                "X-Trace-ID": ctx.correlation_trace_id,
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def get_health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            provider_mode=cfg.alienese_provider_mode,
        )

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        return ModelListResponse(
            data=[
                ModelInfo(id="alienese-default"),
                ModelInfo(id=cfg.controller_model),
                ModelInfo(id=cfg.generator_model),
            ]
        )

    app.include_router(router)
    return app
