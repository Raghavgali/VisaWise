"""OpenTelemetry wiring: traces -> Langfuse Cloud via OTLP/HTTP.

Off by default. With no Langfuse keys / OTLP endpoint in the env,
`init_observability()` returns False without touching global state — no
exporter, no background thread, no network. Local dev and tests run exactly
as before. The OTel SDK imports live inside `init_observability` for the same
reason: an unconfigured process never pays for them.

Configuration (see config.py):
  * LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY  -> endpoint and Basic-auth
    header are derived (Langfuse speaks native OTLP/HTTP).
  * OTEL_EXPORTER_OTLP_ENDPOINT (+ _HEADERS)   -> overrides the derivation,
    so any OTLP backend works without code changes.

Privacy stance: the google-genai instrumentation records gen_ai spans with
model/token metadata but captures NO prompt or completion content unless
OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT is explicitly enabled —
user queries about their immigration status never leave the request path.
"""

from __future__ import annotations

import base64
import functools
import logging
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Set only when init_observability() actually configured a provider; the
# handle shutdown_observability() flushes through.
_tracer_provider = None


def _otlp_traces_config() -> tuple[str, dict[str, str]] | None:
    """Resolve (endpoint, headers) for the trace exporter, or None = disabled."""
    if settings.otel_exporter_otlp_endpoint:
        # Standard OTLP header syntax: "k1=v1,k2=v2".
        headers = {}
        for pair in settings.otel_exporter_otlp_headers.split(","):
            key, sep, value = pair.partition("=")
            if sep:
                headers[key.strip()] = value.strip()
        return settings.otel_exporter_otlp_endpoint, headers

    if settings.langfuse_public_key and settings.langfuse_secret_key:
        token = base64.b64encode(
            f"{settings.langfuse_public_key}:{settings.langfuse_secret_key}".encode()
        ).decode()
        endpoint = f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces"
        return endpoint, {"Authorization": f"Basic {token}"}

    return None


def init_observability(app: FastAPI) -> bool:
    """Configure tracing if the env asks for it. Returns whether it did.

    Called from the app lifespan before serving traffic. Sets up, in order:
    a TracerProvider with service identity, a batching OTLP/HTTP exporter,
    FastAPI server spans (excluding /health — Modal pings it), and gen_ai
    spans for every Gemini call.
    """
    global _tracer_provider

    config = _otlp_traces_config()
    if config is None:
        logger.info("observability disabled: no Langfuse/OTLP configuration in env")
        return False
    endpoint, headers = config

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    try:
        service_version = version("visawise")
    except PackageNotFoundError:
        service_version = "unknown"

    resource = Resource.create(
        {"service.name": settings.otel_service_name, "service.version": service_version}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
    )
    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
    GoogleGenAiSdkInstrumentor().instrument()

    logger.info("observability enabled: traces -> %s", endpoint)
    return True


def shutdown_observability() -> None:
    """Flush pending spans, then tear the provider down.

    Must run in the lifespan `finally:` — Modal freezes scale-to-zero
    containers without warning, and the BatchSpanProcessor holds spans in
    memory for up to its schedule delay. An unflushed final request is a
    trace that never existed.
    """
    global _tracer_provider
    if _tracer_provider is None:
        return
    _tracer_provider.force_flush(timeout_millis=5_000)
    _tracer_provider.shutdown()
    _tracer_provider = None


def get_tracer():
    """The app's tracer. Safe to call when disabled: OTel's global tracer is
    a no-op until a provider is installed, so spans simply vanish for free."""
    from opentelemetry import trace

    return trace.get_tracer("visawise")


def current_span():
    """The active span (the request's server span inside an endpoint), or a
    no-op span when telemetry is disabled — callers never need to check."""
    from opentelemetry import trace

    return trace.get_current_span()


def record_first_token(span, ttft_ms: float) -> None:
    """Mark time-to-first-token on the request span.

    On a streaming endpoint TTFT is the latency users actually feel; the
    span's total duration (which includes streaming the whole answer) would
    overstate it badly.
    """
    span.add_event("first_token", {"ttft_ms": round(ttft_ms, 1)})
    span.set_attribute("visawise.ttft_ms", round(ttft_ms, 1))


def traced_node(name: str, fn: Callable) -> Callable:
    """Wrap a LangGraph node callable in a span named ``rag.<name>``.

    The graph's nodes are sync functions, so an async invocation (the app's
    ``astream``) runs them in a worker thread; langchain-core copies
    contextvars into that thread, which is what lets these spans nest under
    the request's server span — tests/test_observability.py pins that
    behavior so a langgraph upgrade can't silently orphan the spans.

    Only bounded metadata goes on the span (counts, scores, lengths); query
    and chunk text never do.
    """

    @functools.wraps(fn)
    def wrapper(state):
        with get_tracer().start_as_current_span(f"rag.{name}") as span:
            span.set_attribute("rag.stage", name)
            result = fn(state)
            _set_stage_attributes(span, result)
            return result

    return wrapper


def _set_stage_attributes(span, result) -> None:
    if not isinstance(result, dict):
        return
    for key in ("dense_hits", "bm25_hits", "fused", "reranked"):
        hits = result.get(key)
        if isinstance(hits, list):
            span.set_attribute(f"rag.{key}.count", len(hits))
            top_score = getattr(hits[0], "score", None) if hits else None
            if isinstance(top_score, (int, float)):
                span.set_attribute(f"rag.{key}.top_score", float(top_score))
    answer = result.get("answer")
    if isinstance(answer, str):
        span.set_attribute("rag.answer.chars", len(answer))
        citations = result.get("citations")
        if isinstance(citations, list):
            span.set_attribute("rag.citations.count", len(citations))
