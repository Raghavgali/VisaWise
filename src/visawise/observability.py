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
import logging
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
