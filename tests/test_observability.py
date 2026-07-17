"""Observability contracts.

Two things must hold or the telemetry work is worse than useless:
  1. With no Langfuse/OTLP env, everything is a strict no-op (local dev,
     evals, and this test suite pay nothing).
  2. Stage spans nest under the request span even though LangGraph runs the
     sync node callables in a worker thread — langchain-core copies
     contextvars across that hop. Pinned here so a langgraph upgrade can't
     silently orphan every rag.* span into its own parentless trace.
"""

import asyncio
from typing import TypedDict

from fastapi import FastAPI
from langgraph.graph import END, START, StateGraph

from visawise import observability
from visawise.config import settings
from visawise.observability import traced_node


class _State(TypedDict, total=False):
    query: str
    fused: list
    answer: str
    citations: list


def test_init_is_noop_without_env(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_public_key", "")
    monkeypatch.setattr(settings, "langfuse_secret_key", "")
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", "")

    assert observability.init_observability(FastAPI()) is False
    observability.shutdown_observability()  # harmless when nothing was set up


def test_langfuse_keys_derive_otlp_endpoint_and_auth(monkeypatch):
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", "")
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-x")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-y")

    endpoint, headers = observability._otlp_traces_config()

    assert endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert headers["Authorization"].startswith("Basic ")


def test_traced_node_is_transparent_when_disabled():
    def node(state):
        return {"answer": "ok", "citations": []}

    wrapped = traced_node("generate", node)

    assert wrapped({"query": "q"}) == {"answer": "ok", "citations": []}


def test_stage_spans_nest_under_request_span_across_langgraph_threads():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # First global set wins process-wide; later tests just export to memory,
    # which is harmless.
    trace.set_tracer_provider(provider)

    def retrieve(state):
        return {"fused": []}

    graph = StateGraph(_State)
    graph.add_node("retrieve", traced_node("retrieve", retrieve))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", END)
    compiled = graph.compile()

    async def run():
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("request") as request_span:
            async for _ in compiled.astream({"query": "hi"}, stream_mode=["updates"]):
                pass
        return request_span

    request_span = asyncio.run(run())

    spans = {span.name: span for span in exporter.get_finished_spans()}
    stage = spans["rag.retrieve"]
    assert stage.parent is not None, "stage span was orphaned across the thread hop"
    assert stage.parent.span_id == request_span.get_span_context().span_id
    assert stage.context.trace_id == request_span.get_span_context().trace_id
    assert stage.attributes["rag.stage"] == "retrieve"
    assert stage.attributes["rag.fused.count"] == 0
