"""FastAPI surface (Phase 4).

    POST /api/chat        {message} -> SSE stream; final event carries citations
    GET  /api/eval/runs   committed RunRecord summaries (dashboard tab data)
    GET  /health
    /                     static chat UI + Evaluation tab

Serving notes: graph compiled once at startup from the production EngineConfig
(hybrid 0.6/0.4 + local rerank); LanceDB opened from the baked-in data dir;
slowapi rate limiting; DISCLAIMER shown in UI.
"""
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette import EventSourceResponse, ServerSentEvent

from ..config import settings
from ..observability import (
    current_span,
    init_observability,
    record_first_token,
    shutdown_observability,
)
from ..rag.prompts import DISCLAIMER

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from ..embedding import get_embedder
    from ..rag.graph import EngineConfig, build_graph
    from ..rag.rerank import load_local_model
    from ..rag.retrieval import open_table

    app.state.ready = False

    # Serving config, chosen by the eval harness: Gemini 3.1 Flash-Lite (free,
    # fast, judge-independent) + guardrail prompt v4 + thinking off. See
    # docs/learning-notes.md for the model/guardrail selection story.
    production_config = EngineConfig(
        retriever="hybrid",
        vector_weight=0.6,
        reranker="local",
        llm=settings.generation_llm,
        prompt_version="v4",
        thinking_budget=0,
    )

    graph = build_graph(production_config)
    table = await asyncio.to_thread(open_table)

    required_columns = {
        "chunk_id",
        "vector",
        "text",
        "url",
        "title",
        "topic",
        "section",
    }

    missing_columns = required_columns - set(table.schema.names)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"LanceDB table is missing required columns: {missing}")

    # Warm the local models so the first user request doesn't pay cold start.
    if production_config.retriever in {"vector", "hybrid"}:
        await asyncio.to_thread(get_embedder)

    if production_config.reranker == "local":
        await asyncio.to_thread(load_local_model)

    app.state.engine_config = production_config
    app.state.lancedb_table = table
    app.state.graph = graph
    app.state.ready = True

    try:
        yield
    finally:
        app.state.ready = False
        # Flush buffered spans before Modal freezes the scale-to-zero
        # container; otherwise the last requests' traces are silently lost.
        shutdown_observability()


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Instrument at import time, NOT in the lifespan: Starlette finalizes the
# middleware stack before the lifespan runs, and the FastAPI instrumentor
# must add its middleware before that or it raises. No-op (returns False)
# unless the Langfuse/OTLP env is configured.
init_observability(app)


def get_graph(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready",
        )

    graph = getattr(request.app.state, "graph", None)

    if graph is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph is unavailable",
        )

    return graph


@app.get("/health")
def get_health(request: Request) -> dict[str, str]:
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready",
        )
    return {"status": "healthy"}


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    message: str = Field(
        min_length=1,
        max_length=4000,
        strict=True,
    )


def _sse_json(event: str, payload: dict) -> ServerSentEvent:
    # SSE data is text: serialize explicitly (a dict would render as repr()).
    return ServerSentEvent(event=event, data=json.dumps(payload, ensure_ascii=False))


@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_endpoint(payload: ChatRequest, request: Request) -> EventSourceResponse:
    graph = get_graph(request)
    message = payload.message

    # Captured here, where the server span is guaranteed active; the
    # generator body runs later, during response send. No-op span when
    # telemetry is off.
    request_span = current_span()
    request_start = time.perf_counter()

    async def stream_events() -> AsyncIterator[ServerSentEvent]:
        final_update: dict | None = None
        first_token_seen = False

        try:
            # With multiple stream modes, astream yields (mode, data) tuples.
            # "messages" surfaces LLM tokens from inside nodes (even for
            # invoke()-based nodes); "updates" carries each node's state delta.
            async for mode, data in graph.astream(
                {"query": message},
                stream_mode=["messages", "updates"],
            ):
                if await request.is_disconnected():
                    return

                if mode == "messages":
                    message_chunk, metadata = data

                    if metadata.get("langgraph_node") != "generate":
                        continue

                    token = message_chunk.content

                    if isinstance(token, str) and token:
                        if not first_token_seen:
                            first_token_seen = True
                            record_first_token(
                                request_span, (time.perf_counter() - request_start) * 1000.0
                            )
                        yield _sse_json("token", {"text": token})

                elif mode == "updates":
                    generate_update = (data or {}).get("generate")

                    if generate_update is not None:
                        final_update = generate_update

            if final_update is None:
                raise RuntimeError("Graph completed without a generate result")

            yield _sse_json(
                "done",
                {
                    "answer": final_update.get("answer", ""),
                    "citations": final_update.get("citations", []),
                    "disclaimer": DISCLAIMER,
                },
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception("Chat graph streaming failed")

            if not await request.is_disconnected():
                yield _sse_json(
                    "error",
                    {"message": "The answer could not be generated. Please try again."},
                )

    return EventSourceResponse(stream_events())


@app.get("/api/eval/runs")
def eval_runs() -> dict:
    """The committed leaderboard summary (rendered by `visawise-eval report`)."""
    summary_path = settings.runs_dir / "latest_summary.json"
    if not summary_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No eval summary found -- run `visawise-eval report` first.",
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:

    @app.get("/")
    def index() -> dict:
        return {"service": "VisaWise", "docs": "/docs", "disclaimer": DISCLAIMER}


def main() -> None:
    import uvicorn

    uvicorn.run("visawise.app.main:app", host="0.0.0.0", port=8000)
