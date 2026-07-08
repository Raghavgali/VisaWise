"""FastAPI surface (Phase 4).

    POST /api/chat        {message} -> SSE stream; final event carries citations
    GET  /api/eval/runs   committed RunRecord summaries (dashboard tab data)
    GET  /health
    /                     static chat UI + Evaluation tab

Serving notes: graph compiled once at startup from the production EngineConfig
(hybrid 0.6/0.4 + local rerank); LanceDB opened from the baked-in data dir;
slowapi rate limiting; DISCLAIMER shown in UI.
"""

# TODO(pairing): Phase 4
