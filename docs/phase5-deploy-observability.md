# Phase 5 — Deploy (Modal + Vercel) + Observability (OTel → Langfuse)

VisaWise is a free, eval-first USCIS RAG chatbot; the serving config is locked
(`gemini:gemini-3.1-flash-lite` + guardrail prompt v4, hybrid retrieval + local
rerank) and on `main`. This phase produces a public, recruiter-clickable deploy
and adds first-class observability so the live app is watchable like a
production service.

**Decisions (locked):** Dockerfile (portable) wrapped by Modal · Vercel static
frontend + Modal API · instrument now with metrics + logs + **traces** · traces →
**Langfuse Cloud (free)** · Modal **scale-to-zero** (accept ~5–15s cold start for $0).

**Critical constraint:** `settings.lancedb_dir`/`runs_dir` derive from
`REPO_ROOT = Path(config.py).resolve().parents[2]`, so the image MUST keep the
source-tree layout (`/app/src/visawise`, `/app/data`, `/app/evals`) with an
editable install, or the baked LanceDB index won't resolve at runtime.

**Reuse (don't recreate):** `visawise.embedding.get_embedder` &
`visawise.rag.rerank.load_local_model` (HF model loaders — the Docker build
pre-fetches via these); `visawise.app.main:app` (the ASGI app Modal wraps);
`evals/report.py:compare()` (existing quality gate = the quality SLO);
`evals/runs/latest_summary.json` (git-tracked; the dashboard's `/api/eval/runs`
payload — must be in the image).

We work this **phase by phase**: each phase has a Goal, files, steps, and a
Verify checkpoint. Finish and verify one before starting the next.

---

# PART 1 — DEPLOY

## Phase 0 — Accounts & prerequisites (~10 min)
**Goal:** external accounts + keys ready (can run in parallel with Phase 1).
- Modal CLI + account: `uv tool install modal` (or pip) → `modal token new`.
- **Langfuse Cloud** account → project → copy `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` (needed Part 2).
- **Vercel** account (GitHub login).
- Have `GOOGLE_API_KEY` on hand (already in local `.env`).
**Verify:** `modal token new` works; can log into Langfuse + Vercel.

## Phase 1 — CPU-only torch (image-size lever)
**Goal:** avoid the multi-GB CUDA torch wheel (no GPU is used).
**Files:** `pyproject.toml`, `uv.lock`.
- Add `torch` as an explicit dep pinned to the CPU index:
  `[[tool.uv.index]] name="pytorch-cpu" url="https://download.pytorch.org/whl/cpu" explicit=true`
  + `[tool.uv.sources] torch = {index="pytorch-cpu"}`.
- `uv lock` → `uv sync`.
**Verify:** `uv run python -c "import torch; print(torch.__version__)"` shows `+cpu`; `uv run pytest -q` still green (93).

## Phase 2 — Dockerfile + .dockerignore + local smoke
**Goal:** a self-contained image that boots and serves locally.
**Files (new):** `Dockerfile`, `.dockerignore`.
- **Dockerfile:** `python:3.12-slim`; `COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv`; `WORKDIR /app`;
  `apt-get install -y --no-install-recommends libgomp1` (+ clean lists); env
  `UV_PROJECT_ENVIRONMENT=/app/.venv UV_COMPILE_BYTECODE=1 HF_HOME=/opt/hf-cache PATH="/app/.venv/bin:$PATH"`;
  `COPY pyproject.toml uv.lock ./` + `COPY src ./src`; `RUN uv sync --frozen --no-dev`;
  **pre-fetch models (network ON):**
  `RUN HF_HUB_OFFLINE=0 uv run python -c "from visawise.embedding import get_embedder; get_embedder(); from visawise.rag.rerank import load_local_model; load_local_model()"`;
  `COPY data/lancedb ./data/lancedb` + `COPY evals/runs/latest_summary.json ./evals/runs/`;
  `ENV HF_HUB_OFFLINE=1`; `EXPOSE 8000`; `CMD ["visawise-app"]`.
- **.dockerignore** (independent of `.gitignore`): exclude `.git .venv **/__pycache__ *.pyc data/raw data/extracted data/chunks tests research docs configs evals/datasets evals/runs/archive`; **KEEP** `data/lancedb/` (gitignored but required) + `evals/runs/latest_summary.json`.
**Verify:** `docker run --rm -p 8000:8000 -e GOOGLE_API_KEY=… visawise`; `curl localhost:8000/health` → healthy after warm; SSE `curl -N -X POST localhost:8000/api/chat -H 'Content-Type: application/json' -d '{"message":"How many days of unemployment on post-completion OPT?"}'` streams `done` + citations; open `http://localhost:8000/`.

**Build gotchas (verified 2026-07-17 on macOS/arm64):**
- **Space in repo path breaks `docker build .`** — BuildKit's context Dockerfile-reader ingests 2 bytes (`…/Datasets and Projects/…`). Workaround for ALL local builds: pipe the Dockerfile via stdin → `docker build -f - . < Dockerfile`. (Modal is unaffected — it uploads context to its own builders.)
- **`README.md` must be COPY'd** — hatchling reads `readme = "README.md"` at build time (`OSError: Readme file does not exist` otherwise).
- **CPU-torch + `uv` cache-mount** got the image to **3.67 GB** (from 5.43 GB; the retained uv wheel cache was the bloat). HF models are still ~1.64 GB (candidate for a further trim — force safetensors-only).
- **amd64 image segfaults (exit 139) under QEMU emulation on arm64 Macs** — torch/native libs. So local runtime smoke uses a **native arm64** build; the **amd64** image is what ships to Modal (real x86, runs fine). Verified native arm64: health healthy + SSE answer with citations.
- **Disk**: the ML image needs several GB free; a 100%-full disk fails `uv sync` mid-install and can crash Docker Desktop.

## Phase 3 — Frontend/backend split (config + CORS)
**Goal:** make the same-origin app work across two origins.
**Files:** new `src/visawise/app/static/config.js`; modify `config.py`, `app/main.py`, `static/index.html`, `static/app.js`, `.env.example`.
- `config.js`: `window.__VISAWISE_API_BASE__ = "";` (same-origin default; set to Modal URL in Phase 5). Committed, not secret.
- `index.html:278`: add `<script src="config.js"></script>` before `app.js`.
- `app.js`: `const API_BASE = window.__VISAWISE_API_BASE__ || "";` then prefix both fetches (`:67` `/api/chat`, `:404` `/api/eval/runs`).
- `config.py`: add `cors_allowed_origins: list[str]` (env `CORS_ALLOWED_ORIGINS`, JSON array; default localhost).
- `main.py:100-110`: use `settings.cors_allowed_origins`.
- `.env.example`: add `GOOGLE_API_KEY=` and `CORS_ALLOWED_ORIGINS=`.
**Verify:** local run works same-origin (`API_BASE=""`); `pytest` green; `CORS_ALLOWED_ORIGINS` env honored.

## Phase 4 — Modal deploy
**Goal:** the API live on a public Modal URL.
**Files (new):** `deploy/modal_app.py`.
- `image = modal.Image.from_dockerfile("Dockerfile", context_dir=".")`; `app = modal.App("visawise")`;
  fn: `@app.function(image=image, secrets=[modal.Secret.from_name("visawise-secrets")], min_containers=0, scaledown_window=300, memory=4096, cpu=2.0, timeout=600)` + `@modal.concurrent(max_inputs=8)` + `@modal.asgi_app()` → `from visawise.app.main import app; return app`.
- Secret: `modal secret create visawise-secrets GOOGLE_API_KEY=… CORS_ALLOWED_ORIGINS='["http://localhost:5173"]'` (add Vercel origin in Phase 5).
- `modal serve deploy/modal_app.py` to iterate; then `modal deploy`.
**Verify:** Modal URL `/health`, SSE `/api/chat`, `/api/eval/runs`; first cold hit warms within timeout and streams. Record the URL.
**Done 2026-07-17:** live at `https://raghavgali397--visawise-fastapi-app.modal.run` (built on Modal's x86 in ~108s — the amd64 image that segfaults under local QEMU runs fine here). Cold start ~40s.
**Gotcha — secrets:** a generic SSE `error` ("answer could not be generated") with a healthy `/health` = the `generate` step threw. Check `modal app logs visawise`; first failure was `400 API_KEY_INVALID` (a bad `GOOGLE_API_KEY` in the secret — paste with no quotes/trailing whitespace). A changed secret only takes effect on a **fresh container**, so `modal deploy` again (or `modal app stop`) after `modal secret create --force`.

## Phase 5 — Vercel deploy + end-to-end
**Goal:** public frontend calling the Modal API.
**Files (new):** `vercel.json`; edit `config.js`; update Modal secret CORS.
- `vercel.json`: `{ "framework": null, "buildCommand": null, "outputDirectory": "src/visawise/app/static", "cleanUrls": true }`.
- Put the Modal URL in `config.js`; add the Vercel origin to `CORS_ALLOWED_ORIGINS` in the Modal secret; redeploy Modal; deploy Vercel.
**Verify (end-to-end):** Vercel page → ask a starter question → tokens stream + citations render (cross-origin SSE + CORS) → Evaluation tab loads the live docket. Add the public URL to `README.md`; fix the stale "Llama 3.1 8B (NVIDIA NIM)" stack line → Gemini; flip the Phase-5 status row to done.

---

# PART 2 — OBSERVABILITY (OTel → Langfuse) + SLOs

**Architecture:** Traces (OTel FastAPI + custom RAG-stage spans + OpenLLMetry
`gen_ai` spans) → **Langfuse** via OTLP/HTTP. Metrics (golden signals)
instrumented in OTel, viewed via **Modal built-in metrics** + **Langfuse
trace-derived** latency/count/cost (Langfuse isn't an OTel-metrics store —
documented). Logs: structured JSON → stdout → Modal logs, correlated to Langfuse
by trace id. Everything is a **no-op when the OTLP/Langfuse env vars are unset**
(local dev + tests unaffected).

## Phase 6 — Observability scaffold (no-op safe)
**Goal:** OTel init that does nothing until configured; zero local/test impact.
**Files (new):** `src/visawise/observability.py`; modify `config.py`, `pyproject.toml`.
- `config.py`: add `otel_exporter_otlp_endpoint`, `otel_exporter_otlp_headers`, `otel_service_name="visawise"`, `langfuse_public_key`, `langfuse_secret_key` (default `""`).
- `pyproject.toml`: optional group `observability` = `opentelemetry-api/sdk`, `opentelemetry-exporter-otlp-proto-http`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-instrumentation-logging`, `traceloop-sdk`; `uv lock` — **verify no conflict with langchain/langgraph 1.0**.
- `observability.py`: `init_observability(app) -> bool` (early-returns when Langfuse/OTLP env unset); `get_tracer()`, `traced_node()`, `record_stage_latency()` helpers tolerant of no-op providers.
**Verify:** `pytest` green with obs deps installed but env unset (full no-op).

## Phase 7 — Traces → Langfuse (the headline)
**Goal:** one `/api/chat` trace: server → RAG stages → Gemini span.
**Files:** `observability.py`, `app/main.py`, `rag/graph.py`.
- `init_observability`: `TracerProvider(Resource(service.name, version=git sha))` + `BatchSpanProcessor(OTLPSpanExporter(endpoint=Langfuse OTLP, headers=Basic base64(public:secret)))`; `FastAPIInstrumentor.instrument_app(app, excluded_urls="health")`; `Traceloop.init(...)` with **content capture disabled** (`TRACELOOP_TRACE_CONTENT=false`).
- `main.py` lifespan: `init_observability(app)` before `yield`; `force_flush()/shutdown()` in `finally:` (flush before Modal freezes the replica); `first_token`/`ttft` at main.py:188.
- `graph.py` `build_graph`: wrap node callables with `traced_node` at `add_node` (bounded attrs: `rag.stage`, `rag.n_hits`, `rag.top_score`).
- Langfuse keys → Modal secret; redeploy.
**Verify:** one `/api/chat` → Langfuse shows a single trace (server → `dense_retrieve`/`bm25_retrieve`/`fuse`/`rerank`/`generate` → `gen_ai` Gemini span w/ tokens/cost). **Verify stage-span nesting for real** — LangGraph runs sync nodes in a thread executor; if spans orphan, capture+attach the OTel context in `stream_events()` before `graph.astream`.

## Phase 8 — Metrics (golden signals) + structured logs
**Goal:** four golden signals instrumented; correlated JSON logs.
**Files:** `observability.py`, `app/main.py`, `rag/graph.py`.
- Instruments: `visawise.rag.stage.duration{stage}`, `visawise.chat.ttft`, `visawise.chat.requests{outcome}`, `visawise.chat.errors{error.type}`, `visawise.chat.inflight` (up_down), `visawise.startup.duration`; FastAPI gives `http.server.request.duration` (Latency) + counts (Traffic); Saturation = `chat.inflight` + rerank latency + Modal CPU/mem.
- JSON logging via `LoggingInstrumentor` (injects `otelTraceID`/`otelSpanID`); log `query_len`, `n_hits`, `top_rerank_score`, node timings — never raw query/chunk text.
**Verify:** Modal logs show JSON lines carrying the trace id matching the Langfuse trace; Modal dashboard shows invocations/latency/CPU; latency+cost visible in Langfuse.

## Phase 9 — SLOs + docs
**Goal:** documented SLIs/SLOs and the runtime↔eval-gate link.
**Files:** new `docs/observability-slo.md` (or extend `docs/EVALS.md`), `README.md`.
SLIs/SLOs (28-day window; exclude `/health` + cold-start):
| # | SLI | SLO |
|---|---|---|
| 1 | Availability — non-5xx `/api/chat` ÷ total | 99.0% (error budget 1%) |
| 2 | Latency — p95 `/api/chat` | ≤ 4.0s (p50 ≤ 2.5s) |
| 3 | Chat error rate — `chat.errors ÷ chat.requests` (excl. 429) | < 2% |
| 4 | Quality gate (offline) — faithfulness ≥ 0.85 + safety-slice abstention, enforced by `evals/report.py:compare()` | release gate, not a runtime budget |

Document the error-budget concept; note runtime p95 (2) vs eval-time `timings.p95_ms` = a drift alarm.

---

## Risks (watch during execution)
- **Image size** — CPU torch pin is mandatory (Phase 1) or the image is multi-GB.
- **Cold start** (scale-to-zero) — ~5–15s first hit; a `/health` warm-up ping hides most.
- **REPO_ROOT fragility** — keep editable source-tree layout in the image.
- **HF offline** — network ON during build pre-fetch, OFF at runtime.
- **slowapi in-memory** — per-replica, not global (fine for demo).
- **Langfuse ≠ metrics store** — metrics via Modal + trace-derived (documented).
- **Thread-context for stage spans** — verify nesting with a real trace, don't assume.
- **Span-loss on scale-to-zero** — flush in lifespan `finally:`.
- **Free quotas** — Gemini RPD (bursty → 429 as SSE `error`); Langfuse 50k units/mo; Modal credits (scale-to-zero ≈ $0).

## Part 2 gotchas (discovered during Phases 6–8)

- **Langfuse Cloud is two regions.** Keys are region-bound: EU = `cloud.langfuse.com`,
  US = `us.cloud.langfuse.com`. Wrong region = `401` — and on the OTLP export path
  that 401 is *silent* (a dropped batch and a log line). Config accepts
  `LANGFUSE_BASE_URL` (official SDK name) or `LANGFUSE_HOST`.
- **Changing a Modal secret does NOT recycle warm containers, and neither does
  `modal deploy` if the function definition didn't change** (the secret is referenced
  by name). Force it: `modal container list` → `modal container stop --yes <id>`.
- **A developer .env with real Langfuse keys makes the test suite export spans to the
  production project** (app tests import `main.py`, which inits telemetry at import
  time). `tests/conftest.py` blanks the telemetry settings before any test module
  loads. The seven 0-duration junk traces this produced were deleted via the API.
- **`init_observability(app)` must run at import time, not in the lifespan** —
  Starlette freezes the middleware stack before the lifespan runs; the FastAPI
  instrumentor needs to add middleware before that.
- **Metrics deviation:** Langfuse's OTLP endpoint ingests traces only. Golden signals
  are derived: latency/traffic/errors from traces in Langfuse, saturation from
  Modal's built-in CPU/memory dashboard. No OTel metric instruments (no backend
  for them = dead code).
- **First production trace breakdown (13.1s total):** rerank 4.5s (CPU cross-encoder,
  2 vCPU) + generate 8.5s (Gemini, ~4.7k input tokens) — retrieval ~0.1s. The ~8s
  p50 is generation-dominated, not retrieval.
