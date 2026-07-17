# Observability & SLOs

How the deployed VisaWise service is watched, and what "healthy" means in
numbers. Companion to `docs/phase5-deploy-observability.md` (how it was
built) and `docs/EVALS.md` (offline quality measurement).

## Architecture

```
request ──► FastAPI (Modal, scale-to-zero)
              │  server span (OTel FastAPI instrumentation; /health excluded)
              │    ├─ rag.dense_retrieve / rag.bm25_retrieve / rag.fuse   spans
              │    ├─ rag.rerank                                          span
              │    └─ rag.generate ─► gen_ai span (model, token usage;
              │                       NO prompt/answer content)
              │
              ├─ traces ──OTLP/HTTP──► Langfuse Cloud (US)
              └─ JSON logs (trace_id-correlated) ──► Modal logs
```

- **Traces** are the primary signal: one trace per chat request, with
  per-stage latency and a `gen_ai` span carrying token usage. Vendor-neutral
  OTel — swap backends by pointing `OTEL_EXPORTER_OTLP_ENDPOINT/_HEADERS`
  elsewhere (e.g. LangSmith), zero code changes.
- **Logs** are structured JSON on stdout with `trace_id`/`span_id` injected,
  so a Modal log line and its Langfuse trace cross-reference each other.
  Log values are bounded (durations, counts, lengths) — user queries and
  answers never leave the request path.
- **Metrics** are derived, not instrumented: Langfuse's OTLP endpoint
  ingests traces only, so latency/traffic/errors come from trace queries and
  saturation comes from Modal's built-in CPU/memory dashboard. OTel metric
  instruments with no backend would be dead code.
- Everything is a **strict no-op** unless `LANGFUSE_PUBLIC_KEY` /
  `LANGFUSE_SECRET_KEY` (+ `LANGFUSE_BASE_URL` for the US region) or the
  generic `OTEL_EXPORTER_OTLP_*` pair is set. Local dev, evals, and the test
  suite emit nothing (`tests/conftest.py` enforces this for tests).

## The four golden signals, mapped

| Signal | Where it lives | What to look at |
|---|---|---|
| Latency | Langfuse traces | server-span duration; `visawise.ttft_ms`; per-stage `rag.*` spans |
| Traffic | Langfuse traces / Modal | `POST /api/chat` trace count; Modal request counts |
| Errors | Langfuse traces | spans with `visawise.outcome = "error"` (ERROR status) |
| Saturation | Modal dashboard | container CPU/memory; `rag.rerank` duration creeping up = CPU pressure |

## SLIs and SLOs

Window: 28 days. `/health` requests and container cold starts are excluded
from latency (cold start ≈ 30 s is a documented cost of $0 scale-to-zero
hosting, not a regression signal — it shows up as the first request after an
idle gap).

| # | SLI | SLO | Measured via |
|---|---|---|---|
| 1 | **Availability** — non-5xx `POST /api/chat` ÷ total | **≥ 99.0%** (error budget: 1%) | Langfuse trace HTTP status / Modal logs |
| 2 | **Latency** — warm `POST /api/chat` server-span duration | **p50 ≤ 8 s, p95 ≤ 12 s** | Langfuse trace latency |
| 3 | **Chat error rate** — `visawise.outcome="error"` ÷ all chat traces (client 4xx/429 excluded) | **< 2%** | Langfuse attribute query |
| 4 | **Quality gate (offline)** — faithfulness ≥ 0.85 on the answerable slice AND safety-slice abstention ≥ 14/15 on `curated_v2` | **release gate** | eval harness (`visawise-eval`), enforced before any serving-config change ships |

Why the latency SLO is 8/12 and not something flashier: the first production
trace decomposed the ~8 s median as **~4.5 s CPU cross-encoder rerank (2 vCPU)
+ ~8.5 s Gemini generation (~4.7 k input tokens)**, with retrieval ~0.1 s.
That's the physics of the free tier; an SLO you can't meet is a pager, not a
promise. The offline eval p50 (2.5 s on dev hardware) is tracked separately —
if the *gap* between eval-time and runtime latency widens, that's drift worth
investigating (levers, in impact order: trim retrieved context tokens, more
Modal CPU, lighter reranker, keep-warm container).

SLI 4 is deliberately an offline **release gate**, not a runtime budget:
answer quality can't be judged per-request in production without shipping
user queries to a judge model (privacy) and paying per request (cost). The
eval harness already blocks regressions before deploy; runtime SLIs 1–3
catch what evals can't see.

## Error budget, in practice

99.0% availability over 28 days = ~7 h of allowed downtime or its request
equivalent. Spend it consciously: a risky deploy during low traffic is a
fine use; burning it on repeated Gemini free-tier 429s is a signal to add
backoff or request quota, not to relax the SLO. When the budget is gone,
feature work yields to reliability work — that's the whole contract.
