# VisaWise

[![CI](https://github.com/Raghavgali/VisaWise/actions/workflows/ci.yml/badge.svg)](https://github.com/Raghavgali/VisaWise/actions/workflows/ci.yml)

**A production RAG system for U.S. immigration policy (USCIS), built eval-first and run like a service.**

**▶ Live demo: [visa-wise-six.vercel.app](https://visa-wise-six.vercel.app)** — frontend on Vercel, API on Modal
(scale-to-zero: the first request after idle takes ~30 s to warm, then answers stream in seconds).

Ask about F-1/OPT, H-1B, green cards, or status changes and get cited answers from a
re-runnable corpus of official USCIS pages. Nothing in this system is a belief: the
retrieval weights, the chunking, the reranker, the serving model, and the guardrail
prompt were all **chosen by the evaluation harness**, and the same harness
regression-gates every change. The eval suite is the centerpiece; the chatbot is its demo.

> ⚠️ Educational demo — not legal advice. Verify anything important at
> [uscis.gov](https://www.uscis.gov) or with an immigration attorney.

## Why this exists

Immigration policy changes fast and the cost of a stale or overconfident answer is high.
The H-1B page in this corpus leads with a September 2025 proclamation adding a
**$100,000 payment condition** to certain petitions — a policy that didn't exist when the
original 2024 research corpus was built. A chatbot serving last year's policy, or
inventing specifics it never retrieved, is worse than no chatbot. VisaWise treats both as
measurable systems problems: staleness-aware ingestion, and an eval suite with an explicit
safety slice.

## What the eval harness found (three results worth reading)

**1 — Retrieval strategy matters, and we measured how much.** The 2024 research this
builds on ([`research/`](research/)) showed hybrid retrieval (dense + BM25, weighted RRF)
lifting RAGAS faithfulness **0.44 → 0.83** over vector-only, with 0.6/0.4 the optimal
weighting. The production harness re-tested those decisions against the fresh corpus
instead of assuming they still hold: hybrid still wins on retrieval quality
(context precision 0.92 vs 0.84, MRR 0.83 vs 0.78) and serves today.

**2 — The aggregate metric hid a real safety hole; slicing found it.** A 30-sample
human-curated dataset tags each question with a slice (answerable, temporal, stale-source,
out-of-corpus, adversarial-injection, high-risk). First run: content metrics looked great
— and the safety slices scored **~3/15**. The engine answered out-of-corpus questions with
invented specifics, gave confident advice on high-risk questions, and one prompt-injection
sample leaked the system prompt. Four guardrail-prompt iterations (tuned on a held-out
dev set, measured once on the test set) plus a reference-text fix in the safety set
later, the serving config scores **14/15 on the safety slices** with answerable
faithfulness at **0.92** — and the remaining failure is documented, not hidden.

**3 — The serving model was a controlled bake-off, then a refinement — and the two
stages are not comparable, so they're reported separately.** First the **screen**: four
free-tier models, identical conditions (dataset `curated_v1`, guardrail prompt v2, judge
`gpt-4o-mini`):

| Model (free tier) | Faithfulness | Safety | p50 |
|---|---|---|---|
| **Gemini 3.1 Flash-Lite** | **0.85** | **11/15** | **2.2 s** |
| gpt-oss-20B (Groq) | 0.83 | 11/15 | 31.4 s |
| Llama-4-Scout (Groq) | 0.74 | 5/15 | 7.2 s |
| Llama-3.1-8B (NVIDIA NIM) | 0.55 | 7/15 | 2.5 s |

Gemini and gpt-oss tied on safety with comparable faithfulness; Gemini won on latency
(2.2 s vs 31.4 s) and free-quota viability. Then the **refinement, applied to the winner
only**: the guardrail prompt evolved v2 → v4 (tuned on a held-out dev set) and the five
out-of-corpus reference texts were fixed (`curated_v2` — the other 25 samples are
byte-identical). Result for the serving config: **0.92 answerable faithfulness, 14/15
safety, 2.5 s**. That 14/15 is *not* a bake-off column — the other three models never ran
with prompt v4 or the corrected references, and rerunning them wasn't worth the quota.

The judge stays `gpt-4o-mini` (OpenAI) precisely so the serving model isn't graded by its
own family. Eval integrity is enforced mechanically: any judged metric scoring <95% of its
samples **fails the run**, the compare gate **fails closed** on dataset/corpus mismatches,
aborted runs, and differing survivor subsets, and **CI runs a release gate on every push**
(answerable faithfulness ≥ 0.85, safety ≥ 14/15, over the committed run records — no API
keys needed).

Full leaderboard and methodology: [docs/EVALS.md](docs/EVALS.md) · the live
[eval dashboard](https://visa-wise-six.vercel.app) renders the committed run records.

## Architecture

```
             INGESTION (visawise-ingest: idempotent, re-runnable)
  configs/sources.yaml ── fetch ─→ extract ─→ chunk ─→ index
   24 curated USCIS URLs    │         │          │        │
                       raw HTML   markdown   chunks.jsonl  LanceDB (embedded)
                       snapshots  + frontmatter  (single    ├─ bge-768 vectors
                       + hashes   (trafilatura)  source     └─ stemmed BM25 FTS
                       + archive               of truth)      — hybrid in ONE store,
                         detection                              no external DB service
                                                    │
             RAG CORE (LangGraph — one graph factory shared by app & evals)
                      ┌─→ dense_retrieve (cosine kNN) ──┐
             query ──►│                                 ├─→ fuse (weighted RRF)
                      └─→ bm25_retrieve (native FTS) ───┘        │
                                                              rerank (local cross-encoder)
                                                                 │
                                                              generate (Gemini 3.1 Flash-Lite
                                                                 │       + guardrail prompt v4)
             EVALS (visawise-eval)                            answer + citations
             YAML experiments × versioned golden datasets
             → RAGAS 0.4 (judge: gpt-4o-mini) + hit-rate/MRR/NDCG
             → slice-aware safety scoring → committed RunRecords
             → regression gates → live dashboard
```

**Design principles**

- **One source of truth per layer.** `chunks.jsonl` feeds both indexes and eval
  provenance; one engine factory (`build_graph(EngineConfig)`) serves both the app and
  the harness — so benchmark numbers always describe the system that's actually deployed.
- **Framework-thin retrieval.** Direct LanceDB calls and a hand-rolled weighted
  reciprocal-rank fusion (~15 lines) instead of retriever abstractions. RRF fuses
  *ranks*, so the incompatible score scales of the two legs never mix.
- **Loud failures.** Dead links, pages USCIS 301-redirects into `/archive/` (still
  HTTP 200 — but officially stale!), suspiciously short extractions, and under-covered
  judge metrics all fail visibly. Silent decay is how RAG systems rot.
- **Determinism end-to-end.** Content-hashed fetches, deterministic chunk IDs, and a
  `corpus_hash` stamped into every eval run.

## Run like a service: deploy + observability

The backend is a Dockerfile (CPU-pinned torch, models and the LanceDB index baked in,
offline at runtime) that Modal wraps and serves scale-to-zero; the frontend is static on
Vercel. Total hosting cost: ~$0.

Every request emits an **OpenTelemetry trace** (exported OTLP → Langfuse Cloud; the
backend is one env-var swap away — nothing is vendor-locked):

```
POST /api/chat ..................... 13.1 s     ← real production trace
├─ rag.dense_retrieve .............. 0.10 s
├─ rag.bm25_retrieve ............... 0.02 s
├─ rag.fuse ........................ 0.00 s
├─ rag.rerank ...................... 4.5 s      ← cross-encoder on 2 free vCPUs
└─ rag.generate .................... 8.5 s
   └─ gen_ai (Gemini) .............. 8.5 s      ← ~4.7k input tokens, usage captured
```

That one trace settles where the latency lives (generation + CPU rerank; retrieval is
free) — which is why the latency SLO is an honest **p50 ≤ 8 s / p95 ≤ 12 s warm** rather
than a number the free tier can't meet. Logs are structured JSON carrying the `trace_id`,
so a Modal log line and its Langfuse trace cross-reference exactly. Spans record counts,
scores, and durations — **never the user's question or the answer**. Offline answer
quality is deliberately a *release gate* (the eval harness), not a runtime metric: SLIs,
SLOs, and the error-budget contract are in
[docs/observability-slo.md](docs/observability-slo.md).

## Status

| Phase | Scope | State |
|---|---|---|
| 1 — Ingestion pipeline | fetch / extract / chunk / index, CLI, tests | ✅ done |
| 2 — RAG core | hybrid retrieval, reranking, LangGraph pipeline | ✅ done |
| 3 — Eval harness | golden datasets, RAGAS 0.4 + retrieval metrics, YAML experiments, regression gates | ✅ done ([leaderboard](docs/EVALS.md)) |
| 4 — App | FastAPI + SSE chat UI + live eval dashboard | ✅ done |
| 5 — Deploy | Dockerized backend on Modal (scale-to-zero) + static frontend on Vercel | ✅ [live](https://visa-wise-six.vercel.app) |
| 6 — Observability | OTel traces → Langfuse, trace-correlated JSON logs, SLOs | ✅ done ([SLOs](docs/observability-slo.md)) |

## Quickstart

```bash
uv sync                                # Python 3.12, uv-managed
cp .env.example .env                   # add GOOGLE_API_KEY (serving)
                                       #   + OPENAI_API_KEY (eval judge)

uv run visawise-ingest all             # fetch → extract → chunk → index (~35s)
uv run visawise-ingest status          # corpus ⇄ index drift report

uv run python -c "
from visawise.rag.retrieval import hybrid_search
for hit in hybrid_search('Can an F-1 student work off campus?', top_k=4):
    print(f'{hit.score:.4f}  {hit.title}')"

uv run visawise-app                    # chat UI + eval dashboard at :8000
uv run pytest                          # fast, no-network unit tests
```

No vector-DB account needed: LanceDB is embedded — dense vectors and full-text search
live in one local table that ships inside the deploy container. Telemetry is a strict
no-op unless Langfuse/OTLP env vars are set.

## Repo layout

```
configs/            source catalog + experiment definitions (YAML)
src/visawise/
  ingestion/        fetch → extract → chunk → index (+ visawise-ingest CLI)
  rag/              retrieval, rerank, LangGraph engine factory, versioned prompts
  evals/            datasets, metrics, slice-aware runner, reports (+ visawise-eval CLI)
  app/              FastAPI: SSE /api/chat + eval dashboard
  observability.py  OTel wiring: stage spans, gen_ai spans, JSON logs (no-op unless configured)
deploy/             Modal app definition (wraps the Dockerfile)
data/               build artifacts (gitignored) + committed fetch manifest
evals/              versioned golden datasets + committed eval RunRecords
research/           the original 2024 notebooks, eval CSVs, and findings
docs/               evals guide, SLOs, deploy/observability build log
```

## Stack

Python 3.12 · LangChain 1.x + LangGraph · LanceDB (embedded; vectors + tantivy FTS) ·
`BAAI/bge-base-en-v1.5` embeddings · `bge-reranker-base` cross-encoder · Gemini 3.1
Flash-Lite generation (chosen by the eval harness; provider is one flag away) ·
RAGAS 0.4 + gpt-4o-mini judge · FastAPI · Docker + Modal + Vercel ·
OpenTelemetry → Langfuse · uv
