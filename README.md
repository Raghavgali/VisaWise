# VisaWise

[![CI](https://github.com/Raghavgali/VisaWise/actions/workflows/ci.yml/badge.svg)](https://github.com/Raghavgali/VisaWise/actions/workflows/ci.yml)

**A USCIS immigration RAG system where the eval harness is the product and the chatbot is the demo.**

[![The VisaWise chat interface. Click to open the live app.](docs/assets/visawise-chat.png)](https://visa-wise-six.vercel.app)

**Live: [visa-wise-six.vercel.app](https://visa-wise-six.vercel.app)** · scale-to-zero backend,
so the first request after idle takes ~30-45 s (the UI narrates the warm-up); everything after
that streams in seconds.

At first glance this is another RAG chatbot: hybrid retrieval, a cross-encoder reranker, Gemini,
citations. Look closer and the repo is organized around a different center of gravity. Nothing
in the serving path is a belief: the retrieval weights, the chunking, the reranker, the serving
model, and the guardrail prompt were all **chosen by a benchmark**, the benchmark **runs in CI as
a release gate**, and the live Evaluation tab renders the committed run records. The system's
headline numbers, all reproducible from this repo:

**0.92 faithfulness** on answerable questions · **14/15** on an adversarial safety slice ·
**~$0/month** to serve · every push gated on those floors

> ⚠️ Educational demo, not legal advice. Verify anything important at
> [uscis.gov](https://www.uscis.gov) or with an immigration attorney.

## Three results the harness produced

**1. The aggregate hid a real safety hole; slicing found it.** The curated test set tags each
question with a slice: answerable, temporal, stale-source, out-of-corpus, adversarial-injection,
high-risk. The first full run looked great on content metrics and scored **~3/15** on the safety
slices: the engine answered out-of-corpus questions with invented specifics, gave confident
advice on high-risk questions, and one injection sample **leaked the system prompt**. Four
guardrail-prompt iterations later (tuned on a held-out dev split, measured once on the test set)
plus a fix to five inconsistent reference texts, the serving config scores **14/15** with
answerable faithfulness at **0.92**. The remaining failure is documented, not hidden.

**2. The serving model was a controlled bake-off, then a refinement, reported separately because
they are not comparable.** The screen: four free-tier models, identical conditions
(curated_v1, guardrail v2, gpt-4o-mini judge):

| Model (free tier) | Faithfulness | Safety | p50 |
|---|---|---|---|
| **Gemini 3.1 Flash-Lite** | **0.85** | **11/15** | **2.2 s** |
| gpt-oss-20B (Groq) | 0.83 | 11/15 | 31.4 s |
| Llama-4-Scout (Groq) | 0.74 | 5/15 | 7.2 s |
| Llama-3.1-8B (NVIDIA NIM) | 0.55 | 7/15 | 2.5 s |

Gemini and gpt-oss tied on safety with comparable faithfulness; Gemini won on latency and on a
free quota that could actually finish the benchmark. The guardrail v4 + reference-fix refinement
(which produced the 0.92 / 14-15 serving numbers) was applied to the winner only, so it is
reported as a second stage rather than passed off as a fifth bake-off column.

**3. Retrieval strategy is worth more than model size.** The 2024 research this grew from
([`research/`](research/)) measured hybrid retrieval (dense + BM25, weighted RRF) lifting RAGAS
faithfulness **0.44 to 0.83** over vector-only, with 0.6/0.4 the optimal weighting. The
production harness re-verified those decisions against the fresh corpus instead of assuming they
still hold: hybrid still wins on retrieval quality (context precision 0.92 vs 0.84, MRR 0.83 vs
0.78) and serves today. Meanwhile the model bake-off above shows a well-fed Flash-Lite beating a
20B model; retrieval and prompting moved the numbers more than parameters did.

Eval integrity is enforced mechanically, not by convention: a judged metric that covers <95% of
its samples **fails the run** (partial averages must never look complete), the compare gate
**fails closed** on dataset/corpus mismatches, aborted runs, and differing survivor subsets, and
CI runs the release gate (faithfulness ≥ 0.85, safety ≥ 14/15, offline over committed
RunRecords) on every push. Methodology and full leaderboard: [docs/EVALS.md](docs/EVALS.md).

## Architecture

```
             INGESTION (visawise-ingest: idempotent, re-runnable)
  configs/sources.yaml .. fetch --> extract --> chunk --> index
   24 curated USCIS URLs    |         |          |        |
                       raw HTML   markdown   chunks.jsonl  LanceDB (embedded)
                       snapshots  + frontmatter  (single    |- bge-768 vectors
                       + hashes   (trafilatura)  source     |- stemmed BM25 FTS
                       + archive               of truth)    (hybrid in ONE store,
                         detection                           no external DB)
                                                    |
             RAG CORE (LangGraph: one graph factory shared by app & evals)
                      +--> dense_retrieve (cosine kNN) --+
             query -->|                                  +--> fuse (weighted RRF)
                      +--> bm25_retrieve (native FTS) ---+         |
                                                          rerank (local cross-encoder)
                                                                   |
                                                          generate (Gemini 3.1 Flash-Lite
                                                                   |  + guardrail prompt v4)
             EVALS (visawise-eval)                        answer + citations
             YAML experiments x versioned golden datasets
             --> RAGAS 0.4 (judge: gpt-4o-mini) + hit-rate/MRR/NDCG
             --> slice-aware safety scoring --> committed RunRecords
             --> fail-closed gates --> CI --> live dashboard
```

**Design choices that carry weight**

- **One graph factory serves both the app and the harness**, so benchmark numbers always
  describe the system that is actually deployed, not a lookalike config.
- **Framework-thin retrieval.** Direct LanceDB calls and a hand-rolled weighted
  reciprocal-rank fusion (~15 lines) instead of retriever abstractions. RRF fuses ranks, so the
  incompatible score scales of the two legs never mix.
- **The judge is from a different family than the generator** (gpt-4o-mini grading Gemini), so
  the system is never graded by its own relatives.
- **Loud failures.** Dead links, pages USCIS 301-redirects into `/archive/` (still HTTP 200, but
  officially stale), suspiciously short extractions, and under-covered judge metrics all fail
  visibly. Silent decay is how RAG systems rot.
- **Determinism end-to-end.** Content-hashed fetches, deterministic chunk IDs, and a
  `corpus_hash` stamped into every eval run.

## Run like a service

Backend: a Dockerfile (CPU-pinned torch, models and the LanceDB index baked in, offline at
runtime) wrapped by Modal, scale-to-zero. Frontend: static on Vercel. Hosting cost: ~$0.

Every request emits an OpenTelemetry trace (OTLP to Langfuse; the backend is one env-var swap
away, nothing is vendor-locked). A real production trace:

```
POST /api/chat ..................... 13.1 s
|- rag.dense_retrieve .............. 0.10 s
|- rag.bm25_retrieve ............... 0.02 s
|- rag.fuse ........................ 0.00 s
|- rag.rerank ...................... 4.5 s    <- cross-encoder on 2 free vCPUs
|- rag.generate .................... 8.5 s
   |- gen_ai (Gemini) .............. 8.5 s    <- ~4.7k input tokens, usage captured
```

That one trace settles where the latency lives (CPU rerank + generation; retrieval is free),
which is why the latency SLO is an honest **p50 ≤ 8 s / p95 ≤ 12 s warm** rather than a number
the free tier cannot meet. Logs are structured JSON carrying the trace_id, so a Modal log line
and its Langfuse trace cross-reference exactly. Spans record counts, scores, and durations,
never the user's question or the answer. Answer quality is deliberately a release gate (the
harness in CI), not a runtime metric. SLIs, SLOs, and the error-budget contract:
[docs/observability-slo.md](docs/observability-slo.md).

## Status

| Phase | Scope | State |
|---|---|---|
| 1: Ingestion pipeline | fetch / extract / chunk / index, CLI, tests | ✅ done |
| 2: RAG core | hybrid retrieval, reranking, LangGraph pipeline | ✅ done |
| 3: Eval harness | golden datasets, RAGAS + retrieval metrics, YAML experiments, fail-closed gates | ✅ done ([leaderboard](docs/EVALS.md)) |
| 4: App | FastAPI + SSE chat UI + live eval dashboard | ✅ done |
| 5: Deploy | Dockerized backend on Modal (scale-to-zero) + Vercel frontend | ✅ [live](https://visa-wise-six.vercel.app) |
| 6: Observability | OTel traces to Langfuse, trace-correlated JSON logs, SLOs | ✅ done ([SLOs](docs/observability-slo.md)) |

## Quickstart

```bash
uv sync                                # Python 3.12, uv-managed
cp .env.example .env                   # add GOOGLE_API_KEY (serving)
                                       #   + OPENAI_API_KEY (eval judge)

uv run visawise-ingest all             # fetch -> extract -> chunk -> index (~35s)
uv run visawise-ingest status          # corpus vs index drift report

uv run python -c "
from visawise.rag.retrieval import hybrid_search
for hit in hybrid_search('Can an F-1 student work off campus?', top_k=4):
    print(f'{hit.score:.4f}  {hit.title}')"

uv run visawise-app                    # chat UI + eval dashboard at :8000
uv run pytest                          # fast, no-network unit tests
```

No vector-DB account needed: LanceDB is embedded, dense vectors and BM25 in one local table that
ships inside the deploy container. Telemetry is a strict no-op unless the Langfuse env vars are
set.

## Repo layout

```
configs/            source catalog + experiment definitions (YAML)
src/visawise/
  ingestion/        fetch -> extract -> chunk -> index (+ visawise-ingest CLI)
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
bge-base-en-v1.5 embeddings · bge-reranker-base cross-encoder · Gemini 3.1 Flash-Lite generation
(chosen by the eval harness; provider is one flag away) · RAGAS 0.4 + gpt-4o-mini judge ·
FastAPI · Docker + Modal + Vercel · OpenTelemetry + Langfuse · uv
