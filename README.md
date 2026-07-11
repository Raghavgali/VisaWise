# VisaWise

**A production RAG system for U.S. immigration policy (USCIS), built eval-first.**

Ask questions about F-1/OPT, H-1B, green cards, and status changes — answered from a
continuously refreshable corpus of official USCIS pages, with citations. Every retrieval
decision in this system (hybrid weights, chunking strategy, reranking) is a **measured
experiment**, not a belief: the evaluation harness is the product's centerpiece, and the
configuration that serves traffic is the configuration that won on the benchmarks.

> ⚠️ Educational demo — not legal advice. Verify anything important at
> [uscis.gov](https://www.uscis.gov) or with an immigration attorney.

## Why this exists

Immigration policy changes fast and the cost of stale answers is high. The H-1B page in
this corpus, for example, leads with a September 2025 proclamation adding a **$100,000
payment condition** to certain petitions — a policy that did not exist when this project's
2024 research corpus was built. A chatbot serving last year's policy is worse than no
chatbot. VisaWise treats that as a systems problem: a re-runnable ingestion pipeline,
staleness detection, and an eval suite that regression-gates every change.

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
                                                              generate (Groq · Llama 3.3)
                                                                 │
             EVALS (visawise-eval)                            answer + citations
             YAML experiments × versioned golden datasets
             → RAGAS 0.4 (judge: gpt-4o-mini) + hit-rate/MRR/NDCG
             → committed RunRecords → regression gates → live dashboard
```

**Design principles**

- **One source of truth per layer.** `chunks.jsonl` feeds the vector index, the BM25 index,
  and eval provenance. One embedding loader serves both indexing and queries. One engine
  factory (`build_graph(EngineConfig)`) serves both the app and the eval harness — so
  benchmark numbers always describe the system that's actually deployed.
- **Framework-thin retrieval.** Direct LanceDB calls and a hand-rolled weighted
  reciprocal-rank fusion (~15 lines) instead of framework retriever abstractions.
  RRF fuses *ranks*, so the incompatible score scales of the two legs never mix.
- **Loud failures.** Dead links, pages that USCIS 301-redirects into `/archive/`
  (still HTTP 200 — but officially stale!), and suspiciously short extractions all fail
  visibly. Silent data decay is how RAG corpora rot.
- **Determinism end-to-end.** Content-hashed fetches, deterministic chunk IDs, and a
  `corpus_hash` stamped into every eval run — any result is traceable to the exact corpus
  that produced it.

## The research this builds on (2024)

VisaWise began as a masters project (with two teammates) evaluating **how much retrieval
strategy matters in RAG**, using RAGAS over a USCIS corpus. Key findings, preserved in
[`research/`](research/):

| Configuration | Faithfulness | Answer relevancy |
|---|---|---|
| Vector-only retrieval | 0.44 | 0.63 |
| **Hybrid (vector + BM25, RRF)** | **0.83** | **0.76** |

A weight sweep found **0.6 vector / 0.4 BM25** optimal (relevancy 0.89 with Cohere
reranking), and qualitative analysis showed *why* keyword-heavy retrieval fails: it
latches onto surface terms and injects wrong-context chunks. Those validated decisions —
hybrid retrieval at 0.6/0.4, rerank-to-4, 1024-token chunks, bge-base embeddings — are
this system's defaults, and the production eval harness re-tests them against the fresh
corpus rather than assuming they still hold.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 — Ingestion pipeline | fetch / extract / chunk / index, CLI, tests | ✅ done |
| 2 — RAG core | hybrid retrieval, reranking, LangGraph pipeline | ✅ done |
| 3 — Eval harness | golden datasets, RAGAS 0.4 + retrieval metrics, YAML experiments, regression gates | planned |
| 4 — App | FastAPI + chat UI + live eval dashboard | planned |
| 5 — Deploy | single self-contained container (index ships inside) | planned |

## Quickstart

```bash
uv sync                                # Python 3.12, uv-managed
cp .env.example .env                   # add GROQ_API_KEY (+ OPENAI_API_KEY for evals)

uv run visawise-ingest all             # fetch → extract → chunk → index (~35s)
uv run visawise-ingest status          # corpus ⇄ index drift report

uv run python -c "
from visawise.rag.retrieval import hybrid_search
for hit in hybrid_search('Can an F-1 student work off campus?', top_k=4):
    print(f'{hit.score:.4f}  {hit.title}')"

uv run pytest                          # fast, no-network unit tests
```

No vector-DB account needed: LanceDB is embedded — dense vectors and full-text search
live in one local table that ships inside the deploy container.

## Repo layout

```
configs/            source catalog + experiment definitions (YAML)
src/visawise/
  ingestion/        fetch → extract → chunk → index (+ visawise-ingest CLI)
  rag/              retrieval, rerank, LangGraph engine factory, versioned prompts
  evals/            datasets, metrics, runner, reports (+ visawise-eval CLI)
  app/              FastAPI: /api/chat + eval dashboard
data/               build artifacts (gitignored) + committed fetch manifest
evals/              versioned golden datasets + committed eval RunRecords
research/           the original 2024 notebooks, eval CSVs, and findings
docs/               build lessons log
```

## Stack

Python 3.12 · LangChain 1.x + LangGraph · LanceDB (embedded; vectors + tantivy FTS) ·
`BAAI/bge-base-en-v1.5` embeddings · `bge-reranker-base` cross-encoder · Groq
(Llama 3.3 70B) generation · RAGAS 0.4 + gpt-4o-mini judge · FastAPI · uv
