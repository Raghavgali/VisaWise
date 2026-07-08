# VisaWise Build — Lessons & Issues Log

Study notes from productionizing VisaWise (Phase 1: ingestion, Phase 2: RAG
core), July 2026. Each entry: what happened, why it matters, and the
transferable lesson.

---

## 1. The recurring bug: AI-generated code invents schemas instead of reading them

The single most repeated failure across all three hand-written/codex-assisted
stages. Each time, generated code used a *plausible-looking* vocabulary instead
of the actual upstream contract:

| Stage | Invented | Actual contract |
|---|---|---|
| fetch.py | `key, category, label, status, body_sha256` | `doc_id, topic, title, http_status, content_sha256` |
| extract.py | whole parallel `Snapshot` dataclass + `load_manifest()` | should have imported `FetchRecord` + `read_manifest` from fetch.py |
| chunk.py | `metadata["key"] / ["label"] / ["category"] / ["captured_at"]` | frontmatter keys `doc_id / title / topic / fetched_at` |

Every one was a runtime crash (`TypeError`/`KeyError`), invisible until executed.

**Lessons:**
- In a pipeline, each stage consumes the previous stage's schema. There must be
  exactly ONE definition of that schema (a dataclass), and downstream stages
  IMPORT it — never redefine it. (`extract.py` importing `FetchRecord` is the fix
  pattern.)
- When using AI codegen, hand it the real contract (our docstring contracts
  exist for exactly this) and diff its field names against the source of truth
  before running.
- Contract-in-docstring at the top of each module turned every one of these
  from "mystery crash" into "5-second diff against the spec."

## 2. fetch.py — issues caught in review

1. **Dataclass used as an instance**: `FetchSummary["failed"].append(...)` /
   `FetchSummary["fetched"] += 1` — dict operations on the *class object*.
   Needed `summary = FetchSummary(...)` then attribute access. Python won't
   stop you referencing a class where you meant an instance.
2. **Hardcoded config**: `timeout=20`, `sleep(0.5)`, `attempts=3`,
   UA `"ExampleMirror/0.1"` — all of these already existed in `Settings`.
   Hardcoding scattered copies is exactly how the 2024 notebooks became
   unreproducible.
3. **Retry blind spot**: retrying only on transport exceptions
   (`httpx.HTTPError`) means a 502/503 — which returns a *response*, not an
   exception — was treated as a permanent failure. Fix: treat 5xx as retryable;
   4xx as permanent.
4. **Backoff off-by-one**: sleeping after the final failed attempt wastes time
   before raising. Guard with `if attempt < attempts - 1`.
5. **Unused/undeclared import**: `import requests` — not used AND not in
   dependencies; the import itself would crash. Dead imports are not harmless.

**Design wins worth remembering:**
- *Raw-first storage*: fetch stores verbatim HTML; extraction is a separate,
  re-runnable stage. You can improve extraction forever without re-crawling,
  and you can always answer "is the diff from my code or from the source?"
- *Atomic writes*: write to `.tmp`, then `Path.replace()` — a crash mid-write
  can never corrupt the manifest.
- *Non-200s never overwrite a previous good snapshot.*
- *Content-hash skip*: unchanged pages don't rewrite files, so `fetched_at`
  honestly means "when content last changed."

## 3. The archive discovery — link rot is a *system design input*

While checking a source PDF from the 2024 project ("FAQs for Individuals in
H-1B Nonimmigrant Status"), we found USCIS had moved it to `/archive/` with an
explicit "information on this page is out of date" banner — in under two years,
the project's single most important source went stale.

Crucially, USCIS archives pages via **301 redirect into `/archive/`**, which
still returns HTTP 200. A naive fetcher would happily ingest officially-stale
policy. Fix: after following redirects, inspect the final URL; `/archive/`
means archived → record `archived=True`, report loudly, and extract skips it.

**Lesson:** "the fetch succeeded" ≠ "the content is valid." Encode staleness
signals (redirects, banners, dates) into the pipeline, not into your memory.
Corollary: the eval golden dataset generated from that 2024 document is also
invalid against the fresh corpus — data lineage matters end to end.

## 4. extract.py — issues caught in review

1. Parallel `Snapshot` schema (see §1) — delete, import `FetchRecord`.
2. **Path bug**: passed `data_dir` where `raw_dir` was expected → looked for
   HTML at `data/<id>.html` instead of `data/raw/<id>.html`. Every doc would
   "fail" with a misleading missing-file error.
3. **`clean_dir=None` never resolved** to `settings.extracted_dir` →
   `AttributeError` on first use. Pattern for optional-with-default args:
   `x = x if x is not None else settings.default`.
4. **Kwarg drift on our own dataclass**: `ExtractSummary(cleaned=...)` vs field
   `extracted`.
5. **Forgot the archived skip** — the entire point of §3.

## 5. The "Related Links" extraction issue

Symptom noticed by eyeballing extracted output (always do this): some pages
ended with `#### Related Links` / `**Form**` / `**Other USCIS Links**` — empty
scaffolding. Cause: trafilatura (with `include_links=False`) drops anchor
*hrefs/text* but keeps surrounding headings and bold group labels of the
page-end link boxes.

Investigation before fixing (the important part):
- Only 4 of 24 docs had the heading-form pattern; other grep hits were
  legitimate prose ("for more information, see...").
- All 4 sections were pure navigation lists — titles of pages that are already
  separate corpus sources. Zero unique prose. Safe to remove.

Fix: `strip_trailing_link_sections()` — removes a trailing
`Related Links`/`More Information` section **only if** every line after the
heading looks like a link label (no line > 200 chars). A same-named heading
over genuine prose can never be silently deleted.

**Lessons:**
- Extraction quality issues are invisible in summary counts (`extracted=24`
  both before and after). Inspect actual artifacts.
- Write destructive text-cleanup with a conservative guard + unit tests for
  both directions (drops junk / keeps prose).
- Regex subtlety: verifying with bytes (`od -c`) beats squinting — my first
  grep "mismatch" turned out to be two different files, not invisible
  characters.

## 6. chunk.py — issues caught in review

1. `removeprefix("----\n")` — **four dashes**. YAML happened to tolerate the
   leftover `---` (it's a document-start marker), so this *worked by accident*.
   The worst kind of bug: latent, invisible, waiting for a stricter parser.
2. Frontmatter key drift (see §1).
3. `Chunk(...)` missing the new `section` field → `TypeError`. When a shared
   dataclass gains a field, grep for every constructor call.

**Design decisions to remember:**
- **Deterministic chunk IDs**: `sha256(url::seq::text_sha)[:24]`. Same input →
  same IDs → idempotent indexing, diffable corpora, meaningful `corpus_hash`.
- **`corpus_hash` = sha256 over sorted chunk IDs**: one value identifying the
  exact corpus (content AND chunking strategy) any eval run measured.
  Order-independent (sorted), collision-safe enough, cheap.
- **Chunking strategy is an experiment axis, not a belief**: `token` (2024
  baseline, kept for comparability) vs `section` (heading-aware breadcrumbs)
  behind config; the eval harness declares the winner. 62 vs 180 chunks from
  the same 24 docs.
- **Tokenizer parity**: the splitter counts tokens with the *embedder's own
  tokenizer*, so "1024 tokens" means the same thing at split time and embed
  time.
- **The 2024 quirk**: 1024-token chunks were embedded with bge-base whose max
  sequence is 512 — dense retrieval only ever saw the first half of each chunk
  (BM25 saw all of it). Nobody noticed in 2024 because nothing measured it.
  The `section` strategy (≤480 tokens + breadcrumb) fixes it — pending eval.

## 7. index.py / whole-pipeline lessons

- **Layered hashing absorbs upstream noise** (observed live): 5 of 24 pages
  had cosmetic HTML churn on refetch → extracted text identical → same chunk
  IDs → same `corpus_hash` → `index added=0`. Raw-layer hashes are sensitive;
  each downstream layer re-derives stability. No false rebuilds.
- **One embedder loader** (`get_embedder()`, lru_cached): index-side and
  query-side vectors MUST come from the same model/config. Two loading paths
  will eventually disagree silently.
- **Embedded DB as build artifact**: LanceDB gives dense vectors + stemmed
  BM25 FTS in one table, in-process. The deploy container ships the index —
  no external service, no keys, no cold-start dependency.
- **Skip the ANN index below ~50k vectors**: brute-force kNN is exact and
  instant at our scale. Approximate indexes are a scale tool, not a default.

## 8. Dependency resolution war stories

1. `llama-index-vector-stores-pinecone` pinned `pinecone>=7,<8` while we'd
   pinned v9 (per fresh research!). Integration packages' pins override the
   "latest SDK" instinct — the *integration* dictates the client version.
2. `ragas 0.4.3` declares its langchain deps **unpinned** but imports modules
   removed in `langchain-community 0.4` → `ModuleNotFoundError` at import
   time. Fix: our own `langchain-community>=0.3,<0.4` constraint, documented
   with a comment in pyproject.toml.

**Lesson:** a resolver "solution" only proves version *constraints* are
satisfiable, not that the code paths work — always run an import smoke test
after dependency changes. And when a library leaves deps unpinned, pin them
yourself and write down why.

## 9. retrieval.py (Phase 2) — issues caught in review

Six bugs in the first draft. The interesting part: only two would have
crashed. The other four were *silent* — they'd have shipped wrong behavior.

1. **Stale loop variable in `fuse_rrf`** (the headline bug). The sort lambda
   and the top-k loop referenced `hit.chunk_id` — but `hit` was the leftover
   variable from the *earlier* accumulation loop, still in scope. Two
   consequences: the sort key was a constant (so no actual ranking), and every
   returned result was the same chunk repeated N times.

   ```python
   for hits, weight in zip(result_lists, weights):
       for rank, hit in enumerate(hits, 1): ...      # `hit` lives on after this loop

   ranked_ids = sorted(fused_scores, key=lambda hit_id: fused_scores[hit.chunk_id])
   #                                        wanted hit_id ^^^  got stale `hit`
   ```

   **Lesson:** Python loop variables outlive their loop — a name that *looks*
   right can be a corpse from three loops ago, and no linter flags it because
   it's defined. Defenses: (a) don't reuse near-identical names (`hit` vs
   `hit_id`) in one scope, (b) write a regression test that would catch the
   symptom (`len(set(ids)) == len(results)` — now in test_retrieval.py).

2. **Copy-paste field bug**: `topic=row["title"]` — every hit's topic became
   its title. No crash, ever; just silently corrupt metadata flowing into
   citations and eval provenance. Repetitive field-mapping code is where
   copy-paste bugs breed; test one full round-trip, field by field.

3. **Wrong score column**: FTS results carry `_score`, not `_search`
   (`KeyError`). Don't guess a library's magic column names — print one real
   result row and look.

4. **Missing `.metric("cosine")` on dense search** — the quiet killer.
   LanceDB defaults to **L2**; the validated 2024 setup (Pinecone) was cosine,
   and our bge vectors are unnormalized, so L2 and cosine produce *different
   rankings*. Everything would have "worked" — retrieval returns plausible
   results either way — while quietly invalidating every comparison against
   the 2024 baseline. **Lesson: when migrating infrastructure, the new tool's
   defaults are not the old tool's defaults.** Write down the invariants
   (metric, dimensions, normalization) and assert them explicitly in code.

5. **Implicit query routing**: `table.search(query_string)` happened to do FTS
   because an FTS index exists ("auto" mode). Works until someone adds an
   embedding function registration and the same line silently becomes a vector
   search. `query_type="fts"` states intent.

6. **A second embedding loader** — `get_embedder()` re-defined in retrieval.py
   with a *hardcoded* model name, duplicating the ingestion-side loader. This
   is the exact failure mode §7 warned about, written days after writing §7.
   Fix was structural, not disciplinary: a shared `visawise/embedding.py` that
   both sides import, and a rule ("nothing else may construct a
   SentenceTransformer") stated in its docstring. **Lesson: an invariant that
   lives in your memory will be violated; an invariant that lives in the
   architecture can't be.**

**Design notes worth remembering:**
- **RRF fuses ranks, not scores.** The two legs have incomparable scales
  (negative cosine distance vs BM25); reciprocal-rank fusion never mixes them,
  which is *why* it's the right fusion for heterogeneous retrievers.
- Sanity-check the math on real output: top hit scored 0.0164 =
  0.6/(60+1) + 0.4/(60+1) — i.e., ranked #1 by both legs. When you can predict
  a score by hand, you actually understand the system.
- Cached table handle (`lru_cache`) is fine for a read-only serving process;
  document the assumption (re-ingestion needs a new process).
- Heeded a deprecation warning immediately (score columns must be selected
  explicitly) — future-you never has to rediscover it during an upgrade.

## 10. Meta-lessons (the ones that generalize)

- **Contracts first, then code.** Writing the module docstring contract before
  implementation made every review a diff-against-spec instead of a debate.
- **Verify by execution, not by reading.** Every stage was run against the
  real site/data immediately; summary counts + artifact inspection + an
  idempotency re-run. "It parses" proves almost nothing.
- **Loud failures beat graceful degradation** in a data pipeline: non-200s,
  archive redirects, too-short extractions all fail visibly. Silent data
  quality decay is how the 2024 corpus rotted.
- **CLI = thin presentation layer** over library functions that take explicit
  arguments (testable, composable); settings bind at the edge.
