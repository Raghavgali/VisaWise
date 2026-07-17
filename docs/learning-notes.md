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

## 11. graph.py (Phase 2) — issues caught in review

Nine bugs, and a new *class* of bug: stringly-typed graph wiring. Node names,
edge targets, and state-dict keys are all strings — nothing checks them until
runtime (or ever, for dict keys).

1. **Misspelled attribute, twice** (`config.retreiver`) — would have raised
   `AttributeError` on every query. Frozen dataclasses don't autocomplete
   typos away.
2. **Edge to a node that doesn't exist** (`"dense_retriever"` vs registered
   `"dense_retrieve"`) + `add_node(START, ...)` where `add_edge` was meant +
   `previous_node = "bm25_retreive"` typo. LangGraph catches these at
   `compile()` — *if* you compile that branch. Ablation branches that only
   run in eval configs are exactly the ones that never get exercised manually.
3. **Conditional swallowed into an f-string**:
   `f"Section: {hit.section}\n if hit.section else"` — syntactically valid,
   renders the literal text into the LLM prompt. And `list.append(a, b, c, d)`
   — `append` takes one argument.
4. **Type annotation disagreed with the code**: `Literal["dense", "hybrid"]`
   while every dispatch checked `"vector"`/`"bm25"`/`"hybrid"`.

**Lessons:**
- **Compile-shape tests are nearly free and kill the whole wiring-bug class**:
  compiling a LangGraph executes no nodes, so `build_graph(cfg)` +
  `compiled.get_graph().nodes` assertions run in CI with a fake API key —
  every retriever/reranker branch gets its wiring checked on every test run.
- **Validate config before constructing resources.** The unknown-retriever
  check originally sat *below* `ChatGroq(...)`, so a bad retriever with a
  missing key reported the wrong error.

## 12. Eval harness (Phase 3) — RAGAS 0.4 live-fire lessons

Built with two subagents in parallel (opus for datasets/metrics/runner, sonnet
for report/CLI) against pinned contracts; offline tests all passed first try.
Every failure happened in the *live* path — the part tests can't cover.

- **Verify model names against the API, not memory.** "gpt-5.1-mini" doesn't
  exist (`/v1/models` says so); the general-purpose minis are gpt-5-mini and
  gpt-5.4-mini. Same pattern as invented API schemas, one level up. (Settled
  on gpt-4o-mini — for a 50-sample × 4-metric judge run, cost-fit beats
  horsepower, and judge *consistency across runs* matters more than judge IQ.)
- **ragas 0.4.3's `default_transforms` crashes on short docs**: it only
  extracts `headlines` for documents > 500 tokens but runs `HeadlineSplitter`
  on *all* document nodes → `ValueError: 'headlines' property not found`. Fix:
  filter short docs out of the generator input (they're nav hubs — poor exam
  material anyway; they stay in the retrieval corpus). Also needs `rapidfuzz`
  (undeclared optional dep of the transform pipeline).
- **Naive equality never survives contact with generated text.** Mapping
  ragas `reference_contexts` back to source URLs by substring failed 4/6:
  multi-hop contexts get a `<N-hop>\n\n` prefix and the splitter mangles
  punctuation at chunk boundaries. Fix: match on an alphanumeric-only key,
  exact substring first, `rapidfuzz.partial_ratio >= 85` fallback → 49/52
  mapped. When one generator writes and another program reads, normalize
  aggressively and keep a fuzzy fallback.
- **The renamed-metric trap**: ragas 0.4 has no `ResponseRelevancy` — the
  metric class is `AnswerRelevancy` (and it silently *requires* an
  `embeddings=` arg). The subagent caught this by reading the installed
  package source instead of trusting the obvious name — the single
  highest-value instruction in the delegation prompt.
- **Samples with no ground-truth mapping score `None`, never 0** — faking a
  zero would poison aggregates; `None` gets skipped in aggregation and the
  honesty probes (out-of-scope curated questions) stay usable.

## 13. The coverage bug — when "loud failures" aren't loud enough

An external review of the committed RunRecords caught what our own reporting
missed: faithfulness had only scored 24–31 of 52 samples per run. Every
failure was the same — ragas' instructor adapter defaults the judge to
`max_tokens=1024`, and the faithfulness verdict JSON (statement decomposition
+ per-statement verdicts) blows past that on long answers. The harness did
capture each failure as an `{"error": ...}` entry and printed a warning count,
but the *aggregates* then averaged the survivors and the leaderboard rendered
`0.866` with nothing marking it as a 29-of-52 number.

- **Partial averages are biased averages.** The failures weren't random: they
  hit exactly the long, claim-dense answers — the hardest samples. A survivor
  mean over the easy subset silently flatters every run, and each run keeps a
  *different* survivor subset, so cross-run deltas were doubly meaningless.
- **Error capture without coverage accounting is half a safety net.** Catching
  the exception kept runs alive (good) but let a 56%-coverage aggregate wear
  the same clothes as a 100% one (bad). Fix: `coverage: {metric: {scored,
  total}}` in every RunRecord, `⚠k/n` beside every partial aggregate in the
  leaderboard and dashboard, and a runner gate that fails the run when a
  judged metric covers < 95% (records still written first — salvage, then
  scream).
- **A one-run warning is not a report-level guarantee.** The `score_judged`
  warning printed at run time and scrolled away; coverage now travels *with
  the data* so no later reader depends on having watched the console.
- **Read the library's own fine print**: ragas' `llm_factory` docstring
  literally says "If structured output is truncated, increase max_tokens."
  Default 1024; gpt-4o-mini allows 16K out; `judge_max_tokens = 8192` ended
  truncation entirely (52/52 in the reruns).
- The superseded runs moved to `evals/runs/archive/` with a README — a wrong
  number, once published, should leave a paper trail, not vanish.

## 14. Controlled reranker ablation — separating three tangled effects

A review pointed out our headline comparison wasn't controlled: `vector_only`
was (vector retrieval, no reranker) and `hybrid_default` was (hybrid
retrieval, local reranker). Two variables moved at once, so "hybrid wins"
couldn't distinguish *retrieval* from *reranking*. Worse, the reranker itself
moves two things simultaneously — it **reorders** the candidates *and*
**truncates** them (top_k=8 → rerank_top_n=4) — so even "rerank vs no rerank"
tangles reranking with how many contexts reach the LLM.

Fix: a 4-rung ladder where each adjacent pair changes exactly ONE variable.
Everything else (dataset synthetic_v2, 8B LLM, prompt v1, top_k=8, vector
weight 0.6, temperature 0, corpus hash, gpt-4o-mini judge) is held fixed.

| Rung | Config | Contexts to LLM |
|---|---|---|
| 1 `vector_only` | vector, no rerank | 8 |
| 2 `hybrid_no_rerank` | hybrid, no rerank | 8 |
| 3 `hybrid_rerank_full` | hybrid, rerank, top_n=**8** | 8 (reordered) |
| 4 `hybrid_default` | hybrid, rerank, top_n=**4** | 4 (serving) |

- **1→2 isolates the retrieval algorithm** (vector vs hybrid): both pass 8
  contexts, neither reranks.
- **2→3 isolates the reranker's reordering**: identical 8-context *set*, only
  the order differs. This is the key rung the naive 3-arm design omits — it
  holds context count at 8 so reordering doesn't get credit for truncation.
- **3→4 isolates truncation** (8→4 contexts): identical rerank order.

Falsifiable predictions written down *before* reading the results:
- Set-based metrics (`hit_rate`, `context_recall`) are computed over the
  retrieved set, which is byte-identical between rung 2 and rung 3 → they
  **must be equal** across 2 and 3. If they aren't, there's a bug in how the
  runner captures the final context set. (A built-in consistency check.)
- Rank-aware metrics (`MRR`, `context_precision`) should improve 2→3 iff the
  cross-encoder genuinely ranks better than RRF.
- The `hit_rate` / `context_recall` *drop* the reviewer saw at serving should
  appear at **3→4** (truncation), not 2→3 (reranking) — i.e. it's the price of
  passing 4 contexts instead of 8, not a failing of the reranker.

### What we found (synthetic_v2, 52/52 coverage)

Deterministic retrieval metrics (no judge, so no noise — the trustworthy ones):

| Rung | hit_rate | MRR | NDCG |
|---|---|---|---|
| 1 vector | 0.885 | 0.777 | 0.781 |
| 2 hybrid, no rerank | 0.885 | 0.798 | 0.802 |
| 3 hybrid, rerank→8 | 0.885 | **0.838** | **0.826** |
| 4 hybrid, rerank→4 (serving) | 0.865 | 0.833 | 0.813 |

The decomposition, one variable at a time:
- **1→2 retrieval (vector→hybrid):** hit_rate flat (0.885), MRR +0.021, NDCG
  +0.021. Both find a relevant page equally often; hybrid just ranks it higher.
- **2→3 reranking, set held identical:** hit_rate provably flat (0.885 =
  0.885), but **MRR +0.040 and NDCG +0.024**. This is the cleanest result in
  the whole harness — the cross-encoder's reordering lifts ranking by an amount
  *far* above the judge noise floor, with the retrieved set byte-identical. The
  reranker earns its place here, unambiguously.
- **3→4 truncation (8→4 contexts):** hit_rate 0.885→0.865 — the entire
  hit_rate drop the reviewer flagged is *truncation*, exactly as predicted, not
  reranking. It buys context_precision (0.887→0.925) and halves generation
  input tokens (p50 6.1s→4.1s), which is why serving keeps it on a free tier.

**Judge-noise calibration (v2):** the two byte-identical hybrid+rerank+8B runs
differ by |Δ|faithfulness = 0.025, everything else ≤ 0.005. So ~0.03 is the
noise floor — which is *exactly* why the deterministic MRR/NDCG gains matter:
the reranker's value is invisible in the judged metrics (they wiggle within
noise) but crisp in MRR (+0.040, no noise).

**A prediction I got wrong — and the lesson.** I pre-registered that *both*
`hit_rate` and `context_recall` must be exactly equal across rungs 2 and 3
because "they're computed over the retrieved set." `hit_rate` held (0.885 =
0.885) and the per-sample retrieved sets were verified byte-identical. But
`context_recall` came out 0.928 vs 0.937 — because I misclassified it.
`context_recall` is a **RAGAS LLM-judged** metric (it decomposes the reference
answer into claims and asks the judge whether each is supported), so it carries
judge noise even when the context set is identical. Only `hit_rate` / `MRR` /
`NDCG` are deterministic. The consistency check was still valuable — it just
lives entirely in the deterministic metrics, and the 0.009 recall wiggle is
noise, not a bug. Lesson: know which of your metrics have an LLM in the loop
before you claim two of them must be bit-identical.

## 15. Safety slices, the guardrail prompt, and picking a free model

The synthetic set only had ordinary answerable questions. A review pushed for a
human-reviewed `curated_v1` (30 samples, 6 slices × 5) that could test what the
synthetic set can't: temporal/current facts, stale-source correction,
out-of-corpus abstention, prompt-injection resistance, and high-risk
qualification. To score these honestly the runner became **slice-aware**:
answer-bearing slices get the RAGAS content metrics; the three safety slices get
`score_abstention` (a gpt-4o-mini judge that asks "did it correctly abstain /
refuse / qualify?"). `coverage_from_scores` was changed so a metric's `total`
counts only samples where it was *attempted* — so a metric that applies to just
one slice gates against its own denominator, not the whole run.

**First curated run exposed the real gap.** The free 8B (`llama-3.1-8b`) was
strong on content but abstained on only ~2/15 safety cases — it answered
out-of-corpus questions from loosely-related chunks, gave confident advice on
high-risk cases, and could be induced to **print its own system prompt**. A
single blended faithfulness number (0.87) hid all of this; the slices surfaced
it. That's the whole argument for slice-aware eval.

**The guardrail prompt took four tries, each fixing the last one's measured
failure:**
- **v1** (no guardrail): safety 2/15.
- **v2** (add "abstain / qualify / resist injection"): safety up, but it
  over-hedged — it opened answerable answers with a false "the pages don't cover
  this... *however*, <correct answer>", so answer-slice **relevancy collapsed**
  (0.17–0.31).
- **v3** (make it a clean binary: answer directly OR say not-covered, never
  both): fixed the over-hedging (relevancy 0.45→0.93 on dev) but the blunt
  binary **collapsed *refuse* and *qualify* into the "not covered" bucket** —
  fraud/persona attacks got a bland deflection instead of a refusal, and a
  "help me fix my illegal status quietly" got answered. Dev safety 5→3.
- **v4** (explicit **precedence**: REFUSE → QUALIFY → answer/abstain): kept v3's
  directness *and* restored refuse/qualify. The four behaviors stopped poisoning
  each other.

**Held-out discipline mattered and is the reason to trust the numbers.** I tuned
the prompt against a separate `dev_v1` (12 fresh questions, same corpus/slices,
disjoint from curated) and only measured on `curated_v1` once per candidate.
This caught optimism: v4 looked like a clean 6/6 sweep on dev (n=2/slice) but
landed at 10/15 safety and 0.42 answerable relevancy on the 5-per-slice test —
better than v2 on answer quality, not the blowout dev implied. Tuning on the
test set would have hidden that.

**Free tiers of strong models are a quota mirage.** The model is a one-flag swap,
so we ablated it: the free 8B guardrails weakly; `llama-4-scout` (Groq) is fast
but mediocre on safety; `gpt-oss-20b` (Groq) is excellent but 31s/query *and*
died at sample 14 on the 200K-tokens/day cap; `gemini-2.0-flash` is retired and
`gemini-3.5-flash`'s free tier is **20 requests/day** (aborted at 19/30). Only
`gemini-3.1-flash-lite` was fast (2.2s), strong (best content faithfulness), and
had a free quota that finished a 30-sample run. It's also **judge-independent** —
using gpt-4o-mini as both generator and judge would invite self-preference bias,
so a non-OpenAI generator is the methodologically correct choice. A Gemini 3.x
detail that would have silently poisoned every metric: its `.content` is a list
of content blocks, not a string, so the generate node now joins the text parts
(`_message_text`) instead of `str(list)`.

**A grounding gate: hypothesized, tested, rejected.** Both prompts leave
out-of-corpus at ~1/5. Idea: drop retrieved context when the top cross-encoder
rerank score is below a threshold (nothing on-topic) so the prompt abstains.
Dev reranker scores separated cleanly (on-topic ≥0.64, off-topic ≤0.18), but on
the held-out test the gate did **not** move out_of_corpus (1/5→1/5) and nicked
high_risk (5→4) and content. Two reasons it fooled the local analysis: (a) the
hardest out-of-corpus case ("processing time for **I-765**") is *topically*
in-corpus, so the reranker scores it 0.90 and the gate never fires; (b) the
gated model *does* abstain tersely ("the official pages don't cover it"), but the
`out_of_corpus` references demand a redirect (to CDC/DMV/admissions), and the
abstention judge scored the **same wording** as pass for one sample and fail for
three. So the "1/5" is partly a **reference/judge inconsistency**, not the model
answering off-topic — a known limitation to fix in the metric, not the model.
The gate stays as an off-by-default `grounding_threshold` knob; it does not ship.

**Serving config, chosen entirely by measurement:** `gemini:gemini-3.1-flash-lite`
+ guardrail prompt **v4** + thinking off, hybrid 0.6 / rerank→4. Free, ~2.3s,
safety 10/15 (high_risk 5/5, adversarial 4/5), answerable faithfulness 0.92.
