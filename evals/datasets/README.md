# Golden datasets

Frozen, versioned question sets. A dataset is **frozen once a committed run
references it** — any change means a new version file, never an in-place edit
(the loader/`save_dataset` enforce this). Every RunRecord stamps the dataset
`name` + `sha256`, so a result is always traceable to exact bytes.

| File | sha256 (prefix) | Samples | Notes |
|---|---|---|---|
| `synthetic_v1.jsonl` | `584c7810…` | 52 | RAGAS 0.4 TestsetGenerator over the fresh corpus. 3 samples (syn-018/038/047) have empty `source_urls` — their multi-hop reference contexts didn't map to a single page, so they score `None` on retrieval metrics. |
| `synthetic_v2.jsonl` | `e5dfa633…` | 52 | **v1 + source_urls for syn-018/038/047** (exact-text matched to the H-1B specialty-occupations and temporary-nonimmigrant-workers pages). Questions, references, and reference_contexts are byte-identical to v1 — only retrieval ground truth was added, so judged metrics stay directly comparable across the two. This is the active dataset. |

### v1 → v2

v1 was already committed when a review flagged the 3 unmapped samples. Rather
than mutate a frozen file, v2 adds only the missing `source_urls` so retrieval
metrics cover 52/52. v1 stays in the repo unchanged as the lineage record and
for the committed runs that reference it (the aborted 70B run).
