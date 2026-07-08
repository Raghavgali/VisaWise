"""Reusable USCIS corpus ingestion pipeline.

Stages (each idempotent, re-runnable, driven by `visawise-ingest`):

    fetch   configs/sources.yaml -> data/raw/*.html + data/raw/manifest.json
    extract data/raw/            -> data/extracted/*.md (frontmatter + clean markdown)
    chunk   data/extracted/      -> data/chunks/chunks.jsonl   <- single source of truth
    index   data/chunks/         -> LanceDB table: embeddings + tantivy BM25 FTS
                                    (idempotent, prunes stale rows, ships in container)
"""
