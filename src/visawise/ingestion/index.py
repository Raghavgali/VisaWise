"""Stage 4: embed chunks and build the LanceDB table (idempotent).

Contract:
- Embeds with BAAI/bge-base-en-v1.5 (768-dim) [validated 2024] via
  sentence-transformers, batched, CPU/MPS.
- Writes the LanceDB table at settings.lancedb_dir: one row per chunk
  {vector, chunk_id, doc_id, seq, text, url, title, topic, section, fetched_at}.
- Creates the native (Rust) full-text index on `text` with English stemming --
  the BM25 leg of hybrid search, parity with the 2024 BM25Retriever config.
- Idempotent by chunk_id: unchanged chunks are skipped, stale rows (ids no
  longer in chunks.jsonl) are deleted. force=True rebuilds from scratch.
- No ANN index on purpose: at corpus scale (tens to hundreds of rows) LanceDB
  brute-force kNN is exact and instant; revisit if the corpus grows >50k.
- `status()` reports drift between chunks.jsonl and the table.

The embedded DB directory is a build artifact: ship it inside the deploy
container (no external vector service). Heavy imports stay inside functions.
Embedding goes through visawise.embedding (the single loader shared with
query-side code), so index and query vectors always agree.
"""

from dataclasses import asdict, dataclass

from ..config import settings
from ..embedding import embed_texts
from .chunk import Chunk, load_chunks


@dataclass
class IndexSummary:
    added: int = 0
    skipped: int = 0
    pruned: int = 0


def _connect():
    import lancedb

    settings.lancedb_dir.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(settings.lancedb_dir)


def _rows(chunks: list[Chunk]) -> list[dict]:
    vectors = embed_texts([chunk.text for chunk in chunks])
    return [{"vector": vector, **asdict(chunk)} for vector, chunk in zip(vectors, chunks)]


def _existing_ids(table) -> set[str]:
    return set(table.to_arrow().column("chunk_id").to_pylist())


def _create_fts_index(table) -> None:
    table.create_fts_index(
        "text",
        replace=True,
        use_tantivy=False,  # native Rust FTS: persisted, no extra dependency
        language="English",
        stem=True,
        remove_stop_words=True,
    )


def index_chunks(chunks: list[Chunk] | None = None, force: bool = False) -> IndexSummary:
    chunks = chunks if chunks is not None else load_chunks()
    db = _connect()
    summary = IndexSummary()

    if force and settings.lancedb_table in db.table_names():
        db.drop_table(settings.lancedb_table)

    if settings.lancedb_table not in db.table_names():
        table = db.create_table(settings.lancedb_table, data=_rows(chunks))
        summary.added = len(chunks)
        _create_fts_index(table)
        return summary

    table = db.open_table(settings.lancedb_table)
    existing = _existing_ids(table)
    wanted = {chunk.chunk_id for chunk in chunks}

    stale = existing - wanted
    if stale:
        id_list = ", ".join(f"'{chunk_id}'" for chunk_id in stale)
        table.delete(f"chunk_id IN ({id_list})")
        summary.pruned = len(stale)

    to_add = [chunk for chunk in chunks if chunk.chunk_id not in existing]
    summary.skipped = len(chunks) - len(to_add)
    if to_add:
        table.add(_rows(to_add))
        summary.added = len(to_add)

    if to_add or stale:
        _create_fts_index(table)

    return summary


def status() -> dict:
    from .chunk import corpus_hash

    chunks = load_chunks()
    wanted = {chunk.chunk_id for chunk in chunks}

    db = _connect()
    if settings.lancedb_table in db.table_names():
        existing = _existing_ids(db.open_table(settings.lancedb_table))
    else:
        existing = set()

    return {
        "strategy": settings.chunk_strategy,
        "corpus_hash": corpus_hash(chunks)[:16],
        "chunks_jsonl": len(chunks),
        "table_rows": len(existing),
        "missing_from_table": len(wanted - existing),
        "stale_in_table": len(existing - wanted),
        "in_sync": wanted == existing,
    }
