"""Retrieval primitives used by graph nodes. Framework-thin by design:
direct LanceDB calls + hand-rolled weighted reciprocal-rank fusion.

Contract:
- dense_search(query, top_k): embed query via visawise.embedding (same loader
  as ingestion) -> LanceDB kNN with COSINE metric [validated 2024: Pinecone
  was cosine; LanceDB defaults to L2, which ranks differently on our
  unnormalized bge vectors] -> [ScoredChunk], score = -distance.
- bm25_search(query, top_k): LanceDB native FTS (query_type="fts", English
  stemming -- parity with the 2024 BM25Retriever) -> [ScoredChunk],
  score = BM25 _score.
- fuse_rrf(result_lists, weights, top_k): weighted reciprocal-rank fusion,
  score(d) = sum_i weights[i] / (K + rank_i(d)), K=60. Reimplements the
  [validated 2024] QueryFusionRetriever reciprocal_rerank + retriever_weights
  behavior transparently. Fusion uses ranks only, so the incompatible score
  scales of the two legs (-distance vs BM25) never mix.
- hybrid_search(query, top_k, vector_weight): the composed default
  [validated 2024: 0.6 vector / 0.4 bm25]; each leg retrieves top_k
  candidates, fused down to top_k.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

from ..config import settings
from ..embedding import embed_query

RRF_K = 60

_SELECT_COLUMNS = ["chunk_id", "text", "url", "title", "topic", "section"]


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    url: str
    title: str
    topic: str
    section: str
    score: float


@lru_cache(maxsize=4)
def _open_table(db_dir: str, table_name: str):
    import lancedb

    return lancedb.connect(db_dir).open_table(table_name)


def open_table():
    """Cached table handle (read-only serving; re-ingestion needs a new process)."""
    return _open_table(str(settings.lancedb_dir), settings.lancedb_table)


def _row_to_hit(row: dict, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=row["chunk_id"],
        text=row["text"],
        url=row["url"],
        title=row["title"],
        topic=row["topic"],
        section=row["section"],
        score=score,
    )


def dense_search(query: str, top_k: int) -> list[ScoredChunk]:
    rows = (
        open_table()
        .search(embed_query(query))
        .metric("cosine")
        .select([*_SELECT_COLUMNS, "_distance"])
        .limit(top_k)
        .to_list()
    )
    return [_row_to_hit(row, -float(row["_distance"])) for row in rows]


def bm25_search(query: str, top_k: int) -> list[ScoredChunk]:
    rows = (
        open_table()
        .search(query, query_type="fts")
        .select([*_SELECT_COLUMNS, "_score"])
        .limit(top_k)
        .to_list()
    )
    return [_row_to_hit(row, float(row["_score"])) for row in rows]


def fuse_rrf(
    result_lists: list[list[ScoredChunk]],
    weights: list[float],
    top_k: int,
) -> list[ScoredChunk]:
    if len(result_lists) != len(weights):
        raise ValueError("result_lists and weights must have the same length")

    fused_scores: dict[str, float] = {}
    first_hit_by_id: dict[str, ScoredChunk] = {}

    for hits, weight in zip(result_lists, weights):
        for rank, hit in enumerate(hits, start=1):
            fused_scores[hit.chunk_id] = fused_scores.get(hit.chunk_id, 0.0) + weight / (
                RRF_K + rank
            )
            if hit.chunk_id not in first_hit_by_id:
                first_hit_by_id[hit.chunk_id] = hit

    ranked_ids = sorted(fused_scores, key=lambda hit_id: fused_scores[hit_id], reverse=True)

    return [
        replace(first_hit_by_id[hit_id], score=fused_scores[hit_id])
        for hit_id in ranked_ids[:top_k]
    ]


def hybrid_search(
    query: str,
    top_k: int | None = None,
    vector_weight: float | None = None,
) -> list[ScoredChunk]:
    top_k = top_k if top_k is not None else settings.retriever_top_k
    vector_weight = (
        vector_weight if vector_weight is not None else settings.hybrid_vector_weight
    )

    return fuse_rrf(
        [dense_search(query, top_k), bm25_search(query, top_k)],
        [vector_weight, 1.0 - vector_weight],
        top_k,
    )
