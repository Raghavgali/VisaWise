"""Reranker builders (plain callables: (query, [ScoredChunk], top_n) -> [ScoredChunk]).

- "local": BAAI/bge-reranker-base cross-encoder via sentence-transformers
  (CPU-viable at 8->4, no API dependency) -- what the deployed app uses.
- "cohere": Cohere rerank API -- eval-side parity with the 2024 experiments
  only (trial tier is 1k calls/month, non-production; never wire into serving).
- "none": passthrough truncation to top_n (ablation experiments).
"""

from dataclasses import replace
from typing import Callable, Literal
from functools import lru_cache

from .retrieval import ScoredChunk
from ..config import settings

RerankerKind = Literal["local", "cohere", "none"]

Reranker = Callable[[str, list[ScoredChunk], int], list[ScoredChunk]]

COHERE_RERANK_MODEL = "rerank-v4.0-pro"

@lru_cache(maxsize=1)
def load_local_model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.local_reranker_model)

def local_rerank(
        query: str,
        chunks: list[ScoredChunk],
        top_n: int
) -> list[ScoredChunk]:
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    if not chunks or top_n == 0:
        return []
    
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = load_local_model().predict(pairs, show_progress_bar=False)

    rescored = [
        replace(chunk, score=float(score))
        for chunk, score in zip(chunks, scores, strict=True)
    ]

    return sorted(rescored, key=lambda chunk: chunk.score, reverse=True)[:top_n]

@lru_cache(maxsize=1)
def load_cohere_client():
    if not settings.cohere_api_key:
        raise RuntimeError("COHERE_API_KEY is required for the Cohere reranker")
    
    import cohere 

    return cohere.ClientV2(api_key=settings.cohere_api_key)


def cohere_rerank(
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
) -> list[ScoredChunk]:
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    if not chunks or top_n == 0:
        return []
    
    response = load_cohere_client().rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=[chunk.text for chunk in chunks],
        top_n=min(top_n, len(chunks)),
    )

    return [
        replace(
            chunks[result.index],
            score=float(result.relevance_score),
        )
        for result in response.results
    ]


def passthrough_rerank(
        _query: str,
        chunks: list[ScoredChunk],
        top_n: int,
) -> list[ScoredChunk]:
    return chunks[:top_n]

def build_reranker(kind: RerankerKind) -> Reranker:
    if kind == "local":
        return local_rerank
    
    if kind == "cohere":
        return cohere_rerank
    
    if kind == "none":
        return passthrough_rerank
    
    raise ValueError(f"Unknown reranker kind: {kind!r}")