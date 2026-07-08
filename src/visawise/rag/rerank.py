"""Reranker builders (plain callables: (query, [ScoredChunk], top_n) -> [ScoredChunk]).

- "local": BAAI/bge-reranker-base cross-encoder via sentence-transformers
  (CPU-viable at 8->4, no API dependency) -- what the deployed app uses.
- "cohere": Cohere rerank API -- eval-side parity with the 2024 experiments
  only (trial tier is 1k calls/month, non-production; never wire into serving).
- "none": passthrough truncation to top_n (ablation experiments).
"""

from typing import Callable, Literal

from .retrieval import ScoredChunk

RerankerKind = Literal["local", "cohere", "none"]

Reranker = Callable[[str, list[ScoredChunk], int], list[ScoredChunk]]


def build_reranker(kind: RerankerKind) -> Reranker:
    raise NotImplementedError  # TODO(pairing)
