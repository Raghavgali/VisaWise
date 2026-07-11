"""Unit tests for rerank builders -- only the paths that need no model/API."""

import pytest

from visawise.rag.rerank import (
    build_reranker,
    cohere_rerank,
    local_rerank,
    passthrough_rerank,
)
from visawise.rag.retrieval import ScoredChunk


def hit(chunk_id: str, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, text=f"text-{chunk_id}", url="u", title="T",
        topic="x", section="", score=score,
    )


def test_build_reranker_dispatch():
    assert build_reranker("local") is local_rerank
    assert build_reranker("cohere") is cohere_rerank
    assert build_reranker("none") is passthrough_rerank


def test_build_reranker_unknown_kind_raises():
    with pytest.raises(ValueError):
        build_reranker("fancy")


def test_passthrough_truncates_without_reordering():
    hits = [hit("a"), hit("b"), hit("c")]
    assert passthrough_rerank("q", hits, 2) == hits[:2]


@pytest.mark.parametrize("rerank", [local_rerank, cohere_rerank])
def test_empty_and_zero_top_n_short_circuit(rerank):
    # Must return [] before touching the model / API client.
    assert rerank("q", [], 4) == []
    assert rerank("q", [hit("a")], 0) == []


@pytest.mark.parametrize("rerank", [local_rerank, cohere_rerank])
def test_negative_top_n_raises(rerank):
    with pytest.raises(ValueError):
        rerank("q", [hit("a")], -1)
