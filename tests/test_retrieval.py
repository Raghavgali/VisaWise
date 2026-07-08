"""Unit tests for fuse_rrf -- pure function, no DB, no model."""

import pytest

from visawise.rag.retrieval import RRF_K, ScoredChunk, fuse_rrf


def hit(chunk_id: str, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, text=f"text-{chunk_id}", url="u", title="T",
        topic="x", section="", score=score,
    )


def ids(hits: list[ScoredChunk]) -> list[str]:
    return [h.chunk_id for h in hits]


def test_returns_distinct_ranked_chunks():
    # Regression: the stale-loop-variable bug returned N copies of one chunk.
    fused = fuse_rrf([[hit("a"), hit("b")], [hit("c"), hit("d")]], [0.5, 0.5], 4)
    assert sorted(ids(fused)) == ["a", "b", "c", "d"]
    assert len(set(ids(fused))) == 4


def test_full_weight_preserves_that_lists_order():
    fused = fuse_rrf([[hit("a"), hit("b"), hit("c")], [hit("c"), hit("a")]], [1.0, 0.0], 3)
    assert ids(fused) == ["a", "b", "c"]


def test_overlap_outranks_single_list_presence():
    # "b" appears in both lists; with equal weights it must beat "a" and "c".
    fused = fuse_rrf([[hit("a"), hit("b")], [hit("b"), hit("c")]], [0.5, 0.5], 3)
    assert ids(fused)[0] == "b"


def test_scores_match_formula():
    fused = fuse_rrf([[hit("a")], [hit("a")]], [0.6, 0.4], 1)
    expected = 0.6 / (RRF_K + 1) + 0.4 / (RRF_K + 1)
    assert fused[0].score == pytest.approx(expected)


def test_weight_shifts_ranking():
    dense = [hit("d1"), hit("shared")]
    bm25 = [hit("b1"), hit("shared")]
    heavy_dense = ids(fuse_rrf([dense, bm25], [0.9, 0.1], 3))
    heavy_bm25 = ids(fuse_rrf([dense, bm25], [0.1, 0.9], 3))
    assert heavy_dense.index("d1") < heavy_dense.index("b1")
    assert heavy_bm25.index("b1") < heavy_bm25.index("d1")


def test_top_k_truncates():
    fused = fuse_rrf([[hit("a"), hit("b"), hit("c")]], [1.0], 2)
    assert len(fused) == 2


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        fuse_rrf([[hit("a")]], [0.5, 0.5], 1)
