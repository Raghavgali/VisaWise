"""Retrieval-metric math -- deterministic, offline (no LLM, no judged scoring)."""
import math

import pytest

from visawise.evals.metrics import coverage_from_scores, score_retrieval


def test_perfect_hit_at_rank_1():
    (result,) = score_retrieval([{"retrieved_urls": ["a", "b"], "source_urls": ["a"]}])
    assert result["hit_rate"] == 1.0
    assert result["mrr"] == 1.0
    assert result["ndcg"] == 1.0


def test_complete_miss():
    (result,) = score_retrieval([{"retrieved_urls": ["x", "y"], "source_urls": ["a"]}])
    assert result == {"hit_rate": 0.0, "mrr": 0.0, "ndcg": 0.0}


def test_first_relevant_at_rank_2_gives_mrr_half():
    (result,) = score_retrieval([{"retrieved_urls": ["x", "a"], "source_urls": ["a"]}])
    assert result["hit_rate"] == 1.0
    assert result["mrr"] == 0.5
    # dcg = 1/log2(3); idcg = 1/log2(2) = 1
    assert result["ndcg"] == pytest.approx(1.0 / math.log2(3))


def test_ndcg_two_relevant_hand_computed():
    # retrieved [x, a, b], relevant {a, b}
    (result,) = score_retrieval([{"retrieved_urls": ["x", "a", "b"], "source_urls": ["a", "b"]}])
    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert result["ndcg"] == pytest.approx(dcg / idcg)
    assert result["mrr"] == 0.5


def test_duplicate_url_dedup_keeps_first_rank():
    # 'a' repeated then 'b' relevant: dedup -> [a, b], so b sits at rank 2
    (result,) = score_retrieval([{"retrieved_urls": ["a", "a", "b"], "source_urls": ["b"]}])
    assert result["mrr"] == 0.5
    assert result["ndcg"] == pytest.approx(1.0 / math.log2(3))


def test_empty_source_urls_returns_none():
    (result,) = score_retrieval([{"retrieved_urls": ["a", "b"], "source_urls": []}])
    assert result == {"hit_rate": None, "mrr": None, "ndcg": None}


def test_coverage_counts_plain_floats_and_value_dicts():
    per_sample = [
        {"hit_rate": 1.0, "faithfulness": {"value": 0.8, "reason": None}},
        {"hit_rate": 0.0, "faithfulness": {"value": 0.6, "reason": None}},
    ]
    coverage = coverage_from_scores(per_sample)
    assert coverage == {
        "hit_rate": {"scored": 2, "total": 2},
        "faithfulness": {"scored": 2, "total": 2},
    }


def test_coverage_excludes_none_error_bool_and_nan():
    per_sample = [
        {"faithfulness": {"value": 0.8, "reason": None}},
        {"faithfulness": None},
        {"faithfulness": {"error": "boom"}},
        {"faithfulness": True},
        {"faithfulness": float("nan")},
    ]
    coverage = coverage_from_scores(per_sample)
    assert coverage == {"faithfulness": {"scored": 1, "total": 5}}


def test_coverage_from_empty_input():
    assert coverage_from_scores([]) == {}


def test_coverage_total_counts_only_attempted_samples():
    """A metric present on a subset (slice-specific) is gated against that
    subset, not the whole run: faithfulness on the 2 answer samples, abstention
    on the 2 safety samples. hit_rate keeps None sentinels so its key is present
    on all four (total 4)."""
    per_sample = [
        {"hit_rate": 1.0, "faithfulness": {"value": 0.8, "reason": None}},
        {"hit_rate": 0.0, "faithfulness": {"value": 0.6, "reason": None}},
        {"hit_rate": None, "abstention": {"value": 1.0, "reason": None}},
        {"hit_rate": None, "abstention": {"value": 0.0, "reason": None}},
    ]
    coverage = coverage_from_scores(per_sample)
    assert coverage["faithfulness"] == {"scored": 2, "total": 2}
    assert coverage["abstention"] == {"scored": 2, "total": 2}
    assert coverage["hit_rate"] == {"scored": 2, "total": 4}


def test_slice_mode_classifies_safety_slices_as_abstain():
    from visawise.evals.metrics import slice_mode

    assert slice_mode(None) == "answer"
    assert slice_mode("answerable_grounded") == "answer"
    assert slice_mode("temporal_current") == "answer"
    assert slice_mode("stale_source") == "answer"
    assert slice_mode("out_of_corpus") == "abstain"
    assert slice_mode("adversarial_injection") == "abstain"
    assert slice_mode("high_risk_abstain") == "abstain"
