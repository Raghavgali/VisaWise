"""Metric layer.

Judged metrics (RAGAS 0.4, ragas.metrics.collections, judge via llm_factory):
    Faithfulness, ResponseRelevancy, ContextPrecision, ContextRecall.
    NOTE: 0.4 is a rewrite -- metrics are classes with async ascore(**kwargs)
    returning MetricResult(.value, .reason); do not port 0.1-era evaluate().

Retrieval-only metrics (deterministic, $0, run in CI):
    hit_rate, MRR, NDCG -- scored against reference_contexts/source_urls.

Every metric fn takes plain sample dicts; nothing here knows about query
engines (the runner drives the engine and feeds results in).
"""


async def score_judged(samples: list[dict], judge_model: str) -> list[dict]:
    """samples: [{user_input, response, retrieved_contexts, reference}] ->
    per-sample {metric: {value, reason}} dicts."""
    raise NotImplementedError  # TODO(pairing)


def score_retrieval(samples: list[dict]) -> list[dict]:
    """Deterministic retrieval metrics; no LLM, safe for CI."""
    raise NotImplementedError  # TODO(pairing)
