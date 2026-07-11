"""Metric layer.

Judged metrics (RAGAS 0.4, ragas.metrics.collections, judge via llm_factory):
    Faithfulness, AnswerRelevancy (the 0.4 name for "response relevancy"),
    ContextPrecision, ContextRecall.
    NOTE: 0.4 is a rewrite -- metrics are classes with async ascore(**kwargs)
    returning MetricResult(.value, .reason); do not port 0.1-era evaluate().

Retrieval-only metrics (deterministic, $0, run in CI):
    hit_rate, MRR, NDCG -- scored against reference_contexts/source_urls.

Every metric fn takes plain sample dicts; nothing here knows about query
engines (the runner drives the engine and feeds results in).
"""
import asyncio
import math


async def score_judged(samples: list[dict], judge_model: str) -> list[dict]:
    """samples: [{user_input, response, retrieved_contexts, reference}] ->
    per-sample {metric: {value, reason}} dicts (or {error} for a failed metric).

    Bounded concurrency via an asyncio.Semaphore; one bad sample/metric never
    kills the run -- it is captured and counted, and a loud per-metric error
    summary is printed at the end.
    """
    from openai import AsyncOpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    from ..config import settings

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(judge_model, client=client)
    embeddings = embedding_factory("openai", client=client)

    # (metric instance, kwarg-builder) keyed by the name we report under.
    scorers: dict[str, tuple[object, "callable"]] = {
        "faithfulness": (
            Faithfulness(llm=llm),
            lambda s: {
                "user_input": s["user_input"],
                "response": s["response"],
                "retrieved_contexts": s["retrieved_contexts"],
            },
        ),
        "response_relevancy": (
            AnswerRelevancy(llm=llm, embeddings=embeddings),
            lambda s: {"user_input": s["user_input"], "response": s["response"]},
        ),
        "context_precision": (
            ContextPrecision(llm=llm),
            lambda s: {
                "user_input": s["user_input"],
                "reference": s["reference"],
                "retrieved_contexts": s["retrieved_contexts"],
            },
        ),
        "context_recall": (
            ContextRecall(llm=llm),
            lambda s: {
                "user_input": s["user_input"],
                "retrieved_contexts": s["retrieved_contexts"],
                "reference": s["reference"],
            },
        ),
    }

    semaphore = asyncio.Semaphore(8)
    error_counts: dict[str, int] = {name: 0 for name in scorers}

    async def score_one(sample: dict) -> dict:
        result: dict = {}
        for name, (metric, build_kwargs) in scorers.items():
            async with semaphore:
                try:
                    outcome = await metric.ascore(**build_kwargs(sample))
                    result[name] = {"value": float(outcome.value), "reason": outcome.reason}
                except Exception as exc:  # noqa: BLE001 -- resilience is the point
                    error_counts[name] += 1
                    result[name] = {"error": str(exc)}
        return result

    results = await asyncio.gather(*(score_one(sample) for sample in samples))

    for name, count in error_counts.items():
        if count:
            print(
                f"[score_judged] WARNING: metric {name!r} errored on "
                f"{count}/{len(samples)} sample(s)"
            )

    return results


def score_retrieval(samples: list[dict]) -> list[dict]:
    """Deterministic retrieval metrics; no LLM, safe for CI.

    samples: [{retrieved_urls: [str] (rank-ordered, may repeat when several
    chunks share a page), source_urls: [str] (the relevant set)}].
    Per sample -> {hit_rate, mrr, ndcg}. Empty source_urls -> all None
    (unmappable; aggregation skips None -- never faked as 0 or 1).
    """
    results: list[dict] = []

    for sample in samples:
        source_urls = sample["source_urls"]
        if not source_urls:
            results.append({"hit_rate": None, "mrr": None, "ndcg": None})
            continue

        relevant = set(source_urls)

        # dedup retrieved urls, keeping first-occurrence rank
        deduped: list[str] = []
        seen: set[str] = set()
        for url in sample["retrieved_urls"]:
            if url not in seen:
                seen.add(url)
                deduped.append(url)

        hit_rate = 1.0 if any(url in relevant for url in deduped) else 0.0

        mrr = 0.0
        for rank, url in enumerate(deduped, start=1):
            if url in relevant:
                mrr = 1.0 / rank
                break

        dcg = 0.0
        for rank, url in enumerate(deduped, start=1):
            if url in relevant:
                dcg += 1.0 / math.log2(rank + 1)

        ideal_n = min(len(relevant), len(deduped))
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_n + 1))
        ndcg = (dcg / idcg) if idcg > 0 else 0.0

        results.append({"hit_rate": hit_rate, "mrr": mrr, "ndcg": ndcg})

    return results
