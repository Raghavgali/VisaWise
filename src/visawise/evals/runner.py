"""Experiment runner: EngineConfig x GoldenDataset -> RunRecord.

For each sample: invoke the compiled graph from rag.graph.build_graph (the
same factory the app serves), collect contexts/response/latency from the
final graph state, then score.

RunRecord (evals/runs/<utc_ts>_<experiment>.json):
    {run_id, experiment, git_sha, corpus_hash, dataset: {name, sha256},
     engine_config: {...}, judge_model, prompt_version,
     samples: [{id, response, retrieved_chunk_ids, scores}],
     aggregates: {metric: mean}, coverage: {metric: {scored, total}},
     timings: {p50_ms, p95_ms}, cost: {...}}

Experiment YAML (configs/experiments/*.yaml):
    name, dataset, engine: {retriever, top_k, vector_weight, reranker, ...},
    metrics: [judged|retrieval|all]
    sweep:  optional {param: [values]} -- expands to one run per value.
"""
import asyncio
import dataclasses
import hashlib
import json
import math
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..config import REPO_ROOT, settings
from ..ingestion.chunk import corpus_hash, load_chunks
from ..rag.graph import EngineConfig, build_graph
from .datasets import dataset_path, load_dataset
from .metrics import coverage_from_scores, score_judged, score_retrieval

_MIN_INVOKE_INTERVAL_S = 2.5  # Groq free tier: 30 req/min
_METRIC_MODES = {"judged", "retrieval", "all"}
# Judged metrics are gated on coverage (settings.judged_coverage_threshold):
# judge failures skew toward long, hard answers, so a partial average is a
# biased average. Retrieval metrics are exempt -- curated honesty probes have
# empty source_urls by design and score None on purpose.
_COVERAGE_GATED_METRICS = {
    "faithfulness",
    "response_relevancy",
    "context_precision",
    "context_recall",
}


def expand_experiment(spec: dict) -> list[tuple[str, dict]]:
    """(pure, unit-testable) -> [(run_name, engine_overrides)].

    No sweep -> one run, empty overrides. A sweep {param: [values]} -> one run
    per value, name suffixed e.g. 'weight_sweep@vector_weight=0.4'.
    """
    name = spec["name"]
    sweep = spec.get("sweep")
    if not sweep:
        return [(name, {})]
    if len(sweep) != 1:
        raise ValueError(f"sweep must have exactly one parameter, got {sorted(sweep)}")

    (param, values), = sweep.items()
    if not isinstance(values, list) or not values:
        raise ValueError(f"sweep values for {param!r} must be a non-empty list")

    # run_name becomes a filename: slug sweep values ("nvidia:meta/llama-..."
    # would otherwise put a path separator in the RunRecord filename).
    return [(f"{name}@{param}={_slug(value)}", {param: value}) for value in values]


def _slug(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "-", str(value))


def _build_engine_config(block: dict) -> EngineConfig:
    allowed = {field.name for field in dataclasses.fields(EngineConfig)}
    unknown = set(block) - allowed
    if unknown:
        raise ValueError(f"Unknown engine config key(s): {sorted(unknown)}")
    return EngineConfig(**block)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[int(position)]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _numeric_score(value: object) -> float | None:
    """Pull a usable float out of a per-sample score entry, or None to skip
    (None sentinels, judged-metric errors, and NaN are all skipped)."""
    if value is None:
        return None
    if isinstance(value, dict):
        if "value" not in value:  # error entry
            return None
        value = value["value"]
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return None if math.isnan(number) else number
    return None


def _aggregate(per_sample_scores: list[dict]) -> dict[str, float]:
    metric_names: set[str] = set()
    for scores in per_sample_scores:
        metric_names.update(scores)

    aggregates: dict[str, float] = {}
    for name in sorted(metric_names):
        collected = [
            number
            for scores in per_sample_scores
            if (number := _numeric_score(scores.get(name))) is not None
        ]
        if collected:
            aggregates[name] = sum(collected) / len(collected)
    return aggregates


def _write_record(record: dict) -> None:
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    path = settings.runs_dir / f"{record['run_id']}.json"
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def run_experiment(experiment_path: str) -> list[str]:
    """Execute an experiment file (1 run, or N for sweeps); returns run_ids."""
    spec = yaml.safe_load(Path(experiment_path).read_text(encoding="utf-8"))

    experiment = spec["name"]
    dataset_name = spec["dataset"]
    metrics_mode = spec.get("metrics", "retrieval")
    if metrics_mode not in _METRIC_MODES:
        raise ValueError(
            f"metrics must be one of {sorted(_METRIC_MODES)}, got {metrics_mode!r}"
        )
    run_judged = metrics_mode in ("judged", "all")
    engine_block = dict(spec.get("engine", {}))

    samples = load_dataset(dataset_name)
    limit = spec.get("limit")
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"limit must be a positive int, got {limit!r}")
        samples = samples[:limit]
    dataset_sha = hashlib.sha256(dataset_path(dataset_name).read_bytes()).hexdigest()
    corpus = corpus_hash(load_chunks())
    git_sha = _git_sha()

    run_ids: list[str] = []
    coverage_failures: list[str] = []
    for run_name, overrides in expand_experiment(spec):
        config = _build_engine_config({**engine_block, **overrides})
        graph = build_graph(config)

        raw_samples: list[dict] = []
        latencies: list[float] = []
        last_invoke = 0.0
        aborted: dict | None = None

        for sample in samples:
            wait = _MIN_INVOKE_INTERVAL_S - (time.monotonic() - last_invoke)
            if wait > 0:
                time.sleep(wait)

            last_invoke = time.monotonic()
            start = time.perf_counter()
            try:
                state = graph.invoke({"query": sample.user_input})
            except Exception as exc:  # noqa: BLE001 -- salvage partial run
                aborted = {"at_sample": sample.id, "error": f"{type(exc).__name__}: {exc}"}
                print(
                    f"[{run_name}] ABORTED at {sample.id} after "
                    f"{len(raw_samples)}/{len(samples)} samples: {aborted['error']}"
                )
                break
            latency_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(latency_ms)

            final_hits = state.get("reranked") if "reranked" in state else state.get("fused", [])
            final_hits = final_hits or []

            raw_samples.append(
                {
                    "id": sample.id,
                    "user_input": sample.user_input,
                    "reference": sample.reference,
                    "source_urls": sample.source_urls,
                    "response": state.get("answer", ""),
                    "retrieved_chunk_ids": [hit.chunk_id for hit in final_hits],
                    "retrieved_contexts": [hit.text for hit in final_hits],
                    "retrieved_urls": [hit.url for hit in final_hits],
                }
            )
            print(f"[{run_name}] {sample.id}: {latency_ms:.0f}ms")

        if not raw_samples:
            raise RuntimeError(
                f"[{run_name}] no samples completed"
                + (f" ({aborted['error']})" if aborted else "")
            )

        retrieval_scores = score_retrieval(
            [
                {"retrieved_urls": item["retrieved_urls"], "source_urls": item["source_urls"]}
                for item in raw_samples
            ]
        )

        judged_scores: list[dict] = [{} for _ in raw_samples]
        if run_judged:
            judged_scores = asyncio.run(
                score_judged(
                    [
                        {
                            "user_input": item["user_input"],
                            "response": item["response"],
                            "retrieved_contexts": item["retrieved_contexts"],
                            "reference": item["reference"],
                        }
                        for item in raw_samples
                    ],
                    settings.judge_model,
                )
            )

        per_sample_scores = [
            {**retrieval_scores[i], **judged_scores[i]} for i in range(len(raw_samples))
        ]
        coverage = coverage_from_scores(per_sample_scores)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}_{run_name}"

        record = {
            "run_id": run_id,
            "experiment": experiment,
            "git_sha": git_sha,
            "corpus_hash": corpus,
            "dataset": {
                "name": dataset_name,
                "sha256": dataset_sha,
                "limit": limit,
                "n_samples": len(raw_samples),
            },
            "aborted": aborted,
            "engine_config": config.to_dict(),
            "judge_model": settings.judge_model if run_judged else None,
            "prompt_version": config.prompt_version,
            "samples": [
                {
                    "id": item["id"],
                    "response": item["response"],
                    "retrieved_chunk_ids": item["retrieved_chunk_ids"],
                    "scores": per_sample_scores[i],
                }
                for i, item in enumerate(raw_samples)
            ],
            "aggregates": _aggregate(per_sample_scores),
            "coverage": coverage,
            "timings": {
                "p50_ms": _percentile(latencies, 0.50),
                "p95_ms": _percentile(latencies, 0.95),
            },
        }

        _write_record(record)
        run_ids.append(run_id)

        for metric, cov in coverage.items():
            if metric not in _COVERAGE_GATED_METRICS or not cov["total"]:
                continue
            if cov["scored"] / cov["total"] < settings.judged_coverage_threshold:
                coverage_failures.append(
                    f"[{run_name}] {metric}: {cov['scored']}/{cov['total']} scored "
                    f"(< {settings.judged_coverage_threshold:.0%})"
                )

    if coverage_failures:
        # Records are already written (nothing is lost), but the run fails:
        # partial-coverage aggregates must never pass silently as results.
        raise RuntimeError(
            "judged-metric coverage below threshold:\n  " + "\n  ".join(coverage_failures)
        )

    return run_ids
