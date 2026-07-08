"""Experiment runner: EngineConfig x GoldenDataset -> RunRecord.

For each sample: invoke the compiled graph from rag.graph.build_graph (the
same factory the app serves), collect contexts/response/latency from the
final graph state, then score.

RunRecord (evals/runs/<utc_ts>_<experiment>.json):
    {run_id, experiment, git_sha, corpus_hash, dataset: {name, sha256},
     engine_config: {...}, judge_model, prompt_version,
     samples: [{id, response, retrieved_chunk_ids, scores}],
     aggregates: {metric: mean}, timings: {p50_ms, p95_ms}, cost: {...}}

Experiment YAML (configs/experiments/*.yaml):
    name, dataset, engine: {retriever, top_k, vector_weight, reranker, ...},
    metrics: [judged|retrieval|all]
    sweep:  optional {param: [values]} -- expands to one run per value.
"""


def run_experiment(experiment_path: str) -> list[str]:
    """Execute an experiment file (1 run, or N for sweeps); returns run_ids."""
    raise NotImplementedError  # TODO(pairing)
