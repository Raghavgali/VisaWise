"""Runner pure helpers -- sweep expansion + engine-config validation (offline).

Also covers run_experiment's coverage gate end-to-end, with build_graph and
score_judged monkeypatched out (no LLM/network) and a fake compiled graph
standing in for rag.graph.build_graph.
"""
import json

import pytest

from visawise.config import settings
from visawise.evals import runner as runner_module
from visawise.evals.datasets import GoldenSample
from visawise.evals.runner import (
    _aggregate,
    _build_engine_config,
    expand_experiment,
    run_experiment,
)


def test_expand_no_sweep_single_run():
    runs = expand_experiment({"name": "hybrid_default", "engine": {"retriever": "hybrid"}})
    assert runs == [("hybrid_default", {})]


def test_expand_weight_sweep_shape():
    spec = {
        "name": "weight_sweep",
        "engine": {"retriever": "hybrid", "top_k": 8},
        "sweep": {"vector_weight": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]},
    }
    runs = expand_experiment(spec)
    assert [name for name, _ in runs] == [
        "weight_sweep@vector_weight=0.0",
        "weight_sweep@vector_weight=0.2",
        "weight_sweep@vector_weight=0.4",
        "weight_sweep@vector_weight=0.6",
        "weight_sweep@vector_weight=0.8",
        "weight_sweep@vector_weight=1.0",
    ]
    assert runs[2][1] == {"vector_weight": 0.4}


def test_expand_multi_param_sweep_rejected():
    with pytest.raises(ValueError, match="exactly one parameter"):
        expand_experiment({"name": "x", "sweep": {"a": [1], "b": [2]}})


def test_build_engine_config_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown engine config key"):
        _build_engine_config({"retriever": "hybrid", "bogus": 1})


def test_build_engine_config_valid():
    config = _build_engine_config({"retriever": "vector", "reranker": "none", "top_k": 4})
    assert config.retriever == "vector"
    assert config.reranker == "none"
    assert config.top_k == 4


def test_aggregate_skips_none_and_errors_and_nan():
    per_sample = [
        {"hit_rate": 1.0, "faithfulness": {"value": 0.8, "reason": None}},
        {"hit_rate": None, "faithfulness": {"error": "boom"}},
        {"hit_rate": 0.0, "faithfulness": {"value": float("nan"), "reason": None}},
    ]
    agg = _aggregate(per_sample)
    assert agg["hit_rate"] == pytest.approx(0.5)  # mean of [1.0, 0.0]
    assert agg["faithfulness"] == pytest.approx(0.8)  # only the one clean value


def test_sweep_values_with_path_chars_are_slugged():
    # "nvidia:meta/llama-..." must not put a path separator in the run name.
    spec = {
        "name": "model_ablation",
        "sweep": {"llm": ["nvidia:meta/llama-3.1-8b-instruct"]},
    }
    (run_name, overrides), = expand_experiment(spec)
    assert "/" not in run_name and ":" not in run_name
    assert overrides == {"llm": "nvidia:meta/llama-3.1-8b-instruct"}


# --- run_experiment coverage gate: fake graph + fake dataset, offline -------


class _FakeHit:
    def __init__(self, chunk_id: str, text: str, url: str):
        self.chunk_id = chunk_id
        self.text = text
        self.url = url


class _FakeGraph:
    def invoke(self, state: dict) -> dict:
        return {
            "answer": f"answer for {state['query']}",
            "reranked": [_FakeHit("chunk-1", "some context", "https://example.com/a")],
        }


def _fake_samples() -> list[GoldenSample]:
    return [
        GoldenSample(
            id=f"s{i}",
            user_input=f"question {i}",
            reference="reference text",
            reference_contexts=["ctx"],
            source_urls=["https://example.com/a"],
            origin="curated",
        )
        for i in range(3)
    ]


async def _score_judged_partial_failure(samples: list[dict], judge_model: str) -> list[dict]:
    # 1/3 samples errors on faithfulness -> 2/3 = 0.667, below the 0.95 gate.
    return [
        {"faithfulness": {"error": "boom"}} if i == 1 else {"faithfulness": {"value": 0.9, "reason": None}}
        for i in range(len(samples))
    ]


async def _score_judged_all_pass(samples: list[dict], judge_model: str) -> list[dict]:
    return [{"faithfulness": {"value": 0.9, "reason": None}} for _ in samples]


def _write_experiment(tmp_path, metrics: str = "judged"):
    path = tmp_path / "experiment.yaml"
    path.write_text(f"name: judged_exp\ndataset: curated_v1\nmetrics: {metrics}\n", encoding="utf-8")
    return path


@pytest.fixture
def wired_runner(tmp_path, monkeypatch):
    """Wire run_experiment to a fake compiled graph + fake dataset, fully
    offline: no LLM, no retriever, no rate-limit sleeps."""
    dataset_file = tmp_path / "curated_v1.jsonl"
    dataset_file.write_text('{"fake": "dataset bytes, only sha256 of this matters"}\n', encoding="utf-8")

    monkeypatch.setattr(runner_module, "build_graph", lambda config: _FakeGraph())
    monkeypatch.setattr(runner_module, "load_dataset", lambda name: _fake_samples())
    monkeypatch.setattr(runner_module, "dataset_path", lambda name: dataset_file)
    monkeypatch.setattr(runner_module, "load_chunks", lambda: [])
    monkeypatch.setattr(runner_module, "corpus_hash", lambda chunks: "fakecorpus")
    monkeypatch.setattr(runner_module, "_MIN_INVOKE_INTERVAL_S", 0.0)
    monkeypatch.setattr(settings, "runs_dir", tmp_path / "runs")
    return tmp_path


def test_run_experiment_raises_on_low_judged_coverage_but_still_writes_record(
    wired_runner, monkeypatch
):
    monkeypatch.setattr(runner_module, "score_judged", _score_judged_partial_failure)
    experiment_path = _write_experiment(wired_runner)

    with pytest.raises(RuntimeError, match="faithfulness"):
        run_experiment(str(experiment_path))

    records = list(settings.runs_dir.glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["coverage"]["faithfulness"] == {"scored": 2, "total": 3}


def test_run_experiment_full_coverage_passes_and_records_coverage(wired_runner, monkeypatch):
    monkeypatch.setattr(runner_module, "score_judged", _score_judged_all_pass)
    experiment_path = _write_experiment(wired_runner)

    run_ids = run_experiment(str(experiment_path))

    assert len(run_ids) == 1
    records = list(settings.runs_dir.glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["run_id"] == run_ids[0]
    assert record["coverage"]["faithfulness"] == {"scored": 3, "total": 3}
