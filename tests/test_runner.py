"""Runner pure helpers -- sweep expansion + engine-config validation (offline)."""
import pytest

from visawise.evals.runner import _aggregate, _build_engine_config, expand_experiment


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
