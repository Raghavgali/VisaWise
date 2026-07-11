"""Unit tests for build_graph -- config validation + compiled graph shape.

Nodes never execute here (no DB, no Groq call): compiling a LangGraph only
wires nodes and edges, so shape is testable with a fake API key.
"""

import pytest

from visawise.config import settings
from visawise.rag.graph import EngineConfig, build_graph


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "gsk_fake_key_for_tests")
    monkeypatch.setattr(settings, "nvidia_api_key", "nvapi-fake_key_for_tests")


def node_names(compiled) -> set[str]:
    return set(compiled.get_graph().nodes) - {"__start__", "__end__"}


def test_unsupported_prompt_version_raises():
    with pytest.raises(ValueError, match="prompt version"):
        build_graph(EngineConfig(prompt_version="v999"))


@pytest.mark.parametrize("llm", ["openai:gpt-4o", "groq:", "llama-3.3-70b-versatile"])
def test_malformed_llm_raises(llm):
    with pytest.raises(ValueError, match="groq:"):
        build_graph(EngineConfig(llm=llm))


def test_unknown_retriever_raises():
    with pytest.raises(ValueError, match="retriever"):
        build_graph(EngineConfig(retriever="dense"))


def test_missing_groq_key_raises(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", None)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        build_graph(EngineConfig(llm="groq:llama-3.3-70b-versatile"))


def test_missing_nvidia_key_raises(monkeypatch):
    monkeypatch.setattr(settings, "nvidia_api_key", None)
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        build_graph(EngineConfig(llm="nvidia:meta/llama-3.1-8b-instruct"))


def test_hybrid_graph_shape():
    compiled = build_graph(EngineConfig(retriever="hybrid", reranker="local"))
    assert node_names(compiled) == {
        "dense_retrieve", "bm25_retrieve", "fuse", "rerank", "generate",
    }


def test_vector_only_graph_skips_bm25_and_fuse():
    compiled = build_graph(EngineConfig(retriever="vector", reranker="local"))
    assert node_names(compiled) == {"dense_retrieve", "rerank", "generate"}


def test_bm25_only_graph_skips_dense_and_fuse():
    compiled = build_graph(EngineConfig(retriever="bm25", reranker="local"))
    assert node_names(compiled) == {"bm25_retrieve", "rerank", "generate"}


def test_no_reranker_skips_rerank_node():
    compiled = build_graph(EngineConfig(retriever="hybrid", reranker="none"))
    assert node_names(compiled) == {
        "dense_retrieve", "bm25_retrieve", "fuse", "generate",
    }
