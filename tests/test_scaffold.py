"""Scaffold sanity: config loads, contracts import, sources file is well-formed.

Real behavior tests land next to each module as we implement it
(test_fetch.py, test_extract.py, test_chunk.py, ...).
"""

import yaml

from visawise.config import settings


def test_settings_load_validated_defaults():
    assert settings.chunk_size == 1024
    assert settings.chunk_overlap == 50
    assert settings.embed_dim == 768
    assert settings.hybrid_vector_weight == 0.6


def test_sources_yaml_well_formed():
    data = yaml.safe_load(settings.sources_file.read_text())
    sources = data["sources"]
    assert len(sources) >= 20
    urls = [s["url"] for s in sources]
    assert len(urls) == len(set(urls)), "duplicate source URLs"
    for s in sources:
        assert s["url"].startswith("https://www.uscis.gov"), s["url"]
        assert s["topic"] and s["title"]


def test_engine_config_serializes():
    from visawise.rag.graph import EngineConfig

    cfg = EngineConfig()
    d = cfg.to_dict()
    assert d["retriever"] == "hybrid"
    assert d["vector_weight"] == 0.6
    assert d["prompt_version"] == "v1"
