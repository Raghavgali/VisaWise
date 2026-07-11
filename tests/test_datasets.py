"""Dataset load/save/version -- offline (no network, no model loads)."""
import json

import pytest

from visawise.config import settings
from visawise.evals import datasets
from visawise.evals.datasets import GoldenSample, load_dataset, save_dataset


@pytest.fixture(autouse=True)
def isolated_datasets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "datasets_dir", tmp_path / "datasets")


def _sample(sample_id: str = "cur-001", origin: str = "curated") -> GoldenSample:
    return GoldenSample(
        id=sample_id,
        user_input="How long is an H-1B valid?",
        reference="Up to three years, extendable to six.",
        reference_contexts=["The initial H-1B period is up to three years."],
        source_urls=["https://www.uscis.gov/h-1b"],
        origin=origin,
    )


def test_save_load_roundtrip():
    samples = [_sample("cur-001"), _sample("cur-002")]
    path = save_dataset(samples, "curated_v1")
    assert path.exists()

    loaded = load_dataset("curated_v1")
    assert loaded == samples


def test_malformed_line_raises_with_line_number():
    path = settings.datasets_dir
    path.mkdir(parents=True, exist_ok=True)
    good = json.dumps({
        "id": "cur-001", "user_input": "q", "reference": "r",
        "reference_contexts": [], "source_urls": [], "origin": "curated",
    })
    (path / "curated_v1.jsonl").write_text(good + "\n" + "{not valid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"curated_v1\.jsonl:2"):
        load_dataset("curated_v1")


def test_missing_field_raises_with_line_number():
    path = settings.datasets_dir
    path.mkdir(parents=True, exist_ok=True)
    bad = json.dumps({"id": "cur-001", "user_input": "q", "origin": "curated"})
    (path / "curated_v1.jsonl").write_text(bad + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"curated_v1\.jsonl:1.*missing"):
        load_dataset("curated_v1")


def test_bad_origin_raises():
    path = settings.datasets_dir
    path.mkdir(parents=True, exist_ok=True)
    bad = json.dumps({
        "id": "cur-001", "user_input": "q", "reference": "r",
        "reference_contexts": [], "source_urls": [], "origin": "made-up",
    })
    (path / "curated_v1.jsonl").write_text(bad + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="origin"):
        load_dataset("curated_v1")


def test_duplicate_ids_raise():
    save_dataset([_sample("cur-001")], "curated_v1")  # write a clean file first is not needed
    # write two lines with the same id directly
    path = settings.datasets_dir / "curated_v2.jsonl"
    line = json.dumps({
        "id": "dup", "user_input": "q", "reference": "r",
        "reference_contexts": [], "source_urls": [], "origin": "curated",
    })
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate id"):
        load_dataset("curated_v2")


def test_frozen_overwrite_refused_then_forced():
    save_dataset([_sample("cur-001")], "curated_v1")

    with pytest.raises(FileExistsError, match="frozen"):
        save_dataset([_sample("cur-002")], "curated_v1")

    # force=True overwrites
    path = save_dataset([_sample("cur-999")], "curated_v1", force=True)
    assert path.exists()
    assert load_dataset("curated_v1")[0].id == "cur-999"


def test_save_rejects_duplicate_ids_in_memory():
    with pytest.raises(ValueError, match="Duplicate id"):
        save_dataset([_sample("cur-001"), _sample("cur-001")], "curated_v1")


def test_invalid_dataset_name_raises():
    with pytest.raises(ValueError, match="Dataset name"):
        datasets.dataset_path("not-a-valid-name")
