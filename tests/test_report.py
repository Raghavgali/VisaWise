"""Unit tests for evals.report -- leaderboard rendering + regression gate.

Offline, no network: everything is driven off fabricated RunRecord JSON
fixtures written straight into a monkeypatched settings.runs_dir.
"""

import json

import pytest
from typer.testing import CliRunner

from visawise.config import settings
from visawise.evals import cli, report
from visawise.evals.report import _compact_run, _metric_cell


def make_run(
    run_id: str,
    *,
    dataset_name: str,
    dataset_sha: str,
    corpus_hash: str,
    retriever: str,
    vector_weight: float,
    reranker: str,
    aggregates: dict,
    p50_ms: float,
) -> dict:
    return {
        "run_id": run_id,
        "experiment": "test_experiment",
        "git_sha": "deadbeef",
        "corpus_hash": corpus_hash,
        "dataset": {"name": dataset_name, "sha256": dataset_sha},
        "engine_config": {
            "retriever": retriever,
            "top_k": 8,
            "vector_weight": vector_weight,
            "reranker": reranker,
            "rerank_top_n": 4,
            "llm": "groq:llama-3.3-70b-versatile",
            "prompt_version": "v1",
        },
        "judge_model": "gpt-5.4-mini" if "faithfulness" in aggregates else None,
        "prompt_version": "v1",
        "samples": [],
        "aggregates": aggregates,
        "timings": {"p50_ms": p50_ms, "p95_ms": p50_ms * 1.5},
    }


# Two judged runs sharing a dataset/corpus (B clearly worse than A) plus one
# retrieval-only run on a different dataset/corpus.
RUN_A = make_run(
    "20260710T000000Z_hybrid_default",
    dataset_name="curated_v1",
    dataset_sha="sha_common",
    corpus_hash="corpus_common",
    retriever="hybrid",
    vector_weight=0.6,
    reranker="local",
    aggregates={
        "faithfulness": 0.90,
        "response_relevancy": 0.85,
        "context_precision": 0.60,
        "context_recall": 0.65,
        "hit_rate": 0.80,
        "mrr": 0.70,
        "ndcg": 0.75,
    },
    p50_ms=250.4,
)

RUN_B = make_run(
    "20260710T010000Z_hybrid_default",
    dataset_name="curated_v1",
    dataset_sha="sha_common",
    corpus_hash="corpus_common",
    retriever="hybrid",
    vector_weight=0.6,
    reranker="local",
    aggregates={
        "faithfulness": 0.70,
        "response_relevancy": 0.65,
        "context_precision": 0.50,
        "context_recall": 0.55,
        "hit_rate": 0.50,
        "mrr": 0.40,
        "ndcg": 0.45,
    },
    p50_ms=260.0,
)

RUN_C = make_run(
    "20260709T230000Z_vector_only",
    dataset_name="synthetic_v1",
    dataset_sha="sha_other",
    corpus_hash="corpus_other",
    retriever="vector",
    vector_weight=1.0,
    reranker="none",
    aggregates={
        "hit_rate": 0.60,
        "mrr": 0.50,
        "ndcg": 0.55,
    },
    p50_ms=120.0,
)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    directory = tmp_path / "runs"
    directory.mkdir()
    for run in (RUN_A, RUN_B, RUN_C):
        (directory / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")
    monkeypatch.setattr(settings, "runs_dir", directory)
    return directory


@pytest.fixture
def evals_md(tmp_path, monkeypatch):
    path = tmp_path / "docs" / "EVALS.md"
    monkeypatch.setattr(report, "EVALS_MD", path)
    return path


class TestReport:
    def test_writes_leaderboard_and_summary(self, runs_dir, evals_md, capsys):
        report.report()

        assert evals_md.exists()
        markdown = evals_md.read_text(encoding="utf-8")
        for run in (RUN_A, RUN_B, RUN_C):
            assert run["run_id"] in markdown

        # judged runs sort before the unjudged run, by faithfulness desc
        pos_a = markdown.index(RUN_A["run_id"])
        pos_b = markdown.index(RUN_B["run_id"])
        pos_c = markdown.index(RUN_C["run_id"])
        assert pos_a < pos_b < pos_c

        # RUN_C has no faithfulness/response_relevancy/context_* -- rendered as "—"
        c_line = next(line for line in markdown.splitlines() if RUN_C["run_id"] in line)
        assert c_line.count("—") >= 4

        summary_path = runs_dir / "latest_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert len(summary["runs"]) == 3
        assert "generated_at" in summary

        out = capsys.readouterr().out
        assert str(evals_md) in out
        assert str(summary_path) in out

    def test_no_runs_found(self, tmp_path, monkeypatch, evals_md, capsys):
        empty_dir = tmp_path / "empty_runs"
        empty_dir.mkdir()
        monkeypatch.setattr(settings, "runs_dir", empty_dir)

        report.report()

        assert not evals_md.exists()
        out = capsys.readouterr().out
        assert "no runs found" in out.lower()


class TestCompare:
    def test_within_threshold_passes(self, runs_dir):
        assert report.compare(RUN_A["run_id"], RUN_B["run_id"], max_drop=0.5) is True

    def test_beyond_threshold_fails(self, runs_dir):
        assert report.compare(RUN_A["run_id"], RUN_B["run_id"], max_drop=0.05) is False

    def test_missing_run_raises(self, runs_dir):
        with pytest.raises(FileNotFoundError, match="nope"):
            report.compare("nope", RUN_B["run_id"])

    def test_no_common_gated_metrics_raises(self, runs_dir):
        judged_only = dict(RUN_A, run_id="judged_only")
        judged_only["aggregates"] = {"context_precision": 0.5}
        (runs_dir / "judged_only.json").write_text(json.dumps(judged_only), encoding="utf-8")

        with pytest.raises(ValueError, match="gated"):
            report.compare("judged_only", RUN_C["run_id"])

    def test_mismatched_dataset_warns(self, runs_dir, capsys):
        report.compare(RUN_A["run_id"], RUN_C["run_id"], max_drop=0.5)
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "sha_common" in out and "sha_other" in out


class TestCli:
    def test_compare_exits_nonzero_on_regression(self, runs_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli.app, ["compare", RUN_A["run_id"], RUN_B["run_id"], "--max-drop", "0.05"]
        )
        assert result.exit_code == 1

    def test_compare_exits_zero_within_threshold(self, runs_dir):
        runner = CliRunner()
        result = runner.invoke(
            cli.app, ["compare", RUN_A["run_id"], RUN_B["run_id"], "--max-drop", "0.5"]
        )
        assert result.exit_code == 0

    def test_report_command_runs(self, runs_dir, evals_md):
        runner = CliRunner()
        result = runner.invoke(cli.app, ["report"])
        assert result.exit_code == 0
        assert evals_md.exists()


class TestMetricCell:
    def test_partial_coverage_appends_warning(self):
        compact = {
            "faithfulness": 0.866,
            "coverage": {"faithfulness": {"scored": 29, "total": 52}},
        }
        assert _metric_cell(compact, "faithfulness") == "0.866 ⚠29/52"

    def test_full_coverage_has_no_warning(self):
        compact = {
            "faithfulness": 0.864,
            "coverage": {"faithfulness": {"scored": 52, "total": 52}},
        }
        assert _metric_cell(compact, "faithfulness") == "0.864"

    def test_missing_value_renders_dash(self):
        compact = {"faithfulness": None, "coverage": {}}
        assert _metric_cell(compact, "faithfulness") == "—"


class TestCompactRunCoverage:
    def test_uses_stored_coverage_when_present(self):
        record = {
            "aggregates": {"faithfulness": 0.9},
            "coverage": {"faithfulness": {"scored": 10, "total": 10}},
            "samples": [],
        }
        compact = _compact_run(record)
        assert compact["coverage"] == {"faithfulness": {"scored": 10, "total": 10}}

    def test_derives_coverage_from_samples_for_old_records(self):
        # Pre-coverage RunRecords have no top-level "coverage" key.
        record = {
            "aggregates": {"faithfulness": 0.75},
            "samples": [
                {"scores": {"faithfulness": {"value": 0.9, "reason": None}}},
                {"scores": {"faithfulness": {"error": "boom"}}},
            ],
        }
        compact = _compact_run(record)
        assert compact["coverage"] == {"faithfulness": {"scored": 1, "total": 2}}


class TestReportPartialCoverageWarning:
    def test_markdown_shows_warning_annotation(self, runs_dir, evals_md):
        run_d = make_run(
            "20260710T020000Z_partial_cov",
            dataset_name="curated_v1",
            dataset_sha="sha_common",
            corpus_hash="corpus_common",
            retriever="hybrid",
            vector_weight=0.6,
            reranker="local",
            aggregates={"faithfulness": 0.866, "hit_rate": 0.9},
            p50_ms=200.0,
        )
        run_d["coverage"] = {"faithfulness": {"scored": 29, "total": 52}}
        (runs_dir / f"{run_d['run_id']}.json").write_text(json.dumps(run_d), encoding="utf-8")

        report.report()

        markdown = evals_md.read_text(encoding="utf-8")
        assert "⚠29/52" in markdown
