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

    def test_mismatched_dataset_fails_closed(self, runs_dir, capsys):
        assert report.compare(RUN_A["run_id"], RUN_C["run_id"], max_drop=0.5) is False
        out = capsys.readouterr().out
        assert "GATE FAIL" in out
        assert "sha_common" in out and "sha_other" in out

    def test_aborted_run_fails_closed(self, runs_dir):
        aborted = dict(RUN_B, run_id="aborted_run", aborted="rate limit at sample 12")
        (runs_dir / "aborted_run.json").write_text(json.dumps(aborted), encoding="utf-8")

        assert report.compare(RUN_A["run_id"], "aborted_run", max_drop=0.5) is False

    def test_partial_coverage_fails_closed(self, runs_dir):
        partial = dict(RUN_B, run_id="partial_run")
        partial["coverage"] = {"faithfulness": {"scored": 29, "total": 52}}
        (runs_dir / "partial_run.json").write_text(json.dumps(partial), encoding="utf-8")

        assert report.compare(RUN_A["run_id"], "partial_run", max_drop=0.5) is False

    def test_identical_partial_subsets_are_comparable(self, runs_dir, capsys):
        # Structural partiality (e.g. retrieval metrics undefined on
        # citation-less safety samples) excludes the SAME samples in both
        # runs — that must stay comparable, not fail the gate.
        samples = [
            {"id": "s1", "scores": {"hit_rate": 1.0}},
            {"id": "s2", "scores": {"hit_rate": None}},
        ]
        cov = {"hit_rate": {"scored": 1, "total": 2}}
        left = dict(RUN_A, run_id="subset_left", samples=samples, coverage=cov)
        right = dict(RUN_A, run_id="subset_right", samples=samples, coverage=cov)
        for r in (left, right):
            (runs_dir / f"{r['run_id']}.json").write_text(json.dumps(r), encoding="utf-8")

        assert report.compare("subset_left", "subset_right", max_drop=0.5) is True
        assert "same 1-sample subset" in capsys.readouterr().out

    def test_abstention_regression_fails(self, runs_dir):
        base = dict(RUN_A, run_id="abst_base")
        base["aggregates"] = dict(RUN_A["aggregates"], abstention=14 / 15)
        cand = dict(RUN_A, run_id="abst_cand")
        cand["aggregates"] = dict(RUN_A["aggregates"], abstention=11 / 15)
        for r in (base, cand):
            (runs_dir / f"{r['run_id']}.json").write_text(json.dumps(r), encoding="utf-8")

        assert report.compare("abst_base", "abst_cand", max_drop=0.05) is False


def _canonical_run(run_id="20260717T000000Z_serving", *, abstention=14 / 15, ans_faith=0.92, **extra):
    run = make_run(
        run_id,
        dataset_name="curated_v2",
        dataset_sha="sha_curated_v2",
        corpus_hash="corpus_common",
        retriever="hybrid",
        vector_weight=0.6,
        reranker="local",
        aggregates={"faithfulness": 0.94, "abstention": abstention, "hit_rate": 1.0, "mrr": 0.77},
        p50_ms=2500.0,
    )
    run["slices"] = {
        "answerable_grounded": {"n": 5, "aggregates": {"faithfulness": ans_faith}, "coverage": {}}
    }
    run.update(extra)
    return run


class TestGate:
    def test_passes_on_healthy_canonical_run(self, runs_dir, capsys):
        run = _canonical_run()
        (runs_dir / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")

        assert report.gate() is True
        assert "GATE PASS" in capsys.readouterr().out

    def test_fails_below_safety_floor(self, runs_dir, capsys):
        run = _canonical_run(abstention=13 / 15)  # 0.867 < 0.93 floor
        (runs_dir / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")

        assert report.gate() is False
        assert "safety abstention" in capsys.readouterr().out

    def test_fails_below_faithfulness_floor(self, runs_dir):
        run = _canonical_run(ans_faith=0.80)
        (runs_dir / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")

        assert report.gate() is False

    def test_fails_on_aborted_canonical_run(self, runs_dir):
        run = _canonical_run(aborted="quota exhausted")
        (runs_dir / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")

        assert report.gate() is False

    def test_fails_when_no_canonical_runs(self, runs_dir):
        # fixture dir has only curated_v1/synthetic_v1 runs
        assert report.gate() is False

    def test_picks_newest_canonical_run(self, runs_dir):
        old_bad = _canonical_run("20260716T000000Z_serving", abstention=10 / 15)
        new_good = _canonical_run("20260718T000000Z_serving")
        for r in (old_bad, new_good):
            (runs_dir / f"{r['run_id']}.json").write_text(json.dumps(r), encoding="utf-8")

        assert report.gate() is True

    def test_cli_gate_exits_nonzero_on_failure(self, runs_dir):
        run = _canonical_run(abstention=13 / 15)
        (runs_dir / f"{run['run_id']}.json").write_text(json.dumps(run), encoding="utf-8")

        result = CliRunner().invoke(cli.app, ["gate"])
        assert result.exit_code == 1


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
