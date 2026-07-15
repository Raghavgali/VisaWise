"""Reporting: runs/ -> leaderboard markdown + dashboard JSON + regression gate.

- report(): renders a cross-run leaderboard (docs/EVALS.md + README section)
  and evals/runs/latest_summary.json (served by /api/eval/runs for the
  dashboard tab).
- compare(baseline_run_id, candidate_run_id, thresholds): exits nonzero when
  faithfulness or response relevancy regress beyond threshold -- the CI gate.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import REPO_ROOT, settings
from .metrics import coverage_from_scores

EVALS_MD = REPO_ROOT / "docs" / "EVALS.md"

# Metrics rendered in the leaderboard / dashboard JSON, in column order.
_METRIC_KEYS = [
    "faithfulness",
    "response_relevancy",
    "context_precision",
    "context_recall",
    "hit_rate",
    "mrr",
    "ndcg",
]

# Metrics the regression gate is allowed to compare (whichever of these both
# runs actually have).
_GATED_METRICS = ["faithfulness", "response_relevancy", "hit_rate", "mrr"]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def _load_runs(runs_dir: Path) -> list[dict]:
    """Every *.json in runs_dir except the latest_summary.json output file.
    Non-recursive, so subdirectories (e.g. scratch/) are ignored for free."""
    if not runs_dir.exists():
        return []

    records = []
    for path in sorted(runs_dir.glob("*.json")):
        if path.name == "latest_summary.json":
            continue
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _compact_run(record: dict) -> dict:
    engine = record.get("engine_config") or {}
    aggregates = record.get("aggregates") or {}
    timings = record.get("timings") or {}

    # Coverage: stored by the runner since the max_tokens fix; derived from
    # per-sample scores for records that predate it.
    coverage = record.get("coverage")
    if coverage is None:
        coverage = coverage_from_scores(
            [(s.get("scores") or {}) for s in (record.get("samples") or [])]
        )

    compact = {
        "run_id": record.get("run_id"),
        "experiment": record.get("experiment"),
        "dataset": (record.get("dataset") or {}).get("name"),
        "n_samples": (record.get("dataset") or {}).get("n_samples"),
        "aborted": bool(record.get("aborted")),
        "retriever": engine.get("retriever"),
        "vector_weight": engine.get("vector_weight"),
        "reranker": engine.get("reranker"),
        "rerank_top_n": engine.get("rerank_top_n"),
        "top_k": engine.get("top_k"),
        "llm": engine.get("llm"),
        "p50_ms": timings.get("p50_ms"),
        "coverage": coverage,
    }
    for key in _METRIC_KEYS:
        compact[key] = aggregates.get(key)
    return compact


def _metric_cell(compact: dict, key: str, decimals: int = 3) -> str:
    """Aggregate + coverage: '0.864' at full coverage, '0.866 ⚠29/52' when the
    metric only scored a subset (partial averages must never look complete)."""
    value = compact.get(key)
    if value is None:
        return "—"
    cell = f"{value:.{decimals}f}"
    cov = (compact.get("coverage") or {}).get(key)
    if cov and cov["scored"] < cov["total"]:
        cell += f" ⚠{cov['scored']}/{cov['total']}"
    return cell


def _sort_key(compact: dict) -> tuple[int, int, float, float]:
    """Clean runs before aborted fragments; then judged runs (have
    faithfulness) sorted by faithfulness desc; unjudged after, by hit_rate
    desc. Missing metrics sort last within their group."""
    has_faith = compact["faithfulness"] is not None
    faith = compact["faithfulness"] if has_faith else 0.0
    hit_rate = compact["hit_rate"] if compact["hit_rate"] is not None else float("-inf")
    return (1 if compact.get("aborted") else 0, 0 if has_faith else 1, -faith, -hit_rate)


def _fmt(value, decimals: int) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _render_markdown(compacts: list[dict], generated_at: str) -> str:
    lines = [
        "# VisaWise Eval Leaderboard",
        "",
        f"Generated {generated_at} -- {len(compacts)} run(s).",
        "",
        "| run_id | retriever | w_vec | reranker | dataset | n | faith | relevancy | "
        "ctx_prec | ctx_recall | hit_rate | mrr | ndcg | p50_ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in compacts:
        n = c.get("n_samples")
        n_cell = str(n) if n is not None else "—"
        if c.get("aborted"):
            n_cell += " ⚠ aborted"
        lines.append(
            "| {run_id} | {retriever} | {w_vec} | {reranker} | {dataset} | {n} | {faith} | "
            "{relevancy} | {ctx_prec} | {ctx_recall} | {hit_rate} | {mrr} | {ndcg} | {p50_ms} |".format(
                run_id=c["run_id"] or "—",
                n=n_cell,
                retriever=c["retriever"] or "—",
                w_vec=_fmt(c["vector_weight"], 3),
                reranker=c["reranker"] or "—",
                dataset=c["dataset"] or "—",
                faith=_metric_cell(c, "faithfulness"),
                relevancy=_metric_cell(c, "response_relevancy"),
                ctx_prec=_metric_cell(c, "context_precision"),
                ctx_recall=_metric_cell(c, "context_recall"),
                hit_rate=_metric_cell(c, "hit_rate"),
                mrr=_metric_cell(c, "mrr"),
                ndcg=_metric_cell(c, "ndcg"),
                p50_ms=_fmt(c["p50_ms"], 0),
            )
        )
    return "\n".join(lines) + "\n"


def report() -> None:
    runs_dir = settings.runs_dir
    records = _load_runs(runs_dir)
    if not records:
        print(f"No runs found in {runs_dir}; nothing to report.")
        return

    compacts = sorted((_compact_run(r) for r in records), key=_sort_key)
    generated_at = datetime.now(timezone.utc).isoformat()

    markdown = _render_markdown(compacts, generated_at)
    _atomic_write_text(EVALS_MD, markdown)

    summary_path = runs_dir / "latest_summary.json"
    summary = {"generated_at": generated_at, "runs": compacts}
    _atomic_write_text(summary_path, json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {EVALS_MD}")
    print(f"Wrote {summary_path}")


def _load_run(run_id: str) -> dict:
    path = settings.runs_dir / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No run record found for run_id={run_id!r} (expected {path})")
    return json.loads(path.read_text(encoding="utf-8"))


def compare(baseline: str, candidate: str, max_drop: float = 0.05) -> bool:
    base = _load_run(baseline)
    cand = _load_run(candidate)

    base_agg = base.get("aggregates") or {}
    cand_agg = cand.get("aggregates") or {}

    gated = [m for m in _GATED_METRICS if m in base_agg and m in cand_agg]
    if not gated:
        raise ValueError(
            f"No common gated metrics between baseline={baseline!r} and candidate={candidate!r}; "
            f"expected overlap among {_GATED_METRICS}"
        )

    base_dataset = base.get("dataset") or {}
    cand_dataset = cand.get("dataset") or {}
    if base_dataset != cand_dataset:  # sha256, limit, or n_samples differ
        print(
            f"WARNING: baseline dataset {base_dataset} != candidate dataset "
            f"{cand_dataset}; deltas across different datasets/subsets are "
            "not comparable."
        )
    for record, label in ((base, "baseline"), (cand, "candidate")):
        if record.get("aborted"):
            print(f"WARNING: {label} run was aborted mid-run: {record['aborted']}")
        coverage = record.get("coverage") or coverage_from_scores(
            [(s.get("scores") or {}) for s in (record.get("samples") or [])]
        )
        partial = {
            metric: f"{cov['scored']}/{cov['total']}"
            for metric, cov in coverage.items()
            if metric in gated and cov["scored"] < cov["total"]
        }
        if partial:
            print(
                f"WARNING: {label} has partial metric coverage {partial}; "
                "its aggregates average a survivor subset, not the dataset."
            )

    if base.get("corpus_hash") != cand.get("corpus_hash"):
        print(
            f"WARNING: baseline corpus_hash={base.get('corpus_hash')} != "
            f"candidate corpus_hash={cand.get('corpus_hash')}; deltas across "
            "different corpora are not comparable."
        )

    passed = True
    print(f"{'metric':<20}{'baseline':>10}{'candidate':>10}{'delta':>10}  verdict")
    for metric in gated:
        b = base_agg[metric]
        c = cand_agg[metric]
        delta = c - b
        regressed = delta < -max_drop
        if regressed:
            passed = False
        verdict = "REGRESSED" if regressed else "ok"
        print(f"{metric:<20}{b:>10.3f}{c:>10.3f}{delta:>10.3f}  {verdict}")

    return passed
