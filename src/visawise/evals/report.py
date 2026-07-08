"""Reporting: runs/ -> leaderboard markdown + dashboard JSON + regression gate.

- report(): renders a cross-run leaderboard (docs/EVALS.md + README section)
  and evals/runs/latest_summary.json (served by /api/eval/runs for the
  dashboard tab).
- compare(baseline_run_id, candidate_run_id, thresholds): exits nonzero when
  faithfulness or response relevancy regress beyond threshold -- the CI gate.
"""


def report() -> None:
    raise NotImplementedError  # TODO(pairing)


def compare(baseline: str, candidate: str, max_drop: float = 0.05) -> bool:
    raise NotImplementedError  # TODO(pairing)
