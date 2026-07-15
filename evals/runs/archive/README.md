# Archived runs — superseded 2026-07-15

These three 52-sample runs (2026-07-11) were the first full ablations, but
their faithfulness aggregates cover only 24–31 of 52 samples: ragas'
instructor adapter defaults the judge to `max_tokens=1024`, which truncated
the faithfulness verdict JSON on long answers, and every truncation landed in
the error bucket. Because the failures skew toward long, claim-dense answers,
those averages describe a biased survivor subset — and a *different* subset
per run — so they are not comparable to each other or to anything else.

Fixed by raising `judge_max_tokens` (settings) to 8192 and adding per-metric
coverage tracking + a ≥95% coverage gate to the runner. The replacement runs
in `evals/runs/` score 52/52. Kept here (out of the leaderboard glob, which is
non-recursive) as evidence for the coverage-gate design decision.
