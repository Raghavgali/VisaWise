"""The evaluation harness -- this project's centerpiece.

Experiments are YAML configs (configs/experiments/), datasets are versioned
JSONL (evals/datasets/), every run produces a committed, reproducible
RunRecord (evals/runs/) stamped with git sha + corpus hash + judge model.

    visawise-eval generate   # build synthetic golden set from the live corpus
    visawise-eval run <exp>  # run one experiment
    visawise-eval sweep <exp># run a parameter matrix (e.g. weight sweep)
    visawise-eval compare    # diff runs / regression-gate vs baseline
    visawise-eval report     # render leaderboard markdown + dashboard JSON
"""
