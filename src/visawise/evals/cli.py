"""`visawise-eval` -- see package docstring for the command surface."""

import typer

app = typer.Typer(no_args_is_help=True, help=__doc__)


@app.command()
def generate(size: int = 50) -> None:
    """Generate a synthetic golden dataset from the live corpus."""
    raise NotImplementedError  # TODO(pairing)


@app.command()
def run(experiment: str) -> None:
    """Run an experiment file from configs/experiments/."""
    raise NotImplementedError  # TODO(pairing)


@app.command()
def compare(baseline: str, candidate: str, max_drop: float = 0.05) -> None:
    """Regression gate: nonzero exit if candidate regresses vs baseline."""
    raise NotImplementedError  # TODO(pairing)


@app.command()
def report() -> None:
    """Render leaderboard markdown + dashboard JSON from committed runs."""
    raise NotImplementedError  # TODO(pairing)


def main() -> None:
    app()
