"""`visawise-eval` -- see package docstring for the command surface."""

import typer

app = typer.Typer(no_args_is_help=True, help=__doc__)


@app.command()
def generate(size: int = 50, name: str = "synthetic_v1") -> None:
    """Generate a synthetic golden dataset from the live corpus."""
    from .datasets import generate_synthetic, save_dataset

    samples = generate_synthetic(size)
    path = save_dataset(samples, name)
    typer.echo(f"generated {len(samples)} samples -> {path}")


@app.command()
def run(experiment: str) -> None:
    """Run an experiment file from configs/experiments/."""
    from pathlib import Path

    from ..config import REPO_ROOT
    from .runner import run_experiment

    experiment_path = Path(experiment)
    if not experiment_path.exists():
        experiment_path = REPO_ROOT / "configs" / "experiments" / f"{experiment}.yaml"

    run_ids = run_experiment(str(experiment_path))
    for run_id in run_ids:
        typer.echo(run_id)


@app.command()
def compare(baseline: str, candidate: str, max_drop: float = 0.05) -> None:
    """Regression gate: nonzero exit if candidate regresses vs baseline,
    or if the two runs aren't comparable (dataset/corpus mismatch, aborted,
    partial coverage). Fails closed."""
    from . import report as report_

    if not report_.compare(baseline, candidate, max_drop):
        raise typer.Exit(code=1)


@app.command()
def gate(run_id: str = typer.Option(None, help="Specific run to gate (default: newest canonical-dataset run)")) -> None:
    """Release gate: absolute floors (answerable faithfulness, safety
    abstention 14/15) on the canonical serving run. Runs offline on committed
    RunRecords — this is what CI executes. Nonzero exit on failure."""
    from . import report as report_

    if not report_.gate(run_id=run_id):
        raise typer.Exit(code=1)


@app.command()
def report() -> None:
    """Render leaderboard markdown + dashboard JSON from committed runs."""
    from . import report as report_

    report_.report()


def main() -> None:
    app()
