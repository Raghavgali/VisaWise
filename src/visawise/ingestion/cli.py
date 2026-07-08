"""`visawise-ingest` -- run pipeline stages individually or end-to-end.

    visawise-ingest fetch [--force]
    visawise-ingest extract
    visawise-ingest chunk
    visawise-ingest index [--force]
    visawise-ingest all [--force]
    visawise-ingest status
"""

import typer

app = typer.Typer(no_args_is_help=True, help=__doc__)


@app.command()
def fetch(force: bool = typer.Option(False, help="Re-fetch even if content unchanged")) -> None:
    from .fetch import fetch_all

    summary = fetch_all(force=force)
    typer.echo(f"fetched={summary.fetched} unchanged={summary.unchanged}")
    for url in summary.archived:
        typer.secho(f"ARCHIVED (stale, will be excluded): {url}", fg=typer.colors.YELLOW)
    for url in summary.failed:
        typer.secho(f"FAILED: {url}", fg=typer.colors.RED)
    if summary.failed:
        raise typer.Exit(1)


@app.command()
def extract() -> None:
    from .extract import extract_all

    summary = extract_all()
    typer.echo(f"extracted={summary.extracted} skipped_archived={summary.skipped_archived}")
    for reason in summary.failed:
        typer.secho(f"FAILED: {reason}", fg=typer.colors.RED)
    if summary.failed:
        raise typer.Exit(1)


@app.command()
def chunk() -> None:
    from ..config import settings
    from .chunk import chunk_all, corpus_hash

    chunks = chunk_all()
    docs = len({c.doc_id for c in chunks})
    typer.echo(
        f"chunks={len(chunks)} docs={docs} strategy={settings.chunk_strategy} "
        f"corpus_hash={corpus_hash(chunks)[:16]}"
    )


@app.command()
def index(force: bool = typer.Option(False, help="Rebuild the LanceDB table from scratch")) -> None:
    from .index import index_chunks

    summary = index_chunks(force=force)
    typer.echo(f"added={summary.added} skipped={summary.skipped} pruned={summary.pruned}")


@app.command()
def all(force: bool = False) -> None:  # noqa: A001 - CLI verb
    fetch(force=force)
    extract()
    chunk()
    index(force=force)


@app.command()
def status() -> None:
    import json

    from .index import status as index_status

    typer.echo(json.dumps(index_status(), indent=2))


def main() -> None:
    app()
