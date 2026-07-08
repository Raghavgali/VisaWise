"""Stage 2: extract clean main content from raw HTML.

Contract:
- Reads data/raw/*.html + manifest.json (schema owned by fetch.FetchRecord).
- Skips records with archived=True (stale by USCIS's own admission).
- trafilatura main-content extraction (drops the nav/banner boilerplate that
  polluted the 2024 print-to-PDF corpus); tables kept (H-1B scenario charts).
- Writes data/extracted/<doc_id>.md: YAML frontmatter (doc_id, url, title,
  topic, fetched_at) + markdown body.
- Guards: pages whose extracted text is suspiciously short (likely JS-rendered
  or extraction failure) are reported as failures, not silently ingested.
"""

from dataclasses import dataclass, field
from pathlib import Path

import trafilatura
import yaml

import re

from ..config import settings
from .fetch import FetchRecord, read_manifest

MIN_EXTRACTED_CHARS = 500

# USCIS pages end with sidebar link boxes ("Related Links" / "More Information")
# that trafilatura renders as headings + bare anchor-text lists (the hrefs are
# gone). They carry no prose and pollute the final chunk of a doc.
_TRAILING_LINK_HEADING = re.compile(
    r"^#{1,6} *(related links?|more information)\s*$", re.IGNORECASE | re.MULTILINE
)
_MAX_LINK_SECTION_LINE_CHARS = 200  # anything longer is real prose, not a link label


def strip_trailing_link_sections(body: str) -> str:
    """Drop a trailing Related Links/More Information section -- but only if
    everything after the heading looks like link labels (no long prose lines),
    so a genuine content section under a same-named heading is never lost."""
    match = None
    for match in _TRAILING_LINK_HEADING.finditer(body):
        pass  # keep the last occurrence
    if match is None:
        return body

    tail = body[match.end():]
    tail_lines = [line.strip("-* \t") for line in tail.splitlines() if line.strip()]
    if any(len(line) > _MAX_LINK_SECTION_LINE_CHARS for line in tail_lines):
        return body

    return body[: match.start()].rstrip()


@dataclass
class ExtractSummary:
    extracted: int = 0
    skipped_archived: int = 0
    failed: list[str] = field(default_factory=list)  # "doc_id: reason"


def markdown_with_frontmatter(record: FetchRecord, body: str) -> str:
    frontmatter = {
        "doc_id": record.doc_id,
        "url": record.url,
        "title": record.title,
        "topic": record.topic,
        "fetched_at": record.fetched_at,
    }

    header = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    return f"---\n{header}\n---\n\n{body.strip()}\n"


def extract_one(record: FetchRecord, raw_dir: Path, out_dir: Path) -> str | None:
    """Returns an error string, or None on success."""
    html_path = raw_dir / f"{record.doc_id}.html"
    output_path = out_dir / f"{record.doc_id}.md"

    if not html_path.exists():
        return f"{record.doc_id}: missing raw HTML"

    html = html_path.read_text(encoding="utf-8", errors="replace")

    body = trafilatura.extract(
        html,
        url=record.url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
    )

    if body:
        body = strip_trailing_link_sections(body)

    if not body or len(body.strip()) < MIN_EXTRACTED_CHARS:
        return f"{record.doc_id}: extracted text too short ({record.url})"

    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = markdown_with_frontmatter(record, body)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_text(rendered, encoding="utf-8")
    tmp_path.replace(output_path)

    return None


def extract_all(
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
) -> ExtractSummary:
    raw_dir = raw_dir if raw_dir is not None else settings.raw_dir
    out_dir = out_dir if out_dir is not None else settings.extracted_dir

    manifest = read_manifest(raw_dir / "manifest.json")
    if not manifest:
        raise FileNotFoundError(f"No manifest at {raw_dir}/manifest.json -- run fetch first")

    summary = ExtractSummary()

    for record in sorted(manifest.values(), key=lambda item: item.doc_id):
        if record.archived:
            summary.skipped_archived += 1
            continue

        error = extract_one(record, raw_dir, out_dir)
        if error:
            summary.failed.append(error)
        else:
            summary.extracted += 1

    return summary
