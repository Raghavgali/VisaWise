"""Stage 1: fetch source HTML from uscis.gov.

Contract:
- Reads source list from configs/sources.yaml ({url, topic, title} entries).
- Writes raw HTML to data/raw/<doc_id>.html where doc_id = sha256(url)[:16].
- Maintains data/raw/manifest.json: {doc_id: FetchRecord}.
- Skips unchanged pages (content sha256 match) unless force=True.
- Non-200s never overwrite a previous good snapshot; they are reported in the
  summary and make the CLI exit nonzero, so link rot is loud on every refresh.
- Pages that 301 into /archive/ still return 200 but are stale by USCIS's own
  admission (observed with the 2024 H-1B FAQ source): they are snapshotted and
  recorded with archived=True, reported separately, and skipped by extract.
"""

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

from ..config import settings


@dataclass
class FetchRecord:
    doc_id: str
    url: str
    final_url: str  # post-redirect; differs from url when a page has moved
    topic: str
    title: str
    http_status: int
    fetched_at: str  # ISO 8601 UTC
    content_sha256: str
    archived: bool = False  # final_url landed under uscis.gov/archive/


@dataclass
class FetchSummary:
    fetched: int = 0
    unchanged: int = 0
    archived: list[str] = field(default_factory=list)  # urls (stale but snapshotted)
    failed: list[str] = field(default_factory=list)  # urls


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_manifest(path: Path) -> dict[str, FetchRecord]:
    if not path.exists():
        return {}

    raw = json.loads(path.read_text())
    return {key: FetchRecord(**value) for key, value in raw.items()}


def write_manifest(path: Path, records: dict[str, FetchRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: asdict(record) for key, record in records.items()}

    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def get_with_retries(client: httpx.Client, url: str, attempts: int) -> httpx.Response:
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = client.get(url)
            if response.status_code < 500:
                return response
            # 5xx is transient server trouble: retry like a transport error
            last_error = RuntimeError(f"HTTP {response.status_code}: {url}")
        except httpx.HTTPError as exc:
            last_error = exc
        if attempt < attempts - 1:
            time.sleep(2**attempt)

    raise RuntimeError(f"failed after {attempts} attempts: {url}") from last_error


def fetch_all(
    sources_file: Path = settings.sources_file,
    output_dir: Path | None = None,
    force: bool = False,
) -> FetchSummary:
    """Fetch every source with retry/backoff and a polite crawl delay."""
    output_dir = output_dir if output_dir is not None else settings.raw_dir
    sources = yaml.safe_load(sources_file.read_text())["sources"]

    manifest_path = output_dir / "manifest.json"
    manifest = read_manifest(manifest_path)
    summary = FetchSummary()

    with httpx.Client(
        timeout=settings.fetch_timeout_s,
        follow_redirects=True,
        headers={"User-Agent": settings.user_agent},
    ) as client:
        for source in sources:
            url = source["url"]
            doc_id = stable_key(url)

            try:
                response = get_with_retries(client, url, attempts=settings.fetch_retries)
            except RuntimeError:
                summary.failed.append(url)
                continue

            if response.status_code != 200:
                summary.failed.append(url)
                continue

            final_url = str(response.url)
            is_archived = "/archive/" in final_url
            if is_archived:
                summary.archived.append(url)

            body = response.content
            body_hash = sha256_bytes(body)

            previous = manifest.get(doc_id)
            if previous and previous.content_sha256 == body_hash and not force:
                summary.unchanged += 1
            else:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f"{doc_id}.html").write_bytes(body)

                manifest[doc_id] = FetchRecord(
                    doc_id=doc_id,
                    url=url,
                    final_url=final_url,
                    topic=source["topic"],
                    title=source["title"],
                    http_status=response.status_code,
                    fetched_at=utc_now(),
                    content_sha256=body_hash,
                    archived=is_archived,
                )
                summary.fetched += 1

            time.sleep(settings.fetch_delay_s)

    write_manifest(manifest_path, manifest)
    return summary
