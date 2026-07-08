"""Stage 3: chunk extracted documents -> chunks.jsonl (the single source of truth).

Contract:
- Two strategies behind settings.chunk_strategy (an EVAL EXPERIMENT AXIS, not
  a hardcoded pick -- the harness declares the winner):
    "token"   [validated 2024 baseline] token-based recursive splitting,
              chunk_size=1024 tokens, overlap=50, sentence-aware separators
              (langchain-text-splitters + the bge tokenizer, so "token" means
              the same thing to the splitter and the embedder).
    "section" heading-aware: split on markdown headings, prepend the section
              breadcrumb ("Page Title > H2 > H3") to the text, token-cap only
              oversized sections. Sections fit bge's 512-token embed window,
              fixing the 2024 quirk where the dense leg saw only the first
              half of each 1024 chunk (BM25 saw all of it).
- chunk_id = sha256(f"{url}::{seq}::{text_sha}")[:24] -- deterministic, so
  re-runs are idempotent and LanceDB writes/prunes can diff by id.
- Each JSONL line carries full provenance: doc_id, seq, url, title, topic,
  section, fetched_at. Consumers: LanceDB indexing (stage 4: dense + FTS),
  eval corpus-hash stamping (evals/).
- corpus_hash() = sha256 over sorted chunk_ids: one value that identifies the
  exact corpus (content AND strategy) an eval run or index was built from.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ..config import settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    seq: int
    text: str  # for "section" strategy, starts with the breadcrumb line
    url: str
    title: str
    topic: str
    section: str  # heading breadcrumb ("" for token strategy)
    fetched_at: str


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_markdown_with_frontmatter(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")

    if not raw.startswith("---\n"):
        raise ValueError(f"Missing YAML frontmatter: {path}")

    header_text, separator, body = raw.removeprefix("---\n").partition("\n---\n")

    if not separator:
        raise ValueError(f"Unclosed YAML frontmatter: {path}")

    metadata = yaml.safe_load(header_text) or {}
    return metadata, body.strip()


def build_token_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Recursive splitter whose "size" is bge tokenizer tokens, so the splitter
    and the embedder agree on what a token is."""
    import transformers

    transformers.logging.set_verbosity_error()  # silence >512-token length warnings
    tokenizer = transformers.AutoTokenizer.from_pretrained(settings.embed_model_name)

    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


_HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]


def split_token(body: str, splitter: RecursiveCharacterTextSplitter) -> list[tuple[str, str]]:
    """-> [(text, section="")] in document order."""
    return [(text.strip(), "") for text in splitter.split_text(body) if text.strip()]


def split_section(
    body: str,
    title: str,
    cap_splitter: RecursiveCharacterTextSplitter,
) -> list[tuple[str, str]]:
    """-> [(breadcrumb + text, breadcrumb)] in document order; oversized
    sections are token-capped, each part keeping the same breadcrumb."""
    header_splitter = MarkdownHeaderTextSplitter(_HEADERS_TO_SPLIT_ON)
    pieces: list[tuple[str, str]] = []

    for section_doc in header_splitter.split_text(body):
        crumbs = [title]
        for level in ("h1", "h2", "h3", "h4"):
            heading = section_doc.metadata.get(level)
            if heading and heading != crumbs[-1]:
                crumbs.append(heading)
        breadcrumb = " > ".join(crumbs)

        for part in cap_splitter.split_text(section_doc.page_content):
            text = part.strip()
            if text:
                pieces.append((f"{breadcrumb}\n\n{text}", breadcrumb))

    return pieces


def make_piece_id(url: str, seq: int, text: str) -> str:
    text_hash = sha256_text(text)
    return sha256_text(f"{url}::{seq}::{text_hash}")[:24]


def write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(path.name + ".tmp")

    with tmp_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), sort_keys=True) + "\n")

    tmp_path.replace(path)


def chunk_all(
    extracted_dir: Path | None = None,
    output_path: Path | None = None,
) -> list[Chunk]:
    """Chunk every extracted doc per settings.chunk_strategy and write
    data/chunks/chunks.jsonl."""
    extracted_dir = extracted_dir if extracted_dir is not None else settings.extracted_dir
    output_path = output_path if output_path is not None else settings.chunks_file
    strategy = settings.chunk_strategy

    if strategy == "token":
        splitter = build_token_splitter(settings.chunk_size, settings.chunk_overlap)
    elif strategy == "section":
        splitter = build_token_splitter(settings.section_max_tokens, settings.chunk_overlap)
    else:
        raise ValueError(f"Unknown chunk_strategy: {strategy!r} (expected 'token' or 'section')")

    md_files = sorted(extracted_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No extracted docs in {extracted_dir} -- run extract first")

    chunks: list[Chunk] = []
    for path in md_files:
        metadata, body = read_markdown_with_frontmatter(path)

        if strategy == "token":
            pieces = split_token(body, splitter)
        else:
            pieces = split_section(body, metadata["title"], splitter)

        for seq, (text, section) in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=make_piece_id(metadata["url"], seq, text),
                    doc_id=metadata["doc_id"],
                    seq=seq,
                    text=text,
                    url=metadata["url"],
                    title=metadata["title"],
                    topic=metadata["topic"],
                    section=section,
                    fetched_at=metadata["fetched_at"],
                )
            )

    write_jsonl(output_path, chunks)
    return chunks


def load_chunks(path: Path | None = None) -> list[Chunk]:
    """Read chunks.jsonl (used by rag/ and evals/, not just ingestion)."""
    path = path if path is not None else settings.chunks_file
    with path.open(encoding="utf-8") as file:
        return [Chunk(**json.loads(line)) for line in file if line.strip()]


def corpus_hash(chunks: list[Chunk]) -> str:
    return sha256_text("\n".join(sorted(chunk.chunk_id for chunk in chunks)))
