"""Golden dataset build/load/version.

Schema (one JSONL line per sample):
    {id, user_input, reference, reference_contexts: [str], source_urls: [str],
     origin: "synthetic" | "curated"}

- synthetic_vN.jsonl: RAGAS 0.4 TestsetGenerator (knowledge-graph based) over
  the fresh corpus, judge/generator = settings.judge_model. Target ~40-50
  samples (the 2024 set had 4-5 -- too small for stable numbers).
- curated_vN.jsonl: the 10 original 2024 scenario questions (ported from
  research/responses_evaluation.csv) + new ones; references human-reviewed.
- Datasets are frozen once referenced by a committed run: edits mean a new
  version file, never in-place changes.
"""
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import settings
from ..ingestion.chunk import read_markdown_with_frontmatter


@dataclass
class GoldenSample:
    id: str
    user_input: str
    reference: str
    reference_contexts: list[str]
    source_urls: list[str]
    origin: str  # "synthetic" | "curated"


_DATASET_NAME = re.compile(r"^(synthetic|curated)_v(\d+)$")
_ORIGINS = {"synthetic", "curated"}
_WHITESPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Whitespace-collapsed, lowercased -- for robust substring matching of
    reference contexts back to their source document."""
    return _WHITESPACE.sub(" ", text).strip().lower()


# ragas multi-hop contexts are prefixed "<N-hop>\n\n", and its HeadlineSplitter
# mangles punctuation at chunk boundaries -- match on alphanumerics only.
_HOP_PREFIX = re.compile(r"^<\d+-hop>\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _match_key(text: str) -> str:
    return _NON_ALNUM.sub(" ", _HOP_PREFIX.sub("", text.lower())).strip()


def _find_source_url(context: str, docs_index: list[tuple[str, str]]) -> str | None:
    """Map a generated reference context back to the doc it came from:
    exact (normalized) substring first, fuzzy fallback for split-mangled text."""
    key = _match_key(context)
    if not key:
        return None

    for doc_key, url in docs_index:
        if key in doc_key:
            return url

    from rapidfuzz import fuzz

    best_url: str | None = None
    best_score = 0.0
    for doc_key, url in docs_index:
        score = fuzz.partial_ratio(key, doc_key)
        if score > best_score:
            best_score, best_url = score, url
    return best_url if best_score >= 85.0 else None


def dataset_path(name: str) -> Path:
    if not _DATASET_NAME.fullmatch(name):
        raise ValueError("Dataset name must look like 'synthetic_v1' or 'curated_v2'")

    return settings.datasets_dir / f"{name}.jsonl"


def _require(condition: bool, path: Path, lineno: int, message: str) -> None:
    if not condition:
        raise ValueError(f"{path}:{lineno}: {message}")


def _parse_sample(data: object, path: Path, lineno: int) -> GoldenSample:
    _require(isinstance(data, dict), path, lineno, "line is not a JSON object")
    assert isinstance(data, dict)  # for type-checkers

    expected = {"id", "user_input", "reference", "reference_contexts", "source_urls", "origin"}
    missing = expected - data.keys()
    _require(not missing, path, lineno, f"missing field(s): {sorted(missing)}")
    extra = data.keys() - expected
    _require(not extra, path, lineno, f"unexpected field(s): {sorted(extra)}")

    _require(isinstance(data["id"], str) and data["id"].strip() != "", path, lineno,
             "'id' must be a non-empty string")
    _require(isinstance(data["user_input"], str) and data["user_input"].strip() != "", path, lineno,
             "'user_input' must be a non-empty string")
    _require(isinstance(data["reference"], str) and data["reference"].strip() != "", path, lineno,
             "'reference' must be a non-empty string")
    _require(
        isinstance(data["reference_contexts"], list)
        and all(isinstance(item, str) for item in data["reference_contexts"]),
        path, lineno, "'reference_contexts' must be a list of strings",
    )
    _require(
        isinstance(data["source_urls"], list)
        and all(isinstance(item, str) for item in data["source_urls"]),
        path, lineno, "'source_urls' must be a list of strings",
    )
    _require(data["origin"] in _ORIGINS, path, lineno,
             f"'origin' must be one of {sorted(_ORIGINS)}, got {data['origin']!r}")

    return GoldenSample(
        id=data["id"],
        user_input=data["user_input"],
        reference=data["reference"],
        reference_contexts=list(data["reference_contexts"]),
        source_urls=list(data["source_urls"]),
        origin=data["origin"],
    )


def load_dataset(name: str) -> list[GoldenSample]:
    """Read + strictly validate {datasets_dir}/{name}.jsonl. Raises ValueError
    naming the file and 1-based line number on the first bad line."""
    path = dataset_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples: list[GoldenSample] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for lineno, raw_line in enumerate(file, start=1):
            if not raw_line.strip():
                continue
            try:
                data = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc

            sample = _parse_sample(data, path, lineno)
            _require(sample.id not in seen_ids, path, lineno, f"duplicate id {sample.id!r}")
            seen_ids.add(sample.id)
            samples.append(sample)

    if not samples:
        raise ValueError(f"{path}: dataset is empty")

    return samples


def save_dataset(samples: list[GoldenSample], name: str, force: bool = False) -> Path:
    """Atomically write a dataset. REFUSES to overwrite an existing file unless
    force=True -- datasets are frozen once committed; bump the version instead."""
    path = dataset_path(name)
    if path.exists() and not force:
        raise FileExistsError(
            f"Dataset {path} already exists. Datasets are frozen once committed -- "
            f"bump the version (e.g. a new {name.rsplit('_v', 1)[0]}_v<N+1>.jsonl) "
            f"instead of editing in place, or pass force=True to overwrite."
        )

    seen_ids: set[str] = set()
    for sample in samples:
        if sample.origin not in _ORIGINS:
            raise ValueError(f"Invalid origin {sample.origin!r} on sample {sample.id!r}")
        if sample.id in seen_ids:
            raise ValueError(f"Duplicate id {sample.id!r} in samples")
        seen_ids.add(sample.id)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
    tmp_path.replace(path)

    return path


def generate_synthetic(size: int = 50) -> list[GoldenSample]:
    """Build a synthetic golden set with RAGAS 0.4's knowledge-graph
    TestsetGenerator over the extracted corpus. Live only (hits OpenAI)."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to generate a synthetic dataset")

    from langchain_core.documents import Document
    from openai import OpenAI
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory
    from ragas.testset import TestsetGenerator
    from ragas.utils import num_tokens_from_string

    # ragas 0.4.3 default_transforms only extracts headlines for docs > 500
    # tokens but runs HeadlineSplitter on ALL document nodes -- short docs
    # crash it ("'headlines' property not found"). Short pages are nav hubs
    # anyway: skip them as question sources (they stay in the retrieval corpus).
    _MIN_DOC_TOKENS = 500

    md_files = sorted(settings.extracted_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No extracted docs in {settings.extracted_dir} -- run extract first"
        )

    documents: list[Document] = []
    docs_index: list[tuple[str, str]] = []  # (normalised page_content, url)
    skipped: list[str] = []
    for path in md_files:
        metadata, body = read_markdown_with_frontmatter(path)
        if num_tokens_from_string(body) <= _MIN_DOC_TOKENS:
            skipped.append(metadata["title"])
            continue
        documents.append(
            Document(
                page_content=body,
                metadata={
                    "url": metadata["url"],
                    "title": metadata["title"],
                    "doc_id": metadata["doc_id"],
                },
            )
        )
        docs_index.append((_match_key(body), metadata["url"]))

    if skipped:
        print(
            f"[generate_synthetic] skipping {len(skipped)} short doc(s) "
            f"(<= {_MIN_DOC_TOKENS} tokens, unusable as question sources): "
            + ", ".join(skipped)
        )
    if not documents:
        raise RuntimeError("No documents long enough for testset generation")

    client = OpenAI(api_key=settings.openai_api_key)
    llm = llm_factory(settings.judge_model, client=client)
    embeddings = embedding_factory("openai", client=client)

    generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
    testset = generator.generate_with_langchain_docs(documents, testset_size=size)

    samples: list[GoldenSample] = []
    unmapped = 0
    for index, row in enumerate(testset.to_list()):
        reference_contexts = list(row.get("reference_contexts") or [])

        source_urls: list[str] = []
        for context in reference_contexts:
            matched = _find_source_url(context, docs_index)
            if matched is None:
                unmapped += 1
            elif matched not in source_urls:
                source_urls.append(matched)

        samples.append(
            GoldenSample(
                id=f"syn-{index:03d}",
                user_input=row.get("user_input", ""),
                reference=row.get("reference", ""),
                reference_contexts=reference_contexts,
                source_urls=source_urls,
                origin="synthetic",
            )
        )

    if unmapped:
        print(
            f"[generate_synthetic] WARNING: {unmapped} reference context(s) could not "
            f"be mapped to a source URL (left out of source_urls)."
        )

    return samples
