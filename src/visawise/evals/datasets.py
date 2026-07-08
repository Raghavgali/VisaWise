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

from dataclasses import dataclass


@dataclass
class GoldenSample:
    id: str
    user_input: str
    reference: str
    reference_contexts: list[str]
    source_urls: list[str]
    origin: str  # "synthetic" | "curated"


def load_dataset(name: str) -> list[GoldenSample]:
    raise NotImplementedError  # TODO(pairing)


def generate_synthetic(size: int = 50) -> list[GoldenSample]:
    raise NotImplementedError  # TODO(pairing)
