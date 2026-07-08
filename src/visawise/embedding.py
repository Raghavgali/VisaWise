"""The ONE place the embedding model loads.

Index-side (ingestion) and query-side (rag) vectors must come from the same
model and config, or retrieval quality degrades silently. Both import from
here; nothing else may construct a SentenceTransformer.
"""

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(texts, batch_size=32, show_progress_bar=len(texts) > 16)
    return [vector.tolist() for vector in vectors]


def embed_query(query: str) -> list[float]:
    return get_embedder().encode(query).tolist()
