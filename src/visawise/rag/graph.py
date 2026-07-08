"""The shared graph factory. Everything queryable is built here and only here.

EngineConfig is the unit of experimentation: the eval harness sweeps it, the
app serves the winning one, and every RunRecord embeds a serialized copy.

build_graph(EngineConfig) -> compiled LangGraph:
    state: {query, dense_hits, bm25_hits, fused, reranked, answer, citations}
    nodes: dense_retrieve | bm25_retrieve -> fuse -> rerank -> generate
    - retriever="vector"/"bm25" builds single-leg graphs (ablation configs).
    - reranker="none" skips the rerank node.
    - generate uses ChatGroq + prompts v-pinned; streaming supported for the
      app, invoke() for evals; citations = deduped (url, title) of final
      context chunks.
"""

from dataclasses import asdict, dataclass

from ..config import settings
from .rerank import RerankerKind
from .retrieval import ScoredChunk  # noqa: F401 - part of the state contract

RetrieverKind = str  # "vector" | "bm25" | "hybrid"


@dataclass(frozen=True)
class EngineConfig:
    retriever: RetrieverKind = "hybrid"
    top_k: int = settings.retriever_top_k
    vector_weight: float = settings.hybrid_vector_weight  # hybrid only
    reranker: RerankerKind = "local"
    rerank_top_n: int = settings.rerank_top_n
    llm: str = f"groq:{settings.groq_model}"
    prompt_version: str = "v1"

    def to_dict(self) -> dict:
        return asdict(self)


def build_graph(config: EngineConfig):
    """EngineConfig -> compiled LangGraph app (shared by serving and evals)."""
    raise NotImplementedError  # TODO(pairing)
