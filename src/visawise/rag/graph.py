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
from typing import TypedDict, Literal

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from dataclasses import asdict, dataclass

from ..config import settings
from .prompts import QA_PROMPT_VERSION, QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from .rerank import build_reranker, RerankerKind
from .retrieval import ScoredChunk, bm25_search, dense_search, fuse_rrf # noqa: F401 - part of the state contract

RetrieverKind = Literal["vector", "bm25", "hybrid"]

@dataclass(frozen=True)
class EngineConfig:
    retriever: RetrieverKind = "hybrid"
    top_k: int = settings.retriever_top_k
    vector_weight: float = settings.hybrid_vector_weight  # hybrid only
    reranker: RerankerKind = "local"
    rerank_top_n: int = settings.rerank_top_n
    llm: str = settings.generation_llm
    prompt_version: str = "v1"

    def to_dict(self) -> dict:
        return asdict(self)

class GraphState(TypedDict, total=False):
    query: str
    dense_hits: list[ScoredChunk]
    bm25_hits: list[ScoredChunk]
    fused: list[ScoredChunk]
    reranked: list[ScoredChunk]
    answer: str 
    citations: list[dict[str, str]]

def build_graph(config: EngineConfig):
    """EngineConfig -> compiled LangGraph app (shared by serving and evals)."""
    if config.prompt_version != QA_PROMPT_VERSION:
        raise ValueError(f"Unsupported prompt version: {config.prompt_version}")
    
    provider, separator, model_name = config.llm.partition(":")
    if provider not in ("groq", "nvidia") or not separator or not model_name:
        raise ValueError("llm must look like 'groq:<model>' or 'nvidia:<model>'")

    if config.retriever not in ("vector", "bm25", "hybrid"):
        raise ValueError(f"Unknown retriever: {config.retriever!r}")

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set -- llm 'groq:...' requires it (.env)")
        llm = ChatGroq(model=model_name, api_key=settings.groq_api_key, temperature=0)
    else:  # nvidia: hosted NIM, OpenAI-compatible endpoint
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set -- llm 'nvidia:...' requires it (.env)")
        llm = ChatOpenAI(
            model=model_name,
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
            temperature=0,
        )
    rank_chunks = build_reranker(config.reranker)

    def dense_retrieve(state: GraphState) -> dict:
        hits = dense_search(state["query"], config.top_k)

        update = {"dense_hits": hits}
        if config.retriever == "vector":
            update["fused"] = hits 

        return update 
    

    def bm25_retrieve(state: GraphState) -> dict:
        hits = bm25_search(state["query"], config.top_k)

        update = {"bm25_hits": hits}
        if config.retriever == "bm25":
            update["fused"] = hits

        return update 
    
    def fuse(state: GraphState) -> dict:
        hits = fuse_rrf(
            [state["dense_hits"], state["bm25_hits"]],
            [config.vector_weight, 1.0 - config.vector_weight],
            config.top_k,
        )
        return {"fused": hits}
    
    def rerank(state: GraphState) -> dict:
        return {
            "reranked": rank_chunks(
                state["query"],
                state["fused"],
                config.rerank_top_n,
            )
        }
    
    def generate(state: GraphState) -> dict:
        final_hits = state.get("reranked", state["fused"])

        context_parts: list[str] = []
        citations: list[dict[str, str]] = []
        seen_sources: set[tuple[str, str]] = set()
        

        for hit in final_hits:
            section = f"Section: {hit.section}\n" if hit.section else ""
            context_parts.append(
                f"Title: {hit.title}\nURL: {hit.url}\n{section}Content:\n{hit.text}"
            )

            source_key = (hit.url, hit.title)
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                citations.append({"url": hit.url, "title": hit.title})
        
        prompt = QA_USER_TEMPLATE.format(
            context="\n\n".join(context_parts),
            query=state["query"],

        )

        response = llm.invoke(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("human", prompt),
            ]
        )

        answer = response.content if isinstance(response.content, str) else str(response.content)
        return {"answer": answer, "citations": citations}
    
    graph = StateGraph(GraphState)
    graph.add_node("generate", generate)

    if config.retriever == "hybrid":
        graph.add_node("dense_retrieve", dense_retrieve)
        graph.add_node("bm25_retrieve", bm25_retrieve)
        graph.add_node("fuse", fuse)

        graph.add_edge(START, "dense_retrieve")
        graph.add_edge(START, "bm25_retrieve")
        graph.add_edge("dense_retrieve", "fuse")
        graph.add_edge("bm25_retrieve", "fuse")
        previous_node = "fuse"

    elif config.retriever == "vector":
        graph.add_node("dense_retrieve", dense_retrieve)
        graph.add_edge(START, "dense_retrieve")
        previous_node = "dense_retrieve"

    elif config.retriever == "bm25":
        graph.add_node("bm25_retrieve", bm25_retrieve)
        graph.add_edge(START, "bm25_retrieve")
        previous_node = "bm25_retrieve"

    if config.reranker != "none":
        graph.add_node("rerank", rerank)
        graph.add_edge(previous_node, "rerank")
        previous_node = "rerank"

    graph.add_edge(previous_node, "generate")
    graph.add_edge("generate", END)

    return graph.compile()

