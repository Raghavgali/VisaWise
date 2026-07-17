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
from ..observability import traced_node
from .prompts import get_prompt
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
    reasoning_effort: str | None = None  # gpt-oss (Groq): "low"|"medium"|"high"
    thinking_budget: int | None = None  # Gemini 2.5: 0 disables thinking
    # If set, drop retrieved context when the top cross-encoder rerank score is
    # below this (nothing on-topic) so the prompt abstains/refuses/qualifies
    # instead of answering from loosely-related passages. Calibrated for the
    # local bge-reranker (0-1); only meaningful with a cross-encoder reranker.
    grounding_threshold: float | None = None

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

def _message_text(response) -> str:
    """Plain text from an LLM response. `.content` is a str for OpenAI/Groq/
    NVIDIA but a list of content blocks for Gemini 3.x / Anthropic (e.g.
    [{'type': 'text', 'text': ...}]); join the text parts so downstream (stored
    answer, judge, UI) never sees a stringified list."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, (str, dict))
        ]
        return "".join(p for p in parts if isinstance(p, str))
    return str(content)


def build_graph(config: EngineConfig):
    """EngineConfig -> compiled LangGraph app (shared by serving and evals)."""
    system_prompt, user_template = get_prompt(config.prompt_version)
    
    provider, separator, model_name = config.llm.partition(":")
    if provider not in ("groq", "nvidia", "gemini") or not separator or not model_name:
        raise ValueError("llm must look like 'groq:<model>', 'nvidia:<model>' or 'gemini:<model>'")

    if config.retriever not in ("vector", "bm25", "hybrid"):
        raise ValueError(f"Unknown retriever: {config.retriever!r}")

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set -- llm 'groq:...' requires it (.env)")
        groq_kwargs = {}
        if config.reasoning_effort is not None:
            # gpt-oss reasoning models: trade depth for latency. Ignored by
            # non-reasoning Groq models, so only set when asked.
            groq_kwargs["reasoning_effort"] = config.reasoning_effort
        llm = ChatGroq(model=model_name, api_key=settings.groq_api_key, temperature=0, **groq_kwargs)
    elif provider == "gemini":  # Google AI Studio (free tier), independent of the gpt-4o-mini judge
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set -- llm 'gemini:...' requires it (.env)")
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_kwargs = {}
        if config.thinking_budget is not None:
            # Gemini 2.5 thinks by default; 0 disables it (fewer tokens, faster).
            gemini_kwargs["thinking_budget"] = config.thinking_budget
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.google_api_key,
            temperature=0,
            **gemini_kwargs,
        )
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

        # Grounding gate: if nothing retrieved clears the relevance threshold,
        # answer with NO context so the prompt's precedence (abstain / refuse /
        # qualify) fires instead of stretching off-topic passages into an answer.
        if config.grounding_threshold is not None and final_hits:
            if max(hit.score for hit in final_hits) < config.grounding_threshold:
                final_hits = []

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
        
        prompt = user_template.format(
            context="\n\n".join(context_parts),
            query=state["query"],
        )

        response = llm.invoke(
            [
                ("system", system_prompt),
                ("human", prompt),
            ]
        )

        answer = _message_text(response)
        return {"answer": answer, "citations": citations}
    
    # traced_node = per-stage latency + bounded metadata in the trace; a
    # strict no-op when telemetry is unconfigured (evals, tests, local dev).
    graph = StateGraph(GraphState)
    graph.add_node("generate", traced_node("generate", generate))

    if config.retriever == "hybrid":
        graph.add_node("dense_retrieve", traced_node("dense_retrieve", dense_retrieve))
        graph.add_node("bm25_retrieve", traced_node("bm25_retrieve", bm25_retrieve))
        graph.add_node("fuse", traced_node("fuse", fuse))

        graph.add_edge(START, "dense_retrieve")
        graph.add_edge(START, "bm25_retrieve")
        graph.add_edge("dense_retrieve", "fuse")
        graph.add_edge("bm25_retrieve", "fuse")
        previous_node = "fuse"

    elif config.retriever == "vector":
        graph.add_node("dense_retrieve", traced_node("dense_retrieve", dense_retrieve))
        graph.add_edge(START, "dense_retrieve")
        previous_node = "dense_retrieve"

    elif config.retriever == "bm25":
        graph.add_node("bm25_retrieve", traced_node("bm25_retrieve", bm25_retrieve))
        graph.add_edge(START, "bm25_retrieve")
        previous_node = "bm25_retrieve"

    if config.reranker != "none":
        graph.add_node("rerank", traced_node("rerank", rerank))
        graph.add_edge(previous_node, "rerank")
        previous_node = "rerank"

    graph.add_edge(previous_node, "generate")
    graph.add_edge("generate", END)

    return graph.compile()

