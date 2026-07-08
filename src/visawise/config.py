"""Single source of configuration truth.

All tunables live here (env-overridable via .env); modules never hardcode
paths, model names, or retrieval parameters. Values marked [validated 2024]
came out of the original research and should not be changed casually --
changing them invalidates comparability with the committed eval baselines.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API keys ---
    groq_api_key: str = ""
    openai_api_key: str = ""  # RAGAS judge only
    cohere_api_key: str = ""  # eval-side rerank parity only, never serving

    # --- corpus / ingestion ---
    sources_file: Path = REPO_ROOT / "configs" / "sources.yaml"
    data_dir: Path = REPO_ROOT / "data"
    fetch_timeout_s: float = 30.0
    fetch_delay_s: float = 1.0  # polite crawl delay between requests
    fetch_retries: int = 3
    user_agent: str = "VisaWiseBot/0.1 (educational RAG project)"

    # --- chunking ---
    # "token" = validated 2024 baseline; "section" = heading-aware + breadcrumbs.
    # An eval experiment axis: change via env/experiment config, not here.
    chunk_strategy: str = "token"
    chunk_size: int = 1024  # [validated 2024] token strategy
    chunk_overlap: int = 50  # [validated 2024] token strategy
    section_max_tokens: int = 480  # section strategy: breadcrumb + text <= bge's 512 window

    # --- embeddings [validated 2024] ---
    embed_model_name: str = "BAAI/bge-base-en-v1.5"
    embed_dim: int = 768

    # --- vector store: LanceDB (embedded; dense vectors + tantivy BM25 FTS in one table) ---
    lancedb_dir: Path = REPO_ROOT / "data" / "lancedb"
    lancedb_table: str = "chunks"

    # --- retrieval defaults [validated 2024: hybrid 0.6/0.4 + rerank top 4] ---
    retriever_top_k: int = 8
    hybrid_vector_weight: float = 0.6
    rerank_top_n: int = 4
    local_reranker_model: str = "BAAI/bge-reranker-base"

    # --- generation ---
    groq_model: str = "llama-3.3-70b-versatile"

    # --- eval ---
    judge_model: str = "gpt-4o-mini"
    datasets_dir: Path = REPO_ROOT / "evals" / "datasets"
    runs_dir: Path = REPO_ROOT / "evals" / "runs"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def extracted_dir(self) -> Path:
        return self.data_dir / "extracted"

    @property
    def chunks_file(self) -> Path:
        return self.data_dir / "chunks" / "chunks.jsonl"


settings = Settings()
