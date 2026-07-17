"""Single source of configuration truth.

All tunables live here (env-overridable via .env); modules never hardcode
paths, model names, or retrieval parameters. Values marked [validated 2024]
came out of the original research and should not be changed casually --
changing them invalidates comparability with the committed eval baselines.
"""

from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- API keys ---
    groq_api_key: str = ""
    nvidia_api_key: str = ""  # build.nvidia.com NIM (free tier)
    google_api_key: str = ""  # Gemini (Google AI Studio, free tier)
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
    # Provider-prefixed: "nvidia:<model>" (hosted NIM, OpenAI-compatible) or
    # "groq:<model>". Small model by default -- model size is an eval
    # experiment axis (configs/experiments/model_ablation.yaml), not a belief.
    generation_llm: str = "gemini:gemini-3.1-flash-lite"  # eval-selected serving model
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # --- eval ---
    judge_model: str = "gpt-4o-mini"
    # ragas' InstructorLLM defaults to max_tokens=1024, which truncates the
    # faithfulness verdict JSON on long answers (~40% of samples failed silently
    # into the error bucket before this was raised). gpt-4o-mini allows 16K out.
    judge_max_tokens: int = 8192
    # A judged metric scoring fewer than this fraction of samples fails the run:
    # averages over different survivor subsets are not comparable.
    judged_coverage_threshold: float = 0.95
    datasets_dir: Path = REPO_ROOT / "evals" / "datasets"
    runs_dir: Path = REPO_ROOT / "evals" / "runs"

    # ----- backend / serving -----
    # Origins allowed to call the API cross-origin: the deployed frontend
    # (Vercel) plus local dev. NoDecode + the validator let the env var be a
    # plain comma-separated list, e.g.
    #   CORS_ALLOWED_ORIGINS=https://visawise.vercel.app,http://localhost:5173
    # (a JSON array also works), instead of requiring JSON in the Modal secret.
    cors_allowed_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # ----- observability -----
    # All default-empty: telemetry is a strict no-op until configured, so local
    # dev and tests never open network connections or background threads.
    # Setting the two Langfuse keys is enough — observability.py derives the
    # OTLP endpoint + Basic-auth header from them. The generic OTEL_* pair
    # overrides that derivation for any other OTLP backend (vendor-neutral).
    otel_service_name: str = "visawise"
    otel_exporter_otlp_endpoint: str = ""  # e.g. https://host/api/public/otel/v1/traces
    otel_exporter_otlp_headers: str = ""  # e.g. Authorization=Basic <base64 pk:sk>
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    # Default is the EU region; US-region projects must point this at
    # https://us.cloud.langfuse.com (keys 401 against the wrong region).
    # Accepts LANGFUSE_BASE_URL (official Langfuse SDK name) or LANGFUSE_HOST.
    langfuse_host: str = Field(
        "https://cloud.langfuse.com",
        validation_alias=AliasChoices("langfuse_base_url", "langfuse_host"),
    )

    @field_validator("cors_allowed_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        # NoDecode hands us the raw env string; accept a JSON array or a plain
        # comma-separated list.
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                import json

                return json.loads(text)
            return [origin.strip() for origin in text.split(",") if origin.strip()]
        return value

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
