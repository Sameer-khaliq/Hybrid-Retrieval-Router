from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    groq_api_key: str = Field(default="")
    google_api_key: str = Field(default="")
    tavily_api_key: str = Field(default="")

    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="chunks")

    chunk_min_tokens: int = Field(default=200)
    chunk_max_tokens: int = Field(default=500)
    chunk_overlap_min_pct: float = Field(default=0.10)
    chunk_overlap_max_pct: float = Field(default=0.15)

    sparse_top_n: int = Field(default=20)
    dense_top_n: int = Field(default=20)

    rrf_k: int = Field(default=60)

    rerank_top_k: int = Field(default=10)
    rerank_top_k_fast: int = Field(default=5)
    rerank_candidate_pool: int = Field(default=10)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_timeout_s: float = Field(default=5.4)
    reranker_prefer_offline: bool = Field(default=True)
    reranker_cpu_threads: int = Field(default=4)
    reranker_max_length: int = Field(default=256)
    reranker_max_candidate_chars: int = Field(default=800)
    rerank_fast_path_mode: str = Field(default="skip")

    tau_low: float = Field(default=0.3)
    tau_high: float = Field(default=0.6)

    weight_token_count: float = Field(default=0.20)
    weight_keyword: float = Field(default=0.35)
    weight_question_word: float = Field(default=0.25)
    weight_clause_count: float = Field(default=0.20)

    router_timeout_ms: int = Field(default=1500)
    groq_router_model: str = Field(default="openai/gpt-oss-20b")

    groq_fast_model: str = Field(default="openai/gpt-oss-20b")
    groq_deep_model: str = Field(default="openai/gpt-oss-120b")
    groq_mid_model_a: str = Field(default="qwen/qwen3.6-27b")
    groq_mid_model_b: str = Field(default="")

    groq_deep_max_concurrency: int = Field(default=2)

    generation_max_context_chunks: int = Field(default=5)
    generation_max_context_chunks_fast: int = Field(default=5)
    generation_max_context_chunks_deep: int = Field(default=10)
    generation_max_chunk_chars: int = Field(default=700)

    tavily_confidence_floor: float = Field(default=0.02)

    gemini_embedding_model: str = Field(default="gemini-embedding-001")
    gemini_embedding_version: str = Field(default="001")
    gemini_embedding_dimension: int = Field(default=768)

    max_query_length: int = Field(default=2000)
    max_api_calls_per_query: int = Field(default=4)
    max_retries: int = Field(default=2)

    bm25_index_dir: str = Field(default="data/.bm25_index")


settings = Settings()