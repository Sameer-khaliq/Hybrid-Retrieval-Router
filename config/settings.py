"""
Central configuration module (Step 2, FR-19 foundation).

Every tunable parameter in the system lives here and NOWHERE else.
Every module built from Step 5 onward must import `settings` from this
file instead of hardcoding a literal value.

Sanctioned exception to "no custom classes" per REQUIREMENTS.md §0.2:
pydantic-settings' BaseSettings is a data-validation schema, not
business-logic, so it's allowed.

MODEL UPDATE (2026-08-16): Groq decommissioned llama-3.1-8b-instant and
llama-3.3-70b-versatile. Current available chat-completion models on
the free tier: openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.6-27b
(groq/compound* excluded - agentic/tool-using models, would break
NFR-4's deterministic ≤4-call budget; meta-llama/llama-prompt-guard-2-*
excluded - classifiers, not chat models; whisper-large-v3* excluded -
speech-to-text, irrelevant here).

Remapped:
  groq_router_model : llama-3.1-8b-instant  -> openai/gpt-oss-20b
  groq_fast_model    : llama-3.1-8b-instant  -> openai/gpt-oss-20b
  groq_mid_model_a   : llama-3.3-70b-versatile -> qwen/qwen3.6-27b
  groq_mid_model_b   : gpt-oss-20b -> dropped (pool shrank to 3 usable
                        models; no distinct second mid-tier candidate
                        remains under the current free-tier allowance -
                        see groq_mid_model_b's own comment below)
  groq_deep_model    : gpt-oss-120b -> unchanged, still available

Known consequence: fast-path generation and the Layer-1 router now
share one model (gpt-oss-20b). Not a correctness problem, but it means
router-call traffic and fast-path-generation traffic draw from the same
RPM/TPM ceiling on that model - worth watching once load-testing
against NFR-3, not a Phase 4 blocker.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # ---- Env file wiring ----------------------------------------------
    # extra="ignore" so unrelated env vars (e.g. from Docker) don't crash
    # startup with a validation error.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- API keys (no defaults on purpose - missing key should fail loud)
    groq_api_key: str = Field(default="")
    google_api_key: str = Field(default="")  # Gemini
    tavily_api_key: str = Field(default="")

    # ---- Qdrant (Step 3) ------------------------------------------------
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="chunks")

    # ---- Chunking (FR-1) -------------------------------------------------
    chunk_min_tokens: int = Field(default=200)
    chunk_max_tokens: int = Field(default=500)
    chunk_overlap_min_pct: float = Field(default=0.10)
    chunk_overlap_max_pct: float = Field(default=0.15)

    # ---- Retrieval top-N (FR-4 / FR-5) -----------------------------------
    sparse_top_n: int = Field(default=20)
    dense_top_n: int = Field(default=20)

    # ---- RRF fusion (FR-6) ------------------------------------------------
    rrf_k: int = Field(default=60)

    # ---- Reranking top-K / top-K' (FR-6) -----------------------------------
    rerank_top_k: int = Field(default=15)      # deep-path
    rerank_top_k_fast: int = Field(default=5)  # fast-path (reduced or skipped)

    # ---- Cross-encoder reranker model (FR-10, Step 18) ---------------------
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ---- Routing thresholds (FR-12) ----------------------------------------
    tau_low: float = Field(default=0.3)
    tau_high: float = Field(default=0.6)

    # ---- Routing complexity-score weights (§3.4) ---------------------------
    weight_token_count: float = Field(default=0.20)
    weight_keyword: float = Field(default=0.35)
    weight_question_word: float = Field(default=0.25)
    weight_clause_count: float = Field(default=0.20)

    # ---- LLM-router fallback (FR-13, addendum #2) ---------------------------
    router_timeout_ms: int = Field(default=400)
    groq_router_model: str = Field(default="openai/gpt-oss-20b")

    # ---- Groq generation models (fast/deep + FR-27 mid-tier) ----------------
    # Pool shrank to 3 usable models after the 2026-08-16 decommission.
    groq_fast_model: str = Field(default="openai/gpt-oss-20b")
    groq_deep_model: str = Field(default="openai/gpt-oss-120b")
    groq_mid_model_a: str = Field(default="qwen/qwen3.6-27b")
    # groq_mid_model_b intentionally left unset (empty string, not a real
    # model ID) - no distinct second mid-tier candidate exists under the
    # current free-tier allowance. FR-27 (Step 29, not yet built) should
    # treat mid-tier as a single model (groq_mid_model_a) until Groq's
    # available-model list changes again, rather than defaulting this to
    # a duplicate of an existing tier.
    groq_mid_model_b: str = Field(default="")

    # ---- Tavily fallback (FR-9) ---------------------------------------------
    tavily_confidence_floor: float = Field(default=0.02)

    # ---- Gemini embedding config, pinned in one place (Risk #5) -------------
    gemini_embedding_model: str = Field(default="gemini-embedding-001")
    gemini_embedding_version: str = Field(default="001")
    gemini_embedding_dimension: int = Field(default=768)
    # task_type is set per-call (document at ingest, query at query-time) -
    # NOT configurable here, so it can never silently drift; see FR-19/Risk #5.

    # ---- API / interface (FR-15/17) ------------------------------------------
    max_query_length: int = Field(default=2000)

    # ---- Cost / rate-limit budget (NFR-4) -------------------------------------
    max_api_calls_per_query: int = Field(default=4)

    # ---- Retry/backoff (NFR-8, used by Step 11's resilience.py) ---------------
    max_retries: int = Field(default=2)

    # ---- BM25 sparse index persistence (Step 8) ------------------------------
    bm25_index_dir: str = Field(default="data/.bm25_index")


# Module-level singleton - import this, don't re-instantiate Settings()
# all over the codebase (keeps env parsing to one place).
settings = Settings()