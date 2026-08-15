"""
Gemini embedding wrapper — pins model/version/task_type/dimension in one place
(Risk Register #5). Ingestion uses task_type="document"; query path (Step 9)
reuses this module with task_type="query".
"""
from __future__ import annotations

from google import genai
from google.genai import types

from config.settings import settings
from src.observability.logging_setup import get_logger


def get_genai_client() -> genai.Client:
    return genai.Client(api_key=settings.google_api_key)


def embed_texts(
    texts: list[str],
    task_type: str,
    trace_id: str = "embed",
) -> list[list[float]]:
    """
    Embed a batch of texts with the pinned model/dimension/task_type.

    task_type: "document" (ingestion) or "query" (retrieval) — per Risk #5,
    NEVER mix these silently.
    """
    logger = get_logger(trace_id=trace_id)
    client = get_genai_client()

    gemini_task_type = "RETRIEVAL_DOCUMENT" if task_type == "document" else "RETRIEVAL_QUERY"

    result = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=gemini_task_type,
            output_dimensionality=settings.gemini_embedding_dimension,
        ),
    )

    embeddings = [e.values for e in result.embeddings]

    logger.info(
        "embedding_call",
        stage="embedding",
        task_type=task_type,
        model=settings.gemini_embedding_model,
        dimension=settings.gemini_embedding_dimension,
        num_texts=len(texts),
    )

    return embeddings


def get_embedding_config() -> dict:
    """Config snapshot logged alongside every stored vector (feeds Step 10)."""
    return {
        "model": settings.gemini_embedding_model,
        "version": settings.gemini_embedding_version,
        "dimension": settings.gemini_embedding_dimension,
        "task_type": "document",
    }