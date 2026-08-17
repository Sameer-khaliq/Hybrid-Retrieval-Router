"""
Wraps the query-time embedding call in with_retry; on exhausted retries,
falls back to sparse-only and marks degraded=True (FR-8).
"""
from __future__ import annotations

import asyncio

from config.settings import settings
from src.common.resilience import RetriesExhaustedError, with_retry
from src.ingestion.embedder import embed_texts
from src.observability.logging_setup import get_logger
from src.retrieval.sparse_bm25 import query_bm25


async def embed_query_with_fallback(query: str, trace_id: str = "embed_fallback") -> dict:
    """
    Returns {"vector": [...], "degraded": False} on success, or
    {"vector": None, "degraded": True, "sparse_results": [...]} on
    exhausted retries (FR-8's own verify target).
    """
    logger = get_logger(trace_id=trace_id)

    async def _embed():
        return await asyncio.to_thread(embed_texts, [query], "query", trace_id)

    try:
        vectors = await with_retry(_embed, max_retries=settings.max_retries)
        return {"vector": vectors[0], "degraded": False}
    except RetriesExhaustedError:
        logger.warning("embedding_failed_fallback_sparse", stage="embedding_fallback")
        sparse_results = await asyncio.to_thread(query_bm25, query)
        return {"vector": None, "degraded": True, "sparse_results": sparse_results}