"""
Concurrent sparse+dense retrieval orchestration (FR-7).

NON-PAUSABLE: finish as one unbroken unit. A partially-wired asyncio.gather
(only one leg actually awaited, or Step 11's fallback wrapper accidentally
bypassed) runs without error but returns sequential-not-concurrent or
partial results — surfaces later as an unexplained NFR-2 latency miss.
"""
from __future__ import annotations

import asyncio

from config.settings import settings
from src.retrieval.sparse_bm25 import query_bm25
from src.retrieval.fallback import embed_query_with_fallback
from src.common.qdrant_client import get_client
from src.observability.logging_setup import get_logger


async def _dense_leg(query: str, top_n: int, trace_id: str) -> dict:
    """Runs embed_query_with_fallback, then queries Qdrant only if embedding succeeded."""
    embed_result = await embed_query_with_fallback(query, trace_id=trace_id)

    if embed_result["degraded"]:
        return {"results": [], "degraded": True}

    client = get_client()

    def _search():
        return client.query_points(
            collection_name=settings.qdrant_collection,
            query=embed_result["vector"],
            limit=top_n,
            with_payload=True,
        ).points

    points = await asyncio.to_thread(_search)
    results = [{"chunk_id": p.id, "score": p.score, "payload": p.payload} for p in points]
    return {"results": results, "degraded": False}


async def _sparse_leg(query: str, top_n: int) -> list[dict]:
    return await asyncio.to_thread(query_bm25, query, top_n)


async def retrieve_concurrent(
    query: str,
    sparse_top_n: int | None = None,
    dense_top_n: int | None = None,
    trace_id: str = "retrieve",
) -> dict:
    """
    Runs sparse and dense retrieval concurrently via asyncio.gather —
    both legs kicked off in the same event-loop tick, not one awaited
    before the other starts.

    Returns {"sparse": [...], "dense": [...], "degraded": bool}.
    """
    sparse_top_n = sparse_top_n or settings.sparse_top_n
    dense_top_n = dense_top_n or settings.dense_top_n
    logger = get_logger(trace_id=trace_id)

    sparse_task = _sparse_leg(query, sparse_top_n)
    dense_task = _dense_leg(query, dense_top_n, trace_id)

    # Both tasks are already scheduled as coroutines above; gather runs
    # them concurrently, not sequentially.
    sparse_results, dense_result = await asyncio.gather(sparse_task, dense_task)

    logger.info(
        "retrieval_concurrent_done",
        stage="retrieval",
        sparse_count=len(sparse_results),
        dense_count=len(dense_result["results"]),
        degraded=dense_result["degraded"],
    )

    return {
        "sparse": sparse_results,
        "dense": dense_result["results"],
        "degraded": dense_result["degraded"],
    }