"""
Concurrent sparse+dense retrieval orchestration (FR-7), extended in
Step 17 with routing-retrieval concurrency wiring (FR-14).

NON-PAUSABLE: finish as one unbroken unit. A partially-wired asyncio.gather
(only one leg actually awaited, or Step 11's fallback wrapper accidentally
bypassed) runs without error but returns sequential-not-concurrent or
partial results - surfaces later as an unexplained NFR-2 latency miss.
Same risk class applies to Step 17's routing addition below: a routing
coroutine that's merely awaited after retrieval finishes, instead of
gathered alongside it, will pass functionally but silently violate FR-14.
"""
from __future__ import annotations

import asyncio

from config.settings import settings
from src.common.qdrant_client import get_client
from src.observability.logging_setup import get_logger
from src.retrieval.fallback import embed_query_with_fallback
from src.retrieval.sparse_bm25 import query_bm25
from src.routing.layer0_rules import route_query  # Step 17 addition


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
    Step 12 (FR-7): runs sparse and dense retrieval concurrently via
    asyncio.gather - both legs kicked off in the same event-loop tick,
    not one awaited before the other starts.

    Unchanged from Step 12 - kept as its own callable so Step 12's
    existing tests keep testing exactly this function in isolation.

    Returns {"sparse": [...], "dense": [...], "degraded": bool}.
    """
    sparse_top_n = sparse_top_n or settings.sparse_top_n
    dense_top_n = dense_top_n or settings.dense_top_n
    logger = get_logger(trace_id=trace_id)

    sparse_task = _sparse_leg(query, sparse_top_n)
    dense_task = _dense_leg(query, dense_top_n, trace_id)

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


async def retrieve_and_route_concurrent(
    query: str,
    sparse_top_n: int | None = None,
    dense_top_n: int | None = None,
    trace_id: str = "retrieve_and_route",
) -> dict:
    """
    Step 17 (FR-14): kicks off Step 16's full routing cascade
    (route_query) via asyncio.gather ALONGSIDE retrieve_concurrent(),
    rather than awaiting retrieval first and only then starting routing.

    This is the function Step 21's pipeline wiring should call, not
    retrieve_concurrent() directly - it's a strict superset (retrieval +
    routing, both started concurrently) needed for Step 19's conditional
    rerank, which requires the routing decision to already be available.

    Returns:
        {
            "sparse": [...], "dense": [...], "degraded": bool,  # from retrieval
            "routing": {                                          # from routing
                "path": "fast"|"deep", "score": float,
                "confidence": float|None, "reason": str,
                "deciding_layer": str, "degraded": bool,
            },
        }
    """
    logger = get_logger(trace_id=trace_id)

    # Both coroutines are scheduled here, in the same gather() call -
    # this is the load-bearing line for FR-14. Do NOT await retrieval_task
    # separately before constructing routing_task; that would reintroduce
    # sequential execution while still "looking" concurrent at a glance.
    retrieval_task = retrieve_concurrent(
        query, sparse_top_n=sparse_top_n, dense_top_n=dense_top_n, trace_id=trace_id
    )
    routing_task = route_query(query, trace_id=trace_id)

    retrieval_result, routing_result = await asyncio.gather(retrieval_task, routing_task)

    logger.info(
        "retrieve_and_route_done",
        stage="retrieval_routing",
        routing_path=routing_result["path"],
        routing_deciding_layer=routing_result["deciding_layer"],
        degraded=retrieval_result["degraded"] or routing_result["degraded"],
    )

    return {
        **retrieval_result,
        "routing": routing_result,
    }