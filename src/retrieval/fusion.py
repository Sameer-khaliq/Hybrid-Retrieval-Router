"""
RRF fusion (FR-6, first half — Step 13) + conditional cross-encoder
rerank wiring (FR-6, second half — Step 19).

rrf_fuse() below is unchanged from Step 13. apply_conditional_rerank()
is the new Step 19 piece: given a fused list and Step 16/17's routing
decision, deep-path reranks over top-K (default 15); fast-path either
skips reranking or applies it to a reduced top-K' (default 5), per
settings.rerank_fast_path_mode.
"""
from __future__ import annotations

from typing import Any

from config.settings import settings
from src.common.qdrant_client import get_client
from src.observability.logging_setup import get_logger
from src.retrieval.rerank import rerank_async


def rrf_fuse(sparse_ranked: list[dict], dense_ranked: list[dict], k: int | None = None) -> list[dict]:
    """
    sparse_ranked, dense_ranked: lists of {"chunk_id": ..., "score": ..., "payload"?: ...},
    already sorted descending by score (rank = position in list, 0-indexed).

    score = sum over lists containing chunk_id of 1 / (k + rank + 1)
    (rank+1 so the top rank contributes 1/(k+1), not 1/k — standard RRF convention)

    Returns chunks sorted descending by fused score:
    [{"chunk_id": ..., "rrf_score": ..., "payload": ...}, ...]
    """
    k = k if k is not None else settings.rrf_k
    scores: dict = {}
    payloads: dict = {}

    for rank, item in enumerate(sparse_ranked):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if item.get("payload"):
            payloads[cid] = item["payload"]

    for rank, item in enumerate(dense_ranked):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        if item.get("payload"):
            payloads[cid] = item["payload"]

    fused = [
        {"chunk_id": cid, "rrf_score": score, "payload": payloads.get(cid)}
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def _payload_text(payload: dict[str, Any] | None, chunk_id: Any = None) -> str:
    """
    Extract chunk text from a fused item's payload for reranker input.

    Confirmed against src/ingestion/pipeline.py's embed_and_upsert():
    payload = {**chunk, "embedding_config": ...}, and chunk always
    carries "text" straight from chunking.py (Step 5). So "text" is the
    correct, single key to read here - no fallback chain needed for the
    dense/Qdrant-sourced path.

    NOT YET VERIFIED: whether sparse (BM25)-only matches carry the same
    full payload, or just {chunk_id, score}. If sparse_bm25.py's
    retrieval path doesn't attach payload for BM25-only hits (e.g. under
    Step 11's FR-8 degraded sparse-only fallback), this branch will fire
    for those candidates. Rather than crash the whole query on a missing
    key, this logs a visible warning and returns "" so the reranker
    scores it low rather than the request erroring - but a warning
    appearing in practice means something upstream should be fixed, not
    silently tolerated forever.
    """
    if payload and "text" in payload:
        return payload["text"]
    logger = get_logger(trace_id="fusion")
    logger.warning("missing_payload_text", stage="rerank", chunk_id=chunk_id)
    return ""


def _backfill_missing_payloads(items: list[dict], trace_id: str = "fusion") -> list[dict]:
    """
    sparse_bm25.py's query_bm25() returns only {"chunk_id", "score"} - no
    payload at all. A chunk that appears in sparse_ranked but NOT in
    dense_ranked (BM25 surfaced it, dense search didn't) therefore has
    payload=None after rrf_fuse(), even though the chunk's full payload
    genuinely exists in Qdrant - it just wasn't attached at retrieval
    time on the sparse leg.

    This is exactly the "sparse-only match missing payload" case flagged
    as unverified in _payload_text()'s docstring - now confirmed real,
    not hypothetical. Without this backfill, sparse-only chunks would
    silently rerank as empty text (or, worse, reach generation with no
    actual content), quietly defeating BM25's exact-match contribution
    to the hybrid result.

    Only backfills for the items actually passed in (call this AFTER
    slicing to top_k, not on the full fused list) - keeps the extra
    Qdrant round-trip cheap and scoped to what's actually used
    downstream.
    """
    missing_ids = [item["chunk_id"] for item in items if not item.get("payload")]
    if not missing_ids:
        return items

    logger = get_logger(trace_id=trace_id)
    client = get_client()
    points = client.retrieve(
        collection_name=settings.qdrant_collection, ids=missing_ids, with_payload=True
    )
    payload_by_id = {p.id: p.payload for p in points}

    logger.info(
        "backfilled_missing_payloads",
        stage="rerank",
        num_missing=len(missing_ids),
        num_recovered=len(payload_by_id),
    )

    return [
        {**item, "payload": item.get("payload") or payload_by_id.get(item["chunk_id"])}
        for item in items
    ]


def _to_rerank_candidates(fused_slice: list[dict]) -> list[dict[str, Any]]:
    """Adapt rrf_fuse()'s {"chunk_id", "rrf_score", "payload"} shape into
    what src.retrieval.rerank expects: dicts carrying a "text" key,
    with everything else passed through untouched."""
    return [
        {
            "chunk_id": item["chunk_id"],
            "text": _payload_text(item.get("payload"), chunk_id=item["chunk_id"]),
            "rrf_score": item["rrf_score"],
            "payload": item.get("payload"),
        }
        for item in fused_slice
    ]


async def apply_conditional_rerank(
    query: str,
    fused: list[dict],
    routing_decision: dict,
    trace_id: str = "fusion",
) -> dict:
    """
    FR-6's conditional-rerank half, completing what Step 13 deferred.

    routing_decision: the dict returned by layer0_rules.route_query()
    (Step 16/17) — only routing_decision["path"] ("fast" | "deep") is
    read here; the rest of that dict is Step 17/24's concern, not ours.

    Deep-path: rerank the top settings.rerank_top_k (default 15) fused
    candidates, reordering them by cross-encoder relevance.

    Fast-path: behavior depends on settings.rerank_fast_path_mode:
      - "skip"    -> no reranker call at all, just the top
                     settings.rerank_top_k_fast fused-order candidates
                     (FR-6's "reranking SHALL be skipped" branch)
      - "reduced" -> reranker IS called, but only over the smaller
                     settings.rerank_top_k_fast candidate set
                     (FR-6's "applied to a reduced top-K'" branch)
    Both are valid per FR-6's wording ("skipped or applied to a reduced
    top-K', per config") — settings.rerank_fast_path_mode is the config
    switch. Defaults to "reduced" since a config value can't be left
    genuinely undefined; adjust in settings.py if you want fast-path to
    default to zero rerank calls instead.

    Returns:
        {
            "chunks": [...],          # final ordered candidate list
            "reranked": bool,         # whether the cross-encoder ran
            "rerank_top_k": int|None, # K actually used, None if skipped
            "path": "fast" | "deep",
        }
    """
    logger = get_logger(trace_id=trace_id)
    path = routing_decision["path"]

    if path == "deep":
        top_k = settings.rerank_top_k
        sliced = _backfill_missing_payloads(fused[:top_k], trace_id=trace_id)
        candidates = _to_rerank_candidates(sliced)
        reranked_chunks = await rerank_async(query, candidates, top_k=top_k, trace_id=trace_id)
        logger.info(
            "conditional_rerank",
            stage="rerank",
            path=path,
            reranked=True,
            top_k=top_k,
        )
        return {"chunks": reranked_chunks, "reranked": True, "rerank_top_k": top_k, "path": path}

    # path == "fast"
    top_k = settings.rerank_top_k_fast

    if settings.rerank_fast_path_mode == "skip":
        chunks = _backfill_missing_payloads(fused[:top_k], trace_id=trace_id)
        logger.info(
            "conditional_rerank",
            stage="rerank",
            path=path,
            reranked=False,
            top_k=top_k,
        )
        return {"chunks": chunks, "reranked": False, "rerank_top_k": None, "path": path}

    # "reduced" mode - still reranks, just over a smaller K
    sliced = _backfill_missing_payloads(fused[:top_k], trace_id=trace_id)
    candidates = _to_rerank_candidates(sliced)
    reranked_chunks = await rerank_async(query, candidates, top_k=top_k, trace_id=trace_id)
    logger.info(
        "conditional_rerank",
        stage="rerank",
        path=path,
        reranked=True,
        top_k=top_k,
    )
    return {"chunks": reranked_chunks, "reranked": True, "rerank_top_k": top_k, "path": path}