"""
RRF fusion — math only (FR-6, first half). Conditional cross-encoder
invocation deferred to Step 19 — deliberately not built here since it
depends on the routing decision, which doesn't exist until Phase 3.
"""
from __future__ import annotations

from config.settings import settings


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