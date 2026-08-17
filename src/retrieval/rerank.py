"""
Step 18 — Cross-encoder reranker (FR-10).

Scores each (query, candidate) pair independently with a local
sentence-transformers cross-encoder (ms-marco-MiniLM-L-6 class model,
per REQUIREMENTS.md §0.2 — CPU-bound, zero API cost, no rate limits).

Deliberately standalone: this module knows nothing about routing or
fusion. Step 19 decides *whether* and *how much* (top-K vs top-K') to
call this; this module only answers "given these candidates, what's
the relevance order."

Candidate shape expected in / out: any dict with at least a "text" (or
"content") key. All other keys (chunk_id, source metadata, prior
rrf_score, etc.) are passed through untouched. A "rerank_score" key is
added to each returned dict, and the list is sorted descending by it.

The underlying CrossEncoder.predict() call is synchronous/CPU-bound.
Since Step 21+ wires this into an async pipeline, rerank_async() below
offloads the blocking call via asyncio.to_thread so it doesn't block
the event loop (and, by extension, the fast-path branches that skip
reranking entirely and shouldn't be waiting behind a deep-path rerank
running on the same thread).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from sentence_transformers import CrossEncoder

from config.settings import settings
from src.observability.logging_setup import get_logger

# Module-level lazy singleton. Loading a CrossEncoder involves reading
# model weights off disk (or downloading on first run) - do this once
# per process, not once per request.
_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def _candidate_text(candidate: dict[str, Any]) -> str:
    """Candidates may key their chunk text as 'text' or 'content'
    depending on what upstream (fusion.py) settled on. Support both
    rather than guessing wrong and silently scoring empty strings."""
    text = candidate.get("text")
    if text is None:
        text = candidate.get("content", "")
    return text


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
    trace_id: str = "rerank",
) -> list[dict[str, Any]]:
    """
    FR-10: score each (query, candidate) pair independently, return
    candidates ordered by descending relevance score.

    top_k: if given, truncate to the top_k highest-scoring candidates
    after sorting (Step 19 passes settings.rerank_top_k or
    settings.rerank_top_k_fast here depending on routing path). If
    None, all candidates are returned, sorted.

    Synchronous - use rerank_async() from async call sites (Step 19+).
    """
    logger = get_logger(trace_id=trace_id)

    if not candidates:
        return []

    pairs = [(query, _candidate_text(c)) for c in candidates]

    model = _get_reranker()
    start = time.perf_counter()
    scores = model.predict(pairs)
    elapsed_ms = (time.perf_counter() - start) * 1000

    scored = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    result = scored[:top_k] if top_k is not None else scored

    logger.info(
        "rerank_complete",
        stage="rerank",
        num_candidates=len(candidates),
        top_k=top_k,
        latency_ms=round(elapsed_ms, 2),
    )
    return result


async def rerank_async(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
    trace_id: str = "rerank",
) -> list[dict[str, Any]]:
    """Async wrapper - offloads the blocking CrossEncoder.predict() call
    to a worker thread so it doesn't stall the event loop. This is the
    entrypoint Step 19's conditional-rerank wiring should call."""
    return await asyncio.to_thread(rerank, query, candidates, top_k, trace_id)