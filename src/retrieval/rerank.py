from __future__ import annotations

import asyncio
import time
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from config.settings import settings
from src.observability.logging_setup import get_logger

# Set PyTorch CPU threads
num_threads = min(4, torch.get_num_threads())
torch.set_num_threads(num_threads)

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is not None:
        return _reranker

    try:
        # Claude fix: direct runtime flag, ignores import order
        _reranker = CrossEncoder(
            settings.reranker_model,
            max_length=256,
            device="cpu",
            local_files_only=True,
        )
    except Exception:
        # Fallback if first run and not cached locally
        _reranker = CrossEncoder(
            settings.reranker_model,
            max_length=256,
            device="cpu",
            local_files_only=False,
        )

    return _reranker


def preload_reranker() -> None:
    """Eager load and warmup."""
    model = _get_reranker()
    try:
        model.predict([("warmup query", "warmup text")], show_progress_bar=False)
    except Exception:
        pass


def _candidate_text(candidate: dict[str, Any]) -> str:
    text = candidate.get("text")
    if text is None:
        text = candidate.get("content", "")
    return str(text)[:800]


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
    trace_id: str = "rerank",
) -> list[dict[str, Any]]:
    logger = get_logger(trace_id=trace_id)

    if not candidates:
        return []

    # Limit compute pool to top 10 candidates to guarantee <400ms on CPU
    pool = candidates[:10]
    pairs = [(query, _candidate_text(c)) for c in pool]

    model = _get_reranker()
    start = time.perf_counter()
    scores = model.predict(
        pairs,
        batch_size=len(pairs),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    scored = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(pool, scores)
    ]
    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    result = scored[:top_k] if top_k is not None else scored

    logger.info(
        "rerank_complete",
        stage="rerank",
        num_candidates=len(pool),
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
    return await asyncio.to_thread(rerank, query, candidates, top_k, trace_id)