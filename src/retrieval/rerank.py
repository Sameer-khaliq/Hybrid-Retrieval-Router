from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Any

import torch
from sentence_transformers import CrossEncoder

from config.settings import settings
from src.observability.logging_setup import get_logger

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is not None:
        return _reranker

    # Enforce efficient thread execution on CPU
    torch.set_num_threads(min(4, torch.get_num_threads()))

    try:
        _reranker = CrossEncoder(
            settings.reranker_model,
            max_length=256,
            device="cpu",
            local_files_only=True,
        )
    except Exception:
        _reranker = CrossEncoder(
            settings.reranker_model,
            max_length=256,
            device="cpu",
            local_files_only=False,
        )

    # Disable gradient tracking permanently for inference
    if hasattr(_reranker, "model"):
        _reranker.model.eval()

    return _reranker


def preload_reranker() -> None:
    """Eager load and warmup."""
    model = _get_reranker()
    try:
        with torch.inference_mode():
            model.predict([("warmup query", "warmup text")], show_progress_bar=False)
    except Exception:
        pass

def _candidate_text(candidate: dict[str, Any]) -> str:
    payload = candidate.get("payload") or {}
    text = payload.get("text") or candidate.get("text") or candidate.get("content", "")
    return str(text)[:400]


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
    trace_id: str = "rerank",
) -> list[dict[str, Any]]:
    logger = get_logger(trace_id=trace_id)

    if not candidates:
        return []

    pool_size = max(top_k or 0, settings.rerank_candidate_pool)
    pool = candidates[:pool_size]

    pairs = [(query, _candidate_text(c)) for c in pool]

    model = _get_reranker()
    start = time.perf_counter()

    try:
        # torch.inference_mode disables autograd & overhead entirely
        with torch.inference_mode():
            scores = model.predict(
                pairs,
                batch_size=len(pairs),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
    except Exception as e:
        logger.warning("rerank_inference_failed", stage="rerank", error=str(e))
        # Graceful fallback: return un-reranked pool
        return pool[:top_k] if top_k is not None else pool

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


_rerank_queue: queue.Queue = queue.Queue()
_worker_started = False


def _rerank_worker() -> None:
    while True:
        func, args, result_future, loop = _rerank_queue.get()
        try:
            result = func(*args)
            if not result_future.cancelled():
                loop.call_soon_threadsafe(result_future.set_result, result)
        except Exception as e:
            if not result_future.cancelled():
                loop.call_soon_threadsafe(result_future.set_exception, e)
        finally:
            _rerank_queue.task_done()


def _ensure_worker() -> None:
    global _worker_started
    if not _worker_started:
        thread = threading.Thread(target=_rerank_worker, daemon=True, name="RerankWorker")
        thread.start()
        _worker_started = True


async def rerank_async(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
    trace_id: str = "rerank",
) -> list[dict[str, Any]]:
    _ensure_worker()
    loop = asyncio.get_running_loop()
    result_future: asyncio.Future = loop.create_future()

    _rerank_queue.put((rerank, (query, candidates, top_k, trace_id), result_future, loop))

    try:
        return await asyncio.wait_for(result_future, timeout=settings.rerank_timeout_s)
    except TimeoutError:
        logger = get_logger(trace_id=trace_id)
        logger.warning(
            "rerank_timeout_fallback",
            stage="rerank",
            detail="Exceeded timeout cap, skipping rerank",
        )
        return candidates[:top_k] if top_k is not None else candidates