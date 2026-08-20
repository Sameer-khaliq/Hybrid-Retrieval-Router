"""
Step 22 — Generation: streaming & degraded-mode fallback (FR-26, NFR-9
generation half).

FR-26: streams generation token-by-token via Groq's streaming API, for
BOTH fast- and deep-path generation. Gated responses (Step 14) do NOT
go through this module at all - they return as one atomic response via
generate_atomic() below, per FR-26's explicit carve-out.

NFR-9: on deep-path model failure/rate-limit after Step 11's retries
are exhausted, fall back to the fast-path model rather than erroring,
flagging degraded:true. Same default applies to Layer-1's malformed
router response (already handled in layer1_llm.py) - this is the
generation-side half of that shared policy.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal

from groq import AsyncGroq

from config.settings import settings
from src.common.resilience import RetriesExhaustedError, with_retry
from src.observability.logging_setup import get_logger

_groq_client = AsyncGroq(api_key=settings.groq_api_key)

# Caps concurrent in-flight calls to settings.groq_deep_model. Held for
# the full duration of a deep-path generation (including retries and
# token streaming), released early if/when NFR-9's deep->fast fallback
# kicks in, since the fallback call runs on the uncapped fast model.
_groq_deep_semaphore = asyncio.Semaphore(settings.groq_deep_max_concurrency)

Path = Literal["fast", "deep"]

_GENERATION_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using ONLY the "
    "provided context. If the context doesn't contain enough "
    "information to answer confidently, say so plainly rather than "
    "guessing or using outside knowledge. Each context item is tagged "
    "with its source (corpus or web) - prefer corpus-sourced "
    "information when both are available and don't contradict each "
    "other; note explicitly when you're relying on web-sourced "
    "information for anything time-sensitive."
)


def _build_context_block(chunks: list[dict[str, Any]], max_chunks: int) -> str:
    limited_chunks = chunks[:max_chunks]
    lines = []
    for i, chunk in enumerate(limited_chunks, start=1):
        payload = chunk.get("payload") or {}
        source = payload.get("source") or chunk.get("source", "corpus")
        text = payload.get("text") or chunk.get("text") or chunk.get("content", "")
        text = str(text)[: settings.generation_max_chunk_chars]
        lines.append(f"[{i}] (source: {source}) {text}")
    return "\n\n".join(lines)


def _model_for_path(path: Path) -> str:
    return settings.groq_fast_model if path == "fast" else settings.groq_deep_model


async def _create_stream(model: str, query: str, context_block: str, max_tokens: int = 250):
    """The actual Groq call. Wrapped in with_retry by the caller - this
    function itself stays a plain zero-arg-callable-friendly coroutine,
    matching with_retry's expected signature (Step 11)."""
    tokens_limit = getattr(settings, "generation_max_tokens", max_tokens)
    return await _groq_client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=tokens_limit,
        stream=True,
        messages=[
            {"role": "system", "content": _GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
        ],
    )


async def generate_streaming(
    query: str,
    chunks: list[dict[str, Any]],
    path: Path,
    trace_id: str = "generate",
) -> AsyncIterator[dict]:
    """
    FR-26 + NFR-9 combined entrypoint.

    Yields a sequence of dicts:
        {"delta": str}                                  - one per token
        ... (repeated)
        {"done": True, "model_used": str, "degraded": bool}   - final
    """
    logger = get_logger(trace_id=trace_id)

    max_chunks = (
        getattr(settings, "generation_max_context_chunks_deep", 10)
        if path == "deep"
        else getattr(settings, "generation_max_context_chunks_fast", 5)
    )

    if len(chunks) > max_chunks:
        logger.info(
            "generation_context_truncated",
            stage="generation",
            retrieved_chunks=len(chunks),
            chunks_sent_to_llm=max_chunks,
        )

    context_block = _build_context_block(chunks, max_chunks=max_chunks)
    model = _model_for_path(path)
    degraded = False

    # Dynamic token budgeting based on path (fast vs deep)
    tokens_limit = 200 if path == "fast" else 250

    # Acquire the deep-model concurrency cap before the first attempt.
    holding_deep_semaphore = False
    if model == settings.groq_deep_model:
        await _groq_deep_semaphore.acquire()
        holding_deep_semaphore = True

    try:
        try:
            stream = await with_retry(
                lambda: _create_stream(model, query, context_block, max_tokens=tokens_limit),
                max_retries=settings.max_retries,
            )
        except RetriesExhaustedError as exc:
            if path == "deep":
                logger.warning(
                    "deep_model_unavailable_fallback_to_fast",
                    stage="generation",
                    failed_model=model,
                    error=str(exc),
                )
                if holding_deep_semaphore:
                    _groq_deep_semaphore.release()
                    holding_deep_semaphore = False
                model = settings.groq_fast_model
                degraded = True
                try:
                    stream = await with_retry(
                        lambda: _create_stream(model, query, context_block, max_tokens=150),
                        max_retries=settings.max_retries,
                    )
                except RetriesExhaustedError as exc2:
                    logger.error(
                        "fast_model_also_unavailable_no_further_fallback",
                        stage="generation",
                        error=str(exc2),
                    )
                    yield {"error": "generation_unavailable", "degraded": True}
                    yield {"done": True, "model_used": model, "degraded": True}
                    return
            else:
                logger.error(
                    "fast_model_unavailable_no_fallback",
                    stage="generation",
                    error=str(exc),
                )
                yield {"error": "generation_unavailable", "degraded": True}
                yield {"done": True, "model_used": model, "degraded": True}
                return

        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield {"delta": delta}

        logger.info(
            "generation_complete",
            stage="generation",
            model_used=model,
            degraded=degraded,
        )
        yield {"done": True, "model_used": model, "degraded": degraded}
    finally:
        if holding_deep_semaphore:
            _groq_deep_semaphore.release()


async def generate_atomic(text: str, trace_id: str = "gated_response") -> dict:
    """
    FR-26's explicit carve-out: gated responses (Step 14's FR-21-24
    matches) bypass this module's streaming path entirely and return as
    ONE atomic, non-streamed response - never token-by-token.
    """
    return {"answer": text, "streamed": False, "degraded": False}