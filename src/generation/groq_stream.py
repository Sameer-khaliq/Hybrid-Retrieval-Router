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

DESIGN NOTE - why fallback happens at stream-creation, not mid-stream:
the retry/fallback logic wraps only the `.create(..., stream=True)`
call itself, not the token-by-token consumption after it. Groq's
client raises on the create() call for auth/rate-limit/5xx errors
before any tokens are delivered - that's the natural retry boundary.
Retrying or falling back MID-stream (after some tokens have already
reached the client) isn't attempted: there's no clean way to "retry" a
partially-delivered stream without either duplicating already-sent
tokens or confusing the client with a truncated-then-restarted answer.
This also means no separate "probe" call is needed before streaming -
the real generation call IS the reachability check, so NFR-4's
per-query API-call budget isn't spent on a call that does nothing but
check if another call would work.

CONCURRENCY CAP (429 fix, part 1): once the reranker's event-loop-
blocking bug (rerank.py) is fixed, deep-path requests are no longer
artificially serialized by a starved event loop, so they can
genuinely fire concurrently. _groq_deep_semaphore below caps
concurrent in-flight groq_deep_model calls to
settings.groq_deep_max_concurrency as a safety net. Only the deep
model is capped - fast-path (which shares gpt-oss-20b with the
Layer-1 router) hasn't shown 429s and isn't capped here.

CONTEXT-SIZE CAP (429 fix, part 2 - the actual root cause): Groq's
free-tier TPM ceiling for both gpt-oss-120b and gpt-oss-20b is 8K
tokens/minute. A full 15-chunk deep-path prompt runs ~4-6K tokens on
its own, so 1-2 back-to-back deep-path requests exhaust the per-minute
budget regardless of concurrency - the semaphore above limits how many
requests are in flight at once, but does nothing about how large each
one is. _build_context_block() below caps the number of chunks it puts
in the prompt (settings.generation_max_context_chunks) and truncates
each chunk's text (settings.generation_max_chunk_chars). This is
generation-context-only: the reranker still scores and returns the
full settings.rerank_top_k (15) candidates for deep-path, so NFR-5
(Recall@10) is unaffected - only what actually gets sent to Groq
shrinks.
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


def _build_context_block(chunks: list[dict[str, Any]]) -> str:
    """
    Assemble retrieved/reranked (+ optional Tavily web) chunks into a
    single context string for the generation prompt. Each chunk should
    carry a "source" tag ("corpus" | "web") from Step 19/20's tagging -
    defaults to "corpus" if absent so this doesn't hard-fail on chunks
    built before that tagging was wired in (e.g. direct unit tests).

    Capped to settings.generation_max_context_chunks chunks, each
    truncated to settings.generation_max_chunk_chars characters. This
    is deliberately generation-only truncation - the caller's full
    `chunks` list (e.g. all 15 reranked deep-path candidates) is still
    used as-is for `sources` in the API response and for NFR-5's Recall
    measurement; only what actually goes into the Groq prompt shrinks,
    to stay inside the 8K TPM/minute free-tier ceiling.
    """
    limited_chunks = chunks[: settings.generation_max_context_chunks]
    lines = []
    for i, chunk in enumerate(limited_chunks, start=1):
        source = chunk.get("source", "corpus")
        text = chunk.get("text") or chunk.get("content", "")
        text = str(text)[: settings.generation_max_chunk_chars]
        lines.append(f"[{i}] (source: {source}) {text}")
    return "\n\n".join(lines)


def _model_for_path(path: Path) -> str:
    return settings.groq_fast_model if path == "fast" else settings.groq_deep_model


async def _create_stream(model: str, query: str, context_block: str):
    """The actual Groq call. Wrapped in with_retry by the caller - this
    function itself stays a plain zero-arg-callable-friendly coroutine,
    matching with_retry's expected signature (Step 11)."""
    return await _groq_client.chat.completions.create(
        model=model,
        temperature=0.2,
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

    On total failure (fast-path model also exhausted after a deep-path
    fallback, or fast-path itself failing outright with no path was
    ever "deep" to fall back from) - see the docstring note below on
    NFR-9's stated scope. This yields an {"error": ...} sentinel before
    the final {"done": ...} rather than raising, so Step 23's API layer
    always has something to build an HTTP response from (never an
    uncaught exception reaching the client).
    """
    logger = get_logger(trace_id=trace_id)
    if len(chunks) > settings.generation_max_context_chunks:
        logger.info(
            "generation_context_truncated",
            stage="generation",
            retrieved_chunks=len(chunks),
            chunks_sent_to_llm=settings.generation_max_context_chunks,
        )
    context_block = _build_context_block(chunks)
    model = _model_for_path(path)
    degraded = False

    # Acquire the deep-model concurrency cap before the first attempt.
    # Held through retries; released early below if we fall back to the
    # fast model, and always released in the outer finally (including on
    # early generator close, e.g. client disconnect mid-stream).
    holding_deep_semaphore = False
    if model == settings.groq_deep_model:
        await _groq_deep_semaphore.acquire()
        holding_deep_semaphore = True

    try:
        try:
            stream = await with_retry(
                lambda: _create_stream(model, query, context_block),
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
                    # Falling back to the fast model - release the deep
                    # slot now rather than holding it for the duration
                    # of a call that no longer touches the deep model.
                    _groq_deep_semaphore.release()
                    holding_deep_semaphore = False
                model = settings.groq_fast_model
                degraded = True
                try:
                    stream = await with_retry(
                        lambda: _create_stream(model, query, context_block),
                        max_retries=settings.max_retries,
                    )
                except RetriesExhaustedError as exc2:
                    # NFR-9 only specifies deep -> fast fallback. It doesn't
                    # define a further tier for "fast-path model itself is
                    # down" - there is no lower tier to fall back to. This
                    # is a genuine gap in the requirements doc, not
                    # something I'm inventing a default for silently: flag
                    # it back to the caller as an explicit error sentinel
                    # rather than picking an undocumented behavior.
                    logger.error(
                        "fast_model_also_unavailable_no_further_fallback",
                        stage="generation",
                        error=str(exc2),
                    )
                    yield {"error": "generation_unavailable", "degraded": True}
                    yield {"done": True, "model_used": model, "degraded": True}
                    return
            else:
                # path == "fast" failing outright - same "no lower tier"
                # gap as above, just reached directly instead of via a
                # deep-path fallback attempt first.
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

    Exists so Step 23's API layer has one consistent response shape to
    build from regardless of whether the query was gated or went
    through full generation - the HTTP layer shouldn't need two
    different response-construction code paths.
    """
    return {"answer": text, "streamed": False, "degraded": False}