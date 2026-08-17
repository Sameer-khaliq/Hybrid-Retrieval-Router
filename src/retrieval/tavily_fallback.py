"""
Step 20 — Tavily fallback leg (FR-9).

Triggers on either:
  (a) Step 19's fused/reranked top-1 score falling below
      settings.tavily_confidence_floor, or
  (b) a cheap rule-based "needs current/external info" check
      (keyword/pattern signals - deliberately NOT a dedicated LLM call,
      to protect NFR-4's <=4-call budget).

Tags results "web" vs "corpus", reuses resilience.py (Step 11) for
retry/backoff.

DESIGN NOTE (addendum #3, reinforced per the plan's own repeated
warning): this trigger logic reads Step 19's already-computed
fusion/rerank confidence. It must NOT re-derive or share the
domain-relevance signal from Step 14's FR-24 gate. A query reaching
this module has, by construction, already passed FR-24's pre-retrieval
gate - or it would have short-circuited before retrieval ever ran.
FR-9 firing here is strictly a post-retrieval confidence/currency
decision, never a re-litigation of FR-24's relevance gate. This module
therefore takes no dependency on src.gating.prefilter at all - not even
an import - to make that independence structurally enforced, not just
documented.
"""

from __future__ import annotations

import re
from typing import Any

from tavily import AsyncTavilyClient

from config.settings import settings
from src.common.resilience import RetriesExhaustedError, with_retry
from src.observability.logging_setup import get_logger

_tavily_client: AsyncTavilyClient | None = None


def _get_tavily_client() -> AsyncTavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = AsyncTavilyClient(api_key=settings.tavily_api_key)
    return _tavily_client


# Cheap rule-based "needs current/external info" signal (addendum, Sequencing
# Note #3 in the implementation plan: filled gap, no LLM call, protects
# NFR-4's <=4-call budget). Deliberately conservative - false negatives here
# just mean a currency-sensitive query gets answered from a possibly-stale
# corpus, which is a quality issue, not a correctness one; false positives
# cost one extra Tavily call, which is within NFR-4's budget either way.
_CURRENCY_TRIGGER_PATTERN = re.compile(
    r"\b("
    r"today|tonight|this week|this month|this year|"
    r"currently|current(ly)?|right now|"
    r"latest|newest|most recent|recent(ly)?|"
    r"as of \d{4}|in \d{4}\b|"
    r"yesterday|tomorrow"
    r")\b",
    re.IGNORECASE,
)


def needs_current_info(query: str) -> bool:
    """
    Rule-based currency check (FR-9's second trigger). Pure regex, no
    I/O, no LLM call - matches Step 14's gating philosophy but is a
    structurally separate function/module, per the addendum #3 warning
    against sharing state with FR-24's gate.
    """
    return bool(_CURRENCY_TRIGGER_PATTERN.search(query))


def is_low_confidence(reranked_result: dict) -> bool:
    """
    FR-9's first trigger: fused/reranked top-1 score below the
    configured floor. reranked_result is apply_conditional_rerank()'s
    (Step 19) return dict - {"chunks": [...], "reranked": bool, ...}.

    Reads whichever score field is actually present on the top chunk:
    "rerank_score" if the cross-encoder ran, otherwise "rrf_score" (the
    fast-path-skip case from Step 19, where chunks are still in
    rrf_score order, un-reranked).
    """
    chunks = reranked_result.get("chunks") or []
    if not chunks:
        return True  # no results at all is definitionally low-confidence

    top = chunks[0]
    top_score = top.get("rerank_score", top.get("rrf_score", 0.0))
    return top_score < settings.tavily_confidence_floor


async def _search_tavily_raw(query: str) -> list[dict[str, Any]]:
    client = _get_tavily_client()
    response = await client.search(query=query, max_results=settings.dense_top_n)
    return response.get("results", [])


async def _search_tavily_with_retry(query: str) -> list[dict[str, Any]]:
    """Retry/backoff (NFR-8, Step 11) wraps the raw Tavily call. Unlike
    Step 16's router call, there's no hard wall-clock ceiling specified
    for Tavily in the requirements - settings.max_retries governs
    attempt count, same shared policy as every other external call."""
    return await with_retry(
        lambda: _search_tavily_raw(query),
        max_retries=settings.max_retries,
    )


def _tag_source(items: list[dict], source: str) -> list[dict]:
    return [{**item, "source": source} for item in items]


async def apply_tavily_fallback(
    query: str,
    reranked_result: dict,
    trace_id: str = "tavily",
) -> dict:
    """
    FR-9's entrypoint. Call this after Step 19's apply_conditional_rerank().

    Decides whether to invoke Tavily based on (a) low corpus confidence
    or (b) a currency-trigger match in the query text - either condition
    is sufficient, independently of the other.

    Returns:
        {
            "chunks": [...],       # corpus chunks, tagged source="corpus",
                                    # PLUS Tavily results tagged source="web"
                                    # if triggered (corpus chunks always kept -
                                    # FR-9 supplements, doesn't replace)
            "tavily_triggered": bool,
            "tavily_trigger_reason": "low_confidence" | "needs_current_info"
                                      | "both" | None,
            "degraded": bool,      # True if Tavily was triggered but failed
                                    # after retries - corpus-only result
                                    # returned rather than erroring
        }
    """
    logger = get_logger(trace_id=trace_id)

    corpus_chunks = _tag_source(reranked_result.get("chunks") or [], "corpus")

    low_conf = is_low_confidence(reranked_result)
    needs_current = needs_current_info(query)

    if not low_conf and not needs_current:
        return {
            "chunks": corpus_chunks,
            "tavily_triggered": False,
            "tavily_trigger_reason": None,
            "degraded": False,
        }

    reason = "both" if (low_conf and needs_current) else (
        "low_confidence" if low_conf else "needs_current_info"
    )

    try:
        web_results_raw = await _search_tavily_with_retry(query)
    except RetriesExhaustedError as exc:
        logger.warning(
            "tavily_retries_exhausted",
            stage="tavily_fallback",
            trigger_reason=reason,
            error=str(exc),
        )
        return {
            "chunks": corpus_chunks,
            "tavily_triggered": True,
            "tavily_trigger_reason": reason,
            "degraded": True,
        }

    web_chunks = _tag_source(web_results_raw, "web")

    logger.info(
        "tavily_fallback_triggered",
        stage="tavily_fallback",
        trigger_reason=reason,
        num_web_results=len(web_chunks),
    )

    return {
        "chunks": corpus_chunks + web_chunks,
        "tavily_triggered": True,
        "tavily_trigger_reason": reason,
        "degraded": False,
    }