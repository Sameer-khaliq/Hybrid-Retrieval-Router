"""
Layer 1: LLM-router fallback (FR-13), invoked only for the ambiguous
middle band from layer0_rules.py.

TWO DISTINCT FAILURE BRANCHES (do not collapse into one handler):
  (a) Call returns but fails schema validation -> default FAST-path,
      degraded=True, deciding_layer="malformed-fallback"
      (FR-13's stated default, per NFR-9)
  (b) Call exceeds router_timeout_ms (400ms default) entirely, with or
      without exhausting retries first -> default DEEP-path,
      deciding_layer="timeout-fallback"
      (addendum #2, grounded in Risk Register #2's bias toward deep-path
      under routing uncertainty - a slightly slower correct answer beats
      a fast wrong one in a demo)

The retry/backoff (NFR-8) runs INSIDE the asyncio.wait_for timeout
window, not layered on top of it - if a 429 retry-with-backoff would
blow past the ceiling, the outer wait_for still cancels at exactly
router_timeout_ms.

Uses src.common.resilience.with_retry (Step 11), which takes a zero-arg
callable and raises RetriesExhaustedError once all attempts fail. That
exception is treated the same as an asyncio.TimeoutError below - both
map to the deep-path/timeout-fallback branch, per addendum #2's wording
("exceeds the ceiling, with or without exhausting retries first").
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from groq import AsyncGroq
from pydantic import BaseModel, Field, ValidationError

from config.settings import settings
from src.common.resilience import RetriesExhaustedError, with_retry
from src.observability.logging_setup import get_logger

_groq_client = AsyncGroq(api_key=settings.groq_api_key)

_ROUTER_SYSTEM_PROMPT = """You are a query-routing classifier. Given a user query, decide whether it needs a fast, lightweight model or a deep, more capable model to answer well.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"path": "fast" | "deep", "confidence": <float 0-1>, "reason": "<short reason>"}

"fast" = simple factual lookup, single entity/fact, short answer expected.
"deep" = multi-hop reasoning, comparison, analysis, or synthesis across multiple facts."""


class RouterDecision(BaseModel):
    """FR-13's structured decision schema - the sanctioned Pydantic exception."""
    path: Literal["fast", "deep"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


async def _call_groq_router_raw(query: str) -> str:
    """Single Groq call, temperature 0, per FR-13. Returns raw response text."""
    response = await _groq_client.chat.completions.create(
        model=settings.groq_router_model,
        temperature=0,
        max_tokens=150,
        messages=[
            {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content


async def _invoke_groq_with_retry(query: str) -> str:
    """Retry/backoff (NFR-8, Step 11) wraps the raw call, running inside
    the caller's asyncio.wait_for window so retries never push past the
    router_timeout_ms ceiling. with_retry takes a zero-arg callable, so
    the query is bound via closure rather than passed as an extra arg."""
    return await with_retry(
        lambda: _call_groq_router_raw(query),
        max_retries=settings.max_retries,
    )


def _default_malformed_fallback(reason: str) -> dict:
    return {
        "path": "fast",
        "confidence": 0.0,
        "reason": reason,
        "deciding_layer": "malformed-fallback",
        "degraded": True,
    }


def _default_timeout_fallback(reason: str) -> dict:
    return {
        "path": "deep",
        "confidence": 0.0,
        "reason": reason,
        "deciding_layer": "timeout-fallback",
        "degraded": True,
    }


async def call_llm_router(query: str, trace_id: str = "layer1") -> dict:
    """
    Returns:
        {"path": "fast"|"deep", "confidence": float, "reason": str,
         "deciding_layer": "llm-fallback"|"malformed-fallback"|"timeout-fallback",
         "degraded": bool}
    """
    logger = get_logger(trace_id=trace_id)
    timeout_s = settings.router_timeout_ms / 1000.0

    try:
        raw_text = await asyncio.wait_for(
            _invoke_groq_with_retry(query), timeout=timeout_s
        )
    except TimeoutError:
        logger.warning(
            "router_timeout",
            stage="routing",
            deciding_layer="timeout-fallback",
            timeout_ms=settings.router_timeout_ms,
        )
        return _default_timeout_fallback(
            f"Layer-1 router exceeded {settings.router_timeout_ms}ms ceiling"
        )
    except RetriesExhaustedError as exc:
        # All (max_retries + 1) attempts failed within the timeout window -
        # same failure class as an outright timeout per addendum #2's
        # "with or without exhausting retries first" wording, so this
        # takes the same deep-path default, not a third branch.
        logger.warning(
            "router_retries_exhausted",
            stage="routing",
            deciding_layer="timeout-fallback",
            error=str(exc),
        )
        return _default_timeout_fallback(
            f"Layer-1 router exhausted retries: {exc}"
        )

    try:
        parsed = json.loads(raw_text)
        decision = RouterDecision.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "router_malformed_output",
            stage="routing",
            deciding_layer="malformed-fallback",
            error=str(exc),
        )
        return _default_malformed_fallback(
            f"Layer-1 router returned malformed output: {exc}"
        )

    return {
        "path": decision.path,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "deciding_layer": "llm-fallback",
        "degraded": False,
    }