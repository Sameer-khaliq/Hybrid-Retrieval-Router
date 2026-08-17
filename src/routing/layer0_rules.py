"""
Two-layer routing cascade: Layer 0 thresholds (FR-12) + cascade entrypoint.

Layer 0 resolves the confident majority (score < tau_low -> fast,
score > tau_high -> deep) at ~0ms with zero I/O. The ambiguous middle
band (tau_low <= score <= tau_high) falls through to Layer 1 (layer1_llm.py).

route_query() below is the single entrypoint Step 17 wires into
orchestrator.py's asyncio.gather alongside retrieval (FR-14) - it owns
the full cascade (score -> layer0 -> layer1-if-needed) so the concurrency
wiring in Step 17 only has to gather one coroutine for "the routing side."
"""

from __future__ import annotations

from typing import Literal, Optional

from config.settings import settings
from src.observability.logging_setup import get_logger
from src.routing.features import compute_complexity_score
from src.routing.layer1_llm import call_llm_router

Path = Literal["fast", "deep"]


def route_layer0(score: float) -> Optional[Path]:
    """
    FR-12: score < tau_low -> fast, score > tau_high -> deep,
    tau_low <= score <= tau_high -> None (defer to Layer 1).

    Boundary values: a score exactly equal to tau_low or tau_high falls
    into the middle band (Layer 1), not either bucket - the comparisons
    below are strict on both ends by design, matching FR-12's stated
    "score < tau_low" / "score > tau_high" wording.
    """
    if score < settings.tau_low:
        return "fast"
    if score > settings.tau_high:
        return "deep"
    return None


async def route_query(query: str, trace_id: str = "route") -> dict:
    """
    Full routing cascade for one query. Returns:
        {
            "path": "fast" | "deep",
            "score": float,
            "confidence": float | None,   # None for Layer 0 decisions
            "reason": str,
            "deciding_layer": "rule-based" | "llm-fallback"
                               | "malformed-fallback" | "timeout-fallback",
            "degraded": bool,
        }

    FR-14: this coroutine is designed to be scheduled via asyncio.gather
    alongside retrieve_concurrent() in orchestrator.py (Step 17), not
    awaited sequentially after retrieval completes.
    """
    logger = get_logger(trace_id=trace_id)
    score = compute_complexity_score(query)

    layer0_path = route_layer0(score)
    if layer0_path is not None:
        result = {
            "path": layer0_path,
            "score": score,
            "confidence": None,
            "reason": (
                f"rule-based: score={score:.3f} "
                f"{'<' if layer0_path == 'fast' else '>'} "
                f"{'tau_low' if layer0_path == 'fast' else 'tau_high'}"
            ),
            "deciding_layer": "rule-based",
            "degraded": False,
        }
        logger.info(
            "routing_decision",
            stage="routing",
            path=result["path"],
            score=score,
            deciding_layer="rule-based",
        )
        return result

    # Middle band - defer to Layer 1 (LLM fallback, FR-13).
    llm_result = await call_llm_router(query, trace_id=trace_id)
    result = {
        "path": llm_result["path"],
        "score": score,
        "confidence": llm_result["confidence"],
        "reason": llm_result["reason"],
        "deciding_layer": llm_result["deciding_layer"],
        "degraded": llm_result["degraded"],
    }
    logger.info(
        "routing_decision",
        stage="routing",
        path=result["path"],
        score=score,
        deciding_layer=result["deciding_layer"],
        degraded=result["degraded"],
    )
    return result