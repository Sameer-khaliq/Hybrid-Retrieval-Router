"""
Step 21 — Full pipeline wiring: gating -> routing/retrieval -> fusion ->
rerank -> Tavily. The first point where every earlier phase's output
actually has to compose into one call path.

Step 24 addition (FR-20, NFR-11, NFR-4 tracking half): this same file
now also assembles latency_breakdown (per-stage ms), pre/post-rerank
chunk IDs, and api_call_count - per the plan's own instruction that
Step 24's consolidation work lives inside run_query.py rather than a
separate module, since this is the one place every stage's timing is
already visible.

api_call_count here covers everything EXCEPT generation - generation
happens in Step 23's API layer (after this function returns), not
inside the pipeline itself, so the API layer adds +1 for it to get the
final NFR-4 count. Embedding is counted as 1 whenever the non-gated
path is taken (the dense leg always attempts an embed call, degraded
or not - a failed-after-retries embed is still one logical API call
under the shared with_retry policy used everywhere else in this
codebase). Router adds +1 only when Layer 1 was actually invoked
(deciding_layer != "rule-based"). Tavily adds +1 only when triggered.

Deliberately kept separate from the HTTP layer (Step 23) so pipeline
logic is testable without spinning up FastAPI.

NON-PAUSABLE per the plan: a half-wired gating short-circuit (some
FR-21-24 categories bypassing correctly, others not) or a routing
decision that's computed but not actually threaded into Step 19's
conditional-rerank call would run end-to-end without erroring while
silently doing the wrong thing. Built and tested as one unbroken chain.

DESIGN DECISION (flagged during Step 20 review, implemented here): if
apply_tavily_fallback() returns empty chunks - meaning corpus retrieval
was empty/low-confidence AND Tavily either wasn't triggered-with-real-
results or failed outright after retries - this pipeline short-circuits
BEFORE calling generation, returning a fixed insufficient-info response
instead. Two reasons: (1) avoids spending a Groq API call (NFR-4
budget) on a call that has nothing to generate from, (2) avoids the
model hallucinating an answer from empty context. This is a genuine gap
in REQUIREMENTS.md/IMPLEMENTATION_PLAN.md - neither document specifies
this exact case - so it's implemented here explicitly rather than left
to whatever generate_streaming() would do with an empty chunk list.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.gating.prefilter import run_prefilter
from src.observability.logging_setup import get_logger
from src.retrieval.fusion import apply_conditional_rerank, rrf_fuse
from src.retrieval.orchestrator import retrieve_and_route_concurrent
from src.retrieval.tavily_fallback import apply_tavily_fallback

INSUFFICIENT_INFO_RESPONSE = (
    "I wasn't able to find enough information to answer that - both the "
    "knowledge base and live web search came back empty or unavailable. "
    "Try rephrasing, or check back shortly."
)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


async def run_query(query: str, trace_id: str | None = None) -> dict[str, Any]:
    """
    Full pipeline for one query. Returns a dict Step 23's API layer
    builds an HTTP response directly from:

        {
            "trace_id": str,
            "gated": bool,
            "answer_text": str | None,    # set for gated / insufficient-info
                                            # cases only; None means Step 22
                                            # should stream from "chunks" instead
            "chunks": list[dict] | None,   # None for gated responses,
                                            # [] for insufficient-info,
                                            # populated otherwise
            "routing": dict | None,        # None only for gated responses -
                                            # gated queries never reach routing
            "tavily_triggered": bool,
            "tavily_trigger_reason": str | None,
            "degraded": bool,
            "latency_breakdown": {stage: ms},   # Step 24 (FR-20/NFR-11)
            "api_call_count": int,              # Step 24 (NFR-4), excludes generation
            "pre_rerank_chunk_ids": list,        # Step 24 (NFR-11)
            "post_rerank_chunk_ids": list,       # Step 24 (NFR-11)
        }
    """
    trace_id = trace_id or str(uuid.uuid4())
    logger = get_logger(trace_id=trace_id)
    latency_breakdown: dict[str, float] = {}
    overall_start = time.perf_counter()

    # --- Step 14: gating, checked first, short-circuits immediately ---
    # ALL FOUR FR-21-24 categories flow through this single call - there
    # is no per-category branch here to accidentally leave one category
    # unwired. run_prefilter() itself owns the ordering/matching; this
    # pipeline only reacts to matched-or-not.
    t0 = time.perf_counter()
    gate_result = run_prefilter(query, trace_id=trace_id)
    latency_breakdown["gating_ms"] = _ms(t0)

    if gate_result is not None:
        logger.info("pipeline_gated", stage="pipeline", category=gate_result["category"])
        latency_breakdown["pipeline_total_ms"] = _ms(overall_start)
        return {
            "trace_id": trace_id,
            "gated": True,
            "answer_text": gate_result["response"],
            "chunks": None,
            "routing": None,
            "tavily_triggered": False,
            "tavily_trigger_reason": None,
            "degraded": False,
            "latency_breakdown": latency_breakdown,
            "api_call_count": 0,
            "pre_rerank_chunk_ids": [],
            "post_rerank_chunk_ids": [],
        }

    # --- Step 17: routing + retrieval, concurrently (FR-14) ---
    t0 = time.perf_counter()
    retrieval_and_routing = await retrieve_and_route_concurrent(query, trace_id=trace_id)
    latency_breakdown["retrieval_and_routing_ms"] = _ms(t0)
    routing = retrieval_and_routing["routing"]

    # --- Step 13: RRF fusion (math only) ---
    t0 = time.perf_counter()
    fused = rrf_fuse(retrieval_and_routing["sparse"], retrieval_and_routing["dense"])
    latency_breakdown["fusion_ms"] = _ms(t0)
    pre_rerank_chunk_ids = [c["chunk_id"] for c in fused]

    # --- Step 19: conditional rerank - routing["path"] threaded through here,
    # not silently dropped, per the non-pausable warning above ---
    t0 = time.perf_counter()
    reranked_result = await apply_conditional_rerank(query, fused, routing, trace_id=trace_id)
    latency_breakdown["rerank_ms"] = _ms(t0)
    post_rerank_chunk_ids = [c["chunk_id"] for c in reranked_result["chunks"]]

    # --- Step 20: Tavily fallback leg ---
    t0 = time.perf_counter()
    tavily_result = await apply_tavily_fallback(query, reranked_result, trace_id=trace_id)
    latency_breakdown["tavily_ms"] = _ms(t0)

    degraded = (
        retrieval_and_routing["degraded"]
        or routing["degraded"]
        or tavily_result["degraded"]
    )

    router_called = routing["deciding_layer"] != "rule-based"
    api_call_count = 1  # embedding - always attempted once gating passes
    if router_called:
        api_call_count += 1
    if tavily_result["tavily_triggered"]:
        api_call_count += 1

    latency_breakdown["pipeline_total_ms"] = _ms(overall_start)

    if not tavily_result["chunks"]:
        logger.warning(
            "pipeline_insufficient_context",
            stage="pipeline",
            tavily_triggered=tavily_result["tavily_triggered"],
        )
        return {
            "trace_id": trace_id,
            "gated": False,
            "answer_text": INSUFFICIENT_INFO_RESPONSE,
            "chunks": [],
            "routing": routing,
            "tavily_triggered": tavily_result["tavily_triggered"],
            "tavily_trigger_reason": tavily_result["tavily_trigger_reason"],
            "degraded": True,
            "latency_breakdown": latency_breakdown,
            "api_call_count": api_call_count,
            "pre_rerank_chunk_ids": pre_rerank_chunk_ids,
            "post_rerank_chunk_ids": post_rerank_chunk_ids,
        }

    logger.info(
        "pipeline_ready_for_generation",
        stage="pipeline",
        routing_path=routing["path"],
        num_chunks=len(tavily_result["chunks"]),
        tavily_triggered=tavily_result["tavily_triggered"],
        degraded=degraded,
        api_call_count=api_call_count,
    )

    return {
        "trace_id": trace_id,
        "gated": False,
        "answer_text": None,
        "chunks": tavily_result["chunks"],
        "routing": routing,
        "tavily_triggered": tavily_result["tavily_triggered"],
        "tavily_trigger_reason": tavily_result["tavily_trigger_reason"],
        "degraded": degraded,
        "latency_breakdown": latency_breakdown,
        "api_call_count": api_call_count,
        "pre_rerank_chunk_ids": pre_rerank_chunk_ids,
        "post_rerank_chunk_ids": post_rerank_chunk_ids,
    }