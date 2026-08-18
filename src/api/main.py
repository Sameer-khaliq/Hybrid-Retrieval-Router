"""
Step 23 — FastAPI surface (FR-15, FR-16, FR-17).

POST /query: input validation happens at the pydantic layer (schemas.py)
before this route body ever runs - FR-17's "zero downstream calls on
invalid input" is structural, not a runtime check inside the handler.
Distinct from Step 14's content-based gating (FR-21-24), which DOES run
inside the pipeline (run_query -> run_prefilter) - that's about query
INTENT, this is about query FORM (empty/oversized).

Response shape: gated (FR-21-24) and insufficient-info (Step 21's own
empty-context guard) responses return as ONE atomic JSON body, per
FR-26's explicit carve-out for gated responses. Everything else streams
as NDJSON: N lines of {"delta": str}, then one final line matching
FR-15's full documented schema (see schemas.QueryResponse).

GET /health: reachability probes for Qdrant/Groq/Gemini/Tavily (FR-16).

ASSUMPTIONS FLAGGED, NOT VERIFIED: the exact lightweight "are you up"
call for each of Groq/Gemini/Tavily's client libraries wasn't confirmed
against your installed SDK versions (I don't have those client wrapper
files). Each check below is wrapped in a broad try/except so a wrong
method name degrades that one dependency to "unreachable" rather than
crashing /health entirely - but verify these against your actual
installed `groq`, `google-genai`, and `tavily-python` versions before
trusting /health's output in the demo.
"""
from __future__ import annotations

import json
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from config.settings import settings
from src.api.schemas import HealthResponse, QueryRequest
from src.common.qdrant_client import get_client
from src.generation.groq_stream import generate_atomic, generate_streaming
from src.observability.logging_setup import get_logger
from src.pipeline.run_query import run_query

app = FastAPI(title="Hybrid Retrieval & Query Routing System")


def _routing_metadata(routing: dict | None) -> dict:
    if routing is None:
        return {"path": None, "confidence": None, "reason": None}
    return {
        "path": routing["path"],
        "confidence": routing.get("confidence"),
        "reason": routing.get("reason"),
    }


def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id": c.get("chunk_id"),
            "source": c.get("source", "corpus"),
            "score": c.get("rerank_score", c.get("rrf_score")),
        }
        for c in chunks
    ]


@app.post("/query")
async def query_endpoint(payload: QueryRequest):
    """
    FR-15: POST /query. FR-17's validation already happened at the
    pydantic layer before this function body runs.
    """
    trace_id = str(uuid.uuid4())
    pipeline_result = await run_query(payload.query, trace_id=trace_id)

    # --- Atomic path: gated (FR-21-24) or insufficient-info (Step 21) ---
    # FR-26's explicit carve-out - neither of these goes through
    # generate_streaming() at all.
    if pipeline_result["answer_text"] is not None:
        atomic = await generate_atomic(pipeline_result["answer_text"], trace_id=trace_id)
        body = {
            "answer": atomic["answer"],
            "sources": [],
            "routing_metadata": _routing_metadata(pipeline_result["routing"]),
            "latency_breakdown": pipeline_result["latency_breakdown"],
            "degraded": pipeline_result["degraded"],
            "trace_id": trace_id,
            "api_call_count": pipeline_result["api_call_count"],
        }
        return JSONResponse(content=body)

    # --- Streaming path: real generation, FR-26 ---
    chunks = pipeline_result["chunks"]
    if payload.top_k is not None:
        chunks = chunks[: payload.top_k]

    async def _stream():
        full_answer = ""
        async for event in generate_streaming(
            payload.query, chunks, pipeline_result["routing"]["path"], trace_id=trace_id
        ):
            if "delta" in event:
                full_answer += event["delta"]
                yield f"data: {json.dumps({'delta': event['delta']})}\n\n"
            elif "error" in event:
                yield f"data: {json.dumps({'error': event['error'], 'degraded': True})}\n\n"
            elif event.get("done"):
                final_degraded = pipeline_result["degraded"] or event["degraded"]
                final_api_call_count = pipeline_result["api_call_count"] + 1  # + generation
                final_body = {
                    "answer": full_answer,
                    "sources": _sources_from_chunks(chunks),
                    "routing_metadata": _routing_metadata(pipeline_result["routing"]),
                    "latency_breakdown": pipeline_result["latency_breakdown"],
                    "degraded": final_degraded,
                    "trace_id": trace_id,
                    "api_call_count": final_api_call_count,
                    "model_used": event["model_used"],
                }
                yield f"data: {json.dumps(final_body)}\n\n"
        # Standard SSE end-of-stream sentinel, same convention Groq/OpenAI
        # use for their own streaming APIs - lets a client's read loop
        # know to stop without guessing from content shape.
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


async def _check_qdrant() -> str:
    try:
        client = get_client()
        client.get_collections()
        return "ok"
    except Exception as exc:  # noqa: BLE001 - health check, any failure -> unreachable
        return f"unreachable: {exc}"


async def _check_groq() -> str:
    # ASSUMPTION (unverified): groq's AsyncGroq client exposes
    # client.models.list(). If your installed groq-python version
    # differs, swap this for whatever minimal reachability call it
    # actually supports.
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.groq_api_key)
        await client.models.list()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"unreachable: {exc}"


async def _check_gemini() -> str:
    # ASSUMPTION (unverified): google-genai's client exposes
    # client.models.list(). Confirm against your installed version.
    try:
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        client.models.list()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"unreachable: {exc}"


async def _check_tavily() -> str:
    # No cheap no-cost reachability probe confirmed for tavily-python -
    # a real search() call would cost API quota on every /health poll,
    # which is worse than not checking at all. This only verifies the
    # client constructs with a non-empty key - NOT that Tavily is
    # actually reachable. Flag this as a known limitation, not a real
    # reachability check, until a better probe is confirmed.
    try:
        from tavily import AsyncTavilyClient

        AsyncTavilyClient(api_key=settings.tavily_api_key)
        if not settings.tavily_api_key:
            return "unreachable: no API key configured"
        return "ok (key present, not verified reachable)"
    except Exception as exc:  # noqa: BLE001
        return f"unreachable: {exc}"


@app.get("/health")
async def health_endpoint():
    """FR-16: per-dependency reachability status."""
    dependencies = {
        "qdrant": await _check_qdrant(),
        "groq": await _check_groq(),
        "gemini": await _check_gemini(),
        "tavily": await _check_tavily(),
    }
    overall = "ok" if all(v.startswith("ok") for v in dependencies.values()) else "degraded"
    return HealthResponse(status=overall, dependencies=dependencies)