from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from config.settings import settings
from src.api.schemas import HealthResponse, QueryRequest
from src.common.qdrant_client import get_client
from src.generation.groq_stream import generate_atomic, generate_streaming
from src.pipeline.run_query import run_query
from src.retrieval.sparse_bm25 import get_or_build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load active BM25 index into memory
    try:
        get_or_build_index()
        print("[api] BM25 active index loaded into memory successfully.")
    except Exception as exc:  # noqa: BLE001
        print(f"[api] Warning: Could not initialize BM25 on startup: {exc}")
    yield


app = FastAPI(
    title="Hybrid Retrieval & Query Routing System",
    lifespan=lifespan,
)


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
    trace_id = str(uuid.uuid4())
    pipeline_result = await run_query(payload.query, trace_id=trace_id)

    # --- Atomic path: gated or insufficient-info ---
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

    # --- Streaming path ---
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
                final_api_call_count = pipeline_result["api_call_count"] + 1
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
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


async def _check_qdrant() -> str:
    try:
        client = get_client()
        client.get_collections()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"unreachable: {exc}"


async def _check_groq() -> str:
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=settings.groq_api_key)
        await client.models.list()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"unreachable: {exc}"


async def _check_gemini() -> str:
    try:
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        client.models.list()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"unreachable: {exc}"


async def _check_tavily() -> str:
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
    dependencies = {
        "qdrant": await _check_qdrant(),
        "groq": await _check_groq(),
        "gemini": await _check_gemini(),
        "tavily": await _check_tavily(),
    }
    overall = "ok" if all(v.startswith("ok") for v in dependencies.values()) else "degraded"
    return HealthResponse(status=overall, dependencies=dependencies)