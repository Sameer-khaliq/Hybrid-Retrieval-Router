"""
Step 24 tests (FR-20, NFR-11, NFR-4).

Plan's stated Done-when: "After one test query through /query: trace ID
is identical across every log line for that query; all NFR-11 fields
are present. A separate assertion confirms api_call_count <= 4 for a
fixture query exercising the worst case (embedding + LLM-router +
generation + Tavily all firing)."

Mocks the pipeline's internal stages (same level as test_pipeline_e2e.py)
rather than the top-level run_query()/generate_streaming(), so the real
logging calls inside run_query.py actually execute and can be captured
via caplog - this is the one test file in Phase 5 that needs the real
(not top-level-mocked) pipeline logging path to verify anything.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def _parse_sse(response_text: str) -> list[dict]:
    """Parse 'data: {...}\\n\\n' SSE events, skipping the '[DONE]' sentinel."""
    events = [e for e in response_text.split("\n\n") if e.strip()]
    return [json.loads(e[len("data: "):]) for e in events if e != "data: [DONE]"]


# Worst case per NFR-4: embedding (always) + LLM-router (mid-band) +
# Tavily (triggered) + generation (always) = 4 total.
WORST_CASE_ROUTING = {
    "path": "deep",
    "score": 0.45,
    "confidence": 0.7,
    "reason": "llm-fallback: ambiguous middle band",
    "deciding_layer": "llm-fallback",  # router WAS called -> +1
    "degraded": False,
}

WORST_CASE_RETRIEVAL_AND_ROUTING = {
    "sparse": [{"chunk_id": 1, "score": 5.0}],
    "dense": [{"chunk_id": 1, "score": 0.9, "payload": {"text": "some corpus text"}}],
    "degraded": False,
    "routing": WORST_CASE_ROUTING,
}

WORST_CASE_FUSED = [{"chunk_id": 1, "rrf_score": 0.03, "payload": {"text": "some corpus text"}}]

WORST_CASE_RERANKED = {
    "chunks": [{"chunk_id": 1, "text": "some corpus text", "rerank_score": 0.02}],
    "reranked": True,
    "rerank_top_k": 15,
    "path": "deep",
}

WORST_CASE_TAVILY_RESULT = {
    "chunks": [
        {"chunk_id": 1, "text": "some corpus text", "source": "corpus"},
        {"title": "Web result", "url": "https://example.com", "content": "...", "source": "web"},
    ],
    "tavily_triggered": True,  # Tavily WAS called -> +1
    "tavily_trigger_reason": "low_confidence",
    "degraded": False,
}


async def _fake_generate_streaming(*args, **kwargs):
    yield {"delta": "The"}
    yield {"delta": " analysis shows..."}
    yield {"done": True, "model_used": "openai/gpt-oss-120b", "degraded": False}  # generation -> +1


def test_worst_case_api_call_count_is_four():
    with (
        patch("src.pipeline.run_query.run_prefilter", return_value=None),
        patch(
            "src.pipeline.run_query.retrieve_and_route_concurrent",
            new=AsyncMock(return_value=WORST_CASE_RETRIEVAL_AND_ROUTING),
        ),
        patch("src.pipeline.run_query.rrf_fuse", return_value=WORST_CASE_FUSED),
        patch(
            "src.pipeline.run_query.apply_conditional_rerank",
            new=AsyncMock(return_value=WORST_CASE_RERANKED),
        ),
        patch(
            "src.pipeline.run_query.apply_tavily_fallback",
            new=AsyncMock(return_value=WORST_CASE_TAVILY_RESULT),
        ),
        patch("src.api.main.generate_streaming", new=_fake_generate_streaming),
    ):
        response = client.post(
            "/query", json={"query": "What are today's mortgage rates compared to last year?"}
        )

    assert response.status_code == 200
    parsed = _parse_sse(response.text)
    final = parsed[-1]

    # embedding(1) + router(1, llm-fallback) + tavily(1, triggered) + generation(1) = 4
    assert final["api_call_count"] == 4


def test_all_nfr11_fields_present_in_final_response():
    with (
        patch("src.pipeline.run_query.run_prefilter", return_value=None),
        patch(
            "src.pipeline.run_query.retrieve_and_route_concurrent",
            new=AsyncMock(return_value=WORST_CASE_RETRIEVAL_AND_ROUTING),
        ),
        patch("src.pipeline.run_query.rrf_fuse", return_value=WORST_CASE_FUSED),
        patch(
            "src.pipeline.run_query.apply_conditional_rerank",
            new=AsyncMock(return_value=WORST_CASE_RERANKED),
        ),
        patch(
            "src.pipeline.run_query.apply_tavily_fallback",
            new=AsyncMock(return_value=WORST_CASE_TAVILY_RESULT),
        ),
        patch("src.api.main.generate_streaming", new=_fake_generate_streaming),
    ):
        response = client.post("/query", json={"query": "some query"})

    final = _parse_sse(response.text)[-1]

    # NFR-11: trace ID, degraded flags, routing decision + reason +
    # deciding layer, latency breakdown (per-stage), api_call_count.
    assert "trace_id" in final
    assert "degraded" in final
    assert final["routing_metadata"]["path"] == "deep"
    assert final["routing_metadata"]["reason"] is not None
    assert set(final["latency_breakdown"].keys()) >= {
        "gating_ms", "retrieval_and_routing_ms", "fusion_ms", "rerank_ms",
        "tavily_ms", "pipeline_total_ms",
    }
    assert "api_call_count" in final


def test_trace_id_consistent_across_log_lines(caplog):
    """FR-20: every log line for one query carries the same trace_id."""
    with (
        caplog.at_level(logging.INFO, logger="hybrid_retrieval"),
        patch("src.pipeline.run_query.run_prefilter", return_value=None),
        patch(
            "src.pipeline.run_query.retrieve_and_route_concurrent",
            new=AsyncMock(return_value=WORST_CASE_RETRIEVAL_AND_ROUTING),
        ),
        patch("src.pipeline.run_query.rrf_fuse", return_value=WORST_CASE_FUSED),
        patch(
            "src.pipeline.run_query.apply_conditional_rerank",
            new=AsyncMock(return_value=WORST_CASE_RERANKED),
        ),
        patch(
            "src.pipeline.run_query.apply_tavily_fallback",
            new=AsyncMock(return_value=WORST_CASE_TAVILY_RESULT),
        ),
        patch("src.api.main.generate_streaming", new=_fake_generate_streaming),
    ):
        response = client.post("/query", json={"query": "some query"})

    final = _parse_sse(response.text)[-1]
    expected_trace_id = final["trace_id"]

    trace_ids_seen = {
        record.trace_id for record in caplog.records if hasattr(record, "trace_id")
    }
    assert trace_ids_seen, "expected at least one log record carrying trace_id"
    assert trace_ids_seen == {expected_trace_id}, (
        f"inconsistent trace IDs across log lines: {trace_ids_seen}"
    )