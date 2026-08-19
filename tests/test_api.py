"""
Step 23 tests (FR-15, FR-16, FR-17).

Plan's stated Done-when (adapted from curl to TestClient, same contract):
  - A valid query returns the documented response schema.
  - Empty/oversized queries return 4xx with zero downstream calls.
  - /health shows per-dependency status.

run_query, generate_atomic, generate_streaming, and the four _check_*
health probes are all mocked - this file verifies the HTTP CONTRACT,
not pipeline correctness (already covered by test_pipeline_e2e.py and
test_generation_streaming.py).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from config.settings import settings
from src.api.main import app

client = TestClient(app)

FAKE_GATED_RESULT = {
    "trace_id": "t1",
    "gated": True,
    "answer_text": "Hi! Ask me about the knowledge base.",
    "chunks": None,
    "routing": None,
    "tavily_triggered": False,
    "tavily_trigger_reason": None,
    "degraded": False,
    "latency_breakdown": {"gating_ms": 0.5, "pipeline_total_ms": 0.6},
    "api_call_count": 0,
    "pre_rerank_chunk_ids": [],
    "post_rerank_chunk_ids": [],
}

FAKE_STREAMING_RESULT = {
    "trace_id": "t2",
    "gated": False,
    "answer_text": None,
    "chunks": [{"chunk_id": 1, "text": "Paris is the capital of France.", "source": "corpus", "rerank_score": 0.9}],
    "routing": {"path": "fast", "score": 0.1, "confidence": None, "reason": "rule-based", "deciding_layer": "rule-based", "degraded": False},
    "tavily_triggered": False,
    "tavily_trigger_reason": None,
    "degraded": False,
    "latency_breakdown": {"gating_ms": 0.4, "pipeline_total_ms": 50.0},
    "api_call_count": 1,
    "pre_rerank_chunk_ids": [1],
    "post_rerank_chunk_ids": [1],
}


async def _fake_stream(*args, **kwargs):
    yield {"delta": "Paris"}
    yield {"delta": " is the capital."}
    yield {"done": True, "model_used": "openai/gpt-oss-20b", "degraded": False}


# ---- FR-17: input validation --------------------------------------------

def test_empty_query_returns_4xx_no_downstream_call():
    with patch("src.api.main.run_query", new=AsyncMock()) as mock_run_query:
        response = client.post("/query", json={"query": ""})
    assert 400 <= response.status_code < 500
    mock_run_query.assert_not_called()


def test_whitespace_only_query_returns_4xx():
    with patch("src.api.main.run_query", new=AsyncMock()) as mock_run_query:
        response = client.post("/query", json={"query": "   "})
    assert 400 <= response.status_code < 500
    mock_run_query.assert_not_called()


def test_oversized_query_returns_4xx_no_downstream_call():
    with patch("src.api.main.run_query", new=AsyncMock()) as mock_run_query:
        response = client.post("/query", json={"query": "x" * (settings.max_query_length + 1)})
    assert 400 <= response.status_code < 500
    mock_run_query.assert_not_called()

def test_query_at_exact_max_length_is_accepted():
    with (
        patch("src.api.main.run_query", new=AsyncMock(return_value=FAKE_GATED_RESULT)),
        patch("src.api.main.generate_atomic", new=AsyncMock(return_value={"answer": FAKE_GATED_RESULT["answer_text"], "streamed": False, "degraded": False})),
    ):
        response = client.post("/query", json={"query": "x" * settings.max_query_length})
    assert response.status_code == 200

# ---- FR-15/26: gated (atomic) response contract ---------------------------

def test_gated_query_returns_single_atomic_json_body():
    with (
        patch("src.api.main.run_query", new=AsyncMock(return_value=FAKE_GATED_RESULT)),
        patch(
            "src.api.main.generate_atomic",
            new=AsyncMock(return_value={"answer": FAKE_GATED_RESULT["answer_text"], "streamed": False, "degraded": False}),
        ),
    ):
        response = client.post("/query", json={"query": "hi"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == FAKE_GATED_RESULT["answer_text"]
    assert body["sources"] == []
    assert body["routing_metadata"] == {"path": None, "confidence": None, "reason": None}
    assert body["degraded"] is False
    assert "latency_breakdown" in body


# ---- FR-15/26: streaming response contract ---------------------------------

def test_normal_query_streams_ndjson_deltas_then_final_schema():
    with (
        patch("src.api.main.run_query", new=AsyncMock(return_value=FAKE_STREAMING_RESULT)),
        patch("src.api.main.generate_streaming", new=_fake_stream),
    ):
        response = client.post("/query", json={"query": "What is the capital of France?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    # SSE format: "data: {...}\n\n" per event, "data: [DONE]\n\n" as sentinel
    raw_events = [e for e in response.text.split("\n\n") if e.strip()]
    parsed = [json.loads(e[len("data: "):]) for e in raw_events if e != "data: [DONE]"]
    assert raw_events[-1] == "data: [DONE]"

    # first N-1 lines are delta events, in order
    deltas = [p["delta"] for p in parsed if "delta" in p]
    assert deltas == ["Paris", " is the capital."]

    # final line matches FR-15's documented schema
    final = parsed[-1]
    assert final["answer"] == "Paris is the capital."
    assert final["sources"][0]["chunk_id"] == 1
    assert final["sources"][0]["source"] == "corpus"
    assert final["routing_metadata"]["path"] == "fast"
    assert final["degraded"] is False
    assert final["api_call_count"] == 2  # 1 from pipeline + 1 for generation


# ---- FR-16: health endpoint -------------------------------------------------

def test_health_endpoint_reports_all_dependencies_ok():
    with (
        patch("src.api.main._check_qdrant", new=AsyncMock(return_value="ok")),
        patch("src.api.main._check_groq", new=AsyncMock(return_value="ok")),
        patch("src.api.main._check_gemini", new=AsyncMock(return_value="ok")),
        patch("src.api.main._check_tavily", new=AsyncMock(return_value="ok")),
    ):
                    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dependencies"]["qdrant"] == "ok"


def test_health_endpoint_reports_degraded_when_one_dependency_down():
    with (
        patch("src.api.main._check_qdrant", new=AsyncMock(return_value="unreachable: connection refused")),
        patch("src.api.main._check_groq", new=AsyncMock(return_value="ok")),
        patch("src.api.main._check_gemini", new=AsyncMock(return_value="ok")),
        patch("src.api.main._check_tavily", new=AsyncMock(return_value="ok")),
    ):
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert "unreachable" in body["dependencies"]["qdrant"]