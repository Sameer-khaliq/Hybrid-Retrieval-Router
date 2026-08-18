"""
Step 21 tests — full pipeline e2e wiring.

Plan's stated Done-when: "A full fixture run through gating-pass ->
routing -> retrieval -> fusion -> rerank -> Tavily-check asserts every
stage's output correctly feeds the next; a separate gating fixture
confirms a matched FR-21-24 case returns immediately without touching
Steps 17/19/20 at all."

Every stage function is mocked - this file verifies WIRING (data
threaded correctly, short-circuit behavior correct), not each stage's
own internal correctness (already covered by that stage's own test
file from Phase 2-4).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.pipeline.run_query import run_query


GATED_RESULT = {
    "gated": True,
    "category": "FR-21",
    "response": "Hi! Ask me about the knowledge base.",
    "reason": "non_corpus_intent",
}

FAKE_ROUTING = {
    "path": "deep",
    "score": 0.75,
    "confidence": None,
    "reason": "rule-based: score=0.750 > tau_high",
    "deciding_layer": "rule-based",
    "degraded": False,
}

FAKE_RETRIEVAL_AND_ROUTING = {
    "sparse": [{"chunk_id": 1, "score": 5.0}],
    "dense": [{"chunk_id": 1, "score": 0.9, "payload": {"text": "some text"}}],
    "degraded": False,
    "routing": FAKE_ROUTING,
}

FAKE_FUSED = [{"chunk_id": 1, "rrf_score": 0.03, "payload": {"text": "some text"}}]

FAKE_RERANKED = {
    "chunks": [{"chunk_id": 1, "text": "some text", "rerank_score": 0.8}],
    "reranked": True,
    "rerank_top_k": 15,
    "path": "deep",
}

FAKE_TAVILY_RESULT = {
    "chunks": [{"chunk_id": 1, "text": "some text", "source": "corpus"}],
    "tavily_triggered": False,
    "tavily_trigger_reason": None,
    "degraded": False,
}


@pytest.mark.asyncio
async def test_gated_query_short_circuits_all_downstream_stages():
    with patch("src.pipeline.run_query.run_prefilter", return_value=GATED_RESULT):
        with patch("src.pipeline.run_query.retrieve_and_route_concurrent", new=AsyncMock()) as mock_retrieve:
            with patch("src.pipeline.run_query.apply_conditional_rerank", new=AsyncMock()) as mock_rerank:
                with patch("src.pipeline.run_query.apply_tavily_fallback", new=AsyncMock()) as mock_tavily:
                    result = await run_query("Hi there")

    mock_retrieve.assert_not_called()
    mock_rerank.assert_not_called()
    mock_tavily.assert_not_called()

    assert result["gated"] is True
    assert result["answer_text"] == GATED_RESULT["response"]
    assert result["chunks"] is None
    assert result["routing"] is None
    assert result["degraded"] is False
    assert result["api_call_count"] == 0
    assert "gating_ms" in result["latency_breakdown"]


@pytest.mark.asyncio
async def test_full_chain_data_threads_correctly():
    with patch("src.pipeline.run_query.run_prefilter", return_value=None):
        with patch(
            "src.pipeline.run_query.retrieve_and_route_concurrent",
            new=AsyncMock(return_value=FAKE_RETRIEVAL_AND_ROUTING),
        ):
            with patch("src.pipeline.run_query.rrf_fuse", return_value=FAKE_FUSED) as mock_fuse:
                with patch(
                    "src.pipeline.run_query.apply_conditional_rerank",
                    new=AsyncMock(return_value=FAKE_RERANKED),
                ) as mock_rerank:
                    with patch(
                        "src.pipeline.run_query.apply_tavily_fallback",
                        new=AsyncMock(return_value=FAKE_TAVILY_RESULT),
                    ) as mock_tavily:
                        result = await run_query("What is the P/E ratio?")

    # rrf_fuse called with the retrieval legs
    mock_fuse.assert_called_once_with(
        FAKE_RETRIEVAL_AND_ROUTING["sparse"], FAKE_RETRIEVAL_AND_ROUTING["dense"]
    )

    # routing decision was threaded into apply_conditional_rerank, not dropped
    rerank_call_args = mock_rerank.call_args
    assert rerank_call_args.args[2] == FAKE_ROUTING  # (query, fused, routing_decision, ...)

    # reranked_result was threaded into apply_tavily_fallback
    tavily_call_args = mock_tavily.call_args
    assert tavily_call_args.args[1] == FAKE_RERANKED

    assert result["gated"] is False
    assert result["answer_text"] is None
    assert result["chunks"] == FAKE_TAVILY_RESULT["chunks"]
    assert result["routing"] == FAKE_ROUTING
    assert result["degraded"] is False
    # FAKE_ROUTING has deciding_layer="rule-based" (no router call) and
    # FAKE_TAVILY_RESULT has tavily_triggered=False -> only embedding counted
    assert result["api_call_count"] == 1
    assert set(result["latency_breakdown"].keys()) >= {
        "gating_ms", "retrieval_and_routing_ms", "fusion_ms", "rerank_ms",
        "tavily_ms", "pipeline_total_ms",
    }
    assert result["pre_rerank_chunk_ids"] == [1]
    assert result["post_rerank_chunk_ids"] == [1]


@pytest.mark.asyncio
async def test_empty_chunks_after_tavily_short_circuits_before_generation():
    empty_tavily_result = {**FAKE_TAVILY_RESULT, "chunks": [], "degraded": True, "tavily_triggered": True, "tavily_trigger_reason": "low_confidence"}

    with patch("src.pipeline.run_query.run_prefilter", return_value=None):
        with patch(
            "src.pipeline.run_query.retrieve_and_route_concurrent",
            new=AsyncMock(return_value=FAKE_RETRIEVAL_AND_ROUTING),
        ):
            with patch("src.pipeline.run_query.rrf_fuse", return_value=FAKE_FUSED):
                with patch(
                    "src.pipeline.run_query.apply_conditional_rerank",
                    new=AsyncMock(return_value=FAKE_RERANKED),
                ):
                    with patch(
                        "src.pipeline.run_query.apply_tavily_fallback",
                        new=AsyncMock(return_value=empty_tavily_result),
                    ):
                        result = await run_query("some obscure query")

    assert result["gated"] is False
    assert result["answer_text"] is not None  # insufficient-info message set
    assert result["chunks"] == []
    assert result["degraded"] is True


@pytest.mark.asyncio
async def test_degraded_flag_propagates_from_any_stage():
    """degraded should be True if ANY of retrieval/routing/tavily flagged
    degraded, even if the others were clean."""
    retrieval_degraded = {**FAKE_RETRIEVAL_AND_ROUTING, "degraded": True}

    with patch("src.pipeline.run_query.run_prefilter", return_value=None):
        with patch(
            "src.pipeline.run_query.retrieve_and_route_concurrent",
            new=AsyncMock(return_value=retrieval_degraded),
        ):
            with patch("src.pipeline.run_query.rrf_fuse", return_value=FAKE_FUSED):
                with patch(
                    "src.pipeline.run_query.apply_conditional_rerank",
                    new=AsyncMock(return_value=FAKE_RERANKED),
                ):
                    with patch(
                        "src.pipeline.run_query.apply_tavily_fallback",
                        new=AsyncMock(return_value=FAKE_TAVILY_RESULT),
                    ):
                        result = await run_query("query")

    assert result["degraded"] is True