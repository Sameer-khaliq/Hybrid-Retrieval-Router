"""
Step 20 tests (FR-9).

Plan's stated Done-when: "A fixture query designed to score below the
floor triggers a mocked Tavily call, with sources correctly tagged
web/corpus; a second fixture confirms a normal high-confidence corpus
query does NOT trigger Tavily."

Tavily's actual API is mocked throughout - this file is about trigger
logic and result tagging, not Tavily's own search quality.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from config.settings import settings
from src.retrieval.tavily_fallback import (
    apply_tavily_fallback,
    is_low_confidence,
    needs_current_info,
)


def _reranked_result(top_score: float, score_key: str = "rerank_score") -> dict:
    return {
        "chunks": [{"chunk_id": "c1", score_key: top_score, "text": "some corpus text"}],
        "reranked": score_key == "rerank_score",
        "rerank_top_k": 15 if score_key == "rerank_score" else None,
        "path": "deep" if score_key == "rerank_score" else "fast",
    }


# ---- needs_current_info() -------------------------------------------------

def test_needs_current_info_matches_currency_triggers():
    assert needs_current_info("What are the latest FDA guidelines?")
    assert needs_current_info("What is the current interest rate today?")
    assert needs_current_info("Show me recent earnings for this company")


def test_needs_current_info_does_not_match_plain_factual_query():
    assert not needs_current_info("What is the standard deposit interest formula?")
    assert not needs_current_info("Explain compound interest.")


# ---- is_low_confidence() ---------------------------------------------------

def test_is_low_confidence_below_floor():
    result = _reranked_result(top_score=settings.tavily_confidence_floor / 2)
    assert is_low_confidence(result) is True


def test_is_low_confidence_above_floor():
    result = _reranked_result(top_score=settings.tavily_confidence_floor * 10)
    assert is_low_confidence(result) is False


def test_is_low_confidence_no_chunks_is_low_confidence():
    assert is_low_confidence({"chunks": []}) is True


def test_is_low_confidence_reads_rrf_score_when_not_reranked():
    """Fast-path-skip case (Step 19): chunks carry rrf_score, not
    rerank_score - is_low_confidence must fall back correctly."""
    result = _reranked_result(top_score=settings.tavily_confidence_floor * 10, score_key="rrf_score")
    assert is_low_confidence(result) is False


# ---- apply_tavily_fallback() -----------------------------------------------

@pytest.mark.asyncio
async def test_low_confidence_triggers_tavily_and_tags_sources():
    low_conf_result = _reranked_result(top_score=settings.tavily_confidence_floor / 2)

    fake_web_results = [{"title": "Web result", "url": "https://example.com", "content": "..."}]
    with patch(
        "src.retrieval.tavily_fallback._search_tavily_with_retry",
        new=AsyncMock(return_value=fake_web_results),
    ) as mock_search:
        result = await apply_tavily_fallback("some low confidence query", low_conf_result)

    mock_search.assert_called_once()
    assert result["tavily_triggered"] is True
    assert result["tavily_trigger_reason"] == "low_confidence"
    assert result["degraded"] is False
    sources = {c["source"] for c in result["chunks"]}
    assert sources == {"corpus", "web"}


@pytest.mark.asyncio
async def test_high_confidence_non_current_query_does_not_trigger_tavily():
    high_conf_result = _reranked_result(top_score=settings.tavily_confidence_floor * 10)

    with patch(
        "src.retrieval.tavily_fallback._search_tavily_with_retry",
        new=AsyncMock(),
    ) as mock_search:
        result = await apply_tavily_fallback("Explain compound interest.", high_conf_result)

    mock_search.assert_not_called()
    assert result["tavily_triggered"] is False
    assert result["tavily_trigger_reason"] is None
    assert all(c["source"] == "corpus" for c in result["chunks"])


@pytest.mark.asyncio
async def test_currency_trigger_fires_even_with_high_confidence():
    high_conf_result = _reranked_result(top_score=settings.tavily_confidence_floor * 10)

    with patch(
        "src.retrieval.tavily_fallback._search_tavily_with_retry",
        new=AsyncMock(return_value=[]),
    ) as mock_search:
        result = await apply_tavily_fallback("What are today's mortgage rates?", high_conf_result)

    mock_search.assert_called_once()
    assert result["tavily_triggered"] is True
    assert result["tavily_trigger_reason"] == "needs_current_info"


@pytest.mark.asyncio
async def test_tavily_retries_exhausted_degrades_gracefully_not_erroring():
    from src.common.resilience import RetriesExhaustedError

    low_conf_result = _reranked_result(top_score=settings.tavily_confidence_floor / 2)

    with patch(
        "src.retrieval.tavily_fallback._search_tavily_with_retry",
        new=AsyncMock(side_effect=RetriesExhaustedError("mock exhaustion")),
    ):
        result = await apply_tavily_fallback("some query", low_conf_result)

    assert result["degraded"] is True
    assert result["tavily_triggered"] is True
    # corpus chunks still returned, not an empty/errored response
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["source"] == "corpus"
    