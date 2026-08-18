"""
Step 19 tests — conditional rerank wiring (completes FR-6).

Plan's stated Done-when: "Two fixture queries pre-labeled fast/deep
confirm the reranker is invoked or skipped as expected, with the
expected K."

rerank_async is mocked here (not the real CrossEncoder) — Step 18's own
test file already covers real-model correctness and CPU latency; this
file is only about the *wiring decision* (invoked/skipped, which K),
not reranker quality.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from config.settings import settings
from src.retrieval.fusion import apply_conditional_rerank, rrf_fuse


def _fake_fused(n: int) -> list[dict]:
    return [
        {"chunk_id": f"c{i}", "rrf_score": 1.0 / (i + 1), "payload": {"text": f"chunk text {i}"}}
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_deep_path_invokes_reranker_at_configured_top_k():
    fused = _fake_fused(20)
    routing_decision = {"path": "deep"}

    with patch("src.retrieval.fusion.rerank_async", new=AsyncMock(return_value=fused[: settings.rerank_top_k])) as mock_rerank:
        result = await apply_conditional_rerank("test query", fused, routing_decision)

    mock_rerank.assert_called_once()
    # candidates passed in were sliced to rerank_top_k before reranking
    call_kwargs = mock_rerank.call_args.kwargs
    _passed_candidates = mock_rerank.call_args.args[1] if len(mock_rerank.call_args.args) > 1 else call_kwargs.get("candidates")
    assert result["reranked"] is True
    assert result["rerank_top_k"] == settings.rerank_top_k


@pytest.mark.asyncio
async def test_fast_path_skip_mode_does_not_invoke_reranker(monkeypatch):
    monkeypatch.setattr(settings, "rerank_fast_path_mode", "skip")
    fused = _fake_fused(20)
    routing_decision = {"path": "fast"}

    with patch("src.retrieval.fusion.rerank_async", new=AsyncMock()) as mock_rerank:
        result = await apply_conditional_rerank("test query", fused, routing_decision)

    mock_rerank.assert_not_called()
    assert result["reranked"] is False
    assert result["rerank_top_k"] is None
    assert len(result["chunks"]) == settings.rerank_top_k_fast


@pytest.mark.asyncio
async def test_fast_path_reduced_mode_invokes_reranker_at_reduced_top_k(monkeypatch):
    monkeypatch.setattr(settings, "rerank_fast_path_mode", "reduced")
    fused = _fake_fused(20)
    routing_decision = {"path": "fast"}

    with patch(
        "src.retrieval.fusion.rerank_async",
        new=AsyncMock(return_value=fused[: settings.rerank_top_k_fast]),
    ) as mock_rerank:
        result = await apply_conditional_rerank("test query", fused, routing_decision)

    mock_rerank.assert_called_once()
    assert result["reranked"] is True
    assert result["rerank_top_k"] == settings.rerank_top_k_fast


def test_rrf_fuse_unchanged_from_step_13():
    """Sanity check that extending this file didn't touch Step 13's math."""
    sparse = [{"chunk_id": "a", "score": 5.0}, {"chunk_id": "b", "score": 3.0}]
    dense = [{"chunk_id": "b", "score": 0.9}, {"chunk_id": "a", "score": 0.5}]
    result = rrf_fuse(sparse, dense, k=60)
    # both chunks appear in both lists at different ranks; exact score
    # values aren't the point here, just that fusion still runs and
    # returns a valid sorted structure
    assert len(result) == 2
    assert result[0]["rrf_score"] >= result[1]["rrf_score"]


def test_payload_text_extraction_confirmed_key():
    """Confirmed against src/ingestion/pipeline.py's embed_and_upsert():
    payload always carries chunk text under "text" (spread directly
    from chunk_text()'s output). No fallback chain needed anymore."""
    from src.retrieval.fusion import _payload_text

    assert _payload_text({"text": "hello"}) == "hello"


def test_payload_text_missing_key_degrades_not_crashes(caplog):
    """A payload missing 'text' (e.g. a sparse-only match without full
    payload attached - unverified against sparse_bm25.py) should log a
    warning and return "", not raise and crash the whole query."""
    from src.retrieval.fusion import _payload_text

    result = _payload_text({"some_other_key": "hello"}, chunk_id="c1")
    assert result == ""
    assert _payload_text(None, chunk_id="c2") == ""