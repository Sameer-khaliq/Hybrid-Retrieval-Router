"""
FR-7 verify: instrumented timing with an artificial delay shows
retrieval-stage wall-clock ≈ max(sparse_time, dense_time), not their sum.
This is the test that actually proves concurrency, not just "code ran".
"""
import asyncio
import time
from unittest.mock import patch

import pytest

from src.retrieval.orchestrator import retrieve_concurrent

DELAY_S = 0.3  # artificial delay injected into both legs


async def _delayed_sparse(query, top_n):
    await asyncio.sleep(DELAY_S)
    return [{"chunk_id": 1, "score": 4.0}, {"chunk_id": 2, "score": 3.5}]


async def _delayed_dense_embed(query, trace_id="x"):
    await asyncio.sleep(DELAY_S)
    return {"vector": [0.1] * 768, "degraded": False}


@pytest.mark.asyncio
async def test_retrieval_is_concurrent_not_sequential():
    """Core FR-7 assertion: wall-clock ≈ max(delay), not sum(delay + delay)."""
    with patch("src.retrieval.orchestrator._sparse_leg", side_effect=_delayed_sparse), \
         patch("src.retrieval.orchestrator.embed_query_with_fallback", side_effect=_delayed_dense_embed), \
         patch("src.retrieval.orchestrator.get_client") as mock_get_client:

        from unittest.mock import MagicMock

        class FakePoint:
            def __init__(self, id_, score, payload):
                self.id, self.score, self.payload = id_, score, payload

        class FakeResult:
            def __init__(self, points):
                self.points = points

        mock_client = MagicMock()
        mock_client.query_points.return_value = FakeResult(
            [FakePoint(3, 0.9, {"text": "dense result"})]
        )
        mock_get_client.return_value = mock_client

        start = time.monotonic()
        result = await retrieve_concurrent("test query", trace_id="test_concurrency")
        elapsed = time.monotonic() - start

    # If sequential: elapsed ≈ 2 * DELAY_S (0.6s). If concurrent: elapsed ≈ DELAY_S (0.3s).
    # Assert closer to concurrent bound with headroom for scheduling overhead.
    assert elapsed < (DELAY_S * 1.5), (
        f"Expected concurrent execution (~{DELAY_S}s), got {elapsed:.3f}s — "
        f"this smells like sparse and dense legs are running sequentially, not via gather."
    )
    assert elapsed >= DELAY_S  # sanity: can't be faster than the slowest leg

    assert len(result["sparse"]) == 2
    assert len(result["dense"]) == 1
    assert result["degraded"] is False


@pytest.mark.asyncio
async def test_retrieval_marks_degraded_when_dense_leg_fails():
    async def failing_dense_embed(query, trace_id="x"):
        return {"vector": None, "degraded": True, "sparse_results": []}

    fake_sparse = [{"chunk_id": 1, "score": 4.0}]

    async def fast_sparse(query, top_n):
        return fake_sparse

    with patch("src.retrieval.orchestrator._sparse_leg", side_effect=fast_sparse), \
         patch("src.retrieval.orchestrator.embed_query_with_fallback", side_effect=failing_dense_embed):

        result = await retrieve_concurrent("test query", trace_id="test_degraded")

    assert result["degraded"] is True
    assert result["dense"] == []
    assert result["sparse"] == fake_sparse