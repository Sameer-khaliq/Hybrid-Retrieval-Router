"""
FR-8 verify: fault-injection test mocking embedding failure across all
retries; a response is still returned with degraded=True and sparse-only
results.
"""
from unittest.mock import patch

import pytest

from src.common.resilience import RetriesExhaustedError, with_retry
from src.retrieval.fallback import embed_query_with_fallback


@pytest.mark.asyncio
async def test_embed_query_success_path(fake_embedding_vector):
    with patch("src.retrieval.fallback.embed_texts", return_value=[fake_embedding_vector]):
        result = await embed_query_with_fallback("what is a bond?", trace_id="test_success")

    assert result["degraded"] is False
    assert result["vector"] == fake_embedding_vector


@pytest.mark.asyncio
async def test_embed_query_falls_back_to_sparse_after_exhausted_retries():
    def always_fails(*args, **kwargs):
        raise ConnectionError("simulated embedding API outage")

    fake_sparse_results = [{"chunk_id": 1, "score": 4.2}, {"chunk_id": 2, "score": 3.1}]

    with patch("src.retrieval.fallback.embed_texts", side_effect=always_fails), \
         patch("src.retrieval.fallback.query_bm25", return_value=fake_sparse_results):

        result = await embed_query_with_fallback("what is a bond?", trace_id="test_fallback")

    assert result["degraded"] is True
    assert result["vector"] is None
    assert result["sparse_results"] == fake_sparse_results


@pytest.mark.asyncio
async def test_with_retry_raises_after_max_retries():
    call_count = {"n": 0}

    async def flaky():
        call_count["n"] += 1
        raise ConnectionError("always fails")

    with pytest.raises(RetriesExhaustedError):
        await with_retry(flaky, max_retries=2, base_delay_s=0.01)

    assert call_count["n"] == 3  # initial attempt + 2 retries


@pytest.mark.asyncio
async def test_with_retry_succeeds_on_second_attempt():
    call_count = {"n": 0}

    async def flaky_then_ok():
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise ConnectionError("transient")
        return "ok"

    result = await with_retry(flaky_then_ok, max_retries=2, base_delay_s=0.01)
    assert result == "ok"
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_with_retry_does_not_retry_non_retryable_errors():
    call_count = {"n": 0}

    async def fails_with_value_error():
        call_count["n"] += 1
        raise ValueError("not a retryable error")

    with pytest.raises(RetriesExhaustedError):
        await with_retry(fails_with_value_error, max_retries=2, base_delay_s=0.01)

    assert call_count["n"] == 1  # no retries attempted — fails fast