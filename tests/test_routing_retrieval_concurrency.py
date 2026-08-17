"""
Tests for FR-14: routing computation (Step 16) is initiated concurrently
with the retrieval pipeline (Step 12), not sequentially after it.

Same technique as Step 12's test_retrieval_concurrency.py: inject
artificial delays into both sides via mocking, then assert wall-clock
time on retrieve_and_route_concurrent() is ~= max(delay), not the sum -
which would only be true if routing and retrieval actually run
concurrently, not one-after-the-other.
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.retrieval.orchestrator import retrieve_and_route_concurrent

RETRIEVAL_DELAY_S = 0.3
ROUTING_DELAY_S = 0.3
# Generous tolerance for CI/dev-machine scheduling jitter - the point of
# this test is distinguishing "~0.3s" (concurrent) from "~0.6s"
# (sequential), not measuring precise timing.
TOLERANCE_S = 0.15


async def _delayed_retrieve_concurrent(*args, **kwargs):
    await asyncio.sleep(RETRIEVAL_DELAY_S)
    return {"sparse": [], "dense": [], "degraded": False}


async def _delayed_route_query(*args, **kwargs):
    await asyncio.sleep(ROUTING_DELAY_S)
    return {
        "path": "fast",
        "score": 0.1,
        "confidence": None,
        "reason": "rule-based: score=0.100 < tau_low",
        "deciding_layer": "rule-based",
        "degraded": False,
    }


@pytest.mark.asyncio
async def test_routing_retrieval_start_concurrently():
    with patch(
        "src.retrieval.orchestrator.retrieve_concurrent",
        new=_delayed_retrieve_concurrent,
    ), patch(
        "src.retrieval.orchestrator.route_query",
        new=_delayed_route_query,
    ):
        start = time.perf_counter()
        result = await retrieve_and_route_concurrent("test query", trace_id="concurrency_test")
        elapsed = time.perf_counter() - start

    max_delay = max(RETRIEVAL_DELAY_S, ROUTING_DELAY_S)
    sum_delay = RETRIEVAL_DELAY_S + ROUTING_DELAY_S

    assert elapsed < max_delay + TOLERANCE_S, (
        f"elapsed {elapsed:.3f}s is too close to sequential sum "
        f"{sum_delay:.3f}s - routing and retrieval do not appear to be "
        f"running concurrently (FR-14 violation)"
    )
    assert result["routing"]["path"] == "fast"
    assert "sparse" in result and "dense" in result


@pytest.mark.asyncio
async def test_retrieve_and_route_result_shape():
    with patch(
        "src.retrieval.orchestrator.retrieve_concurrent",
        new=_delayed_retrieve_concurrent,
    ), patch(
        "src.retrieval.orchestrator.route_query",
        new=_delayed_route_query,
    ):
        result = await retrieve_and_route_concurrent("test query")

    assert set(result.keys()) == {"sparse", "dense", "degraded", "routing"}
    assert set(result["routing"].keys()) == {
        "path", "score", "confidence", "reason", "deciding_layer", "degraded",
    }