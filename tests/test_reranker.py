"""
Step 18 tests (FR-10, Risk Register #3).

Two things this file verifies, matching the plan's "Done when" exactly:
  1. A synthetic case with a known relevant/irrelevant pair confirms
     correct ordering.
  2. A separate timing test logs raw CPU latency for a top-15 rerank
     against a demo-corpus-sized candidate set, feeding Risk #3's
     "benchmark actual CPU latency in week one" mitigation.

These tests load the real CrossEncoder model (no mocking) since the
whole point of Step 18 / Risk #3 is measuring real CPU behavior, not
mocked behavior. First run will download the model (~80MB) - expect
that to be the slow part, not the predict() calls themselves.
"""

from __future__ import annotations

import time

import pytest

from src.retrieval.rerank import rerank, rerank_async


def test_rerank_orders_relevant_above_irrelevant():
    """FR-10's own verify: known relevant/irrelevant pair, correct order."""
    query = "What is the capital of France?"
    candidates = [
        {
            "chunk_id": "irrelevant",
            "text": "Bananas are a good source of potassium and are commonly eaten as a snack.",
        },
        {
            "chunk_id": "relevant",
            "text": "Paris is the capital and most populous city of France.",
        },
    ]

    result = rerank(query, candidates)

    assert len(result) == 2
    assert result[0]["chunk_id"] == "relevant"
    assert result[1]["chunk_id"] == "irrelevant"
    assert result[0]["rerank_score"] > result[1]["rerank_score"]


def test_rerank_respects_top_k_truncation():
    query = "What is the capital of France?"
    candidates = [
        {"chunk_id": f"c{i}", "text": f"Filler sentence number {i} about nothing in particular."}
        for i in range(10)
    ]
    result = rerank(query, candidates, top_k=5)
    assert len(result) == 5


def test_rerank_empty_candidates_returns_empty():
    assert rerank("any query", []) == []


def test_rerank_handles_content_key_fallback():
    """Some upstream candidate dicts may use 'content' instead of 'text' -
    Step 18 shouldn't silently score an empty string for those."""
    query = "capital of France"
    candidates = [{"chunk_id": "a", "content": "Paris is the capital of France."}]
    result = rerank(query, candidates)
    assert len(result) == 1
    assert result[0]["rerank_score"] != 0.0  # got scored against real text, not ""


@pytest.mark.asyncio
async def test_rerank_async_matches_sync_ordering():
    query = "What is the capital of France?"
    candidates = [
        {"chunk_id": "irrelevant", "text": "Bananas are a good source of potassium."},
        {"chunk_id": "relevant", "text": "Paris is the capital of France."},
    ]
    result = await rerank_async(query, candidates)
    assert result[0]["chunk_id"] == "relevant"


def test_rerank_top15_cpu_latency_benchmark():
    """
    Risk Register #3: 'benchmark actual CPU latency in week one, not
    week four.' Not a pass/fail gate by itself (NFR-1's 200ms/600ms
    p50/p95 targets get validated properly at Step 27's DoD pass) -
    this logs the real number now so a blowup is visible immediately,
    not discovered four weeks later during Phase 6.
    """
    query = "How does the fused retrieval score compare across sparse and dense legs for ambiguous multi-hop queries?"
    candidates = [
        {"chunk_id": f"c{i}", "text": f"This is demo-corpus-sized candidate chunk number {i}, "
                                       f"roughly representative of a 200-500 token retrieval result "
                                       f"used to benchmark cross-encoder CPU latency at top-15."}
        for i in range(15)
    ]

    start = time.perf_counter()
    result = rerank(query, candidates, top_k=15)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(result) == 15
    print(f"\n[Risk #3 benchmark] top-15 CPU rerank latency: {elapsed_ms:.1f}ms "
          f"(NFR-1 targets: p50=200ms, p95=600ms)")