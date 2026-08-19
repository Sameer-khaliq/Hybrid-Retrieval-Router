"""
Step 25 tests — metrics.py, synthetic hand-computed values.
"""
from __future__ import annotations

import pytest

from src.eval.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    routing_accuracy,
)


def test_recall_at_k_full_hit():
    assert recall_at_k([1, 2, 3], [1, 2], k=10) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k([1, 9, 9], [1, 2], k=10) == 0.5


def test_recall_at_k_respects_k_cutoff():
    # relevant chunk "2" is at rank 5, outside k=3
    assert recall_at_k([9, 9, 9, 9, 2], [1, 2], k=3) == 0.0


def test_recall_at_k_no_relevant_chunks_and_nothing_retrieved():
    assert recall_at_k([], [], k=10) == 1.0


def test_recall_at_k_no_relevant_chunks_but_something_retrieved():
    assert recall_at_k([1, 2], [], k=10) == 0.0


def test_mean_recall_at_k_averages_across_queries():
    retrieved = [[1, 2, 3], [9, 9, 9]]
    relevant = [[1, 2], [1]]
    # query 1: recall=1.0, query 2: recall=0.0 -> mean=0.5
    assert mean_recall_at_k(retrieved, relevant, k=10) == 0.5


def test_reciprocal_rank_first_position():
    assert reciprocal_rank([5, 9, 9], [5]) == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank([9, 9, 5], [5]) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_match():
    assert reciprocal_rank([9, 9, 9], [5]) == 0.0


def test_mean_reciprocal_rank_averages_across_queries():
    retrieved = [[5, 9], [9, 5]]
    relevant = [[5], [5]]
    # query 1: RR=1.0 (rank 1), query 2: RR=0.5 (rank 2) -> mean=0.75
    assert mean_reciprocal_rank(retrieved, relevant) == 0.75


def test_routing_accuracy_all_correct():
    assert routing_accuracy(["fast", "deep"], ["fast", "deep"]) == 1.0


def test_routing_accuracy_partial():
    assert routing_accuracy(["fast", "fast", "deep"], ["fast", "deep", "deep"]) == pytest.approx(2 / 3)


def test_routing_accuracy_empty():
    assert routing_accuracy([], []) == 0.0