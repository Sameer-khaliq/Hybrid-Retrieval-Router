"""
Step 25 — Batch evaluation metrics (FR-18).

Pure functions, no I/O - each takes already-collected per-query lists
so they're independently unit-testable against synthetic, hand-computed
expected values (see tests/test_metrics.py), separate from run_eval.py's
job of actually running queries through the live pipeline.
"""
from __future__ import annotations


def recall_at_k(retrieved_chunk_ids: list, relevant_chunk_ids: list, k: int) -> float:
    """
    NFR-5: Recall@10 >= 0.85 is the DoD gate. This computes it for one
    query; mean_recall_at_k averages across a query set.

    Edge case: a query with NO labeled relevant chunks is trivially
    "fully recalled" if nothing was retrieved either (1.0), and 0.0 if
    something WAS retrieved for a query that shouldn't have matched
    anything - this only matters for eval-set rows representing
    genuinely out-of-corpus queries, which is an unusual thing to put
    in a Recall@k set in the first place; flagging rather than hiding.
    """
    if not relevant_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0
    top_k = set(retrieved_chunk_ids[:k])
    relevant = set(relevant_chunk_ids)
    return len(top_k & relevant) / len(relevant)


def mean_recall_at_k(per_query_retrieved: list[list], per_query_relevant: list[list], k: int) -> float:
    if not per_query_retrieved:
        return 0.0
    scores = [
        recall_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(per_query_retrieved, per_query_relevant)
    ]
    return sum(scores) / len(scores)


def reciprocal_rank(retrieved_chunk_ids: list, relevant_chunk_ids: list) -> float:
    """1/rank of the FIRST relevant chunk found in retrieved (1-indexed),
    0.0 if none of the relevant chunks appear at all."""
    relevant = set(relevant_chunk_ids)
    for i, cid in enumerate(retrieved_chunk_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(per_query_retrieved: list[list], per_query_relevant: list[list]) -> float:
    if not per_query_retrieved:
        return 0.0
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(per_query_retrieved, per_query_relevant)
    ]
    return sum(scores) / len(scores)


def routing_accuracy(predicted_paths: list[str], expected_paths: list[str]) -> float:
    """
    NFR-6: >= 80% agreement with human-labeled complexity is the DoD
    gate. Caller (run_eval.py) is responsible for excluding gated
    queries (routing=None) from these lists before calling this -
    gating means routing never ran, so there's no routing decision to
    score accuracy against.
    """
    if not predicted_paths:
        return 0.0
    matches = sum(1 for p, e in zip(predicted_paths, expected_paths) if p == e)
    return matches / len(predicted_paths)