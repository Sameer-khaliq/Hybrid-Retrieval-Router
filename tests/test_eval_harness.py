"""
Step 25 tests — run_eval() end-to-end aggregation.

Plan's stated Done-when: "run against a fixture eval set with
precomputed expected metric values, asserting output matches within
tolerance." run_query() is mocked here (per-query, keyed by query
text) so the expected aggregate numbers are hand-computable and exact,
independent of any real corpus content.

Hand-computed expectations (see comments per row below):
    Recall@10 mean = mean over ALL 4 rows (Q4 included via recall_at_k's
    own trivial edge case: nothing relevant + nothing retrieved -> 1.0)
        = (1.0 + 0.0 + 1.0 + 1.0) / 4 = 0.75
    MRR mean        = mean over ALL 4 rows (Q4's reciprocal_rank = 0.0,
    since its empty relevant set never matches anything, unlike
    recall_at_k's special-cased "nothing vs nothing" branch)
        = (1.0 + 0.0 + 1.0 + 0.0) / 4 = 0.5
    Routing accuracy = ONLY over routing-labeled rows (Q1-Q3; Q4 is
    gated, excluded per run_eval.py's own filter) = 2/3 correct
    (Q1 correct, Q2 wrong, Q3 correct) = 0.6667

    IMPORTANT: recall/MRR average over ALL rows including gated ones,
    but routing_accuracy only averages over LABELED rows - these two
    metrics have different denominators by design (see run_eval.py),
    which is exactly what caught a bug in this test's first draft: the
    original hand-computation used 3 as the denominator for recall/MRR
    too, missing Q4's contribution.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.eval.run_eval import run_eval

FAKE_RESULTS = {
    # Q1: relevant=[1,2], retrieved=[2,3,1] -> both found, first hit rank 1
    #     recall@10 = 2/2 = 1.0, RR = 1/1 = 1.0. Routing: predicted "fast" == expected "fast" -> correct.
    "Q1: What is a dividend?": {
        "gated": False, "routing": {"path": "fast"}, "degraded": False,
        "api_call_count": 1, "post_rerank_chunk_ids": [2, 3, 1],
    },
    # Q2: relevant=[5], retrieved=[9,8,7] -> no hit -> recall=0.0, RR=0.0.
    #     Routing: predicted "fast" != expected "deep" -> incorrect.
    "Q2: Compare tax treatment of dividends vs capital gains.": {
        "gated": False, "routing": {"path": "fast"}, "degraded": False,
        "api_call_count": 1, "post_rerank_chunk_ids": [9, 8, 7],
    },
    # Q3: relevant=[10,11], retrieved=[11,99,10] -> both found, first hit rank 1
    #     recall@10 = 2/2 = 1.0, RR = 1/1 = 1.0. Routing: predicted "fast" == expected "fast" -> correct.
    "Q3: What is an ETF?": {
        "gated": False, "routing": {"path": "fast"}, "degraded": False,
        "api_call_count": 1, "post_rerank_chunk_ids": [11, 99, 10],
    },
    # Q4: gated query - no routing decision, no relevant chunks labeled.
    # Included to verify run_eval() doesn't crash on gated rows and
    # correctly excludes them from routing-accuracy scoring.
    "Q4: Hi there": {
        "gated": True, "routing": None, "degraded": False,
        "api_call_count": 0, "post_rerank_chunk_ids": [],
    },
}


async def _fake_run_query(query: str, trace_id=None):
    return FAKE_RESULTS[query]


def _write_eval_set(tmp_path: Path) -> Path:
    rows = [
        {"query": "Q1: What is a dividend?", "relevant_chunk_ids": [1, 2], "expected_routing_path": "fast"},
        {"query": "Q2: Compare tax treatment of dividends vs capital gains.", "relevant_chunk_ids": [5], "expected_routing_path": "deep"},
        {"query": "Q3: What is an ETF?", "relevant_chunk_ids": [10, 11], "expected_routing_path": "fast"},
        {"query": "Q4: Hi there", "relevant_chunk_ids": [], "expected_routing_path": None},
    ]
    path = tmp_path / "eval_set.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.mark.asyncio
async def test_run_eval_computes_exact_aggregate_metrics(tmp_path):
    eval_set_path = _write_eval_set(tmp_path)

    with patch("src.eval.run_eval.run_query", new=AsyncMock(side_effect=_fake_run_query)):
        report = await run_eval(eval_set_path, recall_k=10)

    assert report["num_queries"] == 4
    assert report["num_routing_labeled"] == 3  # Q4 excluded (gated, no routing decision)

    assert report["recall_at_10"] == pytest.approx(0.75, rel=1e-3)
    assert report["mrr"] == pytest.approx(0.5, rel=1e-3)
    assert report["routing_accuracy"] == pytest.approx(2 / 3, rel=1e-3)

    assert report["nfr5_gate_recall_0.85"] is False  # 0.75 < 0.85
    assert report["nfr6_gate_routing_0.80"] is False  # 0.667 < 0.80


@pytest.mark.asyncio
async def test_run_eval_gated_row_does_not_crash_and_is_excluded():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        eval_set_path = _write_eval_set(Path(tmp))
        with patch("src.eval.run_eval.run_query", new=AsyncMock(side_effect=_fake_run_query)):
            report = await run_eval(eval_set_path, recall_k=10)

    gated_row = next(r for r in report["per_query"] if r["query"] == "Q4: Hi there")
    assert gated_row["gated"] is True
    assert gated_row["predicted_path"] is None


@pytest.mark.asyncio
async def test_run_eval_perfect_scores_pass_dod_gates(tmp_path):
    """Sanity check the gate logic itself with a trivially perfect set,
    independent of the imperfect fixture above."""
    rows = [{"query": "Q1: What is a dividend?", "relevant_chunk_ids": [2], "expected_routing_path": "fast"}]
    path = tmp_path / "perfect_set.jsonl"
    path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")

    with patch("src.eval.run_eval.run_query", new=AsyncMock(side_effect=_fake_run_query)):
        report = await run_eval(path, recall_k=10)

    assert report["recall_at_10"] == 1.0
    assert report["routing_accuracy"] == 1.0
    assert report["nfr5_gate_recall_0.85"] is True
    assert report["nfr6_gate_routing_0.80"] is True