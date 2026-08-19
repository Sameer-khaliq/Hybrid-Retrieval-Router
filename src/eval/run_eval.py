"""
Step 25 — Batch evaluation entrypoint (FR-18).

Runs a labeled query set (JSONL) through Step 21's run_query() pipeline
and reports Recall@k, MRR, and routing-decision accuracy to a report
file.

DESIGN NOTE: this calls run_query() ONLY, not the generation layer
(generate_streaming/generate_atomic). Recall@k, MRR, and routing
accuracy are all retrieval/routing-level metrics - none of them need a
generated answer, so skipping generation here saves real Groq API
calls on every eval run (relevant given free-tier rate limits, NFR-3)
without losing anything the three DoD-gated metrics need.

Eval set format (one JSON object per line):
    {
        "query": "What is a dividend?",
        "relevant_chunk_ids": [10234, 88123],
        "expected_routing_path": "fast" | "deep" | null
    }
"relevant_chunk_ids" and "expected_routing_path" are both optional per
row - a row missing "expected_routing_path" (or null) is excluded from
the routing-accuracy calculation, not treated as a mismatch.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.eval.metrics import (
    mean_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    routing_accuracy,
)
from src.pipeline.run_query import run_query

NFR5_RECALL_GATE = 0.85
NFR6_ROUTING_ACCURACY_GATE = 0.80


def load_eval_set(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def run_eval(eval_set_path: Path, recall_k: int = 10) -> dict[str, Any]:
    rows = load_eval_set(eval_set_path)

    per_query_retrieved: list[list] = []
    per_query_relevant: list[list] = []
    predicted_paths: list[str] = []
    expected_paths: list[str] = []
    per_query_details: list[dict] = []

    for row in rows:
        query = row["query"]
        relevant_chunk_ids = row.get("relevant_chunk_ids", [])
        expected_path = row.get("expected_routing_path")

        result = await run_query(query)

        retrieved = result.get("post_rerank_chunk_ids") or []
        per_query_retrieved.append(retrieved)
        per_query_relevant.append(relevant_chunk_ids)

        predicted_path = result["routing"]["path"] if result["routing"] is not None else None
        if expected_path is not None and predicted_path is not None:
            predicted_paths.append(predicted_path)
            expected_paths.append(expected_path)

        per_query_details.append({
            "query": query,
            "gated": result["gated"],
            "predicted_path": predicted_path,
            "expected_path": expected_path,
            "retrieved_chunk_ids": retrieved,
            "relevant_chunk_ids": relevant_chunk_ids,
            "recall_at_k": round(recall_at_k(retrieved, relevant_chunk_ids, recall_k), 4),
            "reciprocal_rank": round(reciprocal_rank(retrieved, relevant_chunk_ids), 4),
            "degraded": result["degraded"],
            "api_call_count": result.get("api_call_count"),
        })

    recall = mean_recall_at_k(per_query_retrieved, per_query_relevant, recall_k)
    mrr = mean_reciprocal_rank(per_query_retrieved, per_query_relevant)
    routing_acc = routing_accuracy(predicted_paths, expected_paths) if predicted_paths else None

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "eval_set": str(eval_set_path),
        "num_queries": len(rows),
        "num_routing_labeled": len(predicted_paths),
        f"recall_at_{recall_k}": round(recall, 4),
        "mrr": round(mrr, 4),
        "routing_accuracy": round(routing_acc, 4) if routing_acc is not None else None,
        "nfr5_gate_recall_0.85": recall >= NFR5_RECALL_GATE,
        "nfr6_gate_routing_0.80": (routing_acc is not None and routing_acc >= NFR6_ROUTING_ACCURACY_GATE),
        "per_query": per_query_details,
    }


def write_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch evaluation (FR-18): Recall@k, MRR, routing accuracy."
    )
    parser.add_argument("--set", required=True, help="Path to labeled eval set (JSONL)")
    parser.add_argument("--k", type=int, default=10, help="Recall@k cutoff (default 10, per NFR-5)")
    parser.add_argument("--out", default="reports/eval_report.json", help="Output report path")
    args = parser.parse_args()

    report = asyncio.run(run_eval(Path(args.set), recall_k=args.k))
    write_report(report, Path(args.out))

    print(f"[eval] {report['num_queries']} queries evaluated ({report['num_routing_labeled']} with routing labels).")
    print(
        f"[eval] Recall@{args.k}: {report[f'recall_at_{args.k}']} "
        f"(NFR-5 gate >=0.85: {'PASS' if report['nfr5_gate_recall_0.85'] else 'FAIL'})"
    )
    print(f"[eval] MRR: {report['mrr']}")
    if report["routing_accuracy"] is not None:
        print(
            f"[eval] Routing accuracy: {report['routing_accuracy']} "
            f"(NFR-6 gate >=0.80: {'PASS' if report['nfr6_gate_routing_0.80'] else 'FAIL'})"
        )
    else:
        print("[eval] Routing accuracy: n/a (no rows with expected_routing_path)")
    print(f"[eval] Report written to {args.out}")


if __name__ == "__main__":
    main()