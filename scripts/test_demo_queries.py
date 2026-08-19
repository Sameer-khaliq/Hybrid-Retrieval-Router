import asyncio
import json
from pathlib import Path

from src.pipeline.run_query import run_query
from src.retrieval.rerank import preload_reranker
from src.retrieval.sparse_bm25 import get_or_build_index

DEMO_FILE = Path("tests/fixtures/demo_queries.jsonl")


async def test_all_demo_queries():
    # 1. Initialize BM25 in-memory index
    print("[init] Loading BM25 index into memory...")
    get_or_build_index()

    # 2. Preload & warmup cross-encoder reranker
    print("[init] Preloading CrossEncoder reranker...")
    preload_reranker()

    if not DEMO_FILE.exists():
        print(f"File not found: {DEMO_FILE}")
        return

    with open(DEMO_FILE, "r", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f if line.strip()]

    print(f"\n{'='*75}")
    print(f"Running Step 26 Rehearsal on {len(queries)} Demo Queries")
    print(f"{'='*75}\n")

    for i, item in enumerate(queries, 1):
        cat = item["category"]
        query = item["query"]
        expected_path = item.get("expected_routing_path")

        print(f"[{i}/{len(queries)}] Category: {cat}")
        print(f"    Query: '{query}'")

        res = await run_query(query)

        gated = res.get("gated", False)
        routing = res.get("routing")
        path = routing["path"] if routing else None
        tavily = res.get("tavily_triggered", False)
        num_chunks = len(res.get("chunks") or [])
        total_ms = res.get("latency_breakdown", {}).get("pipeline_total_ms", 0)

        print(f"    -> Gated: {gated} | Routing Path: {path} (Expected: {expected_path}) | Tavily: {tavily}")
        print(f"    -> Chunks Returned: {num_chunks} | Pipeline Latency: {total_ms}ms\n")


if __name__ == "__main__":
    asyncio.run(test_all_demo_queries())