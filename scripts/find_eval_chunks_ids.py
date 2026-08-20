"""
scripts/find_eval_chunk_ids.py

Helps fill in real chunk_ids for tests/fixtures/eval_set.jsonl.

For every query in eval_set.jsonl, this runs BOTH your dense (Qdrant) and
sparse (BM25) retrieval paths, merges/dedupes the candidates, and prints
each candidate's chunk_id + a text snippet + which leg(s) surfaced it.

You then manually read the snippets and decide which chunk_id(s) actually
answer the query. Paste those into eval_set.jsonl's relevant_chunk_ids.

Run from project root:
    uv run python scripts/find_eval_chunk_ids.py

Optional flags:
    --file tests/fixtures/eval_set.jsonl   (default)
    --top-k 10                             (candidates shown per query)
    --out tests/fixtures/eval_set.candidates.jsonl
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

from src.common.qdrant_client import get_client
from src.retrieval.dense_qdrant import query_dense
from src.retrieval.sparse_bm25 import get_or_build_index, query_bm25


def load_eval_set(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def call_maybe_async(fn, *args, **kwargs):
    """Call fn regardless of whether it's sync or async, and return the
    resolved (non-coroutine) result. Avoids assuming either shape."""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        import asyncio
        result = asyncio.run(result)
    return result


def get_payload_text(qdrant_client, collection: str, chunk_id) -> str:
    try:
        points = qdrant_client.retrieve(
            collection_name=collection,
            ids=[chunk_id],
            with_payload=True,
        )
        if points:
            payload = points[0].payload or {}
            for key in ("text", "chunk_text", "content", "raw_text"):
                if key in payload:
                    return str(payload[key])[:300]
            return str(payload)[:300]
    except Exception as e:
        return f"<payload fetch failed: {e}>"
    return "<no payload found>"


def extract_id_score(r):
    """Normalize a single result item (dict, object, or tuple) into (chunk_id, score)."""
    if isinstance(r, dict):
        cid = r.get("chunk_id") or r.get("id")
        score = r.get("score")
    elif isinstance(r, (tuple, list)):
        cid, score = r[0], r[1]
    else:
        cid = getattr(r, "chunk_id", None) or getattr(r, "id", None)
        score = getattr(r, "score", None)
    return cid, score


def find_candidates(query: str, top_k: int, collection: str):
    dense_raw = call_maybe_async(query_dense, query, top_n=top_k)
    sparse_raw = call_maybe_async(query_bm25, query, top_k)

    client = get_client()
    seen = {}

    for rank, r in enumerate(dense_raw, start=1):
        cid, score = extract_id_score(r)
        seen.setdefault(cid, {"dense_rank": None, "sparse_rank": None, "dense_score": None, "sparse_score": None})
        seen[cid]["dense_rank"] = rank
        seen[cid]["dense_score"] = score

    for rank, r in enumerate(sparse_raw, start=1):
        cid, score = extract_id_score(r)
        seen.setdefault(cid, {"dense_rank": None, "sparse_rank": None, "dense_score": None, "sparse_score": None})
        seen[cid]["sparse_rank"] = rank
        seen[cid]["sparse_score"] = score

    candidates = []
    for cid, info in seen.items():
        text = get_payload_text(client, collection, cid)
        candidates.append({"chunk_id": cid, "text_preview": text, **info})

    def sort_key(c):
        ranks = [r for r in (c["dense_rank"], c["sparse_rank"]) if r is not None]
        return min(ranks) if ranks else 999

    candidates.sort(key=sort_key)
    return candidates[:top_k]


def main():
    print("Initializing BM25 index...")
    get_or_build_index()
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="tests/fixtures/eval_set.jsonl")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--collection", default="chunks")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: {path} not found. Run from project root.", file=sys.stderr)
        sys.exit(1)

    rows = load_eval_set(path)
    print(f"Loaded {len(rows)} eval queries from {path}\n")

    dump = []
    for i, row in enumerate(rows, start=1):
        query = row.get("query") or row.get("question")
        if not query:
            print(f"[{i}] SKIP — no 'query' field in row: {row}")
            continue

        print("=" * 100)
        print(f"[{i}/{len(rows)}] QUERY: {query}")
        print(f"    current relevant_chunk_ids (placeholder or real): {row.get('relevant_chunk_ids')}")
        print(f"    expected_routing_path: {row.get('expected_routing_path')}")
        print("-" * 100)

        candidates = find_candidates(query, args.top_k, args.collection)
        for c in candidates:
            print(
                f"  chunk_id={c['chunk_id']}  "
                f"dense_rank={c['dense_rank']} (score={c['dense_score']})  "
                f"sparse_rank={c['sparse_rank']} (score={c['sparse_score']})"
            )
            print(f"    text: {c['text_preview']!r}")
        print()

        dump.append({"query": query, "candidates": candidates, "row": row})

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(d) + "\n" for d in dump)
        print(f"\nWrote full candidate dump to {args.out}")

    print(
        "\nNow manually pick the chunk_id(s) that genuinely answer each query "
        "and update relevant_chunk_ids in your eval_set.jsonl by hand."
    )


if __name__ == "__main__":
    main()