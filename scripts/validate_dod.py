"""
Step 27 — DoD-gate validation pass (NFR-2, 4, 5, 6, 8, 9, 11, 12).

No new application code, per the plan's own instruction - this script
measures what Steps 1-24 already built against the documented DoD
numbers. Three different kinds of checks:

  1. NFR-5, NFR-6 - computed fresh via Step 25's eval harness.
  2. NFR-2, NFR-4 - measured LIVE against a running /query endpoint.
     REQUIRES `docker compose up -d qdrant` and
     `uv run uvicorn src.api.main:app` already running in another
     terminal before this script is run - it will not start the server
     for you.
  3. NFR-8, NFR-9, NFR-11 - structural gates: re-runs the existing test
     files that already exercise retry/backoff, degraded-mode fallback,
     and observability completeness. This script doesn't duplicate that
     logic, it just confirms those suites still pass as the final gate.
  4. NFR-12 - not measurable; this script WRITES
     docs/SCALABILITY_BOUNDARY.md, the explicit documentation NFR-12
     itself requires ("must be explicitly documented, not discovered
     by the client").

IMPORTANT CAVEAT on NFR-2's numbers: this takes a small number of live
samples per path (default 5) and reports min/median/max - NOT a
statistically real p50/p95, which needs far more samples under
realistic load. Treat this script's NFR-2 output as a smoke-test sanity
check, not a validated final number. REQUIREMENTS.md's own note about
a prior project's projected numbers coming in ~57x over once actually
load-tested is the reason to distrust small-sample numbers here.

Usage:
    uv run python scripts/validate_dod.py --eval-set tests/fixtures/eval_set.jsonl
    uv run python scripts/validate_dod.py --skip-live   # if server isn't running
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

from src.eval.run_eval import run_eval
from src.retrieval.sparse_bm25 import get_or_build_index
from src.retrieval.rerank import preload_reranker
preload_reranker()
print("[dod] Reranker warmed up.")
# phir eval loop shuru karo
API_BASE = "http://localhost:8000"

STRUCTURAL_GATES = {
    "NFR-8 (retry/backoff)": ["tests/test_embedding_fallback.py"],
    "NFR-9 (degraded-mode fallback)": [
        "tests/test_routing_cascade.py",
        "tests/test_generation_streaming.py",
        "tests/test_tavily_fallback.py",
    ],
    "NFR-11 (observability fields)": ["tests/test_observability.py"],
}

SCALABILITY_BOUNDARY_CONTENT = """\
# Scalability Boundary

**This is not a production-scale system as configured.** Per NFR-12,
documenting this explicitly rather than leaving it for a client to
discover is a DoD gate, not an afterthought.

## Known limits

- **In-memory BM25**: the full corpus must fit in RAM. Fine for low
  tens of thousands of chunks; not enterprise document volumes.
  `scripts/build_demo_corpus.py` warns if the corpus crosses ~20k
  chunks, but the real ceiling depends on available machine RAM, not a
  fixed number this system enforces.
- **Single-node Qdrant** via docker-compose: no horizontal scaling, no
  replication, no failover. A container restart loses in-flight state
  (though the persistent volume preserves vector data across restarts).
- **Free-tier API rate limits** (Groq, Gemini, Tavily) cap real
  concurrent throughput far below what the architecture could
  theoretically support on paid infrastructure. Design target is
  single-digit concurrent users comfortably (NFR-3) - untested and
  unvalidated beyond that.
- **No horizontal scaling, multi-node deployment, or managed cloud
  infra** - explicitly out of scope for v1 (REQUIREMENTS.md §4.1).

## What this project demonstrates

The architecture and reasoning that *would* scale:
- A managed/clustered vector store (e.g. Qdrant Cloud, or a sharded
  self-hosted cluster) in place of single-node docker-compose Qdrant.
- Paid model tiers removing the free-tier rate-limit ceiling.
- A queue + horizontal API replicas in front of the FastAPI surface.
- Semantic caching (FR-30, Step 33) reducing redundant API calls at
  scale, once embedding-config stability is verified (Risk #5).

This is **not** a system that already scales - it's a system whose
design choices don't preclude scaling later, demonstrated at a scope
appropriate for a solo portfolio build under free-tier constraints.
"""


def run_structural_gate(label: str, test_paths: list[str]) -> bool:
    print(f"[dod] Running structural gate: {label} ({' '.join(test_paths)}) ...")
    result = subprocess.run(
        ["uv", "run", "pytest", *test_paths, "-q"],
        capture_output=True, text=True,check=False,
    )
    passed = result.returncode == 0
    print(result.stdout[-1500:])
    if not passed:
        print(result.stderr[-1500:])
    return passed


def measure_latency(query: str, samples: int) -> dict:
    timings_ms: list[float] = []
    api_call_count = None
    degraded = None

    for _ in range(samples):
        start = time.perf_counter()
        with (
            httpx.Client(timeout=30.0) as client,
            client.stream("POST", f"{API_BASE}/query", json={"query": query}) as response,
        ):
            last_data_line = None
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    last_data_line = line
                elapsed_ms = (time.perf_counter() - start) * 1000
                timings_ms.append(elapsed_ms)
                if last_data_line:
                    payload = json.loads(last_data_line[len("data: "):])
                    api_call_count = payload.get("api_call_count", api_call_count)
                    degraded = payload.get("degraded", degraded)

    return {
        "min_ms": round(min(timings_ms), 1),
        "median_ms": round(statistics.median(timings_ms), 1),
        "max_ms": round(max(timings_ms), 1),
        "samples": samples,
        "api_call_count": api_call_count,
        "degraded": degraded,
    }


def write_scalability_doc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SCALABILITY_BOUNDARY_CONTENT, encoding="utf-8")


def main() -> None:
    print("Initializing BM25 index...")
    get_or_build_index()
    parser = argparse.ArgumentParser(description="Step 27 - DoD-gate validation pass")
    parser.add_argument("--eval-set", default="tests/fixtures/eval_set.jsonl")
    parser.add_argument("--fast-query", default="What is a dividend?")
    parser.add_argument(
        "--deep-query",
        default="Compare the tax treatment of dividends versus capital gains for a retirement account.",
    )
    parser.add_argument("--samples", type=int, default=5, help="Live latency samples per path (small-sample, see caveat)")
    parser.add_argument("--skip-live", action="store_true", help="Skip NFR-2/NFR-4 live checks (server not running)")
    args = parser.parse_args()

    results: dict[str, bool | None] = {}

    # --- NFR-5 / NFR-6 via eval harness ---
    print("[dod] Running eval harness for NFR-5/NFR-6 ...")
    report = asyncio.run(run_eval(Path(args.eval_set), recall_k=10))
    results["NFR-5 (Recall@10 >= 0.85)"] = report["nfr5_gate_recall_0.85"]
    results["NFR-6 (Routing accuracy >= 0.80)"] = report["nfr6_gate_routing_0.80"]
    print(f"[dod]   Recall@10={report['recall_at_10']}, routing_accuracy={report['routing_accuracy']}")

    # --- NFR-2 / NFR-4 via live endpoint ---
    if not args.skip_live:
        print(f"[dod] Measuring live latency ({args.samples} samples/path, small-sample - see script docstring) ...")
        try:
            fast_stats = measure_latency(args.fast_query, samples=args.samples)
            deep_stats = measure_latency(args.deep_query, samples=args.samples)
            print(f"[dod]   fast-path: {fast_stats}")
            print(f"[dod]   deep-path: {deep_stats}")
            results["NFR-2 fast-path (median <= 1000ms)"] = fast_stats["median_ms"] <= 1000
            results["NFR-2 deep-path (median <= 2000ms)"] = deep_stats["median_ms"] <= 2000
            worst_case_count = max(fast_stats["api_call_count"] or 0, deep_stats["api_call_count"] or 0)
            results["NFR-4 (api_call_count <= 4)"] = worst_case_count <= 4
        except Exception as exc:  # noqa: BLE001
            print("[dod] Live checks failed - is the server running? uv run uvicorn src.api.main:app")
            print(f"[dod] Error: {exc}")
            results["NFR-2 (skipped - server unreachable)"] = None
            results["NFR-4 (skipped - server unreachable)"] = None
    else:
        results["NFR-2 (skipped by --skip-live)"] = None
        results["NFR-4 (skipped by --skip-live)"] = None

    # --- NFR-8 / NFR-9 / NFR-11 via structural test re-runs ---
    for label, paths in STRUCTURAL_GATES.items():
        results[label] = run_structural_gate(label, paths)

    # --- NFR-12: always write the doc ---
    write_scalability_doc(Path("docs/SCALABILITY_BOUNDARY.md"))
    results["NFR-12 (scalability boundary documented)"] = True

    print("\n=== DoD Gate Summary ===")
    for label, passed in results.items():
        status = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        print(f"  [{status}] {label}")

    any_fail = any(v is False for v in results.values())
    if any_fail:
        print("\n[dod] One or more gates FAILED. This is expected on a first pass -")
        print("[dod] especially NFR-5/NFR-6 against the placeholder eval_set.jsonl,")
        print("[dod] and NFR-2 given the fast-path model swap to gpt-oss-20b.")
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()