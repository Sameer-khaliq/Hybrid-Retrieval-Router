"""
Manual interactive tester for Step 14's gating (prefilter.py).

Run it, type a query, see immediately which gate (if any) it matched.
Not a pytest file - a throwaway CLI for eyeballing behavior by hand.

Usage:
    uv run python scripts/manual_test_prefilter.py

Type 'quit' or 'exit' to stop.
"""

from src.gating.prefilter import run_prefilter

print("Prefilter manual tester. Type a query and press Enter.")
print("Type 'quit' or 'exit' to stop.\n")

while True:
    query = input("Query> ").strip()
    if query.lower() in ("quit", "exit"):
        print("Bye!")
        break
    if not query:
        continue

    result = run_prefilter(query, trace_id="manual_test")

    if result is None:
        print("  -> PASS THROUGH (would go to retrieval pipeline)\n")
    else:
        print(f"  -> GATED [{result['category']}] reason={result['reason']}")
        print(f"     response: {result['response']}\n")
