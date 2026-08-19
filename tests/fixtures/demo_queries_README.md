# Step 26 — Demo corpus & query-set rehearsal

## 1. Ingest the demo corpus

```powershell
uv run python scripts/build_demo_corpus.py --dir data/corpus/fiqa/documents
```

Runs Step 6/7's ingestion (dedup + embed + Qdrant upsert) across every
PDF/TXT/MD file in the folder, then rebuilds BM25 (Step 8) against the
full corpus. Skips files whose content was already ingested (dedup, FR-3).

Watch for the NFR-12 warning if chunk count crosses ~20k - flags it,
doesn't block the build.

## 2. Rehearse the demo query set

`demo_queries.jsonl` has 4 rows marked **PLACEHOLDER** in their `note`
field - these genuinely cannot be filled in without your real corpus
and real routing behavior:

- `mid_band_llm_router` - needs a query that actually lands between
  your real `tau_low`/`tau_high` thresholds. Run it through
  `compute_complexity_score()` directly first to check.
- `tavily_trigger_low_confidence` - needs a query engineered to score
  below `tavily_confidence_floor` against your ACTUAL corpus - I can't
  predict retrieval scores I've never seen.

Everything else (gating categories, fast/deep path, currency-trigger
Tavily) should work close to as-is, but verify against your real
corpus content - "What is a dividend?" only works as a fast-path
example if your FiQA corpus actually has dividend-related documents.

Run the rehearsal:

```powershell
uv run python -m src.eval.run_eval --set tests/fixtures/demo_queries.jsonl --out reports/demo_rehearsal.json
```

Per Step 26's own Done-when: **manually read** `reports/demo_rehearsal.json`
and the structured logs from that run, confirming each row's `category`
actually routed/gated/fell-back the way its `note` describes. This
script measures Recall@k/MRR/routing-accuracy same as a normal eval run,
but that's secondary here - the real check is qualitative, per-category.