# Test fixtures -- not yet populated

- `sample.pdf` -- small multi-page PDF for ingestion tests (Steps 5-7)
- `eval_set.jsonl` -- labeled query set with precomputed expected metric
  values, validates the eval harness itself (Step 25)
- `demo_queries.jsonl` -- rehearsed query set exercising fast-path,
  deep-path, the LLM-router mid-band, the Tavily trigger, and each of
  FR-21-24's gating categories (Step 26)

Left empty deliberately -- fake stand-ins for these would make tests
silently pass or fail in confusing ways rather than clearly signaling
"not yet populated."
