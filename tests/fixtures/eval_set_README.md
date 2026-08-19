# eval_set.jsonl — PLACEHOLDER DATA, replace before trusting Recall@k/NFR-5

`relevant_chunk_ids: [0]` in every row is a **placeholder**, not real
data. I don't have access to your actual ingested FiQA corpus, so I
can't know which real `chunk_id`s are genuinely relevant to each query.

Running `run_eval.py` against this file as-is will produce a
meaningless Recall@k number (either near-0 if chunk_id 0 was never
ingested, or a false-positive 1.0 if it happens to rank highly for
unrelated reasons).

## To fix

1. Ingest your demo corpus (Step 26).
2. For each query below, manually inspect (or query Qdrant directly)
   for the chunk_id(s) that actually answer it correctly.
3. Replace `[0]` with the real chunk_id(s), e.g. `[1234567890123, 987654321]`.
4. Add more rows - 3 queries is nowhere near enough for a trustworthy
   NFR-5/NFR-6 measurement. Aim for at least 15-20 for a real DoD pass.

`expected_routing_path` is a genuine label (not a placeholder) reflecting
what a human would judge fast vs deep for each query - but recalibrate
this too once you see how your actual complexity-score weights (§3.4)
behave on real queries.