"""
One-time demo corpus downloader (Risk #4 mitigation, pulled forward from
Step 26 to run in parallel with early build steps).

Downloads the BEIR FiQA dataset (finance Q&A, free, no signup) and lays
it out as:

    data/corpus/fiqa/documents/{doc_id}.txt   <- individual docs (FR-1 ingests these)
    data/eval/fiqa/queries.jsonl              <- test questions
    data/eval/fiqa/qrels.tsv                  <- relevance judgments (FR-18)

Why a subset, not the full ~57k-doc FiQA corpus:
  - NFR-12 caps in-memory BM25 at "low tens of thousands of chunks" -
    a smaller, deliberately-sized corpus keeps you comfortably inside
    that boundary instead of testing it.
  - Free-tier Gemini embedding rate limits (Risk #1) make embedding
    57k documents slow and quota-risky; a few thousand is enough to
    demo and evaluate against meaningfully.

Subset strategy: every document referenced as "relevant" in the qrels
is KEPT (so Recall@10/MRR stay measurable - FR-18/NFR-5), then padded
with random non-relevant documents up to --corpus-size. Queries are
filtered to only those whose relevant docs all survived the subsample.

This is a one-time local script, not part of the runtime pipeline - run
it with `uv run --with datasets python scripts/download_demo_corpus.py`
so the `datasets` package doesn't get added to pyproject.toml.

Usage:
    uv run --with datasets python scripts/download_demo_corpus.py
    uv run --with datasets python scripts/download_demo_corpus.py --corpus-size 2000
"""

import argparse
import csv
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-size", type=int, default=1200,
        help="Target total number of documents in the demo corpus (default: 1200)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="data",
        help="Project data/ directory (default: data)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed, for a reproducible subsample",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    from datasets import load_dataset

    print("Downloading FiQA corpus, queries, and qrels from HuggingFace (BeIR/fiqa)...")
    corpus_ds = load_dataset("BeIR/fiqa", "corpus")["corpus"]
    queries_ds = load_dataset("BeIR/fiqa", "queries")["queries"]
    qrels_ds = load_dataset("BeIR/fiqa-qrels")["test"]

    print(f"  corpus:  {len(corpus_ds)} documents")
    print(f"  queries: {len(queries_ds)} total (all splits)")
    print(f"  qrels:   {len(qrels_ds)} relevance judgments (test split)")

    # ---- Build query_id -> text lookup ------------------------------
    # IMPORTANT: BeIR's qrels columns (corpus-id, query-id) load as
    # int64, while corpus/queries "_id" columns load as str. Cast
    # everything to str consistently the moment it's read, or lookups
    # below silently mismatch (int 233472 != str "233472" -> KeyError).
    query_text = {str(row["_id"]): row["text"] for row in queries_ds}

    # ---- Figure out which doc_ids MUST survive the subsample -----------
    relevant_doc_ids = {str(row["corpus-id"]) for row in qrels_ds}
    print(f"  {len(relevant_doc_ids)} distinct documents are relevant to at least one test query")

    if len(relevant_doc_ids) > args.corpus_size:
        print(
            f"  WARNING: relevant-doc count ({len(relevant_doc_ids)}) exceeds "
            f"--corpus-size ({args.corpus_size}). Increasing corpus size to fit "
            f"all relevant docs, or Recall@10 would be measured against an "
            f"artificially incomplete corpus."
        )
        args.corpus_size = len(relevant_doc_ids)

    # ---- Build the corpus subset: all relevant docs + random padding ----
    all_doc_ids = [str(row["_id"]) for row in corpus_ds]
    doc_by_id = {str(row["_id"]): row for row in corpus_ds}

    remaining_slots = args.corpus_size - len(relevant_doc_ids)
    candidate_pool = [d for d in all_doc_ids if d not in relevant_doc_ids]
    random.shuffle(candidate_pool)
    padding_ids = set(candidate_pool[:remaining_slots])

    keep_ids = relevant_doc_ids | padding_ids
    print(f"Final corpus subset size: {len(keep_ids)} documents")

    # ---- Write documents as individual .txt files (FR-1 ingests these) ---
    corpus_dir = Path(args.out_dir) / "corpus" / "fiqa" / "documents"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    for doc_id in keep_ids:
        doc = doc_by_id[doc_id]
        title = doc.get("title", "").strip()
        text = doc.get("text", "").strip()
        content = f"{title}\n\n{text}" if title else text
        # doc_id from FiQA is already filesystem-safe (numeric string)
        (corpus_dir / f"{doc_id}.txt").write_text(content, encoding="utf-8")

    print(f"Wrote {len(keep_ids)} documents to {corpus_dir}")

    # ---- Filter qrels + queries to only the surviving corpus subset -----
    eval_dir = Path(args.out_dir) / "eval" / "fiqa"
    eval_dir.mkdir(parents=True, exist_ok=True)

    kept_qrels = [row for row in qrels_ds if str(row["corpus-id"]) in keep_ids]
    kept_query_ids = {str(row["query-id"]) for row in kept_qrels}

    qrels_path = eval_dir / "qrels.tsv"
    with open(qrels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query_id", "doc_id", "relevance"])
        for row in kept_qrels:
            writer.writerow([str(row["query-id"]), str(row["corpus-id"]), row["score"]])
    print(f"Wrote {len(kept_qrels)} qrels rows to {qrels_path}")

    queries_path = eval_dir / "queries.jsonl"
    with open(queries_path, "w", encoding="utf-8") as f:
        for qid in sorted(kept_query_ids):
            if qid in query_text:
                f.write(json.dumps({"query_id": qid, "text": query_text[qid]}) + "\n")
    print(f"Wrote {len(kept_query_ids)} queries to {queries_path}")

    print("\nDone. Next: run this corpus through Step 5's chunking (FR-1) once built.")


if __name__ == "__main__":
    main()