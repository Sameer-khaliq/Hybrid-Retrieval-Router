"""
Step 26 — Demo corpus curation helper (Risk #4 mitigation).

Batch-ingests every document in a folder through the existing Step 6/7
pipeline (process_document + embed_and_upsert), then rebuilds the BM25
index (Step 8) so both retrieval legs are ready against the same corpus.

Not a new pipeline - this is a thin batch wrapper around functions that
already exist and are already tested (Steps 6-8). Several things ARE
new here, all added after real failure modes surfaced on a live
1706-document run:

  1. A Qdrant reachability pre-flight check, BEFORE touching any
     documents - prevents burning through the whole corpus against a
     down Qdrant.
  2. embed_and_upsert() calls wrapped in Step 11's with_retry (NFR-8) -
     protects against transient failures (a single dropped connection,
     a momentary 5xx), though NOT against genuine daily-quota exhaustion
     (retrying a 1000-RPD-exhausted call just burns the retry budget
     for nothing - quota resets on Google's clock, not on a backoff
     timer).
  3. HASH ROLLBACK ON EMBED FAILURE (the important one): process_document()
     (Step 6) unconditionally saves its content-hash store even when the
     downstream embed_and_upsert() call fails afterward - meaning ANY
     embed failure (Qdrant down, rate limit, network blip, Ctrl+C at
     the wrong moment) leaves that document's chunks marked "already
     ingested" for dedup purposes (FR-3) despite never having reached
     Qdrant. A later rerun then silently SKIPS that document forever,
     believing it was already embedded. This script now explicitly
     removes (rolls back) a failed document's chunk hashes from the
     persisted store the moment its embed call fails, so a rerun will
     correctly retry it instead of silently losing it.

KNOWN REMAINING GAP: BM25 rebuild only happens at the very end of a
full run. If this script is interrupted (rate-limit exhaustion, Ctrl+C,
crash) before reaching that point, Qdrant will have the newly-embedded
chunks but BM25's on-disk index will NOT reflect them yet. Run this
manually after ANY partial/interrupted run, before querying:
    uv run python -m src.retrieval.sparse_bm25 --rebuild

Usage:
    uv run python scripts/build_demo_corpus.py --dir data/corpus/fiqa/documents
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from config.settings import settings
from src.common.qdrant_client import get_client
from src.common.resilience import RetriesExhaustedError, with_retry
from src.ingestion.pipeline import (
    HASH_STORE_PATH,
    embed_and_upsert,
    load_hash_store,
    process_document,
    save_hash_store,
)
from src.retrieval.sparse_bm25 import get_or_build_index

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Soft warning threshold, not a hard limit - NFR-12's own wording is
# "fine to low tens of thousands," not a precise number. Flag loudly if
# crossed, but don't refuse to build - the actual ceiling depends on
# available RAM, which this script has no way to check reliably.
CHUNK_COUNT_WARNING_THRESHOLD = 20_000


def find_documents(directory: Path) -> list[Path]:
    return [
        p for p in sorted(directory.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def check_qdrant_reachable() -> None:
    """Fail fast, before any document is touched."""
    try:
        client = get_client()
        client.get_collections()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"\n[demo_corpus] Qdrant is unreachable: {exc}\n"
            f"[demo_corpus] Start it first: docker compose up -d qdrant\n"
            f"[demo_corpus] Refusing to start ingestion - running against a "
            f"down Qdrant would mark documents as 'already ingested' "
            f"(Step 6's dedup hash) without ever actually embedding them, "
            f"silently losing them on every future rerun."
        )


def rollback_failed_document_hashes(new_chunks: list[dict]) -> None:
    """
    Removes this document's content hashes from the persisted dedup
    store after its embed_and_upsert() call failed. Without this, the
    document is marked "already ingested" forever despite never
    reaching Qdrant - see module docstring, item 3.
    """
    if not new_chunks:
        return
    store = load_hash_store(HASH_STORE_PATH)
    for chunk in new_chunks:
        store.discard(chunk["content_hash"])
    save_hash_store(store, HASH_STORE_PATH)


async def _embed_and_upsert_with_retry(new_chunks: list[dict], trace_id: str) -> int:
    return await with_retry(
        lambda: asyncio.to_thread(embed_and_upsert, new_chunks, trace_id),
        max_retries=settings.max_retries,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-ingest a demo corpus directory.")
    parser.add_argument("--dir", required=True, help="Directory containing PDF/TXT/MD source documents")
    parser.add_argument("--rebuild-bm25", action="store_true", default=True,
                         help="Rebuild BM25 index after ingestion (default: yes)")
    args = parser.parse_args()

    source_dir = Path(args.dir)
    if not source_dir.exists():
        raise SystemExit(f"Directory not found: {source_dir}")

    check_qdrant_reachable()

    documents = find_documents(source_dir)
    print(f"[demo_corpus] Found {len(documents)} documents ({', '.join(sorted(SUPPORTED_EXTENSIONS))}) in {source_dir}")

    total_new_chunks = 0
    total_upserted = 0
    failed: list[tuple[Path, str]] = []
    quota_exhausted = False

    for i, doc_path in enumerate(documents, start=1):
        new_chunks: list[dict] | None = None
        try:
            new_chunks = process_document(doc_path)
            total_new_chunks += len(new_chunks)
            if new_chunks:
                upserted = asyncio.run(
                    _embed_and_upsert_with_retry(new_chunks, trace_id=f"demo_corpus_{i}")
                )
                total_upserted += upserted
            print(f"[demo_corpus] ({i}/{len(documents)}) {doc_path.name}: "
                  f"{len(new_chunks)} new chunks, {'upserted' if new_chunks else 'already ingested, skipped'}")
        except RetriesExhaustedError as exc:
            failed.append((doc_path, f"embedding retries exhausted: {exc}"))
            rollback_failed_document_hashes(new_chunks)
            print(f"[demo_corpus] ({i}/{len(documents)}) {doc_path.name}: FAILED (retries exhausted) - "
                  f"hash rolled back, will retry on next run")
            # Heuristic: repeated retry-exhaustion in a row usually means
            # daily quota is gone, not transient blips - stop burning
            # through the rest of the corpus one-by-one for nothing.
            quota_exhausted = True
        except Exception as exc:  # noqa: BLE001 - one bad file shouldn't kill the whole batch
            failed.append((doc_path, str(exc)))
            rollback_failed_document_hashes(new_chunks)
            print(f"[demo_corpus] ({i}/{len(documents)}) {doc_path.name}: FAILED - {exc} (hash rolled back)")

        if quota_exhausted:
            print(
                f"\n[demo_corpus] Stopping early at {i}/{len(documents)} - looks like a quota/rate "
                f"limit wall, not a one-off blip. Re-run this exact command later (quota resets on "
                f"Google's clock, typically US midnight Pacific Time) - already-embedded documents "
                f"will be skipped automatically (dedup), failed ones will retry correctly (rollback fix)."
            )
            break

    print(f"\n[demo_corpus] Ingestion run complete: {total_new_chunks} new chunks, {total_upserted} upserted to Qdrant.")
    if failed:
        print(f"[demo_corpus] {len(failed)} document(s) failed this run:")
        for path, err in failed:
            print(f"  - {path}: {err}")

    client = get_client()
    try:
        total_count = client.count(settings.qdrant_collection).count
    except Exception:# noqa: BLE001
        total_count = None

    if total_count is not None:
        print(f"[demo_corpus] Total corpus size in Qdrant so far: {total_count} chunks.")
        if total_count > CHUNK_COUNT_WARNING_THRESHOLD:
            print(
                f"[demo_corpus] WARNING (NFR-12): {total_count} chunks exceeds the "
                f"{CHUNK_COUNT_WARNING_THRESHOLD} soft-warning threshold for BM25's "
                f"in-memory constraint."
            )

    if args.rebuild_bm25:
        print("[demo_corpus] Rebuilding BM25 index against corpus ingested so far...")
        all_points = client.scroll(collection_name=settings.qdrant_collection, limit=100_000)[0]
        chunks = [{"chunk_id": p.id, "text": p.payload["text"]} for p in all_points]
        get_or_build_index(chunks=chunks, rebuild=True)
        print(f"[demo_corpus] BM25 index rebuilt with {len(chunks)} chunks - safe to query now, "
              f"even if this run stopped early.")


if __name__ == "__main__":
    main()