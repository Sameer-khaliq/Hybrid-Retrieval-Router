"""
Metadata tagging & content-hash dedup (Step 6, FR-2, FR-3).

process_document() is the single entrypoint every later step (Step 7's
Qdrant write, Step 8's BM25 build) calls to go from "a file path" to
"a list of chunk dicts ready to store" - each one already carrying its
full metadata and having already passed the dedup check.

Dedup mechanism (FR-3): a SHA-256 hash of each chunk's normalized text
is checked against a persisted hash set BEFORE the chunk is accepted.
Re-running this on an unchanged file therefore returns zero new chunks
the second time - nothing downstream ever receives a duplicate.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.ingestion.chunking import chunk_text
from src.ingestion.loaders import load_document

# Default on-disk location for the persisted hash set. Tests override
# this via the hash_store_path param (not by monkeypatching this
# constant) so each test can run against an isolated, throwaway store.
HASH_STORE_PATH = Path("data/.ingestion_state/chunk_hashes.json")


def normalize_for_hash(text: str) -> str:
    """Collapse whitespace and lowercase, so trivial formatting
    differences don't produce a different hash for otherwise-identical
    content."""
    return " ".join(text.split()).lower()


def compute_content_hash(text: str) -> str:
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_hash_store(path: Path | None = None) -> set[str]:
    path = path or HASH_STORE_PATH
    if path.exists():
        return set(json.loads(path.read_text(encoding="utf-8")))
    return set()


def save_hash_store(hashes: set[str], path: Path | None = None) -> None:
    path = path or HASH_STORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(hashes)), encoding="utf-8")


def process_document(
    path: str | Path,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    overlap_min_pct: float | None = None,
    overlap_max_pct: float | None = None,
    hash_store_path: Path | None = None,
) -> list[dict]:
    """
    Loads a document, chunks it (Step 5), attaches metadata (FR-2),
    and filters out any chunk whose content hash has already been seen
    (FR-3). Returns only the NEWLY accepted chunks - a second call on
    an unchanged file returns an empty list.

    Each returned chunk dict has:
        text, start_char, end_char, token_count   (from chunking.py)
        source_doc_id       - filename stem, e.g. "233472" for 233472.txt
        chunk_index         - position of this chunk within this document
        ingestion_timestamp - UTC ISO-8601, same for every chunk in this run
        content_hash        - SHA-256 over normalized chunk text
    """
    hash_store_path = hash_store_path or HASH_STORE_PATH

    text = load_document(path)
    raw_chunks = chunk_text(
        text,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_min_pct=overlap_min_pct,
        overlap_max_pct=overlap_max_pct,
    )

    source_doc_id = Path(path).stem
    ingestion_timestamp = datetime.now(timezone.utc).isoformat()

    hash_store = load_hash_store(hash_store_path)
    accepted: list[dict] = []

    for chunk_index, c in enumerate(raw_chunks):
        content_hash = compute_content_hash(c["text"])
        if content_hash in hash_store:
            continue  # FR-3: already ingested, skip - no duplicate emitted

        hash_store.add(content_hash)
        accepted.append({
            **c,
            "source_doc_id": source_doc_id,
            "chunk_index": chunk_index,
            "ingestion_timestamp": ingestion_timestamp,
            "content_hash": content_hash,
        })

    save_hash_store(hash_store, hash_store_path)
    return accepted