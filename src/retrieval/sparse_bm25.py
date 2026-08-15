"""
BM25 sparse index — build, disk persistence, hot-reload (FR-4, addendum #1).

NON-PAUSABLE: build_index/save_index/load_index trio is safe to pause after
individually, but rebuild_and_swap() must be finished as one unbroken unit —
a partial implementation leaves the live process serving a stale index or
crashing against a half-written reference.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

import bm25s

from config.settings import settings
from src.observability.logging_setup import get_logger
from src.common.qdrant_client import get_client

INDEX_DIR = Path(settings.bm25_index_dir)
TOKEN_CACHE_PATH = INDEX_DIR.parent / "bm25_token_cache.json"  # outside INDEX_DIR so atomic swap never touches it

# In-process reference the query path reads from. Swapped atomically by
# rebuild_and_swap(); never mutated in place.
_active_index: bm25s.BM25 | None = None
_active_chunk_ids: list[int] | None = None
_swap_lock = threading.Lock()


def _load_token_cache() -> dict[str, list[str]]:
    if TOKEN_CACHE_PATH.exists():
        return json.loads(TOKEN_CACHE_PATH.read_text())
    return {}


def _save_token_cache(cache: dict[str, list[str]]) -> None:
    TOKEN_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = TOKEN_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache))
    os.replace(tmp, TOKEN_CACHE_PATH)  # atomic


def _tokenize_chunks(chunks: list[dict], logger) -> tuple[list[list[str]], list[int]]:
    """
    Tokenize chunks, reusing cached tokens for chunk_ids already tokenized.
    Returns (tokenized_corpus_in_chunk_order, chunk_ids_in_same_order).
    """
    cache = _load_token_cache()
    cache_hits = 0
    to_tokenize_texts, to_tokenize_ids = [], []

    for chunk in chunks:
        cid = str(chunk["chunk_id"])
        if cid in cache:
            cache_hits += 1
        else:
            to_tokenize_texts.append(chunk["text"])
            to_tokenize_ids.append(cid)

    if to_tokenize_texts:
        new_tok = bm25s.tokenize(to_tokenize_texts, stopwords="en", show_progress=False)
        vocab_inv = {v: k for k, v in new_tok.vocab.items()}
        for cid, id_seq in zip(to_tokenize_ids, new_tok.ids):
            cache[cid] = [vocab_inv[i] for i in id_seq]

    tokenized_corpus, chunk_ids = [], []
    for chunk in chunks:
        cid = str(chunk["chunk_id"])
        tokenized_corpus.append(cache[cid])
        chunk_ids.append(chunk["chunk_id"])

    _save_token_cache(cache)

    logger.info(
        "bm25_tokenize", stage="bm25_tokenize",
        total_chunks=len(chunks), cache_hits=cache_hits,
        newly_tokenized=len(to_tokenize_texts),
    )
    return tokenized_corpus, chunk_ids


def build_index(chunks: list[dict], trace_id: str = "bm25_build") -> tuple[bm25s.BM25, list[int]]:
    logger = get_logger(trace_id=trace_id)
    tokenized_corpus, chunk_ids = _tokenize_chunks(chunks, logger)

    retriever = bm25s.BM25()
    retriever.index(tokenized_corpus)

    logger.info("bm25_index_built", stage="bm25_build", num_docs=len(chunks))
    return retriever, chunk_ids


def save_index(retriever: bm25s.BM25, chunk_ids: list[int], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    retriever.save(str(path))
    (path / "chunk_ids.json").write_text(json.dumps(chunk_ids))


def load_index(path: Path) -> tuple[bm25s.BM25, list[int]]:
    retriever = bm25s.BM25.load(str(path), load_corpus=False)
    chunk_ids = json.loads((path / "chunk_ids.json").read_text())
    return retriever, chunk_ids


def get_or_build_index(
    chunks: list[dict] | None = None, rebuild: bool = False, trace_id: str = "bm25_startup"
) -> tuple[bm25s.BM25, list[int]]:
    """Startup path: load from disk unless missing or --rebuild passed."""
    global _active_index, _active_chunk_ids
    logger = get_logger(trace_id=trace_id)

    if INDEX_DIR.exists() and not rebuild:
        retriever, chunk_ids = load_index(INDEX_DIR)
        logger.info("bm25_loaded_from_disk", stage="bm25_startup", num_docs=len(chunk_ids))
    else:
        if chunks is None:
            raise ValueError("No persisted index found and no chunks provided to build one.")
        retriever, chunk_ids = build_index(chunks, trace_id=trace_id)
        save_index(retriever, chunk_ids, INDEX_DIR)
        logger.info("bm25_built_fresh", stage="bm25_startup", num_docs=len(chunk_ids))

    with _swap_lock:
        _active_index = retriever
        _active_chunk_ids = chunk_ids
    return retriever, chunk_ids


def rebuild_and_swap(all_chunks: list[dict], trace_id: str = "bm25_rebuild") -> None:
    """
    Build off the request-serving path, persist to a temp dir, atomically
    rename into place, then swap the in-process pointer. A query arriving
    mid-rebuild is served by the last-good index.
    """
    global _active_index, _active_chunk_ids
    logger = get_logger(trace_id=trace_id)

    retriever, chunk_ids = build_index(all_chunks, trace_id=trace_id)

    tmp_dir = Path(tempfile.mkdtemp(prefix="bm25_build_", dir=INDEX_DIR.parent))
    save_index(retriever, chunk_ids, tmp_dir)

    old_marker = INDEX_DIR.with_name(INDEX_DIR.name + ".old")
    if INDEX_DIR.exists():
        os.replace(INDEX_DIR, old_marker)
    os.replace(tmp_dir, INDEX_DIR)
    if old_marker.exists():
        shutil.rmtree(old_marker)

    with _swap_lock:
        _active_index = retriever
        _active_chunk_ids = chunk_ids

    logger.info("bm25_rebuild_swapped", stage="bm25_rebuild", num_docs=len(chunk_ids))


def get_active_index() -> tuple[bm25s.BM25, list[int]]:
    with _swap_lock:
        if _active_index is None:
            raise RuntimeError("BM25 index not initialized — call get_or_build_index() first.")
        return _active_index, _active_chunk_ids


def query_bm25(query: str, top_n: int | None = None) -> list[dict]:
    """Returns [{chunk_id, score}, ...] sorted descending by score."""
    top_n = top_n or settings.sparse_top_n
    retriever, chunk_ids = get_active_index()

    query_tokens = bm25s.tokenize([query], stopwords="en", return_ids=False, show_progress=False)
    results, scores = retriever.retrieve(query_tokens, k=min(top_n, len(chunk_ids)))
    return [
        {"chunk_id": chunk_ids[idx], "score": float(score)}
        for idx, score in zip(results[0], scores[0])
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.build or args.rebuild:
        client = get_client()
        all_points = client.scroll(collection_name=settings.qdrant_collection, limit=100000)[0]
        chunks = [{"chunk_id": p.id, "text": p.payload["text"]} for p in all_points]
        get_or_build_index(chunks=chunks, rebuild=True)
        print(f"BM25 index built with {len(chunks)} chunks.")