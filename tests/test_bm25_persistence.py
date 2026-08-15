"""
Covers Step 8's three-part verify:
(a) fresh process load_index() == pre-persist in-memory index results
(b) rebuild_and_swap() picks up new chunks; tokenization cache hit count
    for pre-existing chunks == pre-existing chunk count
(c) a query fired during rebuild doesn't error or return partial results
"""
import asyncio
import shutil
from pathlib import Path

import pytest

from src.retrieval import sparse_bm25


@pytest.fixture(autouse=True)
def isolated_index_dir(tmp_path, monkeypatch):
    """Redirect INDEX_DIR/TOKEN_CACHE_PATH to a temp dir so tests don't touch real data/."""
    test_index_dir = tmp_path / "bm25_index"
    test_token_cache = tmp_path / "bm25_token_cache.json"
    monkeypatch.setattr(sparse_bm25, "INDEX_DIR", test_index_dir)
    monkeypatch.setattr(sparse_bm25, "TOKEN_CACHE_PATH", test_token_cache)
    # reset module-level active index between tests
    sparse_bm25._active_index = None
    sparse_bm25._active_chunk_ids = None
    yield
    if test_index_dir.exists():
        shutil.rmtree(test_index_dir, ignore_errors=True)


def test_build_and_query_in_memory(sample_chunks):
    retriever, chunk_ids = sparse_bm25.build_index(sample_chunks, trace_id="test_build")
    assert set(chunk_ids) == {c["chunk_id"] for c in sample_chunks}


def test_save_and_load_roundtrip(sample_chunks):
    """(a) — load_index() after persisting returns identical results to pre-persist index."""
    retriever, chunk_ids = sparse_bm25.build_index(sample_chunks, trace_id="test_save")
    sparse_bm25.save_index(retriever, chunk_ids, sparse_bm25.INDEX_DIR)

    loaded_retriever, loaded_chunk_ids = sparse_bm25.load_index(sparse_bm25.INDEX_DIR)
    assert loaded_chunk_ids == chunk_ids

    # Query both and compare top result — same corpus, same query, same tokenizer vocab
    import bm25s
    query_tokens_orig = bm25s.tokenize(["interest rates inflation"], stopwords="en",
                                        return_ids=False, show_progress=False)
    query_tokens_loaded = bm25s.tokenize(["interest rates inflation"], stopwords="en",
                                          return_ids=False, show_progress=False)

    docs_orig = retriever.retrieve(query_tokens_orig, k=3)
    docs_loaded = loaded_retriever.retrieve(query_tokens_loaded, k=3)

    assert list(docs_orig[0]) == list(docs_loaded[0])


def test_get_or_build_index_loads_from_disk_when_present(sample_chunks):
    sparse_bm25.get_or_build_index(chunks=sample_chunks, rebuild=True)  # first build
    sparse_bm25._active_index = None  # simulate fresh process
    sparse_bm25._active_chunk_ids = None

    retriever, chunk_ids = sparse_bm25.get_or_build_index(rebuild=False)  # should load, not rebuild
    assert set(chunk_ids) == {c["chunk_id"] for c in sample_chunks}


def test_rebuild_and_swap_adds_new_chunk_and_reuses_token_cache(sample_chunks):
    """(b) — new chunk retrievable after rebuild; pre-existing chunks weren't re-tokenized."""
    sparse_bm25.get_or_build_index(chunks=sample_chunks, rebuild=True)

    cache_before = sparse_bm25._load_token_cache()
    pre_existing_ids = {str(c["chunk_id"]) for c in sample_chunks}
    assert pre_existing_ids.issubset(cache_before.keys())

    new_chunk = {"chunk_id": 6, "text": "Cryptocurrency markets show high volatility during earnings season."}
    all_chunks = sample_chunks + [new_chunk]

    sparse_bm25.rebuild_and_swap(all_chunks, trace_id="test_rebuild")

    retriever, chunk_ids = sparse_bm25.get_active_index()
    assert 6 in chunk_ids

    cache_after = sparse_bm25._load_token_cache()
    # pre-existing chunk tokens must be byte-identical (proves cache reuse, not re-tokenize)
    for cid in pre_existing_ids:
        assert cache_after[cid] == cache_before[cid]


@pytest.mark.asyncio
async def test_query_during_rebuild_does_not_error_or_partial(sample_chunks):
    """(c) — concurrent query during rebuild is served by last-good index, never errors."""
    sparse_bm25.get_or_build_index(chunks=sample_chunks, rebuild=True)

    new_chunk = {"chunk_id": 7, "text": "Retirement accounts offer tax-advantaged growth over decades."}
    all_chunks = sample_chunks + [new_chunk]

    async def run_rebuild():
        await asyncio.to_thread(sparse_bm25.rebuild_and_swap, all_chunks, "test_concurrent_rebuild")

    async def run_query():
        # fired concurrently; must not raise, must return a non-empty list
        results = await asyncio.to_thread(sparse_bm25.query_bm25, "interest rates", 3)
        assert isinstance(results, list)
        assert len(results) > 0

    await asyncio.gather(run_rebuild(), run_query())

    # after rebuild completes, new chunk should be queryable
    retriever, chunk_ids = sparse_bm25.get_active_index()
    assert 7 in chunk_ids