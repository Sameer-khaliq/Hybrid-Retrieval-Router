"""
Step 6 tests - metadata tagging & dedup (FR-2, FR-3).

Verify criteria straight from the implementation plan:
  FR-2: "query the store after ingesting a fixture; assert none of these
  fields are null."
  FR-3: "ingest the same file twice; assert chunk count is unchanged
  after the second run."
"""

from pathlib import Path

from src.ingestion.pipeline import process_document

FIXTURE = Path(__file__).parent / "fixtures" / "sample.md"


def test_every_chunk_has_complete_metadata(tmp_path):
    hash_store_path = tmp_path / "hashes.json"
    chunks = process_document(FIXTURE, hash_store_path=hash_store_path)

    assert len(chunks) > 0
    for c in chunks:
        assert c["source_doc_id"] is not None and c["source_doc_id"] != ""
        assert c["chunk_index"] is not None
        assert c["start_char"] is not None
        assert c["end_char"] is not None
        assert c["ingestion_timestamp"] is not None and c["ingestion_timestamp"] != ""
        assert c["content_hash"] is not None and c["content_hash"] != ""


def test_source_doc_id_is_filename_stem(tmp_path):
    hash_store_path = tmp_path / "hashes.json"
    chunks = process_document(FIXTURE, hash_store_path=hash_store_path)
    assert all(c["source_doc_id"] == "sample" for c in chunks)


def test_chunk_index_is_sequential_per_document(tmp_path):
    hash_store_path = tmp_path / "hashes.json"
    chunks = process_document(FIXTURE, hash_store_path=hash_store_path)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_reingest_unchanged_file_produces_zero_new_chunks(tmp_path):
    hash_store_path = tmp_path / "hashes.json"

    first_run = process_document(FIXTURE, hash_store_path=hash_store_path)
    assert len(first_run) > 0, "first ingest should accept chunks - nothing seen yet"

    second_run = process_document(FIXTURE, hash_store_path=hash_store_path)
    assert len(second_run) == 0, (
        "re-ingesting an unchanged file must not produce new/duplicate chunks (FR-3)"
    )


def test_dedup_persists_across_separate_process_document_calls(tmp_path):
    # Simulates two truly independent ingestion runs (e.g. two separate
    # CLI invocations) sharing the same on-disk hash store, not just two
    # calls in the same Python process.
    hash_store_path = tmp_path / "hashes.json"

    process_document(FIXTURE, hash_store_path=hash_store_path)
    assert hash_store_path.exists(), "hash store should be persisted to disk after a run"

    second_run = process_document(FIXTURE, hash_store_path=hash_store_path)
    assert len(second_run) == 0