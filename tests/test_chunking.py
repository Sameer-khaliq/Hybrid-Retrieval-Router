"""
Step 5 tests - chunking (FR-1).

Verify criteria straight from the implementation plan:
  "run ingestion on a fixture document; assert every chunk's token count
  falls in range and measured overlap matches config ±1 token."
"""

from pathlib import Path

from config.settings import settings
from src.ingestion.chunking import chunk_text, count_tokens
from src.ingestion.loaders import load_document

FIXTURE = Path(__file__).parent / "fixtures" / "sample.md"


def test_fixture_produces_multiple_chunks():
    text = load_document(FIXTURE)
    chunks = chunk_text(text)
    # If this fails, the fixture document is too short to meaningfully
    # test overlap/range behavior - grow tests/fixtures/sample.md.
    assert len(chunks) > 2, (
        f"expected multiple chunks from the fixture, got {len(chunks)} - "
        f"fixture may be too short"
    )


def test_every_chunk_token_count_in_range():
    text = load_document(FIXTURE)
    chunks = chunk_text(text)
    for i, c in enumerate(chunks):
        assert settings.chunk_min_tokens <= c["token_count"] <= settings.chunk_max_tokens, (
            f"chunk {i} has {c['token_count']} tokens, outside "
            f"[{settings.chunk_min_tokens}, {settings.chunk_max_tokens}]"
        )


def test_chunk_text_matches_char_offsets_exactly():
    # Sanity check on the offset math itself: text[start_char:end_char]
    # must equal the chunk's own text field exactly, or FR-2's stored
    # offset metadata would be lying about where the chunk came from.
    text = load_document(FIXTURE)
    chunks = chunk_text(text)
    for i, c in enumerate(chunks):
        assert text[c["start_char"]:c["end_char"]] == c["text"], f"offset mismatch at chunk {i}"


def test_overlap_matches_config_within_one_token():
    text = load_document(FIXTURE)
    chunks = chunk_text(text)

    target_pct = (settings.chunk_overlap_min_pct + settings.chunk_overlap_max_pct) / 2
    expected_overlap_tokens = round(settings.chunk_max_tokens * target_pct)

    # Skip the last adjacent pair: the final chunk may have its start
    # pulled back (see chunking.py's last-chunk edge case) specifically
    # to satisfy the min_tokens floor, which changes that one gap's
    # overlap on purpose. Every other adjacent pair should match exactly.
    for i in range(len(chunks) - 2):
        c1, c2 = chunks[i], chunks[i + 1]
        measured_overlap_tokens = c1["end_token"] - c2["start_token"]
        assert abs(measured_overlap_tokens - expected_overlap_tokens) <= 1, (
            f"overlap between chunk {i} and {i+1} is {measured_overlap_tokens} tokens, "
            f"expected {expected_overlap_tokens} ± 1"
        )


def test_custom_bounds_are_respected():
    # FR-19: bounds must be configurable per-call, not hardcoded.
    text = load_document(FIXTURE)
    chunks = chunk_text(text, min_tokens=50, max_tokens=100, overlap_min_pct=0.10, overlap_max_pct=0.10)
    for c in chunks:
        assert 50 <= c["token_count"] <= 100


def test_count_tokens_matches_chunk_token_count():
    text = load_document(FIXTURE)
    chunks = chunk_text(text)
    for c in chunks:
        assert count_tokens(c["text"]) == c["token_count"]