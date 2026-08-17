"""
FR-6 (fusion half) verify: unit test with synthetic rank lists confirms
score = Σ 1/(k + rank + 1) exactly.
"""
from src.retrieval.fusion import rrf_fuse


def test_rrf_fuse_exact_formula():
    sparse = [
        {"chunk_id": 1, "score": 5.0},
        {"chunk_id": 2, "score": 3.0},
    ]
    dense = [
        {"chunk_id": 2, "score": 0.9},
        {"chunk_id": 3, "score": 0.8},
    ]

    result = rrf_fuse(sparse, dense, k=60)
    scores = {r["chunk_id"]: r["rrf_score"] for r in result}

    # chunk 1: sparse rank 0 only -> 1/(60+0+1)
    assert abs(scores[1] - (1 / 61)) < 1e-9
    # chunk 2: sparse rank 1 + dense rank 0 -> 1/(60+1+1) + 1/(60+0+1)
    assert abs(scores[2] - (1 / 62 + 1 / 61)) < 1e-9
    # chunk 3: dense rank 1 only -> 1/(60+1+1)
    assert abs(scores[3] - (1 / 62)) < 1e-9


def test_rrf_fuse_sorted_descending():
    sparse = [{"chunk_id": 1, "score": 5.0}]
    dense = [{"chunk_id": 1, "score": 0.9}, {"chunk_id": 2, "score": 0.8}]

    result = rrf_fuse(sparse, dense, k=60)
    scores = [r["rrf_score"] for r in result]
    assert scores == sorted(scores, reverse=True)
    assert result[0]["chunk_id"] == 1  # appears in both lists, should rank first


def test_rrf_fuse_preserves_payload():
    sparse = [{"chunk_id": 1, "score": 5.0, "payload": {"text": "hello"}}]
    dense = []

    result = rrf_fuse(sparse, dense, k=60)
    assert result[0]["payload"] == {"text": "hello"}


def test_rrf_fuse_default_k_from_settings():
    from config.settings import settings

    sparse = [{"chunk_id": 1, "score": 5.0}]
    dense = []

    result = rrf_fuse(sparse, dense)  # no k passed — should use settings.rrf_k
    expected = 1 / (settings.rrf_k + 0 + 1)
    assert abs(result[0]["rrf_score"] - expected) < 1e-9


def test_rrf_fuse_empty_lists():
    result = rrf_fuse([], [])
    assert result == []