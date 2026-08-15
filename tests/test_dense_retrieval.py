"""
FR-5 verify: returned chunk IDs match expected nearest neighbors within
tolerance. Uses a mocked embedder + mocked Qdrant client for a fast unit
test; a live_api-marked test hits real Gemini + Qdrant if you want to run it.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.retrieval.dense_qdrant import query_dense


class FakePoint:
    def __init__(self, id_, score, payload):
        self.id = id_
        self.score = score
        self.payload = payload


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


def test_query_dense_returns_expected_shape_and_order(fake_embedding_vector):
    fake_points = [
        FakePoint(3, 0.92, {"text": "Quarterly earnings reports show revenue growth."}),
        FakePoint(1, 0.81, {"text": "The Federal Reserve raised interest rates."}),
        FakePoint(5, 0.75, {"text": "A mutual fund pools capital from investors."}),
    ]

    with patch("src.retrieval.dense_qdrant.embed_texts", return_value=[fake_embedding_vector]) as mock_embed, \
         patch("src.retrieval.dense_qdrant.get_client") as mock_get_client:

        mock_client = MagicMock()
        mock_client.query_points.return_value = FakeQueryResult(fake_points)
        mock_get_client.return_value = mock_client

        results = query_dense("what were the earnings this quarter?", top_n=3)

    mock_embed.assert_called_once()
    assert mock_embed.call_args.kwargs["task_type"] == "query"  # task_type must be "query", not "document"

    assert len(results) == 3
    assert [r["chunk_id"] for r in results] == [3, 1, 5]  # order preserved, descending score
    assert results[0]["score"] == 0.92
    assert "payload" in results[0]


@pytest.mark.live_api
def test_query_dense_live_against_real_corpus():
    """Requires real GOOGLE_API_KEY + populated Qdrant collection. Skipped in CI by default."""
    results = query_dense("interest rate policy", top_n=5)
    assert len(results) > 0
    assert all("chunk_id" in r and "score" in r for r in results)