"""
Deliberately mismatching config between ingest-time and query-time causes
check_embedding_config_consistency() to raise before any further Qdrant
call is made; matching configs pass silently.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.common.config_check import (
    check_embedding_config_consistency,
    EmbeddingConfigMismatchError,
)


class FakePoint:
    def __init__(self, payload):
        self.payload = payload


def test_matching_config_passes_silently():
    from config.settings import settings

    matching_payload = {
        "embedding_config": {
            "model": settings.gemini_embedding_model,
            "dimension": settings.gemini_embedding_dimension,
        }
    }

    with patch("src.common.config_check.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([FakePoint(matching_payload)], None)
        mock_get_client.return_value = mock_client

        check_embedding_config_consistency(trace_id="test_match")  # should not raise


def test_dimension_mismatch_raises():
    mismatched_payload = {
        "embedding_config": {
            "model": "gemini-embedding-001",
            "dimension": 1536,  # deliberately wrong vs settings default (768)
        }
    }

    with patch("src.common.config_check.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([FakePoint(mismatched_payload)], None)
        mock_get_client.return_value = mock_client

        with pytest.raises(EmbeddingConfigMismatchError):
            check_embedding_config_consistency(trace_id="test_mismatch")


def test_model_mismatch_raises():
    from config.settings import settings

    mismatched_payload = {
        "embedding_config": {
            "model": "text-embedding-004",  # wrong model name
            "dimension": settings.gemini_embedding_dimension,
        }
    }

    with patch("src.common.config_check.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([FakePoint(mismatched_payload)], None)
        mock_get_client.return_value = mock_client

        with pytest.raises(EmbeddingConfigMismatchError):
            check_embedding_config_consistency(trace_id="test_model_mismatch")


def test_empty_collection_raises_runtime_error():
    with patch("src.common.config_check.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)
        mock_get_client.return_value = mock_client

        with pytest.raises(RuntimeError, match="No points in collection"):
            check_embedding_config_consistency(trace_id="test_empty")