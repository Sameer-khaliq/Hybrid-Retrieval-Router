"""Shared fixtures for Phase 2 test suite."""
import pytest


@pytest.fixture
def sample_chunks():
    """Small synthetic corpus — int chunk_ids matching your 88.txt-style ingestion."""
    return [
        {"chunk_id": 1, "text": "The Federal Reserve raised interest rates to combat inflation."},
        {"chunk_id": 2, "text": "Diversifying a portfolio reduces exposure to single-asset risk."},
        {"chunk_id": 3, "text": "Quarterly earnings reports show revenue growth for tech firms."},
        {"chunk_id": 4, "text": "Bond yields fluctuate inversely with bond prices in the market."},
        {"chunk_id": 5, "text": "A mutual fund pools capital from many investors into securities."},
    ]


@pytest.fixture
def fake_embedding_vector():
    return [0.1] * 768  