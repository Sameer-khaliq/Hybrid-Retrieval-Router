"""
Embedding-config consistency check (Risk #5 mitigation).
Compares the config logged at ingestion time against the config the query
path is about to use. Fails loudly on mismatch — never silently returns
meaningless similarity scores.
"""
from __future__ import annotations

from config.settings import settings
from src.common.qdrant_client import get_client
from src.observability.logging_setup import get_logger


class EmbeddingConfigMismatchError(RuntimeError):
    pass


def get_ingest_time_config(trace_id: str = "config_check") -> dict:
    """Pull the embedding_config payload logged on a sample stored point."""
    client = get_client()
    points, _ = client.scroll(
        collection_name=settings.qdrant_collection, limit=1, with_payload=True
    )
    if not points:
        raise RuntimeError("No points in collection — cannot verify embedding config, run ingestion first.")
    return points[0].payload["embedding_config"]


def check_embedding_config_consistency(trace_id: str = "config_check") -> None:
    logger = get_logger(trace_id=trace_id)

    ingest_config = get_ingest_time_config(trace_id=trace_id)
    query_config = {
        "model": settings.gemini_embedding_model,
        "dimension": settings.gemini_embedding_dimension,
    }

    mismatches = {
        k: (ingest_config.get(k), query_config.get(k))
        for k in ("model", "dimension")
        if ingest_config.get(k) != query_config.get(k)
    }

    if mismatches:
        logger.error("embedding_config_mismatch", stage="config_check", mismatches=mismatches)
        raise EmbeddingConfigMismatchError(
            f"Ingest-time and query-time embedding config diverge: {mismatches}"
        )

    logger.info("embedding_config_consistent", stage="config_check")