"""Dense retrieval query path (FR-5 query half)."""
from __future__ import annotations

from config.settings import settings
from src.common.qdrant_client import get_client
from src.ingestion.embedder import embed_texts
from src.observability.logging_setup import get_logger


def query_dense(query: str, top_n: int | None = None, trace_id: str = "dense_query") -> list[dict]:
    """Returns [{chunk_id, score, payload}, ...] sorted descending by cosine similarity."""
    top_n = top_n or settings.dense_top_n
    logger = get_logger(trace_id=trace_id)

    query_vector = embed_texts([query], task_type="query", trace_id=trace_id)[0]

    client = get_client()
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_n,
        with_payload=True,
    ).points

    output = [{"chunk_id": r.id, "score": r.score, "payload": r.payload} for r in results]
    logger.info("dense_retrieval", stage="dense_retrieval", num_results=len(output))
    return output