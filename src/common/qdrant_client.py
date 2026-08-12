"""
Qdrant connection factory (Step 3).

Everything downstream (ingestion write path Step 7, dense query path
Step 9) imports get_client() from here instead of instantiating
QdrantClient directly, so host/port stay centrally configurable (FR-19).
"""

from functools import lru_cache

from qdrant_client import QdrantClient

from config.settings import settings


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """
    Returns a cached, singleton QdrantClient connected to the host/port
    configured in Settings. Cached so repeated calls (e.g. across
    concurrent asyncio.gather branches in Step 12/17) reuse one
    connection rather than opening a new one per call.
    """
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )