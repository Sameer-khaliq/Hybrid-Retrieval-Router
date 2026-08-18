"""
Step 23 — Request/response schemas for the FastAPI surface (FR-15, FR-17).

QueryRequest's field constraints enforce FR-17 at the pydantic-validation
layer: an empty/whitespace-only or over-length query fails FastAPI's
request parsing BEFORE the route function body ever runs - so the
endpoint's own code never executes, and consequently no downstream call
(embedding, Groq, Tavily) is ever reachable for an invalid request. This
is what "without invoking any downstream API" (FR-17's verify) actually
guarantees structurally, not just by convention.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from config.settings import settings


class QueryRequest(BaseModel):
    query: str = Field(..., max_length=settings.max_query_length)
    top_k: int | None = Field(default=None, ge=1)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class RoutingMetadata(BaseModel):
    path: str | None
    confidence: float | None
    reason: str | None


class QueryResponse(BaseModel):
    """
    FR-15's documented response contract. For the streaming path (non-
    gated, non-insufficient-info queries), this exact schema is what the
    FINAL NDJSON line contains - earlier lines are {"delta": str} token
    events, per FR-26. For gated/insufficient-info responses, this is
    the entire (single, atomic) response body.
    """
    answer: str
    sources: list[dict[str, Any]]
    routing_metadata: RoutingMetadata
    latency_breakdown: dict[str, float]
    degraded: bool = False
    trace_id: str
    api_call_count: int | None = None  # None for gated responses (0, but
                                         # omitted from the documented
                                         # FR-15 schema for those - see main.py)


class DependencyStatus(BaseModel):
    status: str  # "ok" | "unreachable: <error summary>"


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    dependencies: dict[str, str]