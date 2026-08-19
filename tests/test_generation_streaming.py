"""
Step 22 tests (FR-26, NFR-9).

Plan's stated Done-when:
  - Client receives multiple ordered partial chunks before the final
    chunk, for both paths (FR-26's verify).
  - Fault-injection on deep-path failure confirms fallback to fast-path
    output with degraded:true (NFR-9's verify).
  - A gated-response fixture confirms single, atomic, non-chunked
    delivery.

Groq's client is fully mocked - this file verifies the streaming/
fallback CONTRACT, not live model behavior.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.common.resilience import RetriesExhaustedError
from src.generation.groq_stream import generate_atomic, generate_streaming


class _FakeDelta:
    def __init__(self, content: str | None):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str | None):
        self.choices = [_FakeChoice(content)]


async def _fake_stream(tokens: list[str]):
    for t in tokens:
        yield _FakeChunk(t)


@pytest.mark.asyncio
async def test_streaming_yields_ordered_deltas_then_done_fast_path():
    tokens = ["Paris", " is", " the", " capital"]
    chunks = [{"chunk_id": "c1", "text": "Paris is the capital of France.", "source": "corpus"}]

    with patch(
        "src.generation.groq_stream._create_stream",
        new=AsyncMock(return_value=_fake_stream(tokens)),
    ):
        events = [e async for e in generate_streaming("capital of France?", chunks, "fast")]

    deltas = [e["delta"] for e in events if "delta" in e]
    assert deltas == tokens
    assert events[-1]["done"] is True
    assert events[-1]["degraded"] is False
    assert events[-1]["model_used"] == "openai/gpt-oss-20b"


@pytest.mark.asyncio
async def test_streaming_yields_ordered_deltas_then_done_deep_path():
    tokens = ["The", " analysis", " shows..."]
    chunks = [{"chunk_id": "c1", "text": "some deep context", "source": "corpus"}]

    with patch(
        "src.generation.groq_stream._create_stream",
        new=AsyncMock(return_value=_fake_stream(tokens)),
    ):
        events = [e async for e in generate_streaming("complex query", chunks, "deep")]

    deltas = [e["delta"] for e in events if "delta" in e]
    assert deltas == tokens
    assert events[-1]["done"] is True
    assert events[-1]["degraded"] is False


@pytest.mark.asyncio
async def test_deep_path_failure_falls_back_to_fast_model_degraded_true():
    """NFR-9's own verify: deep-path exhausts retries -> fallback to
    fast-path model, degraded:true, generation still succeeds."""
    fallback_tokens = ["fast", " path", " answer"]

    call_count = {"n": 0}

    async def _side_effect(model, query, context_block):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated deep model failure")
        return _fake_stream(fallback_tokens)

    with (
        patch("src.generation.groq_stream._create_stream", side_effect=_side_effect),
        patch(
            "src.generation.groq_stream.with_retry",
            side_effect=[
                RetriesExhaustedError("deep model exhausted"),
                _fake_stream(fallback_tokens),
            ],
        ),
    ):
        events = [
            e async for e in generate_streaming("complex query", [{"text": "ctx"}], "deep")
        ]

    deltas = [e["delta"] for e in events if "delta" in e]
    assert deltas == fallback_tokens
    final = events[-1]
    assert final["degraded"] is True
    assert final["model_used"] != ""


@pytest.mark.asyncio
async def test_fast_path_total_failure_yields_error_sentinel_not_exception():
    """No lower tier exists below fast-path per NFR-9's stated scope -
    total failure should yield an error sentinel, never raise past this
    function into the API layer uncaught."""
    with patch(
        "src.generation.groq_stream.with_retry",
        side_effect=RetriesExhaustedError("fast model down"),
    ):
        events = [e async for e in generate_streaming("query", [{"text": "ctx"}], "fast")]

    assert any(e.get("error") == "generation_unavailable" for e in events)
    assert events[-1]["done"] is True
    assert events[-1]["degraded"] is True


@pytest.mark.asyncio
async def test_generate_atomic_returns_single_non_streamed_response():
    result = await generate_atomic("Hi there! How can I help?")
    assert result["answer"] == "Hi there! How can I help?"
    assert result["streamed"] is False
    assert result["degraded"] is False