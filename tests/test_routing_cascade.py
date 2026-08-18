"""
Tests for src.routing.layer0_rules and src.routing.layer1_llm (FR-12, FR-13).

Covers:
  - Layer 0 boundary values at and around tau_low/tau_high (FR-12)
  - Layer 1 malformed-output -> fast-path default (FR-13 / NFR-9)
  - Layer 1 timeout -> deep-path default (addendum #2)
  - Layer 1 retries-exhausted -> deep-path default (same branch as timeout,
    per addendum #2's "with or without exhausting retries first")

The two failure-branch tests are the important ones per Step 16's
non-pausable warning: they must resolve to DIFFERENT defaults
(fast vs deep), not collapse into one shared fallback.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from config.settings import settings
from src.common.resilience import RetriesExhaustedError
from src.routing.layer0_rules import route_layer0
from src.routing.layer1_llm import call_llm_router

# ---------------------------------------------------------------------------
# FR-12: Layer 0 boundary values
# ---------------------------------------------------------------------------

def test_layer0_below_tau_low_is_fast():
    assert route_layer0(settings.tau_low - 0.01) == "fast"


def test_layer0_at_tau_low_exactly_is_middle_band():
    # FR-12: "score < tau_low" is strict - exactly tau_low falls in the
    # middle band (deferred to Layer 1), not fast-path.
    assert route_layer0(settings.tau_low) is None


def test_layer0_above_tau_high_is_deep():
    assert route_layer0(settings.tau_high + 0.01) == "deep"


def test_layer0_at_tau_high_exactly_is_middle_band():
    assert route_layer0(settings.tau_high) is None


def test_layer0_mid_band_defers_to_layer1():
    midpoint = (settings.tau_low + settings.tau_high) / 2
    assert route_layer0(midpoint) is None


def test_layer0_far_below_zero_bound_is_fast():
    assert route_layer0(0.0) == "fast"


def test_layer0_far_above_one_bound_is_deep():
    assert route_layer0(1.0) == "deep"


# ---------------------------------------------------------------------------
# FR-13 / NFR-9: Layer 1 malformed-output -> fast-path default
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layer1_malformed_json_defaults_to_fast():
    with patch(
        "src.routing.layer1_llm._invoke_groq_with_retry",
        new=AsyncMock(return_value="not valid json at all"),
    ):
        result = await call_llm_router("some ambiguous query")

    assert result["path"] == "fast"
    assert result["deciding_layer"] == "malformed-fallback"
    assert result["degraded"] is True


@pytest.mark.asyncio
async def test_layer1_schema_violation_defaults_to_fast():
    # Valid JSON, but "path" isn't one of the allowed Literal values -
    # should fail Pydantic validation, not JSON parsing.
    with patch(
        "src.routing.layer1_llm._invoke_groq_with_retry",
        new=AsyncMock(return_value='{"path": "medium", "confidence": 0.5, "reason": "x"}'),
    ):
        result = await call_llm_router("some ambiguous query")

    assert result["path"] == "fast"
    assert result["deciding_layer"] == "malformed-fallback"
    assert result["degraded"] is True


# ---------------------------------------------------------------------------
# addendum #2: Layer 1 timeout -> deep-path default
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layer1_timeout_defaults_to_deep():
    async def _hangs_forever(query: str) -> str:
        await asyncio.sleep(10)
        return '{"path": "fast", "confidence": 0.9, "reason": "unreachable"}'

    with patch("src.routing.layer1_llm.settings.router_timeout_ms", 50), patch(
        "src.routing.layer1_llm._invoke_groq_with_retry", new=_hangs_forever
    ):
        result = await call_llm_router("some ambiguous query")

    assert result["path"] == "deep"
    assert result["deciding_layer"] == "timeout-fallback"
    assert result["degraded"] is True


@pytest.mark.asyncio
async def test_layer1_retries_exhausted_defaults_to_deep():
    # Same branch as timeout, per addendum #2 - retries exhausted WITHIN
    # the window still maps to deep-path, not a third outcome.
    with patch(
        "src.routing.layer1_llm._invoke_groq_with_retry",
        new=AsyncMock(side_effect=RetriesExhaustedError("all attempts failed")),
    ):
        result = await call_llm_router("some ambiguous query")

    assert result["path"] == "deep"
    assert result["deciding_layer"] == "timeout-fallback"
    assert result["degraded"] is True


# ---------------------------------------------------------------------------
# FR-13: well-formed success path (sanity check the happy path too)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_layer1_valid_response_passes_through():
    with patch(
        "src.routing.layer1_llm._invoke_groq_with_retry",
        new=AsyncMock(
            return_value='{"path": "deep", "confidence": 0.87, "reason": "multi-hop comparison"}'
        ),
    ):
        result = await call_llm_router("compare AAPL and MSFT dividend policy")

    assert result["path"] == "deep"
    assert result["confidence"] == pytest.approx(0.87)
    assert result["deciding_layer"] == "llm-fallback"
    assert result["degraded"] is False