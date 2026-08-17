"""
Tests for src.routing.features.compute_complexity_score (FR-11).

Labeled query set with expected complexity bands. Bands are deliberately
wide (not pinned to exact float values) since the score is a weighted
heuristic, not a deterministic formula with one "correct" output per
query - FR-11's verify criterion is "falls in the expected band", not
"matches an exact value".
"""

import pytest

from src.routing.features import (
    compute_complexity_score,
    token_count_feature,
    keyword_feature,
    question_word_feature,
    clause_count_feature,
)

# (query, expected_band) where band is "low" (<0.3), "mid" (0.3-0.6),
# "high" (>0.6) - matches tau_low/tau_high defaults.
LABELED_QUERIES = [
    ("What is AAPL?", "low"),
    ("Who is the CEO of Tesla?", "low"),
    ("When was the Fed's last rate hike?", "low"),
    (
        "Compare the dividend yields of AAPL and MSFT over the last five "
        "years, and explain why the difference exists given their "
        "respective payout ratios and free cash flow trends.",
        "high",
    ),
    (
        "Why did the yield curve invert in 2023, and how does that "
        "relate to the Fed's rate policy and the broader impact on "
        "regional bank lending, which in turn affected credit "
        "availability for small businesses?",
        "high",
    ),
    ("What is the relationship between inflation and bond prices?", "mid"),
]


@pytest.mark.parametrize("query,expected_band", LABELED_QUERIES)
def test_complexity_score_falls_in_expected_band(query, expected_band):
    score = compute_complexity_score(query)
    assert 0.0 <= score <= 1.0

    if expected_band == "low":
        assert score < 0.3, f"expected low band, got {score:.3f} for: {query!r}"
    elif expected_band == "high":
        assert score > 0.6, f"expected high band, got {score:.3f} for: {query!r}"
    else:
        assert 0.3 <= score <= 0.6, f"expected mid band, got {score:.3f} for: {query!r}"


def test_token_count_feature_saturates_at_one():
    long_query = "word " * 200
    assert token_count_feature(long_query) == 1.0


def test_token_count_feature_short_query_low():
    assert token_count_feature("What is AAPL?") < 0.3


def test_keyword_feature_detects_comparison_terms():
    assert keyword_feature("Compare AAPL versus MSFT") == 1.0
    assert keyword_feature("What is AAPL's price?") == 0.0


def test_question_word_feature_why_how_is_complex():
    assert question_word_feature("Why did the market crash?") == 1.0
    assert question_word_feature("How does compound interest work?") == 1.0


def test_question_word_feature_what_who_is_simple():
    assert question_word_feature("What is AAPL?") == 0.0
    assert question_word_feature("Who is the CEO?") == 0.0


def test_clause_count_feature_multi_clause_higher():
    simple = clause_count_feature("What is AAPL?")
    complex_ = clause_count_feature(
        "What is AAPL, and how does it compare to MSFT, which has a "
        "different dividend policy because of its cash reserves?"
    )
    assert complex_ > simple


def test_complexity_score_weights_sum_to_expected_defaults():
    # Sanity check that Settings' weights (§3.4) sum to 1.0, so the score
    # stays bounded without needing the defensive clamp to actually fire
    # in the default configuration.
    from config.settings import settings

    total = (
        settings.weight_token_count
        + settings.weight_keyword
        + settings.weight_question_word
        + settings.weight_clause_count
    )
    assert abs(total - 1.0) < 1e-9