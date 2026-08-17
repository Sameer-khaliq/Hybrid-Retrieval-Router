"""
Composite query-complexity score (FR-11, §3.4).

score = 0.20*token_count_feat + 0.35*keyword_feat
      + 0.25*question_word_feat + 0.20*clause_count_feat

All weights are pulled from Settings (FR-19), not hardcoded here, even
though §3.4's starting values match the defaults - so a future
eval-set-calibration pass (FR-18, NFR-6) only touches .env, not this file.

Pure function of query text only - no retrieval/gating dependency, so
this could have been built in Phase 1, but is sequenced adjacent to its
only consumer (Step 16) per the implementation plan.
"""

from __future__ import annotations

import re

import tiktoken

from config.settings import settings

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Normalization ceiling for token-count feature: queries at or above this
# length saturate to 1.0. 40 tokens ~= a fairly long, multi-clause question;
# picked as a starting point, tune against FR-18's eval set.
_TOKEN_COUNT_SATURATION = 40

# Comparison / multi-hop trigger terms - presence signals the query needs
# reasoning across multiple facts/entities, not a single lookup.
_KEYWORD_TRIGGERS_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference\s+between|"
    r"relationship\s+between|correlat(e|ion)|trend|over\s+time|"
    r"both\s+.+\s+and\b|either\s+.+\s+or\b|impact\s+of|effect\s+of|"
    r"why\s+did|how\s+does\s+.+\s+affect|multi[\s-]?step)\b",
    re.IGNORECASE,
)

# Question-word category: "why"/"how" tend toward multi-hop reasoning;
# "what"/"who"/"when"/"where" tend toward single-fact lookup.
_COMPLEX_QUESTION_WORDS = re.compile(r"^\s*(why|how)\b", re.IGNORECASE)
_SIMPLE_QUESTION_WORDS = re.compile(r"^\s*(what|who|when|where|which)\b", re.IGNORECASE)

# Clause-boundary signals: subordinating/coordinating conjunctions and
# relative pronouns that typically introduce an additional clause, plus
# comma/semicolon punctuation as a cheap proxy for clause breaks.
_CLAUSE_MARKERS_RE = re.compile(
    r"\b(and|but|because|although|while|which|that|who|if|when|"
    r"since|whereas|unless)\b",
    re.IGNORECASE,
)
_CLAUSE_PUNCT_RE = re.compile(r"[,;]")

# Normalization ceiling for clause-count feature.
_CLAUSE_COUNT_SATURATION = 5


def token_count_feature(query: str) -> float:
    n_tokens = len(_ENCODING.encode(query))
    return min(n_tokens / _TOKEN_COUNT_SATURATION, 1.0)


def keyword_feature(query: str) -> float:
    return 1.0 if _KEYWORD_TRIGGERS_RE.search(query) else 0.0


def question_word_feature(query: str) -> float:
    if _COMPLEX_QUESTION_WORDS.search(query):
        return 1.0
    if _SIMPLE_QUESTION_WORDS.search(query):
        return 0.0
    # No recognized question word (e.g. an imperative or a fragment) -
    # treat as neutral rather than confidently simple or complex.
    return 0.5


def clause_count_feature(query: str) -> float:
    marker_hits = len(_CLAUSE_MARKERS_RE.findall(query))
    punct_hits = len(_CLAUSE_PUNCT_RE.findall(query))
    total = marker_hits + punct_hits
    return min(total / _CLAUSE_COUNT_SATURATION, 1.0)


def compute_complexity_score(query: str) -> float:
    """
    Returns a composite complexity score in [0, 1]. FR-12 thresholds this
    against tau_low/tau_high to pick fast/deep/LLM-fallback.
    """
    score = (
        settings.weight_token_count * token_count_feature(query)
        + settings.weight_keyword * keyword_feature(query)
        + settings.weight_question_word * question_word_feature(query)
        + settings.weight_clause_count * clause_count_feature(query)
    )
    # Weighted sum of four features already in [0,1] can't exceed 1.0 given
    # weights sum to 1.0, but clamp defensively in case weights are
    # reconfigured to not sum to exactly 1.0.
    return max(0.0, min(score, 1.0))