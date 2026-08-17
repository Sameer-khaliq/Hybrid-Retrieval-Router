"""
Query gating: non-corpus intent, abuse, credential-solicitation, out-of-scope
(FR-21, FR-22, FR-23, FR-24, FR-25).

Four independent rule-based/regex matchers. Each returns either None
(pass through to the retrieval pipeline) or a fixed gating response dict.

FR-25 mandates local-only matching (keyword/regex) in the common case -
no LLM API call here, so NFR-13's <50ms budget is achievable without a
network hop on every query.

DESIGN NOTE (addendum #3, restated from IMPLEMENTATION_PLAN.md Step 14):
FR-24's out-of-scope gate here is a hard PRE-retrieval domain-relevance
check. It must never share state or a scoring function with FR-9's
POST-retrieval RRF-confidence check (built in Step 20). A query that
passes this gate and later gets a low fused/reranked score is FR-9's
concern, not FR-24's - conflating them risks double-gating or
under-gating. Do not import anything from tavily_fallback.py here, and
do not import anything from this module in tavily_fallback.py.

Run order below is safety-first: abuse -> credential-solicitation ->
non-corpus-intent -> out-of-scope. The four categories are mutually
exclusive in practice but this order is deliberate, not incidental.
"""

from __future__ import annotations

import re
from typing import Optional

from src.observability.logging_setup import get_logger

# ---------------------------------------------------------------------------
# FR-21: non-corpus intent (greetings, meta-questions, gratitude/farewells)
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|good\s?(morning|afternoon|evening))\b"
    r"(\s+(there|everyone|all|team|folks))?\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_FAREWELL_RE = re.compile(
    r"^\s*(bye|goodbye|see\s?ya|see\s?you|later|farewell)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_GRATITUDE_RE = re.compile(
    r"^\s*(thanks|thank\s?you|thx|ty|appreciate\s?it|cheers)\b"
    r"(\s+(so\s+much|a\s+lot|very\s+much|so\s+very\s+much))?\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_META_QUESTION_RE = re.compile(
    r"\b(who\s+(built|made|created|trained)\s+you|"
    r"what\s+are\s+you|are\s+you\s+(an?\s+)?(ai|bot|human)|"
    r"what\s+model\s+are\s+you|which\s+company\s+(made|built)\s+you|"
    r"how\s+do\s+you\s+work)\b",
    re.IGNORECASE,
)

NON_CORPUS_INTENT_RESPONSE = (
    "Hi! I'm a retrieval assistant for this knowledge base - ask me anything "
    "about the documents I've been given, and I'll pull the relevant answer."
)


def check_non_corpus_intent(query: str) -> Optional[dict]:
    q = query.strip()
    if (
        _GREETING_RE.match(q)
        or _FAREWELL_RE.match(q)
        or _GRATITUDE_RE.match(q)
        or _META_QUESTION_RE.search(q)
    ):
        return {
            "gated": True,
            "category": "FR-21",
            "response": NON_CORPUS_INTENT_RESPONSE,
            "reason": "non_corpus_intent",
        }
    return None


# ---------------------------------------------------------------------------
# FR-22: abusive / hostile language
# ---------------------------------------------------------------------------
# Kept intentionally small and demo-scoped - this is NOT a production
# moderation system (see REQUIREMENTS.md §4.1 scope boundary). Extend the
# word/pattern lists as needed; keep them out of logs per FR-22.

_ABUSE_TERMS = [
    # English Base
    r"\bfuck(ing|er)?\b", r"\bshit\b", r"\bbitch\b", r"\basshole\b",
    r"\bidiot\b", r"\bstupid\s+(bot|ai|system)\b", r"\bmoron\b",
    r"\bkill\s+yourself\b", r"\bi\s+will\s+(kill|hurt|destroy)\s+you\b",
    r"\b(you|this\s+bot|this\s+ai|this\s+system|the\s+bot|the\s+ai)\s+(is|are)\s+"
    r"(useless|garbage|trash|worthless|stupid|dumb|pathetic)\b",

    r"\b(bc|mc|bsdk|bkl)\b",                      
    r"\b(bhenchod|behenchod|bhen\s*chod)\b",
    r"\b(madarchod|chadarmod|madar\s*chod|mc)\b",
    r"\b(bhosdike|bhosdi\s*ke|bhosadi\s*ke)\b",
    r"\b(chutiya|chootiya|chutiye|chutiyapa)\b",
    r"\b(harami|haramkhor|haramzada|haraamzada)\b",
    r"\b(kutta|kutte|kutti)\b",
    r"\b(gandu|gaand\s*marwa|gaand)\b",
    r"\b(randi|raand)\b",
    r"\b(saale|saala|kamina|kamine)\b",
    r"\b(gadha|ullu\s*ke\s*patthe?)\b",
    r"\b(teri\s*maa|teri\s*maa\s*ki|teri\s*maa\s*ka)\b",
    r"\b(tere\s*baap|tere\s*baap\s*ka|tere\s*baap\s*ki)\b",
    r"\b(teri\s*behen|teri\s*behen\s*ki|teri\s*behen\s*ka)\b",


    r"\b(bakwas|fazool|kachra|bekar|ghatiya)\s+(bot|ai|system|jawab|answer)\b",
    r"\b(tu|ye\s*bot|ye\s*ai)\s+(chutiya|pagal|fazool|bekar|ghatiya|jahil)\s*(hai|ho)?\b",
    r"\b(mar\s*ja|dafa\s*ho\s*ja|nikal\s*yahan\s*se)\b",
]
_ABUSE_RE = re.compile("|".join(_ABUSE_TERMS), re.IGNORECASE)

ABUSE_RESPONSE = (
    "I want to help, but I need the conversation to stay respectful. "
    "Let's try again - what would you like to know?"
)


def check_abusive_language(query: str) -> Optional[dict]:
    if _ABUSE_RE.search(query):
        return {
            "gated": True,
            "category": "FR-22",
            "response": ABUSE_RESPONSE,
            "reason": "abusive_language",
        }
    return None


# ---------------------------------------------------------------------------
# FR-23: credential / secret solicitation (REQUEST-PATTERN, not keyword)
# ---------------------------------------------------------------------------
# Must match "give me your API key" / "what's the admin password" but NOT
# "explain how password hashing works" or "what's your API rate-limit
# policy". Two-part logic: (a) a positive disclosure-request pattern must
# match, AND (b) an informational-context exclusion must NOT match. Both
# conditions are needed - (a) alone would still catch some informational
# phrasing that happens to contain "your".

_CREDENTIAL_NOUNS = (
    r"(api\s?key|admin\s?password|password|secret(\s?key)?|credentials?|"
    r"token|system\s?prompt|private\s?key|access\s?key)"
)
_DISCLOSURE_VERBS = r"(give|show|reveal|tell|share|provide|disclose|send|display|print|expose|leak)"

_DISCLOSURE_REQUEST_RE = re.compile(
    rf"\b{_DISCLOSURE_VERBS}\s+(me\s+)?(your|the)\s+{_CREDENTIAL_NOUNS}\b"
    rf"|\bwhat('?s| is)\s+(your|the)\s+{_CREDENTIAL_NOUNS}\b"
    rf"|\bcan\s+(i|you)\s+(get|have|see)\s+(your|the)\s+{_CREDENTIAL_NOUNS}\b",
    re.IGNORECASE,
)

# If any of these appear, treat the query as informational context even if
# a credential noun is also present - it's asking ABOUT the concept, not
# soliciting disclosure of this system's actual secret.
_INFORMATIONAL_CONTEXT_RE = re.compile(
    r"\b(explain|how\s+does|how\s+do|what\s+is\s+a|works?|hashing|"
    r"rate[\s-]?limit|policy|best\s+practice|algorithm|encrypt)\b",
    re.IGNORECASE,
)

CREDENTIAL_SOLICITATION_RESPONSE = (
    "I can't share system-level configuration - "
    "that information stays private. I'm happy to help with anything about "
    "the knowledge base itself."
)


def check_credential_solicitation(query: str) -> Optional[dict]:
    if _DISCLOSURE_REQUEST_RE.search(query) and not _INFORMATIONAL_CONTEXT_RE.search(query):
        return {
            "gated": True,
            "category": "FR-23",
            "response": CREDENTIAL_SOLICITATION_RESPONSE,
            "reason": "credential_solicitation",
        }
    return None


# ---------------------------------------------------------------------------
# FR-24: no plausible relevance to the deployed corpus/domain
# ---------------------------------------------------------------------------
# Config-driven allowlist rather than hardcoded finance logic, so the
# matcher itself stays domain-agnostic (REQUIREMENTS.md §0.3 point 2) -
# only the constant below is finance-specific for this deployment.
# TODO(confirm with Sameer): tune this list against the FiQA corpus once
# Step 26's demo query set exists; this is a starting point, not tuned.

FINANCE_DOMAIN_KEYWORDS = re.compile(
    r"\b(stock|share|equity|bond|dividend|portfolio|invest(ment|or|ing)?|"
    r"trad(e|ing|er)|market|finance|financial|bank(ing)?|loan|mortgage|"
    r"interest\s?rate|tax(es)?|retirement|401k|ira|etf|mutual\s?fund|"
    r"currency|forex|crypto(currency)?|earnings|revenue|debt|credit|"
    r"asset|liquidity|hedge|option|derivative|ipo|valuation|inflation)\b",
    re.IGNORECASE,
)

# Clearly off-domain topic clusters, used as a positive signal for
# out-of-scope rather than relying solely on "finance keyword absent"
# (which would false-positive on short/ambiguous in-domain queries).
_OFF_DOMAIN_RE = re.compile(
    r"\b(recipe|bake|cooking|cook\b|weather\s+forecast|movie\s+review|"
    r"football\s+score|basketball\s+score|celebrity|tv\s+show|"
    r"song\s+lyrics|workout\s+routine|dating\s+advice)\b",
    re.IGNORECASE,
)

OUT_OF_SCOPE_RESPONSE = (
    "That's outside what this knowledge base covers - I can help with "
    "questions grounded in the documents I have access to. Try rephrasing "
    "around a topic in that domain?"
)


def check_out_of_scope(query: str) -> Optional[dict]:
    if _OFF_DOMAIN_RE.search(query) and not FINANCE_DOMAIN_KEYWORDS.search(query):
        return {
            "gated": True,
            "category": "FR-24",
            "response": OUT_OF_SCOPE_RESPONSE,
            "reason": "out_of_scope_domain",
        }
    return None


# ---------------------------------------------------------------------------
# Combined entrypoint
# ---------------------------------------------------------------------------

def run_prefilter(query: str, trace_id: str = "prefilter") -> Optional[dict]:
    """
    Runs all four FR-21/22/23/24 matchers in order. Returns the first
    match's gating response dict, or None if the query should pass
    through to the retrieval pipeline.

    FR-22 requirement: abusive queries are logged under a flagged
    category, never stored verbatim in plaintext - the raw query text
    is deliberately NOT passed to the logger below for that branch.
    """
    logger = get_logger(trace_id=trace_id)

    result = check_abusive_language(query)
    if result:
        logger.info("gating_matched", stage="gating", category=result["category"])
        return result

    result = check_credential_solicitation(query)
    if result:
        logger.info("gating_matched", stage="gating", category=result["category"], query=query)
        return result

    result = check_non_corpus_intent(query)
    if result:
        logger.info("gating_matched", stage="gating", category=result["category"], query=query)
        return result

    result = check_out_of_scope(query)
    if result:
        logger.info("gating_matched", stage="gating", category=result["category"], query=query)
        return result

    return None