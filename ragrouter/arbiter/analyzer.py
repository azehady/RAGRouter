"""Query signal extraction: classifies queries into intents and features.

Must be fast (<50ms). Uses regex patterns with optional LLM fallback.
"""

from __future__ import annotations

import re

from ragrouter.schemas import QueryIntent, QuerySignals


# ---------------------------------------------------------------------------
# Pattern banks for intent classification
# ---------------------------------------------------------------------------

_RELATIONSHIP_PATTERNS = [
    r"\bhow\s+(?:is|are|does|do)\s+\w+\s+(?:related|connected|linked)\b",
    r"\brelationship\s+between\b",
    r"\bcompare\b",
    r"\bdifference\s+between\b",
    r"\bwhat\s+connects\b",
    r"\binteract\s+with\b",
    r"\bdepend(?:s|ency|encies)\s+(?:on|between)\b",
]

_SYNTHESIS_PATTERNS = [
    r"\bsummar(?:y|ize|ise)\b",
    r"\bexplain\s+(?:how|why|the)\b",
    r"\boverview\s+of\b",
    r"\bdescribe\s+(?:the\s+)?(?:process|workflow|architecture)\b",
    r"\bhow\s+does\s+(?:the\s+)?\w+\s+work\b",
    r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:best\s+practice|recommendation)\b",
    r"\bstep.by.step\b",
]

_SECTION_PATTERNS = [
    r"\bsection\b",
    r"\bchapter\b",
    r"\bpage\b",
    r"\bwhere\s+(?:is|can\s+I\s+find|does)\b",
    r"\bin\s+which\s+(?:doc|document|file)\b",
    r"\bunder\s+(?:which|what)\s+(?:heading|section)\b",
]

_EXPLORATORY_PATTERNS = [
    r"\bwhat\s+(?:all|kind|types?\s+of)\b",
    r"\blist\s+(?:all|the)\b",
    r"\bshow\s+me\s+(?:all|everything)\b",
    r"\btell\s+me\s+about\b",
    r"\bwhat\s+(?:features|capabilities|integrations)\b",
]

_MULTI_HOP_PATTERNS = [
    r"\band\s+(?:also|then|how)\b",
    r"\bcompare\s+\w+\s+(?:and|with|to|vs)\b",
    r"\bboth\s+\w+\s+and\b",
    r"\brelate\b",
    r"\bacross\s+(?:different|multiple)\b",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _extract_keywords(query: str) -> list[str]:
    """Extract significant keywords (nouns/proper nouns heuristic)."""
    stop = {
        "a", "an", "the", "is", "are", "was", "were", "do", "does", "did",
        "how", "what", "where", "when", "why", "which", "who", "can", "could",
        "will", "would", "should", "i", "me", "my", "we", "our", "you", "your",
        "it", "its", "they", "them", "their", "this", "that", "these", "those",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
        "and", "or", "but", "not", "if", "then", "else", "all", "any", "some",
        "has", "have", "had", "be", "been", "being", "get", "got", "getting",
    }
    words = re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())
    return [w for w in words if w not in stop]


def _count_entities(query: str) -> int:
    """Count likely named entities (capitalised words not at sentence start)."""
    words = query.split()
    count = 0
    for i, word in enumerate(words):
        cleaned = re.sub(r"[^a-zA-Z]", "", word)
        if cleaned and cleaned[0].isupper() and i > 0:
            count += 1
    return count


def analyze_query(query: str) -> QuerySignals:
    """Extract routing signals from a query. Fast, deterministic, no LLM."""
    q = query.strip()

    # Intent classification (priority order)
    if _matches_any(q, _RELATIONSHIP_PATTERNS):
        intent = QueryIntent.RELATIONSHIP
    elif _matches_any(q, _SYNTHESIS_PATTERNS):
        intent = QueryIntent.SYNTHESIS
    elif _matches_any(q, _EXPLORATORY_PATTERNS):
        intent = QueryIntent.EXPLORATORY
    else:
        intent = QueryIntent.LOOKUP

    keywords = _extract_keywords(q)
    entity_count = _count_entities(q)

    # Specificity: more keywords + entities = more specific
    specificity = min(1.0, (len(keywords) + entity_count * 2) / 10.0)

    return QuerySignals(
        query=q,
        intent=intent,
        specificity=specificity,
        multi_hop=_matches_any(q, _MULTI_HOP_PATTERNS),
        section_reference=_matches_any(q, _SECTION_PATTERNS),
        entity_count=entity_count,
        keywords=keywords,
    )
