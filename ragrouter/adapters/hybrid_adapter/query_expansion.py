"""LLM-based query expansion for improved recall.

Generates multiple query variants from a single user query:
- lexical: keyword/synonym variation (optimized for BM25)
- semantic: rephrased for embedding search
- hypothetical: HyDE - what the ideal answer document looks like

Single LLM call via LiteLLM proxy, ~300ms budget.
Falls back to original query on any failure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx
import structlog

logger = structlog.get_logger()

# Feature flag
QUERY_EXPANSION_ENABLED = os.environ.get(
    "HYBRID_QUERY_EXPANSION", "true"
).lower() in ("true", "1", "yes")

EXPANSION_MODEL = os.environ.get("HYBRID_EXPANSION_MODEL", "gpt-4o-mini")
EXPANSION_TIMEOUT_S = float(os.environ.get("HYBRID_EXPANSION_TIMEOUT", "5.0"))

_EXPANSION_PROMPT = """\
You are a search query expansion expert. Given a user question, generate 3 alternative \
query formulations to maximize retrieval recall across keyword and semantic search.

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "lexical": "<keyword/synonym variation optimized for BM25 keyword matching>",
  "semantic": "<rephrased query optimized for embedding similarity search>",
  "hypothetical": "<a short paragraph that an ideal answer document would contain>"
}}

User question: {query}"""


@dataclass
class ExpandedQuery:
    """Container for original + expanded query variants."""

    original: str
    lexical: str = ""
    semantic: str = ""
    hypothetical: str = ""

    @property
    def all_variants(self) -> list[str]:
        """Return all non-empty query variants (always includes original)."""
        variants = [self.original]
        for v in [self.lexical, self.semantic, self.hypothetical]:
            if v and v != self.original:
                variants.append(v)
        return variants

    @property
    def expanded(self) -> bool:
        """Whether expansion produced any additional variants."""
        return bool(self.lexical or self.semantic or self.hypothetical)


@dataclass
class ExpansionStats:
    """Telemetry for expansion step."""

    enabled: bool = True
    success: bool = False
    latency_ms: float = 0.0
    variant_count: int = 1
    error: str = ""


async def expand_query(
    query: str,
    litellm_api_base: str | None = None,
    model: str | None = None,
) -> tuple[ExpandedQuery, ExpansionStats]:
    """Expand a query into multiple variants via LLM.

    Args:
        query: Original user query.
        litellm_api_base: LiteLLM proxy URL. Defaults to HYBRID_LITELLM_API_BASE env var.
        model: LLM model to use. Defaults to HYBRID_EXPANSION_MODEL env var.

    Returns:
        Tuple of (ExpandedQuery, ExpansionStats).
        On failure, ExpandedQuery contains only the original query.
    """
    import time

    stats = ExpansionStats()

    if not QUERY_EXPANSION_ENABLED:
        stats.enabled = False
        return ExpandedQuery(original=query), stats

    api_base = litellm_api_base or os.environ.get(
        "HYBRID_LITELLM_API_BASE", "http://localhost:4000"
    )
    llm_model = model or EXPANSION_MODEL
    start = time.monotonic()

    try:
        prompt = _EXPANSION_PROMPT.format(query=query)

        async with httpx.AsyncClient(timeout=EXPANSION_TIMEOUT_S) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                json={
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.3,
                    "response_format": {"type": "json_object"},
                },
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}",
                },
            )
            resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        expanded = ExpandedQuery(
            original=query,
            lexical=parsed.get("lexical", "").strip(),
            semantic=parsed.get("semantic", "").strip(),
            hypothetical=parsed.get("hypothetical", "").strip(),
        )

        stats.success = True
        stats.variant_count = len(expanded.all_variants)
        stats.latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "query_expansion.success",
            variants=stats.variant_count,
            latency_ms=f"{stats.latency_ms:.0f}",
        )

        return expanded, stats

    except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
        stats.latency_ms = (time.monotonic() - start) * 1000
        stats.error = f"{type(e).__name__}: {e}"
        logger.warning("query_expansion.failed", error=stats.error)
        return ExpandedQuery(original=query), stats

    except Exception as e:
        stats.latency_ms = (time.monotonic() - start) * 1000
        stats.error = str(e)
        logger.warning("query_expansion.failed", error=stats.error)
        return ExpandedQuery(original=query), stats
