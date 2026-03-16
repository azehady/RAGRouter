"""LLM-based reranking with position-aware score blending.

Takes top candidates from RRF fusion and scores them via LLM for relevance
to the original query. Blends LLM scores with RRF scores for final ranking.

~300ms budget per call via LiteLLM proxy.
Falls back to input order on any failure.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx
import structlog

logger = structlog.get_logger()

# Feature flag
RERANKING_ENABLED = os.environ.get("HYBRID_LLM_RERANKING", "true").lower() in (
    "true",
    "1",
    "yes",
)

RERANK_MODEL = os.environ.get("HYBRID_RERANK_MODEL", "gpt-4o-mini")
RERANK_TIMEOUT_S = float(os.environ.get("HYBRID_RERANK_TIMEOUT", "8.0"))
RERANK_CANDIDATES = int(os.environ.get("HYBRID_RERANK_CANDIDATES", "15"))
RERANK_TOP_K = int(os.environ.get("HYBRID_RERANK_TOP_K", "5"))
RERANK_BLEND_ALPHA = float(os.environ.get("HYBRID_RERANK_BLEND_ALPHA", "0.5"))

_RERANK_PROMPT = """\
You are a relevance scoring expert. Score each candidate document's relevance to the query.

Query: {query}

Candidates:
{candidates}

For each candidate, assign a relevance score from 0 to 10 where:
- 0 = completely irrelevant
- 5 = somewhat relevant, partially answers the query
- 10 = perfectly relevant, directly answers the query

Respond with ONLY a JSON array of objects (no markdown, no explanation):
[{{"id": "<candidate_id>", "score": <0-10>, "reason": "<brief reason>"}}]"""


@dataclass
class RerankCandidate:
    """A candidate for reranking."""

    id: str
    text: str = ""
    rrf_score: float = 0.0
    llm_score: float = 0.0
    blended_score: float = 0.0
    reason: str = ""
    payload: dict = field(default_factory=dict)


@dataclass
class RerankStats:
    """Telemetry for the reranking step."""

    enabled: bool = True
    success: bool = False
    input_count: int = 0
    output_count: int = 0
    latency_ms: float = 0.0
    error: str = ""


async def rerank_results(
    query: str,
    candidates: list[dict],
    top_k: int | None = None,
    blend_alpha: float | None = None,
    litellm_api_base: str | None = None,
    model: str | None = None,
) -> tuple[list[RerankCandidate], RerankStats]:
    """Rerank candidates using LLM scoring with RRF score blending.

    Args:
        query: Original user query.
        candidates: List of candidate dicts with at least {id, text, rrf_score}.
        top_k: Number of results to return after reranking.
        blend_alpha: Blending weight. final = alpha * rrf_normalized + (1-alpha) * llm_normalized.
            0.0 = pure LLM, 1.0 = pure RRF.
        litellm_api_base: LiteLLM proxy URL.
        model: LLM model for scoring.

    Returns:
        Tuple of (sorted RerankCandidate list, RerankStats).
        On failure, returns candidates in original order.
    """
    import time

    _top_k = top_k if top_k is not None else RERANK_TOP_K
    _alpha = blend_alpha if blend_alpha is not None else RERANK_BLEND_ALPHA
    stats = RerankStats(input_count=len(candidates))

    if not RERANKING_ENABLED:
        stats.enabled = False
        return _passthrough(candidates, _top_k), stats

    if not candidates:
        return [], stats

    api_base = litellm_api_base or os.environ.get(
        "HYBRID_LITELLM_API_BASE", "http://localhost:4000"
    )
    llm_model = model or RERANK_MODEL
    start = time.monotonic()

    # Limit candidates sent to LLM
    max_candidates = min(len(candidates), RERANK_CANDIDATES)
    to_score = candidates[:max_candidates]

    try:
        # Build candidate text for prompt
        candidate_text = ""
        for i, c in enumerate(to_score, start=1):
            text_preview = c.get("text", "")[:300]
            candidate_text += f"\n[{c.get('id', f'doc-{i}')}]\n{text_preview}\n"

        prompt = _RERANK_PROMPT.format(query=query, candidates=candidate_text)

        async with httpx.AsyncClient(timeout=RERANK_TIMEOUT_S) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                json={
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.1,
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

        # Handle both {"results": [...]} and [...] formats
        if isinstance(parsed, dict):
            scores_list = parsed.get("results", parsed.get("scores", []))
        else:
            scores_list = parsed

        # Build LLM score map
        llm_scores: dict[str, tuple[float, str]] = {}
        for item in scores_list:
            doc_id = item.get("id", "")
            score = float(item.get("score", 0)) / 10.0  # Normalize 0-10 to 0-1
            reason = item.get("reason", "")
            llm_scores[doc_id] = (min(1.0, max(0.0, score)), reason)

        # Normalize RRF scores to 0-1 range
        rrf_scores = [c.get("rrf_score", c.get("score", 0.0)) for c in to_score]
        max_rrf = max(rrf_scores) if rrf_scores else 1.0
        min_rrf = min(rrf_scores) if rrf_scores else 0.0
        rrf_range = max_rrf - min_rrf if max_rrf != min_rrf else 1.0

        # Blend scores
        results = []
        for c in to_score:
            doc_id = c.get("id", "")
            raw_rrf = c.get("rrf_score", c.get("score", 0.0))
            normalized_rrf = (raw_rrf - min_rrf) / rrf_range

            llm_score, reason = llm_scores.get(doc_id, (0.5, "not scored"))
            blended = _alpha * normalized_rrf + (1.0 - _alpha) * llm_score

            results.append(
                RerankCandidate(
                    id=doc_id,
                    text=c.get("text", ""),
                    rrf_score=raw_rrf,
                    llm_score=llm_score,
                    blended_score=blended,
                    reason=reason,
                    payload={
                        k: v
                        for k, v in c.items()
                        if k not in ("id", "text", "score", "rrf_score")
                    },
                )
            )

        # Sort by blended score descending
        results.sort(key=lambda r: r.blended_score, reverse=True)
        results = results[:_top_k]

        stats.success = True
        stats.output_count = len(results)
        stats.latency_ms = (time.monotonic() - start) * 1000

        logger.info(
            "llm_reranking.success",
            scored=len(llm_scores),
            output=stats.output_count,
            latency_ms=f"{stats.latency_ms:.0f}",
        )

        return results, stats

    except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
        stats.latency_ms = (time.monotonic() - start) * 1000
        stats.error = f"{type(e).__name__}: {e}"
        logger.warning("llm_reranking.failed", error=stats.error)
        return _passthrough(candidates, _top_k), stats

    except Exception as e:
        stats.latency_ms = (time.monotonic() - start) * 1000
        stats.error = str(e)
        logger.warning("llm_reranking.failed", error=stats.error)
        return _passthrough(candidates, _top_k), stats


def _passthrough(candidates: list[dict], top_k: int) -> list[RerankCandidate]:
    """Convert raw candidates to RerankCandidate without LLM scoring."""
    results = []
    for c in candidates[:top_k]:
        results.append(
            RerankCandidate(
                id=c.get("id", ""),
                text=c.get("text", ""),
                rrf_score=c.get("rrf_score", c.get("score", 0.0)),
                blended_score=c.get("rrf_score", c.get("score", 0.0)),
                payload={
                    k: v
                    for k, v in c.items()
                    if k not in ("id", "text", "score", "rrf_score")
                },
            )
        )
    return results
