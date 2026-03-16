"""Response scoring and arbitration: evaluates engine outputs, picks the best."""

from __future__ import annotations

from ragrouter.schemas import (
    ArbiterResult,
    EngineResponse,
    RoutingDecision,
    ScoredResponse,
)


def _compute_arbiter_score(
    response: EngineResponse,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute a weighted composite score for an engine response.

    Scoring dimensions:
    - groundedness: is the answer supported by cited passages?
    - coverage: how much of the answer is backed by citations?
    - relevance: does the answer address the query?
    - confidence: engine's self-reported certainty
    - latency_penalty: penalise slow responses
    - cost_penalty: penalise expensive responses
    """
    w = weights or {
        "groundedness": 0.30,
        "coverage": 0.25,
        "relevance": 0.25,
        "confidence": 0.10,
        "latency_penalty": 0.05,
        "cost_penalty": 0.05,
    }

    s = response.scores

    # Positive signals
    score = (
        w["groundedness"] * s.groundedness
        + w["coverage"] * s.coverage
        + w["relevance"] * s.relevance
        + w["confidence"] * s.confidence
    )

    # Latency penalty: linear decay from 0 (instant) to -1.0 (>=10s)
    latency_factor = min(response.usage.latency_ms / 10_000.0, 1.0)
    score -= w["latency_penalty"] * latency_factor

    # Cost penalty: linear decay from 0 ($0) to -1.0 (>=$0.01)
    cost_factor = min(response.usage.estimated_cost_usd / 0.01, 1.0)
    score -= w["cost_penalty"] * cost_factor

    return max(0.0, min(1.0, score))


def _has_citations(response: EngineResponse) -> bool:
    return len(response.citations) > 0


def _detect_low_quality(response: EngineResponse) -> str | None:
    """Return a reason string if the response is low quality, else None."""
    if not response.answer or len(response.answer.strip()) < 20:
        return "Answer too short"
    if response.scores.groundedness < 0.2:
        return "Very low groundedness"
    if response.scores.confidence < 0.1:
        return "Very low confidence"
    return None


def score_responses(
    responses: list[EngineResponse],
    routing_decision: RoutingDecision,
    weights: dict[str, float] | None = None,
    confidence_threshold: float = 0.6,
) -> ArbiterResult:
    """Score all engine responses and pick the best one.

    Mode: Winner-Takes-All (default).
    Pick the highest scoring response that passes quality checks.
    """
    if not responses:
        return ArbiterResult(
            answer="No engines returned a response.",
            chosen_engine="none",
            routing_decision=routing_decision,
        )

    scored: list[ScoredResponse] = []
    for resp in responses:
        quality_issue = _detect_low_quality(resp)
        arbiter_score = _compute_arbiter_score(resp, weights)

        # Penalise responses without citations when citations are required
        if routing_decision.budget.must_cite and not _has_citations(resp):
            arbiter_score *= 0.5

        # Penalise low-quality responses
        if quality_issue:
            arbiter_score *= 0.3

        scored.append(
            ScoredResponse(
                response=resp,
                arbiter_score=arbiter_score,
                reason=quality_issue or f"Score: {arbiter_score:.3f}",
            )
        )

    # Sort by score descending
    scored.sort(key=lambda s: s.arbiter_score, reverse=True)

    # Pick the winner
    winner = scored[0]
    winner.chosen = True

    # Check if confidence is below threshold (for cascade fallback signalling)
    if winner.arbiter_score < confidence_threshold and len(scored) > 1:
        winner.reason += f" (below threshold {confidence_threshold}, consider fallback)"

    total_latency = sum(r.response.usage.latency_ms for r in scored)

    return ArbiterResult(
        answer=winner.response.answer,
        citations=winner.response.citations,
        chosen_engine=winner.response.engine,
        routing_decision=routing_decision,
        engine_responses=scored,
        total_latency_ms=total_latency,
    )
