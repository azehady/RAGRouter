"""Engine selection logic: deterministic rules mapping signals to engines."""

from __future__ import annotations

from ragrouter.schemas import (
    Constraints,
    CorpusSignals,
    DispatchMode,
    EngineCandidate,
    EngineType,
    QueryIntent,
    QuerySignals,
    RoutingDecision,
)

from .registry import EngineRegistry


def select_engines(
    query_signals: QuerySignals,
    corpus_signals: CorpusSignals,
    budget: Constraints,
    registry: EngineRegistry,
) -> RoutingDecision:
    """Select engines based on query + corpus signals. Returns ordered candidates."""
    candidates: list[EngineCandidate] = []

    # TOC engine: structured long docs + section-seeking queries
    if (
        registry.get("toc")
        and registry.get("toc").enabled
        and corpus_signals.structure_score > 0.6
        and corpus_signals.avg_doc_tokens > 5000
    ):
        if query_signals.section_reference or query_signals.intent == QueryIntent.LOOKUP:
            candidates.append(
                EngineCandidate(
                    engine=EngineType.TOC,
                    priority=1,
                    reason="Structured corpus + section-seeking query",
                )
            )

    # Graph engine: entity-dense corpus + relationship queries
    if (
        registry.get("graph")
        and registry.get("graph").enabled
        and corpus_signals.entity_density > 0.3
        and (query_signals.multi_hop or query_signals.intent == QueryIntent.RELATIONSHIP)
    ):
        candidates.append(
            EngineCandidate(
                engine=EngineType.GRAPH,
                priority=2 if candidates else 1,
                reason="Entity-dense corpus + relationship/multi-hop query",
            )
        )

    # Hybrid vector: always a good general-purpose choice
    if registry.get("hybrid") and registry.get("hybrid").enabled:
        is_fallback = len(candidates) > 0
        candidates.append(
            EngineCandidate(
                engine=EngineType.HYBRID,
                priority=2 if is_fallback else 1,
                reason="Best general-purpose retrieval" if not is_fallback else "Hybrid fallback",
            )
        )

    # Agent: complex synthesis, only if budget allows
    if (
        registry.get("agent")
        and registry.get("agent").enabled
        and query_signals.intent == QueryIntent.SYNTHESIS
        and budget.max_latency_ms >= 5000
    ):
        candidates.append(
            EngineCandidate(
                engine=EngineType.AGENT,
                priority=len(candidates) + 1,
                reason="Synthesis query with sufficient latency budget",
            )
        )

    # Long-context: tiny corpus, just stuff it all in
    if (
        registry.get("longcontext")
        and registry.get("longcontext").enabled
        and corpus_signals.total_tokens < 50_000
    ):
        if not candidates:
            candidates.append(
                EngineCandidate(
                    engine=EngineType.LONGCONTEXT,
                    priority=1,
                    reason="Small corpus fits in context window",
                )
            )

    # Baseline as last resort if nothing else matched
    if not candidates:
        if registry.get("baseline") and registry.get("baseline").enabled:
            candidates.append(
                EngineCandidate(
                    engine=EngineType.BASELINE,
                    priority=1,
                    reason="Fallback baseline",
                )
            )

    # Always ensure hybrid is present as fallback
    engine_types = {c.engine for c in candidates}
    if EngineType.HYBRID not in engine_types:
        hybrid_cfg = registry.get("hybrid")
        if hybrid_cfg and hybrid_cfg.enabled:
            candidates.append(
                EngineCandidate(
                    engine=EngineType.HYBRID,
                    priority=len(candidates) + 1,
                    reason="Always-on hybrid fallback",
                )
            )

    candidates.sort(key=lambda c: c.priority)

    # Dispatch mode: parallel if budget is generous and multiple candidates
    mode = DispatchMode.CASCADE
    if len(candidates) >= 3 and budget.max_latency_ms >= 5000:
        mode = DispatchMode.PARALLEL_TOPK

    return RoutingDecision(
        query_signals=query_signals,
        candidates=candidates,
        mode=mode,
        budget=budget,
    )
