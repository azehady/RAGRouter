"""Standard engine contract: the uniform interface all RAG engines must implement."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QueryIntent(str, Enum):
    LOOKUP = "lookup"
    RELATIONSHIP = "relationship"
    SYNTHESIS = "synthesis"
    EXPLORATORY = "exploratory"
    COMPARISON = "comparison"


class DispatchMode(str, Enum):
    CASCADE = "cascade"
    PARALLEL_TOPK = "parallel_topk"


class EngineType(str, Enum):
    HYBRID = "hybrid"
    BASELINE = "baseline"
    TOC = "toc"
    GRAPH = "graph"
    AGENT = "agent"
    LONGCONTEXT = "longcontext"


# ---------------------------------------------------------------------------
# Engine Request / Response — every engine adapter normalises to this
# ---------------------------------------------------------------------------


class Constraints(BaseModel):
    max_latency_ms: int = Field(default=5000, description="Max acceptable latency in ms")
    must_cite: bool = Field(default=True, description="Require citations in response")
    max_tokens: int = Field(default=2000, description="Max output tokens")


class EngineRequest(BaseModel):
    query: str
    corpus_id: str = Field(default="ciroos-docs", description="Target document corpus")
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    doc_id: str
    section: str | None = None
    span: str | None = None
    text: str | None = None
    score: float | None = None


class Trace(BaseModel):
    retrieval_method: str
    docs_searched: int = 0
    docs_retrieved: int = 0
    reranking_applied: bool = False
    steps: list[str] = Field(default_factory=list, description="Pipeline steps executed")
    extra: dict[str, Any] = Field(default_factory=dict)


class Scores(BaseModel):
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    groundedness: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class Usage(BaseModel):
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0.0


class EngineResponse(BaseModel):
    engine: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    trace: Trace
    scores: Scores = Field(default_factory=Scores)
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# Arbiter-level models — routing, scoring, final output
# ---------------------------------------------------------------------------


class CorpusSignals(BaseModel):
    """Computed once at index time and cached per corpus."""

    corpus_id: str
    doc_count: int = 0
    avg_doc_tokens: float = 0.0
    total_tokens: int = 0
    structure_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Heading density (H1/H2/H3 per 1K tokens)"
    )
    entity_density: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Named entities per 1K tokens"
    )
    doc_types: dict[str, int] = Field(default_factory=dict, description="File type distribution")


class QuerySignals(BaseModel):
    """Extracted per query, must be fast (<50ms)."""

    query: str
    intent: QueryIntent = QueryIntent.LOOKUP
    specificity: float = Field(default=0.5, ge=0.0, le=1.0)
    multi_hop: bool = False
    section_reference: bool = False
    entity_count: int = 0
    keywords: list[str] = Field(default_factory=list)


class EngineCandidate(BaseModel):
    """A scored engine candidate from the planner."""

    engine: EngineType
    priority: int = 1
    reason: str = ""


class RoutingDecision(BaseModel):
    """The planner's output: which engines to use and in what order."""

    query_signals: QuerySignals
    candidates: list[EngineCandidate]
    mode: DispatchMode = DispatchMode.CASCADE
    budget: Constraints = Field(default_factory=Constraints)


class ScoredResponse(BaseModel):
    """An engine response with arbiter-assigned scores."""

    response: EngineResponse
    arbiter_score: float = Field(default=0.0, ge=0.0, le=1.0)
    chosen: bool = False
    reason: str = ""


class ArbiterResult(BaseModel):
    """Final output from the arbiter: the best answer plus full trace."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chosen_engine: str
    routing_decision: RoutingDecision
    engine_responses: list[ScoredResponse] = Field(default_factory=list)
    total_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Engine registry config
# ---------------------------------------------------------------------------


class EngineConfig(BaseModel):
    """Per-engine configuration loaded from YAML."""

    name: str
    type: EngineType
    url: str
    enabled: bool = True
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    latency_profile_ms: int = Field(default=1000, description="Typical p50 latency")
    cost_rank: int = Field(default=1, ge=1, le=5, description="1=cheapest, 5=most expensive")
    max_concurrent: int = Field(default=5, description="Max concurrent requests")
