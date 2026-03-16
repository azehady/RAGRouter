"""Arbiter FastAPI service: the central router that dispatches to engines."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import structlog
import yaml
from fastapi import FastAPI, HTTPException

from ragrouter.schemas import (
    ArbiterResult,
    Constraints,
    CorpusSignals,
    DispatchMode,
    EngineRequest,
    EngineResponse,
    EngineType,
)

from .analyzer import analyze_query
from .planner import select_engines
from .registry import EngineRegistry
from .scorer import score_responses

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Globals (initialised in lifespan)
# ---------------------------------------------------------------------------

_registry: EngineRegistry | None = None
_config: dict[str, Any] = {}
_http_client: httpx.AsyncClient | None = None

# Default corpus signals for ciroos-docs (would be computed from real data)
_corpus_signals: dict[str, CorpusSignals] = {
    "ciroos-docs": CorpusSignals(
        corpus_id="ciroos-docs",
        doc_count=22,
        avg_doc_tokens=1800,
        total_tokens=40_000,
        structure_score=0.45,
        entity_density=0.2,
        doc_types={"md": 22},
    ),
}


def _load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or Path(__file__).parent / "config.yaml"
    with config_path.open() as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _registry, _config, _http_client
    _config = _load_config()
    _registry = EngineRegistry.from_yaml()
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    logger.info("arbiter.started", engines=len(_registry.enabled_engines()))
    yield
    await _http_client.aclose()
    logger.info("arbiter.stopped")


app = FastAPI(
    title="RAGRouter Arbiter",
    version="0.1.0",
    description="Meta-RAG orchestration: routes queries to the best RAG engine",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Engine dispatch
# ---------------------------------------------------------------------------


async def _call_engine(engine_name: str, request: EngineRequest) -> EngineResponse | None:
    """Call a single engine via HTTP POST /query."""
    cfg = _registry.get(engine_name)
    if not cfg or not cfg.enabled:
        return None

    url = f"{cfg.url.rstrip('/')}/query"
    start = time.monotonic()

    try:
        resp = await _http_client.post(
            url,
            json=request.model_dump(),
            timeout=min(request.constraints.max_latency_ms / 1000.0, 30.0),
        )
        resp.raise_for_status()
        data = resp.json()
        engine_resp = EngineResponse(**data)
        engine_resp.usage.latency_ms = (time.monotonic() - start) * 1000
        return engine_resp
    except Exception as e:
        elapsed = (time.monotonic() - start) * 1000
        logger.warning("engine.call_failed", engine=engine_name, error=str(e), latency_ms=elapsed)
        return None


async def _dispatch_cascade(
    candidates: list[tuple[str, int]],
    request: EngineRequest,
    confidence_threshold: float,
    max_fallbacks: int,
) -> list[EngineResponse]:
    """Cascade mode: try best-fit first, fallback on low confidence."""
    responses: list[EngineResponse] = []
    fallbacks = 0

    for engine_name, _priority in candidates:
        resp = await _call_engine(engine_name, request)
        if resp:
            responses.append(resp)
            # If good enough, stop early
            if resp.scores.confidence >= confidence_threshold:
                break
        fallbacks += 1
        if fallbacks >= max_fallbacks:
            break

    return responses


async def _dispatch_parallel(
    candidates: list[tuple[str, int]],
    request: EngineRequest,
    top_k: int,
) -> list[EngineResponse]:
    """Parallel mode: run top-K concurrently, collect all results."""
    selected = candidates[:top_k]
    tasks = [_call_engine(name, request) for name, _priority in selected]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "engines": len(_registry.enabled_engines()) if _registry else 0}


@app.post("/ask", response_model=ArbiterResult)
async def ask(request: EngineRequest) -> ArbiterResult:
    """Main endpoint: route query to best engine(s), return arbitrated result."""
    start = time.monotonic()

    # 1. Analyze query
    query_signals = analyze_query(request.query)
    logger.info("query.analyzed", intent=query_signals.intent, keywords=query_signals.keywords)

    # 2. Get corpus signals
    corpus = _corpus_signals.get(
        request.corpus_id,
        CorpusSignals(corpus_id=request.corpus_id),
    )

    # 3. Plan: select engines
    routing = select_engines(query_signals, corpus, request.constraints, _registry)
    logger.info(
        "routing.planned",
        mode=routing.mode,
        candidates=[c.engine.value for c in routing.candidates],
    )

    # 4. Dispatch to engines
    candidate_list = [(c.engine.value, c.priority) for c in routing.candidates]
    arbiter_cfg = _config.get("arbiter", {})

    if routing.mode == DispatchMode.CASCADE:
        responses = await _dispatch_cascade(
            candidate_list,
            request,
            confidence_threshold=arbiter_cfg.get("confidence_threshold", 0.6),
            max_fallbacks=arbiter_cfg.get("max_cascade_fallbacks", 2),
        )
    else:
        responses = await _dispatch_parallel(
            candidate_list,
            request,
            top_k=arbiter_cfg.get("parallel_topk", 3),
        )

    # 5. Score and arbitrate
    scoring_weights = _config.get("scoring", {}).get("weights")
    result = score_responses(
        responses,
        routing,
        weights=scoring_weights,
        confidence_threshold=arbiter_cfg.get("confidence_threshold", 0.6),
    )
    result.total_latency_ms = (time.monotonic() - start) * 1000

    logger.info(
        "arbiter.result",
        chosen=result.chosen_engine,
        score=result.engine_responses[0].arbiter_score if result.engine_responses else 0,
        total_latency_ms=result.total_latency_ms,
    )

    return result


@app.post("/route")
async def route(query: str, corpus_id: str = "ciroos-docs") -> dict:
    """Debug endpoint: show routing decision without executing engines."""
    query_signals = analyze_query(query)
    corpus = _corpus_signals.get(corpus_id, CorpusSignals(corpus_id=corpus_id))
    routing = select_engines(query_signals, corpus, Constraints(), _registry)
    return {
        "query_signals": query_signals.model_dump(),
        "routing": routing.model_dump(),
    }


@app.get("/engines")
async def list_engines():
    """List all registered engines and their status."""
    return {
        "engines": [
            {
                "name": e.name,
                "type": e.type.value,
                "enabled": e.enabled,
                "url": e.url,
                "cost_rank": e.cost_rank,
                "latency_profile_ms": e.latency_profile_ms,
                "strengths": e.strengths,
            }
            for e in _registry.all_engines()
        ]
    }


@app.put("/corpus/{corpus_id}/signals")
async def update_corpus_signals(corpus_id: str, signals: CorpusSignals):
    """Update cached corpus signals (normally computed during ingestion)."""
    _corpus_signals[corpus_id] = signals
    return {"status": "updated", "corpus_id": corpus_id}
