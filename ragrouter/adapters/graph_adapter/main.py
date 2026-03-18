"""Graph engine adapter: Webster + KGraph for relationship and multi-hop queries.

Wraps Webster (Cypher) and KGraph (path-finding) behind the standard engine contract.
Pipeline: Query → LLM intent extraction → Graph query (Webster or KGraph) → LLM synthesis → EngineResponse

Selection logic:
- Simple path queries → KGraph connected_paths / shortest_paths API
- Complex Cypher → Webster direct query
- Fallback: KGraph path API if Webster is unavailable
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI
from pydantic_settings import BaseSettings

from ragrouter.adapters.graph_adapter.kgraph_client import KGraphClient
from ragrouter.adapters.graph_adapter.query_builder import (
    build_cypher_from_template,
    extract_graph_intent,
)
from ragrouter.adapters.graph_adapter.webster_client import WebsterClient
from ragrouter.schemas import (
    Citation,
    EngineRequest,
    EngineResponse,
    Scores,
    Trace,
    Usage,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class GraphEngineSettings(BaseSettings):
    webster_url: str = "http://webster.ciroos.svc.cluster.local:8000"
    kgraph_url: str = "http://kgraph.ciroos.svc.cluster.local:8000"
    litellm_api_base: str = "http://localhost:4000"
    litellm_model: str = "gpt-4o-mini"
    default_depth: int = 3
    max_results: int = 25
    default_source: str = "SERVICE_NOW"

    model_config = {"env_prefix": "GRAPH_"}


settings = GraphEngineSettings()


# ---------------------------------------------------------------------------
# Clients (lazy-initialized)
# ---------------------------------------------------------------------------

_webster: WebsterClient | None = None
_kgraph: KGraphClient | None = None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


async def _execute_graph_query(request: EngineRequest) -> EngineResponse:
    """Full graph query pipeline: intent → graph query → synthesis."""
    start = time.monotonic()
    pipeline_steps = []
    org_id = request.metadata.get("org_id", "system")

    # Step 1: Get graph schema for context
    schema = await _kgraph.get_schema(org_id=org_id, source=settings.default_source)
    schema_types = schema.node_types or []
    pipeline_steps.append(f"schema_discovery ({len(schema_types)} types)")

    # Step 2: Extract intent from natural language
    intent = await extract_graph_intent(
        query=request.query,
        schema_types=schema_types,
        litellm_api_base=settings.litellm_api_base,
    )
    intent_ms = (time.monotonic() - start) * 1000
    pipeline_steps.append(
        f"intent_extraction ({intent_ms:.0f}ms, type={intent.query_type})"
    )

    if intent.error:
        logger.warning("graph.intent_failed", error=intent.error)
        return _error_response(
            f"Could not understand graph query: {intent.error}",
            steps=pipeline_steps,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    # Step 3: Execute graph query
    graph_results = []
    query_method = "unknown"

    if intent.use_cypher and intent.cypher_query:
        # Direct Cypher via Webster
        query_method = "webster_cypher"
        result = await _webster.query_cypher(
            cypher=intent.cypher_query,
            org_id=org_id,
        )
        if result.status == "ready":
            graph_results = result.results
        else:
            pipeline_steps.append(f"webster_cypher_failed ({result.error})")
            # Fallback to KGraph path API
            query_method = "kgraph_paths_fallback"
            graph_results = await _execute_kgraph_fallback(intent, org_id)
    elif intent.query_type in ("dependencies", "impacts", "connections", "path_between"):
        # Template-based Cypher via Webster, with KGraph fallback
        src = intent.source_entity
        tgt = intent.target_entity
        cypher, params = build_cypher_from_template(
            query_type=intent.query_type,
            source_type=src.get("type", "Entity"),
            source_name=src.get("name", ""),
            target_type=tgt.get("type", ""),
            target_name=tgt.get("name", ""),
            depth=settings.default_depth,
            limit=settings.max_results,
        )
        query_method = "webster_template"
        result = await _webster.query_cypher(
            cypher=cypher,
            org_id=org_id,
            parameters=params,
        )
        if result.status == "ready" and result.results:
            graph_results = result.results
        else:
            # Fallback to KGraph
            query_method = "kgraph_paths"
            graph_results = await _execute_kgraph_fallback(intent, org_id)
    else:
        # Default: KGraph path API
        query_method = "kgraph_paths"
        graph_results = await _execute_kgraph_fallback(intent, org_id)

    query_ms = (time.monotonic() - start) * 1000 - intent_ms
    pipeline_steps.append(
        f"graph_query ({query_ms:.0f}ms, method={query_method}, results={len(graph_results)})"
    )

    if not graph_results:
        return _error_response(
            "No graph results found for your query. The entities may not exist in the knowledge graph.",
            steps=pipeline_steps,
            latency_ms=(time.monotonic() - start) * 1000,
        )

    # Step 4: LLM synthesis
    answer, citations, tokens_in, tokens_out = await _synthesize_answer(
        query=request.query,
        graph_results=graph_results,
        max_tokens=request.constraints.max_tokens,
    )
    synthesis_ms = (time.monotonic() - start) * 1000 - intent_ms - query_ms
    pipeline_steps.append(f"llm_synthesis ({synthesis_ms:.0f}ms)")

    total_ms = (time.monotonic() - start) * 1000

    return EngineResponse(
        engine="graph",
        answer=answer,
        citations=citations,
        trace=Trace(
            retrieval_method=query_method,
            docs_searched=0,
            docs_retrieved=len(graph_results),
            reranking_applied=False,
            steps=pipeline_steps,
        ),
        scores=Scores(
            confidence=min(1.0, len(graph_results) / settings.max_results * 2),
            groundedness=0.9 if graph_results else 0.0,  # Graph data is factual
            coverage=min(1.0, len(graph_results) / 5),
            relevance=0.8 if graph_results else 0.0,
        ),
        usage=Usage(
            latency_ms=total_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost_usd=(tokens_in * 0.15 + tokens_out * 0.6) / 1_000_000,
        ),
    )


async def _execute_kgraph_fallback(intent, org_id: str) -> list[dict]:
    """Execute query via KGraph path-finding API."""
    src = intent.source_entity
    tgt = intent.target_entity

    if not src.get("name") and not src.get("type"):
        # Try entity search if we have entities from intent
        if intent.entities:
            src = intent.entities[0]
            if len(intent.entities) > 1:
                tgt = intent.entities[1]

    if not src.get("name") and not src.get("type"):
        return []

    # First, find the actual entity IDs by searching
    entities = await _kgraph.search_entities(
        org_id=org_id,
        entity_type=src.get("type"),
        entity_name=src.get("name"),
        source=settings.default_source,
    )

    if not entities:
        return []

    source_entity = entities[0]

    if tgt and (tgt.get("type") or tgt.get("name")):
        # Path between two entities
        paths = await _kgraph.find_paths(
            org_id=org_id,
            source_entity_type=source_entity.entity_type,
            source_entity_id=source_entity.entity_id,
            dest_entity_type=tgt.get("type"),
            source=settings.default_source,
            depth=settings.default_depth,
        )
        return [{"nodes": p.nodes, "edges": p.edges, "depth": p.depth} for p in paths]

    # General connected entities
    paths = await _kgraph.find_paths(
        org_id=org_id,
        source_entity_type=source_entity.entity_type,
        source_entity_id=source_entity.entity_id,
        source=settings.default_source,
        depth=settings.default_depth,
    )
    return [{"nodes": p.nodes, "edges": p.edges, "depth": p.depth} for p in paths]


async def _synthesize_answer(
    query: str,
    graph_results: list[dict],
    max_tokens: int = 2000,
) -> tuple[str, list[Citation], int, int]:
    """Use LLM to synthesize a natural language answer from graph results."""
    # Format graph results for LLM
    results_text = ""
    for i, r in enumerate(graph_results[:20], start=1):
        results_text += f"\nResult {i}: {_format_graph_result(r)}"

    prompt = (
        "You are answering a question about infrastructure relationships using knowledge graph data.\n"
        "Synthesize a clear, structured answer from the graph query results below.\n"
        "Cite specific entities and relationships found in the data.\n\n"
        f"Question: {query}\n\n"
        f"Graph Results:{results_text}\n\n"
        "Answer:"
    )

    citations = []
    for r in graph_results[:10]:
        nodes = r.get("nodes", [])
        for node in (nodes if isinstance(nodes, list) else []):
            name = node.get("name", "") if isinstance(node, dict) else str(node)
            if name:
                citations.append(Citation(
                    doc_id=node.get("entity_id", name) if isinstance(node, dict) else name,
                    section=node.get("entity_type", "entity") if isinstance(node, dict) else "entity",
                    text=name,
                    score=0.9,
                ))

    # Deduplicate citations by doc_id
    seen = set()
    unique_citations = []
    for c in citations:
        if c.doc_id not in seen:
            seen.add(c.doc_id)
            unique_citations.append(c)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.litellm_api_base}/chat/completions",
                json={
                    "model": settings.litellm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}",
                },
            )
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return answer, unique_citations, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    except Exception as e:
        logger.error("graph.synthesis_failed", error=str(e))
        # Fallback: structured dump of results
        answer = f"Graph query returned {len(graph_results)} results:\n"
        for i, r in enumerate(graph_results[:10], start=1):
            answer += f"\n{i}. {_format_graph_result(r)}"
        return answer, unique_citations, 0, 0


def _format_graph_result(result: dict) -> str:
    """Format a single graph result for display."""
    parts = []
    if "source" in result and "dependency" in result:
        parts.append(f"{result['source']} → {result.get('dep_type', '')} {result['dependency']}")
    elif "source" in result and "impacted" in result:
        parts.append(f"{result['impacted']} depends on {result['source']}")
    elif "nodes" in result:
        nodes = result["nodes"]
        if isinstance(nodes, list):
            node_names = [
                n.get("name", str(n)) if isinstance(n, dict) else str(n)
                for n in nodes
            ]
            parts.append(" → ".join(node_names))
    elif "node_names" in result:
        parts.append(" → ".join(result["node_names"]))

    if not parts:
        # Fallback: dump keys
        parts.append(str({k: v for k, v in result.items() if k != "edges"}))

    return "; ".join(parts)


def _error_response(message: str, steps: list[str], latency_ms: float) -> EngineResponse:
    return EngineResponse(
        engine="graph",
        answer=message,
        citations=[],
        trace=Trace(
            retrieval_method="graph",
            docs_searched=0,
            docs_retrieved=0,
            reranking_applied=False,
            steps=steps,
        ),
        scores=Scores(),
        usage=Usage(latency_ms=latency_ms),
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _webster, _kgraph
    _webster = WebsterClient(base_url=settings.webster_url)
    _kgraph = KGraphClient(base_url=settings.kgraph_url)
    logger.info("graph_engine.started", webster=settings.webster_url, kgraph=settings.kgraph_url)
    yield
    logger.info("graph_engine.stopped")


app = FastAPI(
    title="RAGRouter Graph Engine",
    version="0.1.0",
    description="Graph RAG: Webster (Cypher) + KGraph (path-finding) for relationship queries",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    webster_health = await _webster.health() if _webster else {"status": "not_initialized"}
    kgraph_health = await _kgraph.health() if _kgraph else {"status": "not_initialized"}
    return {
        "status": "ok",
        "engine": "graph",
        "webster": webster_health.get("status", "unknown"),
        "kgraph": kgraph_health.get("status", "unknown"),
    }


@app.post("/query", response_model=EngineResponse)
async def query(request: EngineRequest) -> EngineResponse:
    """Execute a graph query and return standardized response."""
    logger.info("graph.query", query=request.query[:100], corpus=request.corpus_id)
    return await _execute_graph_query(request)
