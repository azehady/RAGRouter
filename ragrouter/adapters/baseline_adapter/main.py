"""Baseline engine adapter: thin wrapper around current Qdrant-only vector search.

Gives the arbiter a simple comparison point for the hybrid engine.
~80 lines of adapter code.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI
from pydantic_settings import BaseSettings

from ragrouter.schemas import (
    Citation,
    EngineRequest,
    EngineResponse,
    Scores,
    Trace,
    Usage,
)

logger = structlog.get_logger()


class BaselineSettings(BaseSettings):
    vectorstore_url: str = "http://localhost:2526"
    collection_name: str = "ciroos-docs"
    org_id: str = "system"
    search_limit: int = 5
    score_threshold: float = 0.35
    litellm_api_base: str = "http://localhost:4000"
    litellm_model: str = "gpt-4o-mini"

    model_config = {"env_prefix": "BASELINE_"}


settings = BaselineSettings()
_http: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    _http = httpx.AsyncClient(timeout=30.0)
    yield
    await _http.aclose()


app = FastAPI(
    title="RAGRouter Baseline Engine",
    version="0.1.0",
    description="Baseline: pure vector search via existing vectorstore service",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "baseline"}


@app.post("/query", response_model=EngineResponse)
async def query(request: EngineRequest) -> EngineResponse:
    start = time.monotonic()

    # 1. Search via existing vectorstore API
    search_resp = await _http.post(
        f"{settings.vectorstore_url}/api/v1/collections/{settings.collection_name}/search",
        json={
            "query": request.query,
            "limit": settings.search_limit,
            "score_threshold": settings.score_threshold,
            "with_payload": ["text", "source_file", "title", "section", "heading_hierarchy"],
        },
        headers={"x-organization-id": settings.org_id},
    )
    search_data = search_resp.json()
    results = search_data.get("results", [])
    retrieval_ms = (time.monotonic() - start) * 1000

    # 2. Build context + citations
    context_parts = []
    citations = []
    for r in results:
        payload = r.get("payload", {})
        context_parts.append(f"[{payload.get('source_file', 'unknown')}] {payload.get('text', '')}")
        citations.append(
            Citation(
                doc_id=payload.get("source_file", "unknown"),
                section=payload.get("section"),
                text=payload.get("text", "")[:200],
                score=r.get("score", 0),
            )
        )

    # 3. LLM answer generation
    context = "\n---\n".join(context_parts)
    answer = ""
    tokens_in = tokens_out = 0
    try:
        llm_resp = await _http.post(
            f"{settings.litellm_api_base}/chat/completions",
            json={
                "model": settings.litellm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Answer based on context. Cite sources.\n\nContext:\n{context}\n\nQuestion: {request.query}\n\nAnswer:",
                    }
                ],
                "max_tokens": request.constraints.max_tokens,
                "temperature": 0.3,
            },
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}"},
        )
        data = llm_resp.json()
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
    except Exception as e:
        logger.error("baseline.llm_failed", error=str(e))
        answer = "Retrieved documents:\n" + "\n".join(
            f"- [{c.doc_id}] {c.text[:100]}..." for c in citations
        )

    total_ms = (time.monotonic() - start) * 1000
    avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
    top_score = results[0].get("score", 0) if results else 0

    return EngineResponse(
        engine="baseline",
        answer=answer,
        citations=citations,
        trace=Trace(
            retrieval_method="dense_cosine",
            docs_searched=settings.search_limit,
            docs_retrieved=len(results),
            reranking_applied=False,
            steps=[
                f"vectorstore_search ({retrieval_ms:.0f}ms, {len(results)} docs)",
                f"llm_generation ({total_ms - retrieval_ms:.0f}ms)",
            ],
        ),
        scores=Scores(
            confidence=min(1.0, top_score),
            groundedness=min(1.0, avg_score),
            coverage=min(1.0, len(results) / settings.search_limit),
            relevance=min(1.0, top_score),
        ),
        usage=Usage(
            latency_ms=total_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost_usd=(tokens_in * 0.15 + tokens_out * 0.6) / 1_000_000,
        ),
    )
