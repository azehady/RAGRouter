"""Hybrid engine adapter: Haystack + Qdrant (dense+sparse) + FlashRank reranking.

Wraps the hybrid RAG pipeline behind the standard engine contract.

Enhanced pipeline (when HYBRID_QUERY_EXPANSION / HYBRID_LLM_RERANKING are enabled):
  Query -> Query Expansion (LLM, ~300ms)
        -> Multi-variant Retrieval (parallel, ~200ms)
        -> Cross-query RRF Fusion (~1ms)
        -> LLM Reranking (~300ms)
        -> LLM Answer Generation

Fallback pipeline (when expansion/reranking disabled or fail):
  Query -> Dense Retrieval -> FlashRank Reranking -> LLM Answer
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from ragrouter.adapters.hybrid_adapter.fusion import reciprocal_rank_fusion
from ragrouter.adapters.hybrid_adapter.query_expansion import expand_query
from ragrouter.adapters.hybrid_adapter.reranker import rerank_results
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


class HybridEngineSettings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_https: bool = False

    collection_name: str = "ciroos-docs"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Sparse (BM25-like)
    sparse_model: str = "prithivida/Splade_PP_en_v1"

    # Reranking
    rerank_model: str = "ms-marco-MiniLM-L-12-v2"
    rerank_top_k: int = 5

    # Retrieval
    dense_top_k: int = 20
    sparse_top_k: int = 20

    # LLM for answer generation
    litellm_api_base: str = "http://localhost:4000"
    litellm_model: str = "gpt-4o-mini"

    model_config = {"env_prefix": "HYBRID_"}


settings = HybridEngineSettings()


# ---------------------------------------------------------------------------
# Pipeline components (lazy-initialised)
# ---------------------------------------------------------------------------

_pipeline = None


def _build_pipeline():
    """Build the Haystack hybrid retrieval + reranking pipeline."""
    from haystack import Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.joiners.document_joiner import DocumentJoiner
    from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

    # Qdrant document store
    document_store = QdrantDocumentStore(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=settings.qdrant_https,
        index=settings.collection_name,
        embedding_dim=384,  # all-MiniLM-L6-v2
        recreate_index=False,
        wait_result_from_api=True,
    )

    # Dense retriever
    dense_retriever = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.dense_top_k,
    )

    # Document joiner (RRF fusion)
    joiner = DocumentJoiner(
        join_mode="reciprocal_rank_fusion",
        top_k=settings.rerank_top_k * 3,  # send more to reranker
    )

    # FlashRank reranker
    try:
        from haystack_integrations.components.rankers.flashrank import FlashRankRanker

        ranker = FlashRankRanker(
            model=settings.rerank_model,
            top_k=settings.rerank_top_k,
        )
    except ImportError:
        from flashrank import Ranker

        ranker = None
        logger.warning("FlashRank Haystack integration not found, using standalone")

    # LLM for answer generation
    generator = OpenAIGenerator(
        api_base_url=settings.litellm_api_base,
        model=settings.litellm_model,
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )

    # Prompt template
    prompt_builder = PromptBuilder(
        template="""Answer the question based on the provided context documents.
Always cite specific documents using [Source: filename] format.
If the context doesn't contain enough information, say so clearly.

Context:
{% for doc in documents %}
[{{ doc.meta.get('source_file', 'unknown') }}] {{ doc.content }}
---
{% endfor %}

Question: {{ query }}

Answer:"""
    )

    # Build pipeline
    pipe = Pipeline()
    pipe.add_component("dense_retriever", dense_retriever)
    pipe.add_component("joiner", joiner)
    if ranker and hasattr(ranker, "run"):
        pipe.add_component("ranker", ranker)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("generator", generator)

    # Connect
    pipe.connect("dense_retriever.documents", "joiner.documents")
    if ranker and hasattr(ranker, "run"):
        pipe.connect("joiner.documents", "ranker.documents")
        pipe.connect("ranker.documents", "prompt_builder.documents")
    else:
        pipe.connect("joiner.documents", "prompt_builder.documents")
    pipe.connect("prompt_builder", "generator")

    return pipe, document_store


# ---------------------------------------------------------------------------
# Standalone hybrid search (works without full Haystack pipeline)
# ---------------------------------------------------------------------------


async def _standalone_hybrid_search(request: EngineRequest) -> EngineResponse:
    """Hybrid search using qdrant-client directly + FlashRank standalone.

    This is the fallback when Haystack integration isn't fully set up.
    Uses Qdrant's native prefetch + RRF fusion.
    """
    from qdrant_client import AsyncQdrantClient, models

    start = time.monotonic()

    client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=settings.qdrant_https,
    )

    # Generate query embedding
    try:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(settings.embedding_model)
        query_vectors = list(embedder.embed([request.query]))
        query_vector = query_vectors[0].tolist()
    except ImportError:
        # Fallback: use OpenAI embeddings via the existing vectorstore
        import httpx

        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{settings.litellm_api_base}/embeddings",
                json={"model": "text-embedding-3-small", "input": request.query},
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}"},
            )
            data = resp.json()
            query_vector = data["data"][0]["embedding"]

    # Search Qdrant
    results = await client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=settings.dense_top_k,
        with_payload=True,
        score_threshold=0.3,
    )

    retrieval_ms = (time.monotonic() - start) * 1000
    docs_retrieved = len(results.points)

    # Extract documents for reranking
    candidates = []
    for pt in results.points:
        candidates.append({
            "id": str(pt.id),
            "text": pt.payload.get("text", ""),
            "score": pt.score,
            "source_file": pt.payload.get("source_file", "unknown"),
            "section": pt.payload.get("section", ""),
            "title": pt.payload.get("title", ""),
        })

    # FlashRank reranking
    reranking_applied = False
    try:
        from flashrank import Ranker, RerankRequest

        ranker = Ranker(model_name=settings.rerank_model)
        passages = [{"id": c["id"], "text": c["text"]} for c in candidates]
        rerank_req = RerankRequest(query=request.query, passages=passages)
        reranked = ranker.rerank(rerank_req)
        reranking_applied = True

        # Rebuild candidates with reranked order
        id_to_candidate = {c["id"]: c for c in candidates}
        candidates = []
        for item in reranked[: settings.rerank_top_k]:
            cand = id_to_candidate.get(item["id"], {})
            cand["score"] = item["score"]
            candidates.append(cand)
    except ImportError:
        logger.info("FlashRank not available, using raw search scores")
        candidates = candidates[: settings.rerank_top_k]

    rerank_ms = (time.monotonic() - start) * 1000 - retrieval_ms

    # Build context for LLM
    context_parts = []
    citations = []
    for c in candidates:
        context_parts.append(f"[{c['source_file']}] {c['text']}")
        citations.append(
            Citation(
                doc_id=c["source_file"],
                section=c.get("section"),
                text=c["text"][:200],
                score=c["score"],
            )
        )

    context = "\n---\n".join(context_parts)

    # LLM answer generation
    import httpx

    answer = ""
    tokens_in = 0
    tokens_out = 0
    try:
        prompt = f"""Answer the question based on the provided context documents.
Always cite specific documents using [Source: filename] format.
If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {request.query}

Answer:"""

        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.post(
                f"{settings.litellm_api_base}/chat/completions",
                json={
                    "model": settings.litellm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": request.constraints.max_tokens,
                    "temperature": 0.3,
                },
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}"},
            )
            data = resp.json()
            answer = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
    except Exception as e:
        logger.error("llm.generation_failed", error=str(e))
        answer = "Error generating answer. Retrieved documents:\n" + "\n".join(
            f"- [{c['source_file']}] {c['text'][:100]}..." for c in candidates
        )

    total_ms = (time.monotonic() - start) * 1000

    await client.close()

    # Compute confidence from retrieval scores
    avg_score = sum(c["score"] for c in candidates) / len(candidates) if candidates else 0
    top_score = candidates[0]["score"] if candidates else 0

    return EngineResponse(
        engine="hybrid",
        answer=answer,
        citations=citations,
        trace=Trace(
            retrieval_method="dense+flashrank_rerank" if reranking_applied else "dense",
            docs_searched=settings.dense_top_k,
            docs_retrieved=docs_retrieved,
            reranking_applied=reranking_applied,
            steps=[
                f"dense_search ({retrieval_ms:.0f}ms, {docs_retrieved} docs)",
                f"flashrank_rerank ({rerank_ms:.0f}ms)" if reranking_applied else "no_reranking",
                f"llm_generation ({total_ms - retrieval_ms - rerank_ms:.0f}ms)",
            ],
        ),
        scores=Scores(
            confidence=min(1.0, top_score),
            groundedness=min(1.0, avg_score * 1.2),
            coverage=min(1.0, docs_retrieved / settings.dense_top_k),
            relevance=min(1.0, top_score * 1.1),
        ),
        usage=Usage(
            latency_ms=total_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            estimated_cost_usd=(tokens_in * 0.15 + tokens_out * 0.6) / 1_000_000,
        ),
    )


# ---------------------------------------------------------------------------
# Enhanced pipeline: expansion -> multi-query search -> RRF -> LLM rerank
# ---------------------------------------------------------------------------


async def _enhanced_hybrid_search(request: EngineRequest) -> EngineResponse:
    """Full enhanced pipeline: query expansion + multi-query retrieval + RRF + LLM reranking.

    Falls back to _standalone_hybrid_search on critical failures.
    """
    from qdrant_client import AsyncQdrantClient

    start = time.monotonic()
    pipeline_steps = []

    client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=settings.qdrant_https,
    )

    try:
        # Step 1: Query Expansion
        expanded, expansion_stats = await expand_query(
            query=request.query,
            litellm_api_base=settings.litellm_api_base,
        )
        expansion_ms = expansion_stats.latency_ms
        query_variants = expanded.all_variants
        pipeline_steps.append(
            f"query_expansion ({expansion_ms:.0f}ms, {len(query_variants)} variants, "
            f"{'ok' if expansion_stats.success else 'fallback'})"
        )

        # Step 2: Generate embeddings for all variants in parallel
        embeddings = await _embed_queries(query_variants)
        embed_ms = (time.monotonic() - start) * 1000 - expansion_ms
        pipeline_steps.append(f"embedding ({embed_ms:.0f}ms, {len(embeddings)} queries)")

        # Step 3: Multi-query retrieval in parallel
        search_tasks = [
            _search_qdrant(
                client=client,
                query_vector=emb,
                limit=settings.dense_top_k,
            )
            for emb in embeddings
        ]
        all_results = await asyncio.gather(*search_tasks)
        retrieval_ms = (time.monotonic() - start) * 1000 - expansion_ms - embed_ms

        total_docs = sum(len(r) for r in all_results)
        pipeline_steps.append(
            f"multi_retrieval ({retrieval_ms:.0f}ms, {len(all_results)} queries, {total_docs} docs)"
        )

        # Step 4: FlashRank reranking per result list (if available)
        flashrank_applied = False
        try:
            from flashrank import Ranker, RerankRequest

            ranker = Ranker(model_name=settings.rerank_model)
            reranked_lists = []
            for variant_idx, (variant_query, result_list) in enumerate(
                zip(query_variants, all_results)
            ):
                if not result_list:
                    reranked_lists.append([])
                    continue
                passages = [{"id": c["id"], "text": c["text"]} for c in result_list]
                rerank_req = RerankRequest(query=variant_query, passages=passages)
                reranked = ranker.rerank(rerank_req)
                id_to_cand = {c["id"]: c for c in result_list}
                reranked_list = []
                for item in reranked:
                    cand = id_to_cand.get(item["id"], {})
                    cand["score"] = item["score"]
                    reranked_list.append(cand)
                reranked_lists.append(reranked_list)
            all_results = reranked_lists
            flashrank_applied = True
        except ImportError:
            logger.info("FlashRank not available, skipping per-list reranking")

        flashrank_ms = (time.monotonic() - start) * 1000 - expansion_ms - embed_ms - retrieval_ms
        if flashrank_applied:
            pipeline_steps.append(f"flashrank_rerank ({flashrank_ms:.0f}ms)")

        # Step 5: Cross-query RRF Fusion
        fused_results, fusion_stats = reciprocal_rank_fusion(
            result_lists=all_results,
            top_k=15,  # Send top 15 to LLM reranker
        )
        pipeline_steps.append(
            f"rrf_fusion ({fusion_stats.latency_ms:.1f}ms, "
            f"{fusion_stats.unique_candidates} unique -> {fusion_stats.output_count} fused)"
        )

        # Step 6: LLM Reranking
        fused_dicts = [
            {
                "id": r.id,
                "text": r.text,
                "rrf_score": r.rrf_score,
                "score": r.score,
                **r.payload,
            }
            for r in fused_results
        ]
        reranked, rerank_stats = await rerank_results(
            query=request.query,
            candidates=fused_dicts,
            top_k=settings.rerank_top_k,
            litellm_api_base=settings.litellm_api_base,
        )
        pipeline_steps.append(
            f"llm_reranking ({rerank_stats.latency_ms:.0f}ms, "
            f"{'ok' if rerank_stats.success else 'fallback'})"
        )

        # Step 7: Build context and generate answer
        final_candidates = [
            {
                "id": r.id,
                "text": r.text,
                "score": r.blended_score,
                "source_file": r.payload.get("source_file", "unknown"),
                "section": r.payload.get("section", ""),
            }
            for r in reranked
        ]

        context_parts = []
        citations = []
        for c in final_candidates:
            context_parts.append(f"[{c['source_file']}] {c['text']}")
            citations.append(
                Citation(
                    doc_id=c["source_file"],
                    section=c.get("section"),
                    text=c["text"][:200],
                    score=c["score"],
                )
            )

        context = "\n---\n".join(context_parts)

        # LLM answer generation
        import httpx

        answer = ""
        tokens_in = 0
        tokens_out = 0
        try:
            prompt = (
                "Answer the question based on the provided context documents.\n"
                "Always cite specific documents using [Source: filename] format.\n"
                "If the context doesn't contain enough information, say so clearly.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {request.query}\n\n"
                "Answer:"
            )

            async with httpx.AsyncClient(timeout=30.0) as http:
                resp = await http.post(
                    f"{settings.litellm_api_base}/chat/completions",
                    json={
                        "model": settings.litellm_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": request.constraints.max_tokens,
                        "temperature": 0.3,
                    },
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}"
                    },
                )
                data = resp.json()
                answer = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens_in = usage.get("prompt_tokens", 0)
                tokens_out = usage.get("completion_tokens", 0)
        except Exception as e:
            logger.error("llm.generation_failed", error=str(e))
            answer = "Error generating answer. Retrieved documents:\n" + "\n".join(
                f"- [{c['source_file']}] {c['text'][:100]}..." for c in final_candidates
            )

        total_ms = (time.monotonic() - start) * 1000
        gen_ms = total_ms - expansion_ms - embed_ms - retrieval_ms - flashrank_ms
        pipeline_steps.append(f"llm_generation ({gen_ms:.0f}ms)")

        avg_score = (
            sum(c["score"] for c in final_candidates) / len(final_candidates)
            if final_candidates
            else 0
        )
        top_score = final_candidates[0]["score"] if final_candidates else 0

        retrieval_method_parts = ["dense"]
        if flashrank_applied:
            retrieval_method_parts.append("flashrank")
        if expansion_stats.success:
            retrieval_method_parts.append("expanded")
        if rerank_stats.success:
            retrieval_method_parts.append("llm_reranked")

        return EngineResponse(
            engine="hybrid",
            answer=answer,
            citations=citations,
            trace=Trace(
                retrieval_method="+".join(retrieval_method_parts),
                docs_searched=settings.dense_top_k * len(query_variants),
                docs_retrieved=total_docs,
                reranking_applied=flashrank_applied or rerank_stats.success,
                steps=pipeline_steps,
            ),
            scores=Scores(
                confidence=min(1.0, top_score),
                groundedness=min(1.0, avg_score * 1.2),
                coverage=min(
                    1.0,
                    fusion_stats.unique_candidates / (settings.dense_top_k * len(query_variants)),
                ),
                relevance=min(1.0, top_score * 1.1),
            ),
            usage=Usage(
                latency_ms=total_ms,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                estimated_cost_usd=(tokens_in * 0.15 + tokens_out * 0.6) / 1_000_000,
            ),
        )

    except Exception as e:
        logger.error("enhanced_pipeline.failed, falling back to standalone", error=str(e))
        return await _standalone_hybrid_search(request)
    finally:
        await client.close()


async def _embed_queries(queries: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple queries."""
    try:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(settings.embedding_model)
        return [v.tolist() for v in embedder.embed(queries)]
    except ImportError:
        import httpx

        embeddings = []
        async with httpx.AsyncClient(timeout=15.0) as http:
            for q in queries:
                resp = await http.post(
                    f"{settings.litellm_api_base}/embeddings",
                    json={"model": "text-embedding-3-small", "input": q},
                    headers={
                        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}"
                    },
                )
                data = resp.json()
                embeddings.append(data["data"][0]["embedding"])
        return embeddings


async def _search_qdrant(
    client: Any,
    query_vector: list[float],
    limit: int = 20,
) -> list[dict]:
    """Search Qdrant and return normalized candidate dicts."""
    results = await client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
        score_threshold=0.3,
    )
    candidates = []
    for pt in results.points:
        candidates.append({
            "id": str(pt.id),
            "text": pt.payload.get("text", ""),
            "score": pt.score,
            "source_file": pt.payload.get("source_file", "unknown"),
            "section": pt.payload.get("section", ""),
            "title": pt.payload.get("title", ""),
        })
    return candidates


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    try:
        _pipeline, _ = _build_pipeline()
        logger.info("hybrid_engine.haystack_pipeline_ready")
    except Exception as e:
        logger.warning("hybrid_engine.haystack_unavailable, using standalone", error=str(e))
        _pipeline = None
    yield
    logger.info("hybrid_engine.stopped")


app = FastAPI(
    title="RAGRouter Hybrid Engine",
    version="0.1.0",
    description="Hybrid RAG: dense+sparse retrieval, RRF fusion, FlashRank reranking",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "hybrid", "pipeline": "haystack" if _pipeline else "standalone"}


class SearchRequest(BaseModel):
    """Vectorstore-compatible search request."""

    query: str
    limit: int = 5
    score_threshold: float = 0.3
    with_payload: list[str] | None = None
    filter: dict | None = None


class SearchResultItem(BaseModel):
    id: str
    score: float
    payload: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResultItem]


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """Retrieval-only endpoint returning results in vectorstore-compatible format.

    This allows DocsAgent to use the hybrid engine as a drop-in replacement
    for the vectorstore, getting BM25 + dense + FlashRank reranking without
    changing its result-parsing logic.
    """
    from qdrant_client import AsyncQdrantClient

    start = time.monotonic()
    logger.info("hybrid.search", query=request.query[:100], limit=request.limit)

    client = AsyncQdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
        https=settings.qdrant_https,
    )

    try:
        # Generate query embedding via OpenAI (must match collection's 1536-dim vectors)
        import httpx as _httpx

        embedding_api_base = os.getenv(
            "HYBRID_EMBEDDING_API_BASE", "https://api.openai.com/v1"
        )
        embedding_model = os.getenv(
            "HYBRID_EMBEDDING_MODEL", "text-embedding-3-small"
        )

        async with _httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.post(
                f"{embedding_api_base}/embeddings",
                json={"model": embedding_model, "input": request.query},
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}"
                },
            )
            resp.raise_for_status()
            emb_data = resp.json()
            query_vector = emb_data["data"][0]["embedding"]

        # Search Qdrant
        results = await client.query_points(
            collection_name=settings.collection_name,
            query=query_vector,
            limit=settings.dense_top_k,
            with_payload=True,
            score_threshold=request.score_threshold,
        )

        # Extract candidates
        candidates = []
        for pt in results.points:
            candidates.append({
                "id": str(pt.id),
                "text": pt.payload.get("text", ""),
                "score": pt.score,
                "source_file": pt.payload.get("source_file", "unknown"),
                "section": pt.payload.get("section", ""),
                "title": pt.payload.get("title", ""),
                "heading_hierarchy": pt.payload.get("heading_hierarchy", []),
                "doc_type": pt.payload.get("doc_type", ""),
            })

        # FlashRank reranking
        try:
            from flashrank import Ranker, RerankRequest

            ranker = Ranker(model_name=settings.rerank_model)
            passages = [{"id": c["id"], "text": c["text"]} for c in candidates]
            rerank_req = RerankRequest(query=request.query, passages=passages)
            reranked = ranker.rerank(rerank_req)

            id_to_candidate = {c["id"]: c for c in candidates}
            candidates = []
            for item in reranked[: request.limit]:
                cand = id_to_candidate.get(item["id"], {})
                cand["score"] = item["score"]
                candidates.append(cand)
        except ImportError:
            logger.info("FlashRank not available, using raw search scores")
            candidates = candidates[: request.limit]

        # Apply doc_type filter if provided
        if request.filter:
            must_conditions = request.filter.get("must", [])
            for condition in must_conditions:
                key = condition.get("key")
                match_val = condition.get("match", {}).get("value")
                if key and match_val:
                    candidates = [c for c in candidates if c.get(key) == match_val]

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("hybrid.search_done", results=len(candidates), latency_ms=f"{elapsed_ms:.0f}")

        # Build vectorstore-compatible response
        items = []
        for c in candidates:
            items.append(
                SearchResultItem(
                    id=c["id"],
                    score=c["score"],
                    payload={
                        "text": c["text"],
                        "source_file": c["source_file"],
                        "title": c["title"],
                        "section": c["section"],
                        "heading_hierarchy": c["heading_hierarchy"],
                    },
                )
            )

        return SearchResponse(results=items)
    finally:
        await client.close()


@app.post("/query", response_model=EngineResponse)
async def query(request: EngineRequest) -> EngineResponse:
    """Execute hybrid search and return standardised response.

    Pipeline selection order:
    1. Enhanced pipeline (query expansion + multi-query + RRF + LLM reranking)
    2. Haystack pipeline (if initialized)
    3. Standalone fallback (dense + FlashRank)
    """
    logger.info("hybrid.query", query=request.query[:100], corpus=request.corpus_id)

    # Primary: enhanced pipeline with expansion + RRF + LLM reranking
    try:
        return await _enhanced_hybrid_search(request)
    except Exception as e:
        logger.error("hybrid.enhanced_failed, trying haystack", error=str(e))

    if _pipeline:
        # Haystack pipeline path
        start = time.monotonic()
        try:
            from haystack.components.embedders import SentenceTransformersTextEmbedder

            embedder = SentenceTransformersTextEmbedder(model=settings.embedding_model)
            embedder.warm_up()
            embedding_result = embedder.run(text=request.query)
            query_embedding = embedding_result["embedding"]

            result = _pipeline.run({
                "dense_retriever": {"query_embedding": query_embedding},
                "prompt_builder": {"query": request.query},
            })
            total_ms = (time.monotonic() - start) * 1000

            # Extract from Haystack result
            answer = result.get("generator", {}).get("replies", [""])[0]
            docs = result.get("ranker", result.get("joiner", {})).get("documents", [])

            citations = [
                Citation(
                    doc_id=d.meta.get("source_file", "unknown"),
                    section=d.meta.get("section"),
                    text=d.content[:200] if d.content else None,
                    score=d.score,
                )
                for d in docs[: settings.rerank_top_k]
            ]

            return EngineResponse(
                engine="hybrid",
                answer=answer,
                citations=citations,
                trace=Trace(
                    retrieval_method="haystack_dense+rrf+flashrank",
                    docs_searched=settings.dense_top_k,
                    docs_retrieved=len(docs),
                    reranking_applied=True,
                    steps=["dense_retrieval", "rrf_fusion", "flashrank_rerank", "llm_generation"],
                ),
                scores=Scores(
                    confidence=docs[0].score if docs else 0,
                    groundedness=0.8,
                    coverage=min(1.0, len(docs) / settings.rerank_top_k),
                    relevance=docs[0].score if docs else 0,
                ),
                usage=Usage(latency_ms=total_ms),
            )
        except Exception as e:
            logger.error("hybrid.haystack_failed, falling back to standalone", error=str(e))

    # Final fallback: standalone
    return await _standalone_hybrid_search(request)
