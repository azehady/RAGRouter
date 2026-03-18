"""Integration tests for the enhanced hybrid pipeline.

Tests the full flow: query expansion → multi-query search → RRF fusion → LLM reranking.
All external calls (LLM, Qdrant) are mocked at the httpx boundary.
"""

from __future__ import annotations

import json

import httpx
import pytest

from ragrouter.adapters.hybrid_adapter.fusion import reciprocal_rank_fusion
from ragrouter.adapters.hybrid_adapter.query_expansion import expand_query
from ragrouter.adapters.hybrid_adapter.reranker import rerank_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_response(content: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 80},
        },
        request=httpx.Request("POST", "http://test"),
    )


def _sample_search_results(prefix: str, n: int = 5) -> list[dict]:
    """Generate mock search results as returned from Qdrant."""
    return [
        {
            "id": f"{prefix}-{i}",
            "text": f"Document {prefix}-{i} about topic {i}",
            "score": 0.95 - i * 0.05,
            "source_file": f"docs/{prefix}-{i}.md",
            "section": f"Section {i}",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """Test the complete pipeline: expansion → search → fusion → reranking."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_expansion_and_reranking(self, monkeypatch):
        """Happy path: expansion produces 4 variants, fusion merges, LLM reranks."""
        call_count = {"n": 0}

        expansion_json = json.dumps({
            "lexical": "kubernetes pod crash restart CrashLoopBackOff",
            "semantic": "Why does my Kubernetes pod keep crashing and restarting?",
            "hypothetical": "When a pod enters CrashLoopBackOff state, the kubelet...",
        })

        rerank_json = json.dumps([
            {"id": "original-0", "score": 9, "reason": "directly answers"},
            {"id": "lexical-0", "score": 8, "reason": "relevant keywords"},
            {"id": "semantic-0", "score": 7, "reason": "good match"},
            {"id": "original-1", "score": 5, "reason": "partial match"},
            {"id": "hypothetical-0", "score": 4, "reason": "tangential"},
        ])

        async def mock_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _mock_llm_response(expansion_json)
            return _mock_llm_response(rerank_json)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        # Step 1: Expand
        expanded, exp_stats = await expand_query(
            query="pod crashloopbackoff",
            litellm_api_base="http://test:4000",
        )
        assert exp_stats.success is True
        assert len(expanded.all_variants) == 4

        # Step 2: Simulate parallel search (4 variants × Qdrant)
        search_results = [
            _sample_search_results("original", n=5),
            _sample_search_results("lexical", n=4),
            _sample_search_results("semantic", n=3),
            _sample_search_results("hypothetical", n=3),
        ]

        # Step 3: Fuse
        fused, fusion_stats = reciprocal_rank_fusion(
            result_lists=search_results,
            top_k=15,
        )
        assert fusion_stats.input_lists == 4
        assert fusion_stats.unique_candidates == 15  # 5+4+3+3 all unique
        assert len(fused) == 15

        # Step 4: Rerank
        fused_dicts = [
            {"id": r.id, "text": r.text, "rrf_score": r.rrf_score, **r.payload}
            for r in fused
        ]
        reranked, rerank_stats = await rerank_results(
            query="pod crashloopbackoff",
            candidates=fused_dicts,
            top_k=5,
            blend_alpha=0.3,  # Favor LLM scores
            litellm_api_base="http://test:4000",
        )
        assert rerank_stats.success is True
        assert len(reranked) == 5
        # original-0 should be top (highest LLM score)
        assert reranked[0].id == "original-0"

    @pytest.mark.asyncio
    async def test_pipeline_expansion_disabled(self, monkeypatch):
        """When expansion is disabled, pipeline uses original query only."""
        monkeypatch.setattr(
            "ragrouter.adapters.hybrid_adapter.query_expansion.QUERY_EXPANSION_ENABLED",
            False,
        )

        expanded, stats = await expand_query(query="test query")
        assert stats.enabled is False
        assert len(expanded.all_variants) == 1
        assert expanded.all_variants[0] == "test query"

        # Single query → single result list → fusion is a passthrough
        results = [_sample_search_results("single", n=5)]
        fused, _ = reciprocal_rank_fusion(result_lists=results, top_k=5)
        assert len(fused) == 5
        assert fused[0].id == "single-0"

    @pytest.mark.asyncio
    async def test_pipeline_expansion_fails_gracefully(self, monkeypatch):
        """When expansion LLM call fails, pipeline continues with original query."""
        async def mock_post(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        expanded, stats = await expand_query(
            query="test query",
            litellm_api_base="http://test:4000",
        )
        assert stats.success is False
        assert len(expanded.all_variants) == 1

        # Pipeline continues with single variant
        results = [_sample_search_results("fallback", n=5)]
        fused, _ = reciprocal_rank_fusion(result_lists=results, top_k=5)
        assert len(fused) == 5

    @pytest.mark.asyncio
    async def test_pipeline_reranking_fails_gracefully(self, monkeypatch):
        """When reranking fails, pipeline returns RRF-fused results."""
        # Expansion succeeds
        expansion_json = json.dumps({
            "lexical": "keyword version",
            "semantic": "rephrased version",
            "hypothetical": "ideal doc",
        })

        call_count = {"n": 0}

        async def mock_post(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _mock_llm_response(expansion_json)
            # Reranking call fails
            raise httpx.TimeoutException("reranking timed out")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        expanded, _ = await expand_query(
            query="test",
            litellm_api_base="http://test:4000",
        )
        assert expanded.expanded is True

        # Search + fuse
        results = [_sample_search_results(f"v{i}", n=3) for i in range(4)]
        fused, _ = reciprocal_rank_fusion(result_lists=results, top_k=10)

        # Reranking fails → passthrough
        fused_dicts = [{"id": r.id, "text": r.text, "rrf_score": r.rrf_score} for r in fused]
        reranked, stats = await rerank_results(
            query="test",
            candidates=fused_dicts,
            top_k=5,
            litellm_api_base="http://test:4000",
        )
        assert stats.success is False
        assert len(reranked) == 5
        # Order preserved from RRF
        assert reranked[0].rrf_score >= reranked[1].rrf_score

    @pytest.mark.asyncio
    async def test_pipeline_everything_fails(self, monkeypatch):
        """When all LLM calls fail, pipeline still returns results from search."""
        async def mock_post(*args, **kwargs):
            raise ConnectionError("everything is down")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        # Expansion fails → original only
        expanded, _ = await expand_query(
            query="help",
            litellm_api_base="http://test:4000",
        )
        assert len(expanded.all_variants) == 1

        # Search returns results (mocked at this level)
        results = [_sample_search_results("raw", n=5)]
        fused, _ = reciprocal_rank_fusion(result_lists=results, top_k=5)

        # Reranking fails → passthrough
        fused_dicts = [{"id": r.id, "text": r.text, "rrf_score": r.rrf_score} for r in fused]
        reranked, _ = await rerank_results(
            query="help",
            candidates=fused_dicts,
            top_k=5,
            litellm_api_base="http://test:4000",
        )
        assert len(reranked) == 5
        assert reranked[0].id == "raw-0"

    @pytest.mark.asyncio
    async def test_fusion_deduplication_across_variants(self):
        """Documents appearing in multiple variant searches get boosted."""
        # Same doc "shared-0" appears in all 3 lists
        list_a = [{"id": "shared-0", "text": "shared", "score": 0.9}, {"id": "a-1", "text": "a", "score": 0.8}]
        list_b = [{"id": "shared-0", "text": "shared", "score": 0.85}, {"id": "b-1", "text": "b", "score": 0.7}]
        list_c = [{"id": "shared-0", "text": "shared", "score": 0.88}, {"id": "c-1", "text": "c", "score": 0.6}]

        fused, stats = reciprocal_rank_fusion([list_a, list_b, list_c], top_k=5)

        assert fused[0].id == "shared-0"
        assert fused[0].appearances == 3
        assert stats.unique_candidates == 4  # shared + a-1 + b-1 + c-1
