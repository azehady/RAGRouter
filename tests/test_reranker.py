"""Tests for LLM reranker module."""

from __future__ import annotations

import json

import httpx
import pytest

from ragrouter.adapters.hybrid_adapter.reranker import (
    RerankCandidate,
    RerankStats,
    rerank_results,
    _passthrough,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_candidates(n: int = 5) -> list[dict]:
    return [
        {
            "id": f"doc-{i}",
            "text": f"Document {i} content about topic {i}",
            "rrf_score": 1.0 - i * 0.1,
            "source_file": f"file-{i}.md",
        }
        for i in range(n)
    ]


def _mock_llm_response(content: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
        request=httpx.Request("POST", "http://test"),
    )


# ---------------------------------------------------------------------------
# Passthrough tests
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_passthrough_preserves_order(self):
        candidates = _sample_candidates(5)
        results = _passthrough(candidates, top_k=3)
        assert len(results) == 3
        assert results[0].id == "doc-0"
        assert results[2].id == "doc-2"

    def test_passthrough_preserves_payload(self):
        candidates = [{"id": "d1", "text": "t", "rrf_score": 0.5, "source_file": "x.md"}]
        results = _passthrough(candidates, top_k=5)
        assert results[0].payload["source_file"] == "x.md"

    def test_passthrough_empty(self):
        assert _passthrough([], top_k=5) == []


# ---------------------------------------------------------------------------
# rerank_results with mocked HTTP
# ---------------------------------------------------------------------------


class TestRerankResults:
    @pytest.mark.asyncio
    async def test_successful_reranking(self, monkeypatch):
        candidates = _sample_candidates(5)
        # LLM reverses the order: doc-4 is most relevant
        llm_response = json.dumps([
            {"id": "doc-4", "score": 10, "reason": "most relevant"},
            {"id": "doc-3", "score": 8, "reason": "very relevant"},
            {"id": "doc-2", "score": 6, "reason": "relevant"},
            {"id": "doc-1", "score": 3, "reason": "somewhat relevant"},
            {"id": "doc-0", "score": 1, "reason": "barely relevant"},
        ])

        async def mock_post(*args, **kwargs):
            return _mock_llm_response(llm_response)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        results, stats = await rerank_results(
            query="test query",
            candidates=candidates,
            top_k=3,
            blend_alpha=0.0,  # Pure LLM scoring
            litellm_api_base="http://test:4000",
        )

        assert stats.success is True
        assert stats.output_count == 3
        # With alpha=0.0 (pure LLM), doc-4 should be first
        assert results[0].id == "doc-4"
        assert results[0].llm_score == 1.0  # 10/10
        assert stats.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_pure_rrf_scoring(self, monkeypatch):
        """With alpha=1.0, LLM scores are ignored."""
        candidates = _sample_candidates(5)
        llm_response = json.dumps([
            {"id": "doc-4", "score": 10, "reason": "best"},
            {"id": "doc-0", "score": 1, "reason": "worst"},
        ])

        async def mock_post(*args, **kwargs):
            return _mock_llm_response(llm_response)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        results, stats = await rerank_results(
            query="test",
            candidates=candidates,
            top_k=5,
            blend_alpha=1.0,  # Pure RRF
            litellm_api_base="http://test:4000",
        )

        assert stats.success is True
        # With alpha=1.0 (pure RRF), original order preserved: doc-0 first
        assert results[0].id == "doc-0"

    @pytest.mark.asyncio
    async def test_blended_scoring(self, monkeypatch):
        """With alpha=0.5, both RRF and LLM contribute equally."""
        candidates = [
            {"id": "doc-0", "text": "t0", "rrf_score": 1.0},
            {"id": "doc-1", "text": "t1", "rrf_score": 0.5},
        ]
        # LLM says doc-1 is more relevant
        llm_response = json.dumps([
            {"id": "doc-0", "score": 2, "reason": "low"},
            {"id": "doc-1", "score": 9, "reason": "high"},
        ])

        async def mock_post(*args, **kwargs):
            return _mock_llm_response(llm_response)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        results, stats = await rerank_results(
            query="test",
            candidates=candidates,
            top_k=2,
            blend_alpha=0.5,
            litellm_api_base="http://test:4000",
        )

        assert stats.success is True
        # doc-0: 0.5 * 1.0 (normalized RRF) + 0.5 * 0.2 (LLM) = 0.6
        # doc-1: 0.5 * 0.0 (normalized RRF) + 0.5 * 0.9 (LLM) = 0.45
        # So doc-0 should still win with blended
        assert results[0].id == "doc-0"

    @pytest.mark.asyncio
    async def test_disabled_reranking(self, monkeypatch):
        monkeypatch.setattr(
            "ragrouter.adapters.hybrid_adapter.reranker.RERANKING_ENABLED",
            False,
        )

        candidates = _sample_candidates(3)
        results, stats = await rerank_results(
            query="test",
            candidates=candidates,
            top_k=2,
        )

        assert stats.enabled is False
        assert len(results) == 2
        assert results[0].id == "doc-0"  # Original order

    @pytest.mark.asyncio
    async def test_timeout_fallback(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        candidates = _sample_candidates(3)
        results, stats = await rerank_results(
            query="test",
            candidates=candidates,
            top_k=2,
            litellm_api_base="http://test:4000",
        )

        assert stats.success is False
        assert "TimeoutException" in stats.error
        # Falls back to passthrough
        assert len(results) == 2
        assert results[0].id == "doc-0"

    @pytest.mark.asyncio
    async def test_http_error_fallback(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return httpx.Response(
                status_code=500,
                json={"error": "server error"},
                request=httpx.Request("POST", "http://test"),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        candidates = _sample_candidates(3)
        results, stats = await rerank_results(
            query="test",
            candidates=candidates,
            top_k=2,
            litellm_api_base="http://test:4000",
        )

        assert stats.success is False
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_invalid_json_fallback(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return _mock_llm_response("not valid json {{{")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        candidates = _sample_candidates(3)
        results, stats = await rerank_results(
            query="test",
            candidates=candidates,
            top_k=2,
            litellm_api_base="http://test:4000",
        )

        assert stats.success is False
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_candidates(self):
        results, stats = await rerank_results(
            query="test",
            candidates=[],
            top_k=5,
        )
        assert results == []
        assert stats.input_count == 0

    @pytest.mark.asyncio
    async def test_dict_response_format(self, monkeypatch):
        """LLM wraps scores in a dict like {"results": [...]}."""
        candidates = _sample_candidates(2)
        llm_response = json.dumps({
            "results": [
                {"id": "doc-1", "score": 9, "reason": "best"},
                {"id": "doc-0", "score": 3, "reason": "ok"},
            ]
        })

        async def mock_post(*args, **kwargs):
            return _mock_llm_response(llm_response)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        results, stats = await rerank_results(
            query="test",
            candidates=candidates,
            top_k=2,
            blend_alpha=0.0,
            litellm_api_base="http://test:4000",
        )

        assert stats.success is True
        assert results[0].id == "doc-1"

    @pytest.mark.asyncio
    async def test_score_clamping(self, monkeypatch):
        """LLM returns out-of-range scores, should be clamped to 0-1."""
        candidates = [{"id": "doc-0", "text": "t", "rrf_score": 0.5}]
        llm_response = json.dumps([{"id": "doc-0", "score": 15, "reason": "over max"}])

        async def mock_post(*args, **kwargs):
            return _mock_llm_response(llm_response)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        results, stats = await rerank_results(
            query="test",
            candidates=candidates,
            top_k=1,
            blend_alpha=0.0,
            litellm_api_base="http://test:4000",
        )

        assert stats.success is True
        assert results[0].llm_score == 1.0  # Clamped to max 1.0
