"""Tests for query expansion module."""

from __future__ import annotations

import json

import httpx
import pytest

from ragrouter.adapters.hybrid_adapter.query_expansion import (
    ExpandedQuery,
    ExpansionStats,
    expand_query,
)


# ---------------------------------------------------------------------------
# ExpandedQuery dataclass tests
# ---------------------------------------------------------------------------


class TestExpandedQuery:
    def test_all_variants_only_original(self):
        eq = ExpandedQuery(original="test query")
        assert eq.all_variants == ["test query"]
        assert eq.expanded is False

    def test_all_variants_with_expansions(self):
        eq = ExpandedQuery(
            original="test query",
            lexical="test keyword synonym",
            semantic="rephrased test question",
            hypothetical="a document about testing",
        )
        assert len(eq.all_variants) == 4
        assert eq.all_variants[0] == "test query"
        assert eq.expanded is True

    def test_all_variants_deduplicates_original(self):
        eq = ExpandedQuery(
            original="test query",
            lexical="test query",  # Same as original
            semantic="different phrasing",
        )
        # lexical is same as original, should be excluded
        assert len(eq.all_variants) == 2
        assert "different phrasing" in eq.all_variants

    def test_all_variants_skips_empty(self):
        eq = ExpandedQuery(original="test", lexical="", semantic="rephrase", hypothetical="")
        assert len(eq.all_variants) == 2

    def test_expanded_false_all_empty(self):
        eq = ExpandedQuery(original="test", lexical="", semantic="", hypothetical="")
        assert eq.expanded is False


# ---------------------------------------------------------------------------
# expand_query with mocked HTTP
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


class TestExpandQuery:
    @pytest.mark.asyncio
    async def test_successful_expansion(self, monkeypatch):
        expansion_json = json.dumps({
            "lexical": "kubernetes pod crash restart loop",
            "semantic": "Why does my Kubernetes pod keep crashing and restarting?",
            "hypothetical": "When a pod enters CrashLoopBackOff, it indicates the container ...",
        })

        async def mock_post(*args, **kwargs):
            return _mock_llm_response(expansion_json)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        expanded, stats = await expand_query(
            query="pod crashloopbackoff",
            litellm_api_base="http://test:4000",
        )

        assert expanded.original == "pod crashloopbackoff"
        assert expanded.lexical == "kubernetes pod crash restart loop"
        assert expanded.semantic.startswith("Why does")
        assert expanded.hypothetical.startswith("When a pod")
        assert stats.success is True
        assert stats.variant_count == 4
        assert stats.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_expansion_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "ragrouter.adapters.hybrid_adapter.query_expansion.QUERY_EXPANSION_ENABLED",
            False,
        )

        expanded, stats = await expand_query(query="test query")

        assert expanded.original == "test query"
        assert expanded.expanded is False
        assert stats.enabled is False

    @pytest.mark.asyncio
    async def test_expansion_timeout_fallback(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        expanded, stats = await expand_query(
            query="test query",
            litellm_api_base="http://test:4000",
        )

        assert expanded.original == "test query"
        assert expanded.expanded is False
        assert stats.success is False
        assert "TimeoutException" in stats.error

    @pytest.mark.asyncio
    async def test_expansion_http_error_fallback(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return httpx.Response(
                status_code=500,
                json={"error": "internal"},
                request=httpx.Request("POST", "http://test"),
            )

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        expanded, stats = await expand_query(
            query="test query",
            litellm_api_base="http://test:4000",
        )

        assert expanded.original == "test query"
        assert expanded.expanded is False
        assert stats.success is False

    @pytest.mark.asyncio
    async def test_expansion_invalid_json_fallback(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            return _mock_llm_response("not valid json {{{")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        expanded, stats = await expand_query(
            query="test query",
            litellm_api_base="http://test:4000",
        )

        assert expanded.original == "test query"
        assert expanded.expanded is False
        assert stats.success is False

    @pytest.mark.asyncio
    async def test_expansion_partial_response(self, monkeypatch):
        """LLM returns only some variants."""
        expansion_json = json.dumps({"lexical": "keyword variation"})

        async def mock_post(*args, **kwargs):
            return _mock_llm_response(expansion_json)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        expanded, stats = await expand_query(
            query="test query",
            litellm_api_base="http://test:4000",
        )

        assert expanded.original == "test query"
        assert expanded.lexical == "keyword variation"
        assert expanded.semantic == ""
        assert expanded.hypothetical == ""
        assert stats.success is True
        assert stats.variant_count == 2  # original + lexical

    @pytest.mark.asyncio
    async def test_expansion_connection_error_fallback(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            raise ConnectionError("refused")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        expanded, stats = await expand_query(
            query="test query",
            litellm_api_base="http://test:4000",
        )

        assert expanded.original == "test query"
        assert stats.success is False
        assert "refused" in stats.error
