"""Tests for graph engine adapter: Webster client, KGraph client, query builder."""

from __future__ import annotations

import json

import httpx
import pytest

from ragrouter.adapters.graph_adapter.kgraph_client import (
    GraphEntity,
    GraphPath,
    GraphSchema,
    KGraphClient,
)
from ragrouter.adapters.graph_adapter.query_builder import (
    GraphIntent,
    build_cypher_from_template,
    extract_graph_intent,
)
from ragrouter.adapters.graph_adapter.webster_client import (
    DiffRecord,
    WebsterClient,
    WebsterQueryResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, json_data: dict) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", "http://test"),
    )


def _mock_llm_response(content: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={"choices": [{"message": {"content": content}}], "usage": {}},
        request=httpx.Request("POST", "http://test"),
    )


# ---------------------------------------------------------------------------
# WebsterClient tests
# ---------------------------------------------------------------------------


class TestWebsterClient:
    @pytest.fixture
    def webster(self) -> WebsterClient:
        return WebsterClient(base_url="http://test-webster:8000")

    def test_init_defaults(self):
        client = WebsterClient()
        assert "http" in client.base_url

    def test_init_custom_url(self):
        client = WebsterClient(base_url="webster:8000")
        assert client.base_url == "http://webster:8000"

    def test_validate_read_only_allows_match(self):
        WebsterClient.validate_read_only("MATCH (n) RETURN n LIMIT 10")

    def test_validate_read_only_rejects_create(self):
        with pytest.raises(ValueError, match="Write operations not allowed"):
            WebsterClient.validate_read_only("CREATE (n:Service {name: 'test'})")

    def test_validate_read_only_rejects_delete(self):
        with pytest.raises(ValueError):
            WebsterClient.validate_read_only("MATCH (n) DELETE n")

    def test_validate_read_only_rejects_set(self):
        with pytest.raises(ValueError):
            WebsterClient.validate_read_only("MATCH (n) SET n.name = 'x'")

    @pytest.mark.asyncio
    async def test_query_cypher_success(self, webster, monkeypatch):
        mock_data = {
            "status": "ready",
            "results": [{"name": "ServiceA"}, {"name": "ServiceB"}],
            "total": 2,
        }

        async def mock_post(*args, **kwargs):
            return _mock_response(200, mock_data)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        result = await webster.query_cypher(
            cypher="MATCH (n:Service) RETURN n.name LIMIT 10",
            org_id="org-1",
        )
        assert result.status == "ready"
        assert len(result.results) == 2
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_query_cypher_rejects_write(self, webster):
        with pytest.raises(ValueError, match="Write operations not allowed"):
            await webster.query_cypher(
                cypher="CREATE (n:Service {name: 'bad'})",
                org_id="org-1",
            )

    @pytest.mark.asyncio
    async def test_query_cypher_http_error(self, webster, monkeypatch):
        async def mock_post(*args, **kwargs):
            return _mock_response(500, {"error": "internal"})

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        result = await webster.query_cypher(
            cypher="MATCH (n) RETURN n LIMIT 5",
            org_id="org-1",
        )
        assert result.status == "failed"
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_query_cypher_timeout(self, webster, monkeypatch):
        async def mock_post(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        result = await webster.query_cypher(
            cypher="MATCH (n) RETURN n LIMIT 5",
            org_id="org-1",
        )
        assert result.status == "failed"
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_query_cypher_pit_ready(self, webster, monkeypatch):
        mock_data = {
            "status": "ready",
            "results": [{"name": "ServiceA"}],
            "total": 1,
        }

        async def mock_post(*args, **kwargs):
            return _mock_response(200, mock_data)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        result = await webster.query_cypher_pit(
            cypher="MATCH (n) RETURN n LIMIT 5",
            org_id="org-1",
            at_timestamp="2024-01-15T10:00:00Z",
        )
        assert result.status == "ready"
        assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_query_cypher_pit_provisioning(self, webster, monkeypatch):
        async def mock_post(*args, **kwargs):
            return _mock_response(202, {"status": "provisioning"})

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        result = await webster.query_cypher_pit(
            cypher="MATCH (n) RETURN n LIMIT 5",
            org_id="org-1",
            at_timestamp="2024-01-15T10:00:00Z",
        )
        assert result.status == "provisioning"
        assert result.results == []

    @pytest.mark.asyncio
    async def test_get_diff(self, webster, monkeypatch):
        mock_data = {
            "records": [
                {
                    "entity_id": "svc-1",
                    "name": "WebApp",
                    "entity_type": "Service",
                    "change_type": "updated",
                    "changed_at": "2024-01-15T10:00:00Z",
                    "old_values": {"status": "active"},
                    "new_values": {"status": "degraded"},
                },
            ],
        }

        async def mock_get(*args, **kwargs):
            return _mock_response(200, mock_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        records = await webster.get_diff(
            org_id="org-1",
            from_ts="2024-01-15T00:00:00Z",
            to_ts="2024-01-15T12:00:00Z",
        )
        assert len(records) == 1
        assert records[0].entity_id == "svc-1"
        assert records[0].change_type == "updated"

    @pytest.mark.asyncio
    async def test_health_ok(self, webster, monkeypatch):
        async def mock_get(*args, **kwargs):
            return _mock_response(200, {"status": "ok"})

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        result = await webster.health()
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_unreachable(self, webster, monkeypatch):
        async def mock_get(*args, **kwargs):
            raise ConnectionError("refused")

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        result = await webster.health()
        assert result["status"] == "unreachable"


# ---------------------------------------------------------------------------
# KGraphClient tests
# ---------------------------------------------------------------------------


class TestKGraphClient:
    @pytest.fixture
    def kgraph(self) -> KGraphClient:
        return KGraphClient(base_url="http://test-kgraph:8000")

    @pytest.mark.asyncio
    async def test_get_schema(self, kgraph, monkeypatch):
        mock_data = {
            "node_types": ["Service", "Database", "Host"],
            "edge_types": ["DEPENDS_ON", "RUNS_ON", "HOSTS"],
        }

        async def mock_get(*args, **kwargs):
            return _mock_response(200, mock_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        schema = await kgraph.get_schema(org_id="org-1")
        assert "Service" in schema.node_types
        assert "DEPENDS_ON" in schema.edge_types

    @pytest.mark.asyncio
    async def test_search_entities(self, kgraph, monkeypatch):
        mock_data = {
            "entities": [
                {"entity_id": "svc-1", "entity_type": "Service", "name": "WebApp", "properties": {}},
                {"entity_id": "svc-2", "entity_type": "Service", "name": "API", "properties": {}},
            ],
        }

        async def mock_get(*args, **kwargs):
            return _mock_response(200, mock_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        entities = await kgraph.search_entities(
            org_id="org-1",
            entity_type="Service",
        )
        assert len(entities) == 2
        assert entities[0].name == "WebApp"
        assert entities[0].entity_id == "svc-1"

    @pytest.mark.asyncio
    async def test_find_paths(self, kgraph, monkeypatch):
        mock_data = {
            "paths": [
                {
                    "nodes": [
                        {"name": "WebApp", "entity_type": "Service"},
                        {"name": "PostgreSQL", "entity_type": "Database"},
                    ],
                    "edges": [{"type": "DEPENDS_ON"}],
                    "depth": 1,
                },
            ],
        }

        async def mock_get(*args, **kwargs):
            return _mock_response(200, mock_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        paths = await kgraph.find_paths(
            org_id="org-1",
            source_entity_type="Service",
            source_entity_id="svc-1",
            dest_entity_type="Database",
        )
        assert len(paths) == 1
        assert len(paths[0].nodes) == 2
        assert paths[0].depth == 1

    @pytest.mark.asyncio
    async def test_find_paths_empty(self, kgraph, monkeypatch):
        async def mock_get(*args, **kwargs):
            return _mock_response(200, {"paths": []})

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        paths = await kgraph.find_paths(
            org_id="org-1",
            source_entity_type="Service",
            source_entity_id="svc-1",
        )
        assert len(paths) == 0

    @pytest.mark.asyncio
    async def test_find_paths_error(self, kgraph, monkeypatch):
        async def mock_get(*args, **kwargs):
            raise ConnectionError("refused")

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        paths = await kgraph.find_paths(
            org_id="org-1",
            source_entity_type="Service",
            source_entity_id="svc-1",
        )
        assert paths == []

    @pytest.mark.asyncio
    async def test_execute_cypher(self, kgraph, monkeypatch):
        mock_data = {"results": [{"n.name": "WebApp"}, {"n.name": "API"}]}

        async def mock_post(*args, **kwargs):
            return _mock_response(200, mock_data)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        results = await kgraph.execute_cypher(
            org_id="org-1",
            cypher="MATCH (n:Service) RETURN n.name LIMIT 10",
        )
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_health(self, kgraph, monkeypatch):
        async def mock_get(*args, **kwargs):
            return _mock_response(200, {"status": "ok"})

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

        result = await kgraph.health()
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# Query builder tests
# ---------------------------------------------------------------------------


class TestQueryBuilder:
    def test_build_cypher_dependencies(self):
        cypher, params = build_cypher_from_template(
            query_type="dependencies",
            source_type="Service",
            source_name="WebApp",
        )
        assert "DEPENDS_ON" in cypher
        assert params["source_name"] == "WebApp"
        assert "LIMIT" in cypher

    def test_build_cypher_impacts(self):
        cypher, params = build_cypher_from_template(
            query_type="impacts",
            source_type="Database",
            source_name="PostgreSQL",
        )
        assert "DEPENDS_ON" in cypher
        assert params["source_name"] == "PostgreSQL"

    def test_build_cypher_path_between(self):
        cypher, params = build_cypher_from_template(
            query_type="path_between",
            source_type="Service",
            source_name="WebApp",
            target_type="Database",
            target_name="PostgreSQL",
        )
        assert "shortestPath" in cypher
        assert params["source_name"] == "WebApp"
        assert params["target_name"] == "PostgreSQL"

    def test_build_cypher_unknown_type_fallback(self):
        cypher, params = build_cypher_from_template(
            query_type="nonexistent",
            source_type="Service",
            source_name="Test",
        )
        # Falls back to connections template
        assert "MATCH" in cypher
        assert params["source_name"] == "Test"

    @pytest.mark.asyncio
    async def test_extract_graph_intent_success(self, monkeypatch):
        intent_json = json.dumps({
            "entities": [{"type": "Service", "name": "WebApp"}],
            "relationship_type": "depends_on",
            "query_type": "dependencies",
            "use_cypher": False,
            "cypher_query": "",
            "source_entity": {"type": "Service", "name": "WebApp"},
            "target_entity": None,
        })

        async def mock_post(*args, **kwargs):
            return _mock_llm_response(intent_json)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        intent = await extract_graph_intent(
            query="What does WebApp depend on?",
            schema_types=["Service", "Database", "Host"],
            litellm_api_base="http://test:4000",
        )
        assert intent.query_type == "dependencies"
        assert intent.source_entity["name"] == "WebApp"
        assert intent.error == ""

    @pytest.mark.asyncio
    async def test_extract_graph_intent_failure(self, monkeypatch):
        async def mock_post(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        intent = await extract_graph_intent(
            query="test query",
            litellm_api_base="http://test:4000",
        )
        assert intent.error != ""

    @pytest.mark.asyncio
    async def test_extract_graph_intent_with_cypher(self, monkeypatch):
        intent_json = json.dumps({
            "entities": [{"type": "Service", "name": "WebApp"}],
            "relationship_type": "depends_on",
            "query_type": "cypher",
            "use_cypher": True,
            "cypher_query": "MATCH (s:Service {name: 'WebApp'})-[:DEPENDS_ON]->(d) RETURN d LIMIT 10",
            "source_entity": {"type": "Service", "name": "WebApp"},
            "target_entity": None,
        })

        async def mock_post(*args, **kwargs):
            return _mock_llm_response(intent_json)

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        intent = await extract_graph_intent(
            query="Show me WebApp dependencies as a Cypher query",
            litellm_api_base="http://test:4000",
        )
        assert intent.use_cypher is True
        assert "MATCH" in intent.cypher_query
