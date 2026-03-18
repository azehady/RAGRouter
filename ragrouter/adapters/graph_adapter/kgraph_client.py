"""KGraph API client for knowledge graph entity/path queries.

KGraph provides:
- GET /api/v1/schema — Graph schema (entity types, edge labels)
- GET /api/v1/entities — Entity search by type/name
- GET /api/v1/connected_paths — Path finding between entities
- GET /api/v1/shortest_connected_paths — Optimized shortest path
- POST /api/v1/query — Execute Cypher queries with version management

Multi-tenancy: x-organization-id header required on all requests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import httpx
import structlog

logger = structlog.get_logger()

KGRAPH_SERVICE_URL = os.environ.get(
    "KGRAPH_SERVICE_URL", "http://kgraph.ciroos.svc.cluster.local:8000"
)
KGRAPH_TIMEOUT_S = float(os.environ.get("KGRAPH_TIMEOUT", "15.0"))


@dataclass
class GraphEntity:
    """An entity (node) from KGraph."""

    entity_id: str = ""
    entity_type: str = ""
    name: str = ""
    properties: dict = field(default_factory=dict)
    source: str = ""


@dataclass
class GraphPath:
    """A path between entities from KGraph."""

    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    depth: int = 0


@dataclass
class GraphSchema:
    """Graph schema describing available entity and edge types."""

    node_types: list[str] = field(default_factory=list)
    edge_types: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class KGraphClient:
    """Async client for KGraph knowledge graph service."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        raw = base_url or KGRAPH_SERVICE_URL
        if not raw.startswith(("http://", "https://")):
            raw = f"http://{raw}"
        self.base_url = raw.rstrip("/")
        self.timeout = timeout or KGRAPH_TIMEOUT_S

    def _headers(self, org_id: str) -> dict[str, str]:
        return {
            "x-organization-id": org_id,
            "Content-Type": "application/json",
        }

    async def get_schema(
        self,
        org_id: str,
        source: str | None = None,
    ) -> GraphSchema:
        """Get graph schema (entity types and edge labels).

        Args:
            org_id: Organization ID.
            source: Optional filter by source system (e.g., "SERVICE_NOW").
        """
        params = {}
        if source:
            params["source"] = source

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/api/v1/schema",
                    params=params,
                    headers=self._headers(org_id),
                )
                resp.raise_for_status()
                data = resp.json()

            return GraphSchema(
                node_types=data.get("node_types", data.get("nodes", [])),
                edge_types=data.get("edge_types", data.get("edges", [])),
                raw=data,
            )

        except Exception as e:
            logger.error("kgraph.schema_error", error=str(e))
            return GraphSchema()

    async def search_entities(
        self,
        org_id: str,
        entity_type: str | None = None,
        entity_name: str | None = None,
        source: str = "SERVICE_NOW",
        limit: int = 20,
    ) -> list[GraphEntity]:
        """Search for entities by type and/or name.

        Args:
            org_id: Organization ID.
            entity_type: Filter by entity type (e.g., "Service", "Database").
            entity_name: Filter by entity name (partial match).
            source: Source system (default: SERVICE_NOW).
            limit: Max results.
        """
        params: dict = {"source": source, "limit": limit}
        if entity_type:
            params["entity_type"] = entity_type
        if entity_name:
            params["entity_name"] = entity_name

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/api/v1/entities",
                    params=params,
                    headers=self._headers(org_id),
                )
                resp.raise_for_status()
                data = resp.json()

            return [
                GraphEntity(
                    entity_id=e.get("entity_id", e.get("id", "")),
                    entity_type=e.get("entity_type", e.get("type", "")),
                    name=e.get("name", ""),
                    properties=e.get("properties", {}),
                    source=e.get("source", source),
                )
                for e in data.get("entities", data.get("results", []))
            ]

        except Exception as e:
            logger.error("kgraph.search_entities_error", error=str(e))
            return []

    async def find_paths(
        self,
        org_id: str,
        source_entity_type: str,
        source_entity_id: str,
        dest_entity_type: str | None = None,
        dest_entity_id: str | None = None,
        source: str = "SERVICE_NOW",
        depth: int = 3,
        max_paths: int = 10,
    ) -> list[GraphPath]:
        """Find paths between entities.

        Args:
            org_id: Organization ID.
            source_entity_type: Source entity type.
            source_entity_id: Source entity ID.
            dest_entity_type: Optional destination entity type.
            dest_entity_id: Optional destination entity ID.
            source: Source system.
            depth: Max traversal depth.
            max_paths: Max paths to return.
        """
        params: dict = {
            "source_entity_type": source_entity_type,
            "source_entity_id": source_entity_id,
            "source": source,
            "depth": depth,
            "max_paths": max_paths,
        }
        if dest_entity_type:
            params["dest_entity_type"] = dest_entity_type
        if dest_entity_id:
            params["dest_entity_id"] = dest_entity_id

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/api/v1/connected_paths",
                    params=params,
                    headers=self._headers(org_id),
                )
                resp.raise_for_status()
                data = resp.json()

            return [
                GraphPath(
                    nodes=p.get("nodes", []),
                    edges=p.get("edges", p.get("relationships", [])),
                    depth=p.get("depth", len(p.get("nodes", [])) - 1),
                )
                for p in data.get("paths", data.get("results", []))
            ]

        except Exception as e:
            logger.error("kgraph.find_paths_error", error=str(e))
            return []

    async def find_shortest_paths(
        self,
        org_id: str,
        source_entity_type: str,
        source_entity_id: str,
        dest_entity_type: str,
        source: str = "SERVICE_NOW",
    ) -> list[GraphPath]:
        """Find shortest paths using KGraph's cached reachability data.

        Args:
            org_id: Organization ID.
            source_entity_type: Source entity type.
            source_entity_id: Source entity ID.
            dest_entity_type: Destination entity type.
            source: Source system.
        """
        params: dict = {
            "source_entity_type": source_entity_type,
            "source_entity_id": source_entity_id,
            "dest_entity_type": dest_entity_type,
            "source": source,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/api/v1/shortest_connected_paths",
                    params=params,
                    headers=self._headers(org_id),
                )
                resp.raise_for_status()
                data = resp.json()

            return [
                GraphPath(
                    nodes=p.get("nodes", []),
                    edges=p.get("edges", p.get("relationships", [])),
                    depth=p.get("depth", len(p.get("nodes", [])) - 1),
                )
                for p in data.get("paths", data.get("results", []))
            ]

        except Exception as e:
            logger.error("kgraph.shortest_paths_error", error=str(e))
            return []

    async def execute_cypher(
        self,
        org_id: str,
        cypher: str,
        parameters: dict | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Execute a Cypher query via KGraph.

        KGraph auto-adds organization_id and version_id to parameters.

        Args:
            org_id: Organization ID.
            cypher: Cypher query string.
            parameters: Optional query parameters.
            limit: Max results.
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/v1/query",
                    headers=self._headers(org_id),
                    json={
                        "cypher": cypher,
                        "parameters": parameters or {},
                        "limit": limit,
                    },
                )
                resp.raise_for_status()
                data = resp.json()

            return data.get("results", [])

        except Exception as e:
            logger.error("kgraph.cypher_error", error=str(e))
            return []

    async def health(self) -> dict:
        """Check KGraph service health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.json()
        except Exception:
            return {"status": "unreachable"}
