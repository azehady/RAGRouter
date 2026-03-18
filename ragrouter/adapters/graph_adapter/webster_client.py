"""Webster API client for Cypher queries against Memgraph snapshots.

Webster provides:
- POST /api/v1/webster/query — Execute read-only Cypher queries
- POST /api/v1/webster/query?at=TIMESTAMP — Point-in-time queries (snapshot + replay)
- GET /api/v1/webster/diff — Graph changes between timestamps (ClickHouse audit log)

Multi-tenancy: x-organization-id header required on all requests.
Read-only enforcement: rejects CREATE/DELETE/SET/MERGE/REMOVE/DETACH/DROP/FOREACH.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import httpx
import structlog

logger = structlog.get_logger()

WEBSTER_SERVICE_URL = os.environ.get(
    "WEBSTER_SERVICE_URL", "http://webster.ciroos.svc.cluster.local:8000"
)
WEBSTER_TIMEOUT_S = float(os.environ.get("WEBSTER_TIMEOUT", "30.0"))

_WRITE_PATTERN = re.compile(
    r"\b(CREATE|DELETE|SET|DROP|MERGE|REMOVE|DETACH|FOREACH)\b",
    re.IGNORECASE,
)


@dataclass
class WebsterQueryResult:
    """Result from a Webster Cypher query."""

    status: str = "ready"  # ready | provisioning | replaying | failed
    results: list[dict] = field(default_factory=list)
    total: int = 0
    error: str = ""


@dataclass
class DiffRecord:
    """A single change record from Webster diff endpoint."""

    entity_id: str = ""
    name: str = ""
    kind: str = ""
    entity_type: str = ""
    change_type: str = ""  # created | updated | deleted
    source: str = ""
    changed_at: str = ""
    old_values: dict = field(default_factory=dict)
    new_values: dict = field(default_factory=dict)


class WebsterClient:
    """Async client for Webster graph query service."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        raw = base_url or WEBSTER_SERVICE_URL
        if not raw.startswith(("http://", "https://")):
            raw = f"http://{raw}"
        self.base_url = raw.rstrip("/")
        self.timeout = timeout or WEBSTER_TIMEOUT_S

    def _headers(self, org_id: str) -> dict[str, str]:
        return {
            "x-organization-id": org_id,
            "Content-Type": "application/json",
        }

    @staticmethod
    def validate_read_only(cypher: str) -> None:
        """Reject write operations in Cypher queries.

        Raises:
            ValueError: If the query contains write operations.
        """
        if _WRITE_PATTERN.search(cypher):
            raise ValueError(
                f"Write operations not allowed. Found forbidden keyword in: {cypher[:100]}"
            )

    async def query_cypher(
        self,
        cypher: str,
        org_id: str,
        parameters: dict | None = None,
        timeout: float | None = None,
    ) -> WebsterQueryResult:
        """Execute a read-only Cypher query against current graph state.

        Args:
            cypher: Cypher query string (must be read-only).
            org_id: Organization ID.
            parameters: Optional Cypher query parameters.
            timeout: Optional per-request timeout override.
        """
        self.validate_read_only(cypher)

        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/v1/webster/query",
                    headers=self._headers(org_id),
                    json={
                        "cypher": cypher,
                        "parameters": parameters or {},
                        "timeout": int(timeout or self.timeout),
                    },
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return WebsterQueryResult(
                        status="ready",
                        results=data.get("results", []),
                        total=data.get("total", 0),
                    )

                return WebsterQueryResult(
                    status="failed",
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

        except httpx.TimeoutException:
            logger.warning("webster.query_timeout", cypher=cypher[:80])
            return WebsterQueryResult(status="failed", error="Query timed out")
        except Exception as e:
            logger.error("webster.query_error", error=str(e))
            return WebsterQueryResult(status="failed", error=str(e))

    async def query_cypher_pit(
        self,
        cypher: str,
        org_id: str,
        at_timestamp: str,
        parameters: dict | None = None,
        timeout: float | None = None,
    ) -> WebsterQueryResult:
        """Execute a point-in-time Cypher query.

        Webster creates an ephemeral Memgraph instance from a snapshot, replays
        ClickHouse deltas to the requested timestamp, then executes the query.

        Returns status="provisioning" if the instance is still being created (HTTP 202).
        Caller should re-poll after a short delay.

        Args:
            cypher: Cypher query string (must be read-only).
            org_id: Organization ID.
            at_timestamp: ISO 8601 timestamp for point-in-time query.
            parameters: Optional Cypher query parameters.
            timeout: Optional per-request timeout override.
        """
        self.validate_read_only(cypher)

        try:
            async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/api/v1/webster/query",
                    params={"at": at_timestamp},
                    headers=self._headers(org_id),
                    json={
                        "cypher": cypher,
                        "parameters": parameters or {},
                        "timeout": int(timeout or self.timeout),
                    },
                )

                if resp.status_code == 200:
                    data = resp.json()
                    return WebsterQueryResult(
                        status=data.get("status", "ready"),
                        results=data.get("results", []),
                        total=data.get("total", 0),
                    )

                if resp.status_code == 202:
                    data = resp.json()
                    return WebsterQueryResult(
                        status=data.get("status", "provisioning"),
                        results=[],
                        total=0,
                    )

                return WebsterQueryResult(
                    status="failed",
                    error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

        except httpx.TimeoutException:
            return WebsterQueryResult(status="failed", error="PIT query timed out")
        except Exception as e:
            return WebsterQueryResult(status="failed", error=str(e))

    async def get_diff(
        self,
        org_id: str,
        from_ts: str,
        to_ts: str,
        entity_type: str | None = None,
        change_type: str | None = None,
        limit: int = 1000,
    ) -> list[DiffRecord]:
        """Get graph changes between two timestamps from ClickHouse audit log.

        Args:
            org_id: Organization ID.
            from_ts: Start timestamp (ISO 8601).
            to_ts: End timestamp (ISO 8601).
            entity_type: Optional filter by entity type.
            change_type: Optional filter (created|updated|deleted).
            limit: Max records (default 1000).
        """
        params: dict = {"from": from_ts, "to": to_ts, "limit": limit}
        if entity_type:
            params["entity_type"] = entity_type
        if change_type:
            params["change_type"] = change_type

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(
                    f"{self.base_url}/api/v1/webster/diff",
                    params=params,
                    headers=self._headers(org_id),
                )
                resp.raise_for_status()
                data = resp.json()

            return [
                DiffRecord(
                    entity_id=r.get("entity_id", ""),
                    name=r.get("name", ""),
                    kind=r.get("kind", ""),
                    entity_type=r.get("entity_type", ""),
                    change_type=r.get("change_type", ""),
                    source=r.get("source", ""),
                    changed_at=r.get("changed_at", ""),
                    old_values=r.get("old_values", {}),
                    new_values=r.get("new_values", {}),
                )
                for r in data.get("records", data.get("results", []))
            ]

        except Exception as e:
            logger.error("webster.diff_error", error=str(e))
            return []

    async def health(self) -> dict:
        """Check Webster service health."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health")
                return resp.json()
        except Exception:
            return {"status": "unreachable"}
