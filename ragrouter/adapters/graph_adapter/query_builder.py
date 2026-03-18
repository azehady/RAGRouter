"""Natural language to graph query translation.

Uses LLM to:
1. Extract entities and relationship type from user query
2. Decide whether to use Cypher (Webster) or path API (KGraph)
3. Generate Cypher query or select appropriate KGraph endpoint
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx
import structlog

logger = structlog.get_logger()

QUERY_BUILDER_MODEL = os.environ.get("GRAPH_QUERY_BUILDER_MODEL", "gpt-4o-mini")
QUERY_BUILDER_TIMEOUT_S = float(os.environ.get("GRAPH_QUERY_BUILDER_TIMEOUT", "8.0"))


@dataclass
class GraphIntent:
    """Parsed intent from a natural language query for graph operations."""

    entities: list[dict] = field(default_factory=list)  # [{type, name, id?}]
    relationship_type: str = ""  # depends_on, impacts, connects_to, etc.
    use_cypher: bool = False  # True = use Webster Cypher, False = use KGraph path API
    cypher_query: str = ""  # Generated Cypher (if use_cypher=True)
    source_entity: dict = field(default_factory=dict)  # {type, id/name}
    target_entity: dict = field(default_factory=dict)  # {type, id/name}
    query_type: str = "paths"  # paths | shortest | dependencies | impacts | cypher
    error: str = ""


# Common Cypher templates for typical graph questions
CYPHER_TEMPLATES = {
    "dependencies": (
        "MATCH (s:{source_type} {{name: $source_name}})-[:DEPENDS_ON*1..{depth}]->(d) "
        "RETURN s.name AS source, type(d) AS dep_type, d.name AS dependency, "
        "d.entity_id AS dep_id LIMIT {limit}"
    ),
    "impacts": (
        "MATCH (s:{source_type} {{name: $source_name}})<-[:DEPENDS_ON*1..{depth}]-(i) "
        "RETURN s.name AS source, type(i) AS impacted_type, i.name AS impacted, "
        "i.entity_id AS impacted_id LIMIT {limit}"
    ),
    "connections": (
        "MATCH (s:{source_type} {{name: $source_name}})-[r*1..{depth}]-(c) "
        "RETURN s.name AS source, [rel IN r | type(rel)] AS relationships, "
        "c.name AS connected, labels(c) AS types LIMIT {limit}"
    ),
    "path_between": (
        "MATCH path = shortestPath("
        "(a:{source_type} {{name: $source_name}})-[*..{depth}]-"
        "(b:{target_type} {{name: $target_name}}))"
        "RETURN [n IN nodes(path) | n.name] AS node_names, "
        "[r IN relationships(path) | type(r)] AS relationship_types LIMIT {limit}"
    ),
}

_INTENT_EXTRACTION_PROMPT = """\
You are a graph query intent extractor. Given a user question about infrastructure \
relationships, extract the entities and relationship type.

Available entity types from the graph schema:
{schema_types}

Respond with ONLY a JSON object (no markdown):
{{
  "entities": [{{"type": "entity_type", "name": "entity_name"}}],
  "relationship_type": "depends_on|impacts|connects_to|runs_on|hosts|monitors",
  "query_type": "dependencies|impacts|connections|path_between|cypher",
  "use_cypher": true/false,
  "cypher_query": "MATCH ... (only if use_cypher is true, must be read-only)",
  "source_entity": {{"type": "...", "name": "..."}},
  "target_entity": {{"type": "...", "name": "..."}} or null
}}

Rules:
- use_cypher=true only for complex queries that path APIs can't handle
- For "what depends on X" or "what does X depend on", use query_type="dependencies"
- For "what is impacted by X", use query_type="impacts"
- For "how is X connected to Y", use query_type="path_between"
- For general exploration, use query_type="connections"
- Cypher must be READ-ONLY (MATCH only, no CREATE/DELETE/SET)
- Include LIMIT clause in any generated Cypher

User question: {query}"""


async def extract_graph_intent(
    query: str,
    schema_types: list[str] | None = None,
    litellm_api_base: str | None = None,
    model: str | None = None,
) -> GraphIntent:
    """Extract graph query intent from natural language.

    Args:
        query: User's natural language question.
        schema_types: Available entity types from graph schema.
        litellm_api_base: LiteLLM proxy URL.
        model: LLM model to use.

    Returns:
        GraphIntent with extracted entities, relationship, and query plan.
        On failure, returns a default GraphIntent with error set.
    """
    api_base = litellm_api_base or os.environ.get(
        "GRAPH_LITELLM_API_BASE",
        os.environ.get("HYBRID_LITELLM_API_BASE", "http://localhost:4000"),
    )
    llm_model = model or QUERY_BUILDER_MODEL
    types_str = ", ".join(schema_types) if schema_types else "Service, Database, Host, Application, Cluster, Pod, Namespace, Network, Storage"

    try:
        prompt = _INTENT_EXTRACTION_PROMPT.format(
            schema_types=types_str,
            query=query,
        )

        async with httpx.AsyncClient(timeout=QUERY_BUILDER_TIMEOUT_S) as client:
            resp = await client.post(
                f"{api_base}/chat/completions",
                json={
                    "model": llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"},
                },
                headers={
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'dummy')}",
                },
            )
            resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        return GraphIntent(
            entities=parsed.get("entities", []),
            relationship_type=parsed.get("relationship_type", ""),
            use_cypher=parsed.get("use_cypher", False),
            cypher_query=parsed.get("cypher_query", ""),
            source_entity=parsed.get("source_entity", {}),
            target_entity=parsed.get("target_entity") or {},
            query_type=parsed.get("query_type", "connections"),
        )

    except Exception as e:
        logger.warning("graph_intent_extraction.failed", error=str(e))
        return GraphIntent(error=str(e))


def build_cypher_from_template(
    query_type: str,
    source_type: str,
    source_name: str,
    target_type: str = "",
    target_name: str = "",
    depth: int = 3,
    limit: int = 25,
) -> tuple[str, dict]:
    """Build a Cypher query from a template.

    Args:
        query_type: One of: dependencies, impacts, connections, path_between.
        source_type: Entity type of the source node.
        source_name: Name of the source node.
        target_type: Entity type of the target node (for path_between).
        target_name: Name of the target node (for path_between).
        depth: Max traversal depth.
        limit: Max results.

    Returns:
        Tuple of (cypher_string, parameters_dict).
    """
    template = CYPHER_TEMPLATES.get(query_type)
    if not template:
        template = CYPHER_TEMPLATES["connections"]

    cypher = template.format(
        source_type=source_type or "Entity",
        target_type=target_type or "Entity",
        depth=depth,
        limit=limit,
        source_name="{source_name}",  # Keep as placeholder for parameter
        target_name="{target_name}",
    )

    # Replace placeholder back — Cypher uses $param syntax
    params = {"source_name": source_name}
    if target_name:
        params["target_name"] = target_name

    return cypher, params
