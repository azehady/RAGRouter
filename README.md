# RAGRouter

Meta-RAG orchestration service that analyzes queries, routes them to the best retrieval engine, and arbitrates results with multi-dimensional scoring.

## Architecture

```
                         ┌─────────────┐
                         │   Arbiter    │ :8000
                         │  (FastAPI)   │
                         └──────┬───────┘
                                │
                 ┌──────────────┼──────────────┐
                 │              │              │
          ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
          │   Hybrid    │ │  Graph   │ │  Baseline   │
          │   Engine    │ │  Engine  │ │   Engine    │
          │   :8001     │ │  :8004   │ │   :8002     │
          └──────┬──────┘ └────┬─────┘ └──────┬──────┘
                 │              │              │
          ┌──────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
          │   Qdrant    │ │ Webster  │ │ Vectorstore │
          │  (vectors)  │ │ + KGraph │ │   (HTTP)    │
          └─────────────┘ └──────────┘ └─────────────┘
```

### Pipeline

1. **Analyzer** — extracts query signals (intent, specificity, entity count, multi-hop)
2. **Planner** — selects engines based on query signals + corpus signals + engine strengths
3. **Dispatcher** — executes engines in cascade or parallel mode
4. **Scorer** — scores responses on groundedness, coverage, relevance, confidence and picks the best

### Engines

| Engine | Port | Use Case | Backend |
|--------|------|----------|---------|
| **Hybrid** | 8001 | General retrieval with high recall | Qdrant (dense + sparse) + FlashRank reranking |
| **Graph** | 8004 | Relationship queries, multi-hop | Webster (Cypher) + KGraph (path-finding) |
| **Baseline** | 8002 | Simple vector search, low latency | Vectorstore HTTP API |

All engines implement the same contract: `POST /query` accepts `EngineRequest`, returns `EngineResponse`.

## Capabilities

### Hybrid Engine
- **Query expansion** — LLM generates 3 additional variants (lexical, semantic, hypothetical) for broader recall
- **Multi-query search** — runs all variants in parallel against Qdrant
- **RRF fusion** — Reciprocal Rank Fusion (k=60) merges results across variants, boosting documents that appear in multiple lists
- **LLM reranking** — scores top candidates for relevance, blends with RRF scores
- **Fallback chain** — enhanced pipeline → Haystack pipeline → standalone dense + FlashRank

### Graph Engine
- **NL-to-graph** — LLM extracts entities and intent, selects Cypher or path API
- **Webster integration** — Cypher queries against Memgraph snapshots, point-in-time queries, diff audit
- **KGraph integration** — entity search, path finding, shortest paths
- **Read-only enforcement** — rejects CREATE/DELETE/SET/MERGE operations
- **Fallback** — Webster Cypher → KGraph path API → error response

### Arbiter
- **Intent classification** — LOOKUP, RELATIONSHIP, SYNTHESIS, EXPLORATORY, COMPARISON
- **Engine selection** — matches query signals to engine strengths (e.g., relationship queries → graph engine)
- **Cascade dispatch** — tries engines in priority order, stops when confidence threshold is met
- **Parallel dispatch** — runs top-k engines simultaneously for latency-sensitive queries
- **Multi-dimensional scoring** — weighted combination of groundedness (0.30), coverage (0.25), relevance (0.25), confidence (0.10), latency penalty (0.05), cost penalty (0.05)

## API

### Arbiter (port 8000)

```
GET  /health                        Health check
POST /ask                           Full RAG pipeline → best answer with citations
POST /route                         Debug routing decision without executing
GET  /engines                       List registered engines and status
PUT  /corpus/{corpus_id}/signals    Update corpus signals for routing
```

#### POST /ask

```json
{
  "query": "What services depend on the PostgreSQL database?",
  "corpus_id": "knowledge-articles",
  "chat_history": [],
  "constraints": {
    "max_latency_ms": 5000,
    "must_cite": true,
    "max_tokens": 2000
  },
  "metadata": {}
}
```

Response includes the answer, citations, chosen engine, routing decision, all engine scores, and total latency.

### Hybrid Engine (port 8001)

```
GET  /health    Health check
POST /query     Full hybrid retrieval with answer generation
POST /search    Retrieval-only (vectorstore-compatible format)
```

### Graph Engine (port 8004)

```
GET  /health    Health check (includes Webster + KGraph reachability)
POST /query     Graph query with NL-to-Cypher translation
```

## Configuration

### Engine Registry (`ragrouter/arbiter/config.yaml`)

```yaml
engines:
  hybrid:
    name: "Hybrid Vector Engine"
    type: hybrid
    url: "http://hybrid-engine.ciroos-rag.svc.cluster.local:8001"
    enabled: true
    strengths: [large_corpus, keyword_matching, semantic_similarity, general_purpose]
    weaknesses: [structured_navigation, multi_hop_reasoning]
    latency_profile_ms: 800
    cost_rank: 2

  graph:
    name: "Graph/LightRAG Engine"
    type: graph
    url: "http://graph-engine.ciroos-rag.svc.cluster.local:8004"
    enabled: true
    strengths: [relationship_queries, multi_hop, entity_connections]
    weaknesses: [simple_lookups, expensive_indexing]
    latency_profile_ms: 1500
    cost_rank: 3

arbiter:
  default_mode: "cascade"
  max_cascade_fallbacks: 2
  confidence_threshold: 0.6
  timeout_ms: 10000

scoring:
  weights:
    groundedness: 0.30
    coverage: 0.25
    relevance: 0.25
    confidence: 0.10
    latency_penalty: 0.05
    cost_penalty: 0.05
```

### Environment Variables

#### Hybrid Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `HYBRID_QDRANT_HOST` | `localhost` | Qdrant server host |
| `HYBRID_QDRANT_PORT` | `6333` | Qdrant server port |
| `HYBRID_QDRANT_API_KEY` | — | Qdrant API key (optional) |
| `HYBRID_QDRANT_HTTPS` | `false` | Use HTTPS for Qdrant |
| `HYBRID_COLLECTION_NAME` | `ciroos-docs` | Default Qdrant collection |
| `HYBRID_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `HYBRID_SPARSE_MODEL` | `prithivida/Splade_PP_en_v1` | Sparse embedding model |
| `HYBRID_RERANK_MODEL` | `ms-marco-MiniLM-L-12-v2` | FlashRank reranking model |
| `HYBRID_LITELLM_API_BASE` | `http://localhost:4000` | LiteLLM proxy URL |
| `HYBRID_LITELLM_MODEL` | `gpt-4o-mini` | LLM model for expansion/reranking |
| `HYBRID_DENSE_TOP_K` | `20` | Dense retrieval candidates |
| `HYBRID_SPARSE_TOP_K` | `20` | Sparse retrieval candidates |
| `HYBRID_RERANK_TOP_K` | `5` | Final reranked results |

#### Graph Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPH_WEBSTER_URL` | `http://webster.ciroos.svc.cluster.local:8000` | Webster service URL |
| `GRAPH_KGRAPH_URL` | `http://kgraph.ciroos.svc.cluster.local:8000` | KGraph service URL |
| `GRAPH_LITELLM_API_BASE` | `http://localhost:4000` | LiteLLM proxy URL |
| `GRAPH_LITELLM_MODEL` | `gpt-4o-mini` | LLM model for intent extraction |
| `GRAPH_DEFAULT_DEPTH` | `3` | Default path traversal depth |
| `GRAPH_MAX_RESULTS` | `25` | Maximum graph results |
| `GRAPH_DEFAULT_SOURCE` | `SERVICE_NOW` | Default data source filter |

#### Baseline Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `BASELINE_VECTORSTORE_URL` | `http://localhost:2526` | Vectorstore HTTP API URL |
| `BASELINE_COLLECTION_NAME` | `ciroos-docs` | Default collection |
| `BASELINE_LITELLM_API_BASE` | `http://localhost:4000` | LiteLLM proxy URL |
| `BASELINE_LITELLM_MODEL` | `gpt-4o-mini` | LLM model for answer generation |

## Local Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Install dependencies
uv sync --extra dev

# Run tests (71 tests)
uv run pytest tests/ -v

# Run arbiter locally
uv run uvicorn ragrouter.arbiter.main:app --reload --port 8000

# Run hybrid engine locally (requires Qdrant)
uv sync --extra hybrid
uv run uvicorn ragrouter.adapters.hybrid_adapter.main:app --reload --port 8001

# Run graph engine locally (requires Webster + KGraph)
uv run uvicorn ragrouter.adapters.graph_adapter.main:app --reload --port 8004
```

### Tests

```bash
# All tests
uv run pytest tests/ -v

# By component
uv run pytest tests/test_fusion.py -v              # RRF fusion
uv run pytest tests/test_query_expansion.py -v      # Query expansion
uv run pytest tests/test_reranker.py -v             # LLM reranker
uv run pytest tests/test_graph_adapter.py -v        # Graph engine (Webster, KGraph, query builder)
uv run pytest tests/test_integration_pipeline.py -v # End-to-end pipeline
```

## Deployment

### Docker

```bash
# Build images
docker build -f Dockerfile.arbiter -t ragrouter-arbiter .
docker build -f Dockerfile.hybrid -t ragrouter-hybrid .
docker build -f Dockerfile.graph -t ragrouter-graph .
```

### Kubernetes

Deploys to the `ciroos-rag` namespace:

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy services
kubectl apply -f k8s/arbiter.yaml
kubectl apply -f k8s/hybrid-engine.yaml
kubectl apply -f k8s/graph-engine.yaml
```

Services:
- `arbiter-service.ciroos-rag.svc.cluster.local:8000`
- `hybrid-engine.ciroos-rag.svc.cluster.local:8001`
- `graph-engine.ciroos-rag.svc.cluster.local:8004`

### Resource Requirements

| Service | CPU Request | CPU Limit | Memory Request | Memory Limit |
|---------|------------|-----------|----------------|--------------|
| Arbiter | 100m | 500m | 128Mi | 256Mi |
| Hybrid Engine | 200m | 1000m | 256Mi | 512Mi |
| Graph Engine | 100m | 500m | 128Mi | 256Mi |

## Client Integration

### Python Client (common-agents)

```python
from agents.apis.ragrouter import RAGRouterAPI

rag = RAGRouterAPI(org_id="tenant-123")

# Full pipeline: analyze → route → retrieve → score → answer
result = await rag.ask(
    query="How do I troubleshoot connection pool exhaustion?",
    corpus_id="knowledge-articles",
)
print(result.answer)
print(result.format_for_agent())  # Formatted with citations

# Retrieval-only (vectorstore-compatible drop-in)
resp = await rag.search(
    query="connection pool troubleshooting",
    corpus_id="knowledge-articles",
    limit=10,
)
for item in resp.results:
    print(f"{item.score:.2f} - {item.payload.get('title')}")

# Debug routing
routing = await rag.route(query="What depends on PostgreSQL?")
print(routing.query_signals.intent)   # "relationship"
print(routing.candidates[0].engine)   # "graph"
```

### Feature Toggles

Agents can gradually adopt RAGRouter via environment variables:

```bash
# KnowledgeArticleAgent
KNOWLEDGE_USE_RAGROUTER=true

# DocsAgent
DOCS_USE_RAGROUTER=true
```

When disabled, agents fall back to direct vectorstore search.

## Adding a New Engine

1. Create `ragrouter/adapters/<name>_adapter/main.py` with a FastAPI app
2. Implement `POST /query` accepting `EngineRequest`, returning `EngineResponse`
3. Implement `GET /health`
4. Add engine config to `ragrouter/arbiter/config.yaml`
5. Create a Dockerfile and K8s manifest
6. Deploy — the arbiter auto-discovers enabled engines from config

### Engine Contract

```python
from ragrouter.schemas import EngineRequest, EngineResponse

@app.post("/query")
async def query(request: EngineRequest) -> EngineResponse:
    # 1. Retrieve relevant documents
    # 2. Generate answer with citations
    # 3. Return with scores and trace
    return EngineResponse(
        engine="my-engine",
        answer="...",
        citations=[Citation(doc_id="doc-1", section="...", score=0.9)],
        scores=Scores(confidence=0.8, groundedness=0.9, coverage=0.7, relevance=0.85),
        trace=Trace(retrieval_method="my-method", docs_retrieved=5),
        usage=Usage(latency_ms=200, tokens_in=500, tokens_out=300),
    )
```
