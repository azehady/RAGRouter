# Hybrid RAG Pipeline for DocsAgent

## Overview

Upgrade the DocsAgent from pure vector search to a hybrid RAG pipeline inspired by the QMD Query Pipeline architecture.

### Current Pipeline

```
User Query
  -> Embed (text-embedding-3-small, 1536-dim)
  -> Qdrant cosine similarity search
  -> Top 5 results (score threshold 0.35)
  -> LLM generates answer with citations
```

### Target Pipeline

```
User Query
  -> Query Expansion (LLM via LiteLLM)
       -> lex: lexical/keyword variation
       -> vec: semantic variation
       -> hyde: hypothetical document
  -> Dual Retrieval (parallel, per query variant)
       -> BM25/Sparse Search (keyword matching)
       -> Vector Search (semantic similarity)
  -> Reciprocal Rank Fusion (RRF, k=60, top-rank bonuses)
  -> LLM Reranking (position-aware score blending)
  -> Final Top-5 Results
```

---

## Key Architectural Decisions

### 1. Search Backend: Qdrant (default) + SQLite (benchmark)

**Default: Qdrant sparse vectors** for BM25 alongside existing dense vectors. Qdrant already supports named vectors (dense + sparse) and built-in RRF fusion via its `query_points()` API with prefetch.

**SQLite FTS5 + sqlite-vec** as a pluggable alternative backend for benchmarking. Both backends implement the same `SearchBackend` interface so they can be swapped via `SEARCH_BACKEND=qdrant|sqlite` env var.

**Why not replace Qdrant:** Other services (RCA, schema validation) use the vectorstore. Qdrant is already deployed on the cluster. Adding sparse vectors to Qdrant is additive - no migration risk.

### 2. Query Expansion & Reranking: LLM via LiteLLM

The QMD diagram shows fine-tuned Qwen3-1.7B for expansion and Qwen3-Reranker-0.6B for reranking. Without GPU on MicroK8s, we use LLM-based alternatives via the existing LiteLLM proxy:

- **Query expansion:** Single `gpt-4o-mini` call with structured prompt to generate lex/vec/hyde variants (~300ms)
- **Reranking:** Single `gpt-4o-mini` call to score top-15 candidates (~300ms)

Both stages degrade gracefully - if LLM fails, pipeline falls back to the previous stage's output.

### 3. Keep Existing Embeddings

Keep `text-embedding-3-small` (1536-dim) rather than switching to 384-dim. Higher dimensionality gives better embedding quality. The hybrid search compensates for any embedding limitations with BM25 keyword matching. No re-indexing needed for the Qdrant path.

### 4. Pipeline Location: Split Across Services

- **Vectorstore service** owns: BM25 indexing, sparse vectors, hybrid search API, RRF fusion (within a single query)
- **DocsAgent tools layer** owns: query expansion, multi-query orchestration, cross-query RRF, LLM reranking

This keeps retrieval infrastructure in the vectorstore and RAG-specific logic in the agent layer.

---

## Implementation Phases

### Phase 1: Pluggable Search Backend + BM25 Support

**Goal:** Enable hybrid search (BM25 + vector) in the vectorstore service with Qdrant as default and SQLite as benchmark alternative.

#### 1.1 Search backend interface

New file: `vectorstore/app/services/search_backend.py`

```python
class SearchBackend(ABC):
    """Pluggable search backend interface."""
    async def index_document(chunk_id, text, dense_vector, metadata) -> None
    async def search(query_text, query_vector, limit, filters) -> list[SearchResult]
    async def hybrid_search(query_text, query_vector, limit, filters, alpha) -> list[SearchResult]
```

#### 1.2 Qdrant sparse vector support (default backend)

Modify: `vectorstore/app/services/qdrant_client.py`

- Upgrade `create_collection()` to support named vectors:
  - `"dense"`: existing 1536-dim cosine
  - `"sparse"`: `SparseVectorParams` for BM25-like term frequency vectors
- Add `SparseEncoder` class for BM25 tokenization:
  - Lowercase, strip punctuation
  - Remove stop words (175 common English words)
  - Porter stemming
  - Compute TF weights as Qdrant `SparseVector(indices, values)`
- Modify `upsert_points()` to store both dense + sparse vectors
- Add `hybrid_search()` using Qdrant's prefetch + `Fusion.RRF`:

```python
results = await self.client.query_points(
    collection_name=name,
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=20),
        Prefetch(query=sparse_vector, using="sparse", limit=20),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=limit,
)
```

#### 1.3 SQLite backend (benchmark alternative)

New file: `vectorstore/app/services/sqlite_backend.py`

- SQLite FTS5 for BM25 with Porter stemming
- sqlite-vec for vector search (cosine similarity)
- Same `SearchBackend` interface
- Selected via `SEARCH_BACKEND=sqlite` env var

#### 1.4 Hybrid search API endpoint

Modify: `vectorstore/app/api/search.py`

- New endpoint: `POST /api/v1/collections/{name}/hybrid-search`
- Request body: `HybridSearchRequest(query, limit, alpha, score_threshold, filter)`
- `alpha` controls dense vs sparse weight (1.0 = all dense, 0.0 = all BM25)
- Existing `/search` endpoint unchanged (backward compatible)

#### Files changed (Phase 1)

| File | Change |
|------|--------|
| `vectorstore/app/services/qdrant_client.py` | Sparse vectors, hybrid search |
| `vectorstore/app/services/search_backend.py` | New - abstract interface |
| `vectorstore/app/services/sqlite_backend.py` | New - SQLite FTS5 + sqlite-vec |
| `vectorstore/app/services/sparse_encoder.py` | New - BM25 tokenizer/stemmer |
| `vectorstore/app/api/search.py` | Hybrid search endpoint |
| `vectorstore/app/models/search.py` | HybridSearchRequest model |
| `vectorstore/app/config.py` | Backend selection config |
| `vectorstore/pyproject.toml` | Optional: sqlite-vec, nltk |

---

### Phase 2: Query Expansion

**Goal:** Generate multiple query variations to improve recall across both BM25 and vector search.

#### 2.1 Query expansion module

New file: `common-agents/src/agents/agents/docs/query_expansion.py`

```python
@dataclass
class ExpandedQuery:
    original: str       # User's original query
    lexical: str        # Keyword/synonym variation (for BM25)
    semantic: str       # Rephrased for embedding search
    hypothetical: str   # HyDE: what the ideal answer doc looks like
```

- Single LLM call via LiteLLM proxy requesting JSON output
- Prompt generates all 3 variants in one call (~300ms)
- Graceful fallback: returns original query as all variants on failure

#### 2.2 Integration into DocsAgent search tool

Modify: `common-agents/src/agents/agents/docs/tools.py`

Update `_execute_search()`:
1. Expand query -> 4 variants (original + 3 expansions)
2. Fire 4 parallel hybrid searches via `asyncio.gather()`
3. Collect results for cross-query fusion (Phase 3)

Toggle: `DOCS_QUERY_EXPANSION=true|false` (default: true)

#### Files changed (Phase 2)

| File | Change |
|------|--------|
| `common-agents/src/agents/agents/docs/query_expansion.py` | New - LLM expansion |
| `common-agents/src/agents/agents/docs/tools.py` | Wire up expansion |

---

### Phase 3: RRF Fusion + LLM Reranking

**Goal:** Merge multi-query results and rerank for precision.

#### 3.1 Cross-query Reciprocal Rank Fusion

New file: `common-agents/src/agents/agents/docs/fusion.py`

```python
def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    top_rank_bonus: float = 0.1,
) -> list[SearchResult]:
    """Merge ranked lists: score(d) = sum(1/(k + rank_i)) + bonuses."""
```

- Merges 4 result lists (one per expanded query variant)
- Deduplicates by chunk_id
- Top-rank bonus for docs appearing in top-3 of any list

#### 3.2 LLM-based reranking

New file: `common-agents/src/agents/agents/docs/reranker.py`

```python
async def rerank_results(
    query: str,
    candidates: list[SearchResult],
    top_k: int = 5,
) -> list[SearchResult]:
    """Score candidates via LLM, blend with RRF scores."""
```

- Takes top 15 RRF candidates
- LLM scores each 0-10 for relevance to original query
- Position-aware blending: `final = alpha * rrf_score + (1-alpha) * llm_score`
- Falls back to RRF order on LLM failure

Toggle: `DOCS_RERANKING=true|false` (default: true)

#### 3.3 Full pipeline wiring

Modify: `common-agents/src/agents/agents/docs/tools.py`

```
query -> expand (Phase 2) -> 4x hybrid search (Phase 1)
      -> cross-query RRF fusion (3.1)
      -> LLM reranking (3.2)
      -> top-5 results returned to DocsAgent
```

**Graceful degradation:**
- Expansion fails -> hybrid search with original query only
- Hybrid search fails -> pure vector search (current behavior)
- Reranking fails -> return RRF-fused results without reranking

#### Files changed (Phase 3)

| File | Change |
|------|--------|
| `common-agents/src/agents/agents/docs/fusion.py` | New - RRF implementation |
| `common-agents/src/agents/agents/docs/reranker.py` | New - LLM reranking |
| `common-agents/src/agents/agents/docs/tools.py` | Full pipeline wiring |

---

### Phase 4: Re-Ingestion + Migration

**Goal:** Index existing docs with both dense and sparse vectors.

#### 4.1 Update ingestion script

Modify: `common-agents/src/agents/agents/docs/ingest.py`

- `create_collection()`: pass named vectors config (dense + sparse)
- `upsert_chunks()`: include raw text so vectorstore computes sparse vectors
- Add `--backend qdrant|sqlite` flag

#### 4.2 Migration script

New file: `common-agents/scripts/migrate_to_hybrid.py`

- Scrolls existing `ciroos-docs` points
- Computes sparse vectors from existing `text` payloads
- Batch-updates points (avoids re-computing dense embeddings)

#### Files changed (Phase 4)

| File | Change |
|------|--------|
| `common-agents/src/agents/agents/docs/ingest.py` | Named vectors config |
| `common-agents/scripts/migrate_to_hybrid.py` | New - migration script |

---

### Phase 5: Evaluation + Benchmarking

**Goal:** Validate improvements, compare backends, tune parameters.

#### 5.1 Baseline + hybrid evaluation

Run existing `eval_dataset_v2.json` (30 questions) against:
- Current pure-vector pipeline (baseline)
- Hybrid (Qdrant) with each stage progressively enabled
- Hybrid (SQLite) for backend comparison

#### 5.2 Benchmark script

New file: `common-agents/scripts/benchmark_rag.py`

Measures per query:
- Recall@5, Recall@10, MRR
- Latency breakdown: expansion / search / fusion / reranking
- Backend comparison (Qdrant vs SQLite)

#### 5.3 Tuning parameters

| Parameter | Default | Tune Range |
|-----------|---------|------------|
| RRF k | 60 | 20-100 |
| Dense/sparse alpha | 0.5 | 0.0-1.0 |
| Reranking blend weight | 0.5 | 0.0-1.0 |
| Score threshold | 0.35 | 0.2-0.5 |
| Reranker candidates | 15 | 10-25 |

---

## Configuration / Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SEARCH_BACKEND` | `qdrant` | `qdrant` or `sqlite` |
| `DOCS_HYBRID_SEARCH` | `true` | Enable hybrid (BM25 + vector) search |
| `DOCS_QUERY_EXPANSION` | `true` | Enable LLM query expansion |
| `DOCS_RERANKING` | `true` | Enable LLM reranking |
| `DOCS_RRF_K` | `60` | RRF k parameter |
| `DOCS_RERANK_CANDIDATES` | `15` | Candidates passed to reranker |
| `DOCS_RERANK_TOP_K` | `5` | Final results after reranking |

---

## Latency Budget

| Stage | Estimated | Notes |
|-------|-----------|-------|
| Query expansion | ~300ms | Single LLM call via LiteLLM |
| 4x hybrid search | ~200ms | Parallel, Qdrant is fast |
| RRF fusion | ~1ms | In-memory computation |
| LLM reranking | ~300ms | Single LLM call |
| **Total** | **~800ms** | vs ~500ms current |

Degraded mode (no expansion, no reranking): ~200ms

---

## New Files Summary

| File | Purpose |
|------|---------|
| `vectorstore/app/services/search_backend.py` | Abstract search backend interface |
| `vectorstore/app/services/sparse_encoder.py` | BM25-like sparse vector encoder |
| `vectorstore/app/services/sqlite_backend.py` | SQLite FTS5 + sqlite-vec backend |
| `common-agents/src/agents/agents/docs/query_expansion.py` | LLM-based query expansion |
| `common-agents/src/agents/agents/docs/fusion.py` | Reciprocal Rank Fusion |
| `common-agents/src/agents/agents/docs/reranker.py` | LLM-based reranking |
| `common-agents/scripts/migrate_to_hybrid.py` | Migration script for existing docs |
| `common-agents/scripts/benchmark_rag.py` | Eval benchmark script |

## Modified Files Summary

| File | Changes |
|------|---------|
| `vectorstore/app/services/qdrant_client.py` | Sparse vectors, named vectors, hybrid search |
| `vectorstore/app/api/search.py` | Hybrid search endpoint |
| `vectorstore/app/models/search.py` | HybridSearchRequest/Response models |
| `vectorstore/app/config.py` | Backend selection, sparse config |
| `vectorstore/pyproject.toml` | Optional sqlite-vec, nltk deps |
| `common-agents/src/agents/agents/docs/tools.py` | Full pipeline: expand -> search -> fuse -> rerank |
| `common-agents/src/agents/agents/docs/ingest.py` | Named vectors, sparse vector ingestion |

---

## Verification Plan

1. **Unit tests:** Each new module (sparse_encoder, fusion, reranker, query_expansion) with pytest
2. **Integration test:** `just chatter-orchestrated-eval --query '"how do I onboard an AWS account"'` with hybrid search
3. **Eval benchmark:** 30 eval_dataset_v2 questions, compare recall/MRR before and after
4. **Backend comparison:** Same queries through Qdrant and SQLite backends
5. **Degradation test:** Disable stages via env vars, verify graceful fallback
