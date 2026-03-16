# Current DocsAgent vs Hybrid RAG: Comparison Report

## Overview

This report compares the current DocsAgent RAG pipeline (PR #2512, `feature/chat_with_docs`) with the planned hybrid RAG upgrade. The current implementation uses pure dense vector search. The hybrid approach adds keyword matching, query expansion, score fusion, and reranking on top of the same infrastructure.

---

## Current DocsAgent Pipeline

```
User Query
  -> OpenAI embed (text-embedding-3-small, 1536-dim)
  -> Qdrant cosine similarity search (top 5, threshold 0.35)
  -> LLM generates answer with citations
```

### How it works

1. **Single vector type**: Only dense vectors stored in Qdrant. Collection created with `VectorParams(size=1536, distance=Cosine)` — no named vectors, no sparse vectors.

2. **Single query path**: The raw user query is embedded and searched as-is. No expansion, no reformulation, no alternative phrasings.

3. **No keyword matching**: No BM25 or sparse retrieval. Matching is purely semantic (embedding distance). A query like "RBAC" only matches if it's semantically close to the stored text, not by exact keyword presence.

4. **No fusion**: Single retrieval path returns one ranked list. No merging of results from multiple sources.

5. **No reranking**: Results are returned in raw cosine similarity order. No cross-encoder or LLM-based precision refinement.

6. **Hard score cutoff**: Documents below score threshold 0.35 are dropped entirely.

### Key code paths

- **Search tool** (`tools.py`): Sends `POST /api/v1/collections/ciroos-docs/search` with query text, limit, and optional `doc_type` filter.
- **Vectorstore** (`qdrant_client.py`): Calls `client.query_points()` with the query vector, cosine distance, and org_id filter.
- **Ingestion** (`ingest.py`): Creates collection with plain `VectorParams`, uploads chunks with text payload. Vectorstore auto-generates dense embeddings via OpenAI.
- **Chunking** (`chunker.py`): Heading-based (H2 sections), max 1500 words, 200-word overlap.

### Configuration

| Parameter | Value |
|-----------|-------|
| Embedding model | `text-embedding-3-small` |
| Vector dimensions | 1536 |
| Distance metric | Cosine |
| Search limit | 5 (configurable 1-10) |
| Score threshold | 0.35 |
| Collection | `ciroos-docs` |
| Org ID | `system` (shared) |

---

## Hybrid RAG Pipeline (Planned)

```
User Query
  -> Query Expansion (LLM via LiteLLM)
       -> lex: keyword/synonym variation (for BM25)
       -> vec: semantic rephrasing (for embeddings)
       -> hyde: hypothetical document (HyDE)
  -> Dual Retrieval (parallel, per query variant)
       -> BM25/Sparse Search (Qdrant sparse vectors)
       -> Dense Vector Search (existing cosine)
  -> Reciprocal Rank Fusion (RRF, k=60)
  -> Reranking (FlashRank cross-encoder or LLM)
  -> Final Top-5 Results
  -> LLM generates answer with citations
```

### What changes

1. **Dual vector types**: Qdrant collection upgraded to named vectors — `"dense"` (1536-dim cosine) alongside `"sparse"` (BM25-like term frequency vectors). Qdrant natively supports this via `SparseVectorParams`.

2. **Query expansion**: Single LLM call (~300ms via LiteLLM proxy) generates three additional query variants. Each variant is optimised for a different retrieval path.

3. **BM25 keyword matching**: Sparse vectors enable exact keyword matching. Terms like "RBAC", "ServiceNow", "DNAC" match by token presence, not just semantic similarity.

4. **Multi-query retrieval**: 4 query variants x 2 retrieval paths = up to 8 result lists, executed in parallel via `asyncio.gather()`.

5. **RRF fusion**: Reciprocal Rank Fusion merges all result lists. A document that ranks #8 in dense but #2 in BM25 gets a combined score that surfaces it to the top. Formula: `score(d) = sum(1 / (k + rank_i))` with top-rank bonuses.

6. **Cross-encoder reranking**: FlashRank (~4MB, no GPU, ONNX runtime) or LLM-based reranking scores the top 15 RRF candidates for precise relevance. Position-aware blending: `final = alpha * rrf_score + (1-alpha) * rerank_score`.

---

## Stage-by-Stage Comparison

| Stage | Current | Hybrid |
|-------|---------|--------|
| **Query processing** | Raw query as-is | Query expansion: original + lex + semantic + HyDE |
| **Keyword search** | None | BM25 via Qdrant sparse vectors |
| **Vector search** | 1x dense cosine (top 5) | 4x dense cosine (top 20 each, per query variant) |
| **Fusion** | None (single result list) | RRF across all result lists (k=60, top-rank bonus) |
| **Reranking** | None (raw cosine order) | FlashRank cross-encoder or LLM scoring |
| **Final output** | Top 5 by cosine score | Top 5 by fused + reranked score |
| **Estimated latency** | ~500ms | ~800ms (degraded mode: ~200ms) |

---

## Where the Current Approach Loses

### 1. Exact keyword misses

If someone asks "what is the RBAC policy" and the document text says "role-based access control" but the embedding distance is above the threshold, it won't match. BM25 would catch "RBAC" as an exact keyword token regardless of semantic distance.

**Impact**: Queries with acronyms, product names, and technical terms are vulnerable to missed retrieval.

### 2. Single query brittleness

If the user's phrasing diverges from how the documentation is written, the single embedding may miss relevant content. Query expansion generates three additional phrasings — a keyword-heavy variant for BM25, a semantically rephrased variant for embeddings, and a hypothetical answer document (HyDE) that matches the style of the target content.

**Impact**: Queries that are correct but worded differently from the docs may return low-relevance or no results.

### 3. No precision refinement

Cosine similarity ranks by embedding distance, which is a coarse proxy for relevance. Two documents at scores 0.72 and 0.71 may have very different actual relevance to the query. A cross-encoder reranker (FlashRank) does full cross-attention between query and passage — token-level matching that is much more precise than embedding distance.

**Impact**: The top-5 results may not be the most relevant 5. A document at position 8 by cosine may actually be more relevant than position 3.

### 4. No score fusion across retrieval paths

With a single retrieval path, a document either scores well on dense similarity or it doesn't. Hybrid RRF means a document that ranks poorly in vector search but highly in keyword search (or vice versa) still gets a fair combined score. This is especially valuable for queries that mix specific terms with general concepts.

**Impact**: Recall is limited to what a single embedding-based search can find.

---

## What Stays the Same

Both approaches share the same core infrastructure:

| Component | Shared |
|-----------|--------|
| Vector database | Qdrant |
| Vectorstore service | Port 2526, same API |
| Document chunking | Heading-based (H2 sections), 1500 words max, 200-word overlap |
| Collection name | `ciroos-docs` |
| Multi-tenancy | `org_id: system` for shared docs |
| LLM answer generation | Via LiteLLM proxy |
| Dense embeddings | `text-embedding-3-small` (1536-dim) — kept as-is, no re-embedding |

The hybrid approach is additive. Dense vectors remain unchanged. Sparse vectors are added alongside them. Query expansion and reranking are new stages that wrap around the existing search. Each new stage degrades gracefully — if expansion fails, the original query is used; if reranking fails, RRF order is used; if BM25 fails, pure vector search continues as before.

---

## Latency Budget

| Stage | Current | Hybrid | Notes |
|-------|---------|--------|-------|
| Query expansion | — | ~300ms | Single LLM call via LiteLLM |
| Search | ~500ms | ~200ms | Parallel; Qdrant handles dense+sparse natively |
| RRF fusion | — | ~1ms | In-memory computation |
| Reranking | — | ~300ms | FlashRank (ONNX) or single LLM call |
| **Total** | **~500ms** | **~800ms** | |
| **Degraded** | — | **~200ms** | No expansion, no reranking |

---

## Migration Path

The hybrid upgrade is non-destructive:

1. **Phase 1**: Add sparse vectors to Qdrant collection (named vectors config). Migrate existing points by computing sparse vectors from stored text payloads. No re-embedding of dense vectors needed.

2. **Phase 2**: Add query expansion module in the DocsAgent tools layer. Toggle via `DOCS_QUERY_EXPANSION=true|false`.

3. **Phase 3**: Add RRF fusion and reranking modules. Toggle via `DOCS_RERANKING=true|false`.

4. **Phase 4**: Benchmark against `eval_dataset_v2.json` (30 questions) to measure recall, MRR, and latency improvements.

Each stage is independently toggleable via environment variables, so the system can fall back to current behavior at any point.

---

## Summary

The current DocsAgent is a working single-path dense retrieval system. It handles most queries adequately but is vulnerable to keyword misses, single-phrasing brittleness, and lacks precision refinement. The hybrid upgrade adds three layers — wider recall (query expansion + BM25), better fusion (RRF), and better precision (reranking) — while keeping the same infrastructure and degrading gracefully if any stage fails.
