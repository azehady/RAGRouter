# RAGRouter: Implementation Roadmap

## Vision

A meta-RAG orchestration system that routes queries to the best RAG paradigm, executes them as independent K8s services, and arbitrates results into a single best answer.

**Core thesis: Don't build engines. Orchestrate the best OSS ones. Invest in the router, arbiter, and evaluation layer -- that's where the novel value is.**

```
                    ┌─────────────┐
                    │   Arbiter   │  router + planner + scorer
                    │   1 pod     │
                    └──────┬──────┘
                           │
          ┌────────┬───────┼───────┬────────┐
          ▼        ▼       ▼       ▼        ▼
    ┌─────────┐ ┌──────┐ ┌─────┐ ┌──────┐ ┌──────────┐
    │   TOC   │ │Hybrid│ │Graph│ │Agent │ │Long-Ctx  │
    │PageIndex│ │Haystack│ │Light│ │Lang  │ │Contextual│
    │         │ │+Qdrant│ │RAG  │ │Graph │ │Retrieval │
    └─────────┘ └──────┘ └─────┘ └──────┘ └──────────┘
      pod 1      pod 2    pod 3   pod 4     pod 5
```

---

## Phase 1: Foundation -- Arbiter + Hybrid Engine

**Goal:** Working router with one real engine and one baseline. Validates the routing and arbitration layer end-to-end.

### 1.1 Define the standard engine contract

Every engine exposes `POST /query` with a uniform request/response:

```
Request:  { query, corpus_id, chat_history, constraints }
Response: { engine, answer, citations, trace, scores, usage }
```

This contract is the foundation. Get it right first. All adapters normalize to it.

**Deliverable:** `ragrouter/schemas.py` -- Pydantic models for EngineRequest, EngineResponse, Citation, Trace, Scores, Usage.

### 1.2 Build the arbiter service

FastAPI service with three components:

- **Registry** -- loads engine capabilities from YAML config (URL, strengths, latency profile, cost rank)
- **Analyzer** -- extracts query-level signals (intent, specificity, section_reference) using regex + optional lightweight LLM
- **Planner** -- selects engines using deterministic rules, returns ordered list
- **Scorer** -- evaluates engine responses on groundedness, citation coverage, query alignment

Two dispatch modes:
- `cascade` (default): try best-fit first, fallback on low confidence
- `parallel_topk`: run top-K concurrently, pick best

**Deliverable:** Arbiter pod running on MicroK8s, dispatching to engine stubs, returning scored results.

### 1.3 Hybrid engine adapter (first real engine)

Wrap Haystack + Qdrant + FlashRank behind the standard contract:

- [deepset-ai/haystack](https://github.com/deepset-ai/haystack) for orchestration
- [qdrant/qdrant](https://github.com/qdrant/qdrant) for dense + sparse vectors (already deployed)
- [PrithivirajDamodaran/FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) for reranking

This is also the engine that serves the Ciroos DocsAgent hybrid RAG upgrade (BM25 + vector + RRF + reranking).

**Deliverable:** Hybrid engine pod on MicroK8s, searchable via `/query`, returning standardized responses.

### 1.4 Simple vector baseline (second engine)

Wrap the current Qdrant-only vector search as a baseline engine. Same contract, minimal adapter (~50 lines). This gives the arbiter two engines to compare.

**Deliverable:** Baseline engine pod. Arbiter can route between hybrid and baseline, demonstrating routing value.

### 1.5 Evaluation harness

Use RAGAS ([explodinggradients/ragas](https://github.com/explodinggradients/ragas)) to evaluate both engines on the existing `eval_dataset_v2.json` (30 questions).

Metrics per engine per query:
- Faithfulness, context relevancy, answer relevancy (RAGAS)
- Latency, token cost (from engine usage report)
- Arbiter decision accuracy (did it pick the better engine?)

**Deliverable:** Benchmark script that runs all queries through both engines via the arbiter, outputs comparison table.

### Phase 1 Files

```
ragrouter/
  arbiter/
    main.py              # FastAPI app
    schemas.py           # Standard engine contract (Pydantic)
    registry.py          # Engine capability registry (YAML)
    analyzer.py          # Query signal extraction
    planner.py           # Engine selection logic
    scorer.py            # Response scoring + arbitration
    config.yaml          # Engine registry config
  adapters/
    hybrid_adapter/
      main.py            # FastAPI wrapping Haystack + Qdrant + FlashRank
      Dockerfile
    baseline_adapter/
      main.py            # FastAPI wrapping current Qdrant vector search
      Dockerfile
  evaluation/
    benchmark.py         # Run eval dataset through arbiter
    metrics.py           # RAGAS integration
  k8s/
    arbiter.yaml         # K8s deployment + service
    hybrid-engine.yaml
    baseline-engine.yaml
```

---

## Phase 2: TOC Engine -- PageIndex Integration

**Goal:** Add structure-first retrieval for long/structured documents. Tests whether routing to a different paradigm improves results.

### 2.1 PageIndex adapter

Wrap [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) behind the standard contract:

- Index documents into PageIndex tree structure during ingestion
- On query: PageIndex reasons over TOC, returns relevant sections
- Adapter normalizes output to EngineResponse with citations + trace

### 2.2 Routing rules for TOC

Add to the planner:
- If `structure_score > 0.6` and `avg_doc_tokens > 5000` and query has section-seeking intent → route to TOC
- TOC engine as first choice, hybrid as fallback

### 2.3 Evaluate TOC vs hybrid

Run the same eval dataset through both. Measure where TOC wins (structured docs, section lookups) and where it loses (short docs, fuzzy queries).

### Phase 2 Files

```
ragrouter/
  adapters/
    toc_adapter/
      main.py            # FastAPI wrapping PageIndex
      indexer.py          # Document → PageIndex tree ingestion
      Dockerfile
  k8s/
    toc-engine.yaml
```

---

## Phase 3: Graph Engine -- LightRAG Integration

**Goal:** Add entity-relationship retrieval for multi-hop reasoning queries.

### 3.1 LightRAG adapter

Wrap [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) behind the standard contract:

- Ingest documents into LightRAG's dual-level graph (entities + relationships)
- On query: LightRAG traverses graph, returns connected information
- Adapter normalizes to EngineResponse

### 3.2 Routing rules for graph

Add to the planner:
- If query contains relationship patterns ("how is X related to Y", "compare", "what connects") → route to graph
- If `entity_density > 0.3` in corpus → graph as candidate

### 3.3 Evaluate graph vs hybrid vs TOC

Three-way comparison. Identify query types where each paradigm wins.

### Phase 3 Files

```
ragrouter/
  adapters/
    graph_adapter/
      main.py            # FastAPI wrapping LightRAG
      indexer.py          # Document → LightRAG graph ingestion
      Dockerfile
  k8s/
    graph-engine.yaml
```

---

## Phase 4: Agentic Engine -- LangGraph Integration

**Goal:** Add iterative retrieve-read-refine for complex synthesis queries. This is the expensive fallback for hard questions.

### 4.1 LangGraph adapter

Wrap [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) behind the standard contract:

- Define a graph: retrieve → read → evaluate → refine → verify
- Agent can call back into hybrid/TOC/graph engines as tools
- Adapter normalizes final output to EngineResponse

### 4.2 Routing rules for agentic

Add to the planner:
- If query intent is `synthesis` or `research` → agent as candidate
- Only if `budget.allows_agent` (latency budget > 5s, cost tolerance high)
- Always last in cascade order (most expensive)

### Phase 4 Files

```
ragrouter/
  adapters/
    agent_adapter/
      main.py            # FastAPI wrapping LangGraph
      graph_def.py       # Agent workflow graph definition
      Dockerfile
  k8s/
    agent-engine.yaml
```

---

## Phase 5: Long-Context Engine + Contextual Retrieval

**Goal:** Add a simple baseline for small doc sets and enrich all engines with contextual retrieval.

### 5.1 Long-context adapter

Direct LLM call with full document context. For corpora < 50K tokens, just stuff it all in.

Reference: [anthropics/anthropic-retrieval-demo](https://github.com/anthropics/anthropic-retrieval-demo)

### 5.2 Contextual retrieval enrichment

Apply Anthropic's contextual retrieval technique during ingestion across all engines:
- Prepend chunk-specific context (generated by LLM) before embedding
- Improves all vector-based engines by 49% on failed retrievals

### Phase 5 Files

```
ragrouter/
  adapters/
    longcontext_adapter/
      main.py            # FastAPI wrapping direct LLM call
      Dockerfile
  k8s/
    longcontext-engine.yaml
```

---

## Phase 6: Arbiter Sophistication

**Goal:** Make the arbiter smarter once you have 3+ engines producing real outputs.

### 6.1 Benchmark-anchored confidence calibration

Run all engines on a shared eval set. Map each engine's raw confidence to actual correctness rates. Build a calibration curve per engine so scores are comparable.

### 6.2 Merge-and-verify mode

For high-stakes queries:
- Merge claims from top-2 engine answers
- Run a verification pass: check each claim against cited evidence
- Remove unsupported claims
- Produce a composite answer with stronger grounding

### 6.3 Consensus voting

For parallel top-3 execution:
- If 2/3 engines agree on core claims and citations overlap, prefer that consensus
- Useful when individual confidence is borderline

### 6.4 Learning from production

Log every routing decision + engine outputs + user feedback. Over time:
- Track which engine wins for which query types
- Adjust planner weights based on historical accuracy
- Identify query patterns where current routing fails

---

## Phase 7: Publish + Open Source

**Goal:** Release RAGRouter as a pip-installable library and K8s-deployable system.

### 7.1 Package structure

```
pip install ragrouter
```

- `ragrouter.arbiter` -- router, planner, scorer, registry
- `ragrouter.schemas` -- standard engine contract
- `ragrouter.adapters` -- ready-made adapters for PageIndex, Haystack, LightRAG, LangGraph
- `ragrouter.evaluation` -- RAGAS integration, benchmark harness

### 7.2 Helm chart

K8s deployment with configurable engine selection:

```yaml
engines:
  hybrid:
    enabled: true
    replicas: 2
  toc:
    enabled: true
    replicas: 1
  graph:
    enabled: false
  agent:
    enabled: false
  longcontext:
    enabled: true
    replicas: 1
```

### 7.3 Documentation + benchmarks

- Pareto curves: accuracy vs cost vs latency per paradigm
- Router decision accuracy on standard benchmarks
- Comparison tables showing where each paradigm wins

---

## Deployment Architecture (MicroK8s)

```
Namespace: ragrouter
┌──────────────────────────────────────────────────┐
│                                                  │
│  arbiter (1 pod, ~200MB)                         │
│    FastAPI: /ask, /health                        │
│    Dispatches to engines via ClusterIP services  │
│                                                  │
│  hybrid-engine (1-2 pods, ~1GB)                  │
│    Haystack + Qdrant client + FlashRank          │
│    Handles 70-80% of queries                     │
│                                                  │
│  toc-engine (1 pod, ~500MB)                      │
│    PageIndex                                     │
│    Long structured docs, section lookups         │
│                                                  │
│  graph-engine (1 pod, ~1GB)                      │
│    LightRAG                                      │
│    Relationship queries, multi-hop               │
│                                                  │
│  agent-engine (1 pod, ~500MB)                    │
│    LangGraph                                     │
│    Complex synthesis (expensive fallback)         │
│                                                  │
│  longcontext-engine (1 pod, ~200MB)              │
│    Direct LLM call                               │
│    Small doc sets, baseline                      │
│                                                  │
│  qdrant (1 pod, existing)                        │
│  litellm (1 pod, existing)                       │
│                                                  │
│  Estimated total: 4-6GB RAM (Phase 1-2)          │
│                   8-12GB RAM (all engines)        │
└──────────────────────────────────────────────────┘
```

---

## Resource Estimates

| Component | RAM | CPU | Notes |
|-----------|-----|-----|-------|
| Arbiter | 200MB | 0.2 cores | Stateless, lightweight |
| Hybrid engine | 1GB | 0.5 cores | Haystack + FlashRank model |
| TOC engine | 500MB | 0.3 cores | PageIndex tree in memory |
| Graph engine | 1GB | 0.5 cores | LightRAG graph state |
| Agent engine | 500MB | 0.3 cores | LangGraph, mostly LLM calls |
| Long-context engine | 200MB | 0.1 cores | Just an HTTP proxy to LLM |
| **Phase 1 total** | **~2GB** | **~1 core** | Arbiter + hybrid + baseline |
| **All engines total** | **~3.5GB** | **~2 cores** | Excluding Qdrant, LiteLLM |

---

## Success Criteria

### Phase 1 (MVP)
- Arbiter correctly routes docs queries to hybrid engine
- Hybrid engine outperforms baseline on eval_dataset_v2 (30 questions)
- Routing overhead < 50ms
- End-to-end latency < 2s for hybrid path

### Phase 2 (TOC)
- TOC engine outperforms hybrid on at least 20% of structured-doc queries
- Arbiter correctly identifies and routes to TOC when appropriate

### Phase 3 (Graph)
- Graph engine outperforms hybrid on relationship/multi-hop queries
- Three-way routing produces better aggregate scores than any single engine

### Phase 6 (Calibration)
- Arbiter picks the best engine for > 80% of queries (measured against oracle)
- Merge-and-verify mode produces higher groundedness than any single engine

### Phase 7 (Publication)
- Pareto-optimal on accuracy-vs-cost curve compared to single-engine baselines
- Adopted by at least one external team

---

## Key Principle

> Don't build engines. Orchestrate the best open-source ones.
> The router, arbiter, and evaluation layer is where the novel value lives.
> ~850 lines of adapter code. The rest is your product.
