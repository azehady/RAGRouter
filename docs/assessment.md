# RAGRouter: Assessment & Refined Strategy

## What RAGRouter Is

A meta-RAG orchestration system that routes queries to the best RAG paradigm (or top-N paradigms), executes them, and arbitrates the results into a single best answer.

The core thesis: **RAG engines are specialists, not competitors. The winning system routes between them.**

```
Query
  -> Analyzer (classify query + corpus signals)
  -> Planner (select top 1-3 RAG paradigms)
  -> Execute (dispatch to best-fit engines)
  -> Arbiter (score, merge, verify outputs)
  -> Best Answer with citations + trace
```

---

## Five RAG Paradigms to Route Between

| Paradigm | What It Does | Best For | Weakness |
|----------|-------------|----------|----------|
| **TOC/Hierarchical** | LLM reasons over document structure tree | Long structured docs (legal, financial, specs) | Poor web-scale recall |
| **Hybrid Vector** | BM25 + dense embeddings + reranker | Large corpora, enterprise KB, generalist | Higher complexity |
| **GraphRAG** | Entity-relationship graph traversal | Multi-hop "how is X related to Y" | Expensive graph build, brittle |
| **Agentic/Tool** | Iterative retrieve-read-refine loop | Research, synthesis, complex workflows | Latency, cost |
| **Long-Context** | Stuff everything into LLM context | Small doc sets, quick baseline | Cost, hallucination, no traceability |

---

## Assessment: What's Strong

### 1. The gap is real

Research confirms nobody routes between **entire RAG paradigms**. Existing routers only switch between retrievers (BM25 vs dense) or models (small vs large LLM). Multi-paradigm routing with ensemble arbitration is genuinely novel.

### 2. Standard engine interface is the right design

Every engine returns: `answer + citations + trace + confidence_scores + usage_cost`. This contract makes engines swappable, testable, benchmarkable. It's the abstraction that enables everything else.

### 3. Cascade with early exit is production-ready

Running top-3 in parallel every time burns cost. Cascade (try cheapest-best-fit first, fallback on low confidence) is how real systems work. This is the right default mode.

### 4. Two execution modes cover the spectrum

- **Cascade**: cost-conscious, low-latency (production default)
- **Parallel top-K**: accuracy-maximizing (when "be thorough" is requested)

---

## Assessment: Where to Be Careful

### 1. Building five engines is a multi-month undertaking

Each paradigm is a significant system. The hybrid engine alone (BM25 + vector + RRF + reranking) is a multi-phase project. Building all five before validating the routing layer risks over-investment before proving value.

### 2. Query/corpus classification is harder than it looks

Routing signals like `structure_score`, `entity_density`, `query_intent` are not trivial to compute reliably. A regex classifier works for domain routing but may not be precise enough to choose between TOC vs hybrid vs graph. An LLM call adds latency to a system whose value is latency control.

### 3. Confidence calibration across engines is an open problem

"Winner-takes-all by confidence" assumes confidence scores are comparable across engines. A TOC engine's 0.74 and a vector engine's 0.74 don't mean the same thing. Cross-engine calibration needs either:
- Normalized scoring against a shared benchmark
- A learned combiner (lightweight model trained on engine outputs)
- Heuristic blending with citation/groundedness verification

### 4. TOC RAG has a narrow sweet spot

PageIndex-style reasoning excels on single long structured documents (50+ page PDFs). For short markdown docs (like Ciroos's 22-file corpus), hybrid vector RAG will likely outperform because there isn't enough structure to navigate meaningfully.

---

## Refined Strategy: Don't Reinvent Engines, Orchestrate the Best OSS

The key insight: **don't build engines from scratch. Use the best open-source implementation for each paradigm and focus engineering effort on the router, arbiter, and evaluation layer.**

### Best-in-Class OSS Engine per Paradigm

| Paradigm | Best OSS Engine | Why | Alternative |
|----------|----------------|-----|-------------|
| **TOC/Hierarchical** | **PageIndex** | Purpose-built, benchmarked on FinanceBench (98.7%) | LlamaIndex TreeIndex |
| **Hybrid Vector** | **LlamaIndex** + Qdrant/Milvus | Best retrieval speed, hybrid support, reranker integration | Haystack |
| **GraphRAG** | **Microsoft GraphRAG** | Best documented, community-backed | LlamaIndex KG module, Neo4j + LangChain |
| **Agentic/Tool** | **LangChain Agents** or **AutoGen** | Most flexible tool orchestration | LlamaIndex Agents |
| **Long-Context** | Direct LLM call (Claude/GPT) | No framework needed | LlamaIndex SimpleContext |

### What You Actually Build

```
ragrouter/                      # YOUR CODE (the novel part)
  router/
    analyzer.py                 # Query + corpus feature extraction
    planner.py                  # Paradigm selection logic
    arbiter.py                  # Result scoring, merging, verification
    registry.py                 # Engine capability registry (YAML config)
    schemas.py                  # Standard engine interface contracts
  evaluation/
    benchmarks.py               # Run queries through all engines, compare
    calibration.py              # Cross-engine confidence normalization
    metrics.py                  # Recall, MRR, groundedness, cost tracking
  adapters/                     # Thin wrappers to normalize OSS engine outputs
    pageindex_adapter.py        # PageIndex -> standard interface
    llamaindex_adapter.py       # LlamaIndex -> standard interface
    graphrag_adapter.py         # MS GraphRAG -> standard interface
    langchain_adapter.py        # LangChain agents -> standard interface
    direct_llm_adapter.py       # Raw LLM call -> standard interface
```

**You write the router, arbiter, evaluation, and adapters. You don't write the engines.**

The adapters are thin (~100-200 lines each) -- they normalize each OSS engine's input/output to your standard contract.

---

## The Arbiter: Where the Real Value Lives

This is where you should invest the most engineering effort. The arbiter is what turns "run multiple engines" into "get a better answer than any single engine."

### Arbiter Responsibilities

1. **Score each engine's output** on a common scale
2. **Detect low-quality answers** (hallucination, missing citations, off-topic)
3. **Merge complementary answers** when engines found different relevant pieces
4. **Explain the decision** (why this engine's answer was chosen)

### Scoring Dimensions

| Dimension | What It Measures | How to Compute |
|-----------|-----------------|----------------|
| **Groundedness** | Is the answer supported by retrieved context? | Check if answer claims appear in cited passages |
| **Citation coverage** | How much of the answer is backed by citations? | Ratio of cited sentences to total sentences |
| **Query alignment** | Does the answer address what was asked? | Semantic similarity between query and answer |
| **Completeness** | Did the engine find all relevant information? | Number of unique relevant docs retrieved |
| **Confidence** | Engine's self-assessed certainty | Engine-reported score (needs calibration) |
| **Cost** | Tokens consumed, latency | From engine usage report |

### Arbiter Modes

**Mode 1: Winner-Takes-All (default)**
- Score all engine outputs on the dimensions above
- Pick highest weighted score
- Fast, simple, works for most queries

**Mode 2: Merge-and-Verify (for high-stakes queries)**
- Merge claims from top-2 engines
- Run a verification pass: check each claim against cited evidence
- Remove unsupported claims
- Produce a composite answer with stronger grounding

**Mode 3: Consensus Voting (for parallel top-3)**
- If 2/3 engines agree on core claims and citations overlap, prefer that consensus
- Useful when confidence is borderline across engines

### Confidence Calibration Strategy

The hardest part of the arbiter. Three approaches (in order of complexity):

1. **Heuristic normalization**: For each engine, maintain running statistics of its confidence distribution. Normalize to percentiles. Engine A's 0.74 might be its 85th percentile, while Engine B's 0.74 is only its 60th percentile.

2. **Benchmark-anchored calibration**: Run all engines on a shared eval set. Map each engine's raw confidence to actual correctness rates. Build a calibration curve per engine.

3. **Learned combiner**: Train a lightweight model (logistic regression or small neural net) that takes all engines' scores + features and predicts which answer is best. Requires labeled data from production usage.

Start with #1, validate with #2, consider #3 when you have production traffic.

---

## Routing Signals & Planner Logic

### Corpus-Level Signals (computed once at index time)

| Signal | How to Compute | What It Tells You |
|--------|---------------|-------------------|
| `doc_count` | Count of documents | Scale of corpus |
| `avg_doc_tokens` | Average token count per doc | Document length |
| `structure_score` | Heading density (H1/H2/H3 per 1000 tokens) | How structured the docs are |
| `entity_density` | Named entity count per 1000 tokens | Graph-worthiness |
| `doc_types` | Distribution of file types (PDF, md, html) | Format complexity |

### Query-Level Signals (computed per query, must be fast)

| Signal | How to Compute | What It Tells You |
|--------|---------------|-------------------|
| `query_intent` | Regex patterns + optional lightweight LLM | lookup / relationship / synthesis / exploratory |
| `specificity` | Keyword count, named entities in query | Narrow vs broad |
| `multi_hop` | Contains "and", "compare", "relate" patterns | Needs cross-doc reasoning |
| `section_reference` | Contains "section", "chapter", "page", "where" | Needs structural navigation |

### Routing Rules (MVP, deterministic)

```python
def select_engines(corpus_signals, query_signals, budget):
    engines = []

    # TOC engine: structured long docs + section-seeking queries
    if corpus_signals.structure_score > 0.6 and corpus_signals.avg_doc_tokens > 5000:
        if query_signals.section_reference or query_signals.query_intent == "lookup":
            engines.append(("toc", priority=1))

    # Graph engine: entity-dense corpus + relationship queries
    if corpus_signals.entity_density > 0.3 and query_signals.multi_hop:
        engines.append(("graph", priority=2))

    # Hybrid vector: always a good fallback, best for large corpora
    if corpus_signals.doc_count > 100 or not engines:
        engines.append(("hybrid", priority=1 if not engines else 2))

    # Agent: complex synthesis, only if budget allows
    if query_signals.query_intent == "synthesis" and budget.allows_agent:
        engines.append(("agent", priority=3))

    # Long-context: tiny corpus, just stuff it in
    if corpus_signals.total_tokens < 50000 and not engines:
        engines.append(("longcontext", priority=1))

    # Always have at least hybrid as fallback
    if "hybrid" not in [e[0] for e in engines]:
        engines.append(("hybrid", priority=len(engines) + 1))

    return sorted(engines, key=lambda e: e[1])
```

---

## Standard Engine Contract

Every engine adapter must normalize to this interface:

### Request

```json
{
  "query": "How do I onboard an AWS account?",
  "corpus_id": "ciroos-docs",
  "chat_history": [],
  "constraints": {
    "max_latency_ms": 3000,
    "must_cite": true,
    "max_tokens": 2000
  }
}
```

### Response

```json
{
  "engine": "hybrid",
  "answer": "To onboard an AWS account...",
  "citations": [
    {"doc_id": "onboarding/aws.md", "section": "Setup Steps", "span": "lines 12-28"}
  ],
  "trace": {
    "retrieval_method": "bm25+dense+rrf",
    "docs_searched": 22,
    "docs_retrieved": 5,
    "reranking_applied": true
  },
  "scores": {
    "confidence": 0.82,
    "groundedness": 0.91,
    "coverage": 0.75
  },
  "usage": {
    "latency_ms": 820,
    "tokens_in": 3200,
    "tokens_out": 450,
    "estimated_cost_usd": 0.004
  }
}
```

---

## Build Order (Recommended)

### Phase 1: Router Core + Two Engines

Build the router framework with the hybrid engine (your existing vectorstore + BM25 upgrade) and a simple vector engine (your current pipeline) as engines. This validates the routing and arbitration layer with a real A/B comparison.

**What you build:**
- `router/` - analyzer, planner, arbiter, registry, schemas
- `adapters/llamaindex_adapter.py` - wrap your vectorstore hybrid search
- `adapters/direct_llm_adapter.py` - wrap simple vector search (current)
- `evaluation/benchmarks.py` - run eval_dataset_v2 through both engines

**What you validate:**
- Does routing to the right engine actually improve results?
- Does the arbiter correctly pick the better answer?
- What's the latency overhead of routing?

### Phase 2: Add TOC Engine

Integrate PageIndex (or build a lightweight TOC engine using your existing heading-aware chunker). This tests whether TOC routing adds value for your corpus.

**What you add:**
- `adapters/pageindex_adapter.py`
- Routing rules for structure_score-based TOC selection

### Phase 3: Add Graph Engine

Integrate Microsoft GraphRAG for relationship queries. This tests the multi-hop routing path.

**What you add:**
- `adapters/graphrag_adapter.py`
- Entity density analysis in the analyzer

### Phase 4: Arbiter Sophistication

With 3+ engines producing real outputs, refine the arbiter:
- Benchmark-anchored calibration
- Merge-and-verify mode
- Consensus voting

### Phase 5: Evaluation Suite + Publish

- Comprehensive benchmarks across paradigms
- Pareto curves: accuracy vs cost vs latency
- Open-source the router as a pip-installable library

---

## What Makes This Novel

| Existing Systems | RAGRouter |
|-----------------|-----------|
| Route between retrievers (BM25 vs dense) | Route between entire RAG paradigms |
| Single engine per query | Top-K engines with arbitration |
| Fixed pipeline | Adaptive cascade with early exit |
| Engine-specific confidence | Cross-engine calibrated scoring |
| No cost awareness | Budget-constrained routing |

The missing abstraction in the RAG ecosystem is not a better engine -- it's a better way to choose and combine engines. That's what RAGRouter provides.

---

## Key Risk

Building five engines before validating the router. Mitigation: use best OSS for each engine, invest engineering in the router/arbiter/evaluation layer. The adapters are thin. The intelligence is in the routing and arbitration.
