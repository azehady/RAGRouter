# RAGRouter: OSS Engine Landscape (Jan 2026)

Best open-source engines per RAG paradigm for plugging into the RAGRouter.

---

## Paradigm 1: TOC/Hierarchical RAG (Structure-First)

| Engine | Stars | License | GitHub | Key Differentiator |
|--------|-------|---------|--------|-------------------|
| **PageIndex** | 10.4K | MIT | [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) | Vectorless, reasoning-based. Builds JSON tree index (TOC), LLM reasons over structure. 98.7% on FinanceBench. [MCP server](https://github.com/VectifyAI/pageindex-mcp) available. |
| **LlamaIndex TreeIndex** | 46K (framework) | MIT | [run-llama/llama_index](https://github.com/run-llama/llama_index) | `HierarchicalNodeParser` creates coarse-to-fine hierarchy. `AutoMergingRetriever` fetches parent context. Part of broader LlamaIndex framework. |
| **RAGFlow** (TOC mode) | 72K | Apache 2.0 | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | Auto-generates document-level TOC for long-context RAG. Parent-child chunking. Production-ready. |

**Recommendation:** PageIndex for pure structure reasoning. LlamaIndex TreeIndex if you're already in the LlamaIndex ecosystem.

---

## Paradigm 2: Hybrid Vector RAG (BM25 + Dense + Reranker)

### Orchestration Frameworks

| Engine | Stars | License | GitHub | Key Differentiator |
|--------|-------|---------|--------|-------------------|
| **RAGFlow** | 72K | Apache 2.0 | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | All-in-one. Deep doc understanding, fused hybrid retrieval, parent-child chunking, agentic workflows, MCP server. Most feature-complete. |
| **Haystack** | 22K | Apache 2.0 | [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | Production-grade pipelines. Built-in hybrid retrievers, semantic chunking, query expansion. Used by Airbus, Netflix, Apple. |
| **LlamaIndex** | 46K | MIT | [run-llama/llama_index](https://github.com/run-llama/llama_index) | Data framework. 300+ integrations. `QueryPipeline` for DAG orchestration. Broad but less opinionated. |
| **DSPy** | 31.7K | MIT | [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | Programming (not prompting) LLMs. Auto-optimizes prompts/weights. Lowest framework overhead (~3.5ms). Stanford NLP. |
| **FlashRAG** | N/A | N/A | [RUC-NLPIR/FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | Research toolkit. 36 benchmark datasets, 23 SOTA algorithms. Best for reproduction and benchmarking. |

### Vector Databases with Native Hybrid Search

| Database | Stars | License | Lang | GitHub | Key Feature |
|----------|-------|---------|------|--------|-------------|
| **Qdrant** | 27K | Apache 2.0 | Rust | [qdrant/qdrant](https://github.com/qdrant/qdrant) | Dense + sparse + ColBERT. Score-boosting reranking. FastEmbed included. |
| **Milvus** | 40K+ | Apache 2.0 | Go | [milvus-io/milvus](https://github.com/milvus-io/milvus) | Native BM25 + semantic in v2.5. Sub-50ms on billions of vectors. Used by NVIDIA, Salesforce. |
| **Weaviate** | 15.4K | BSD 3 | Go | [weaviate/weaviate](https://github.com/weaviate/weaviate) | Native BM25 + vector hybrid. Relative Score Fusion. Multi-tenancy, RBAC. |

### Rerankers

| Engine | Stars | License | GitHub | Key Differentiator |
|--------|-------|---------|--------|-------------------|
| **FlashRank** | 907 | Apache 2.0 | [PrithivirajDamodaran/FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) | Ultra-lightweight (~4MB). No Torch dependency. ONNX runtime. |
| **rerankers** | 1.2K | Apache 2.0 | [AnswerDotAI/rerankers](https://github.com/AnswerDotAI/rerankers) | Unified API for all reranking models (cross-encoder, ColBERT, API-based). Zero deps by default. |
| **RAGatouille** | 3.7K | N/A | [AnswerDotAI/RAGatouille](https://github.com/AnswerDotAI/RAGatouille) | ColBERT late-interaction retrieval. Token-level matching. Fine-tuning support. |
| **Cohere Rerank v4.0** | N/A | Commercial | API-only | 32K context, 100+ languages. Best commercial option. |
| **BGE-Reranker-v2-m3** | N/A | MIT | [HuggingFace model](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Open-source cross-encoder from BAAI. Strong multilingual baseline. |

**Recommendation:** Haystack or RAGFlow for orchestration + Qdrant for vector DB (you already have it) + FlashRank or rerankers for lightweight reranking.

---

## Paradigm 3: GraphRAG (Knowledge Graph Augmented)

| Engine | Stars | License | GitHub | Key Differentiator |
|--------|-------|---------|--------|-------------------|
| **Microsoft GraphRAG** | 30.5K | MIT | [microsoft/graphrag](https://github.com/microsoft/graphrag) | The reference implementation. Community hierarchy, summaries. 70-80% win rate over naive RAG. v2.7.0. |
| **LightRAG** | 27.6K | MIT | [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | EMNLP 2025 paper. Dual-level retrieval (entities + relationships). Simpler, faster than MS GraphRAG. RAGAS integration. |
| **nano-graphrag** | 2.8K | MIT | [gusye1234/nano-graphrag](https://github.com/gusye1234/nano-graphrag) | ~800 lines. Hackable reimplementation. Three query modes. Neo4j option. |
| **Fast GraphRAG** | N/A | MIT | [circlemind-ai/fast-graphrag](https://github.com/circlemind-ai/fast-graphrag) | PageRank-based. 6x cheaper than MS GraphRAG ($0.08 vs $0.48). Incremental updates. |
| **Neo4j GraphRAG** | N/A | Apache 2.0 | [neo4j/neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python) | Official Neo4j package. `SimpleKGPipeline` for streamlined setup. Enterprise graph infra. |

**Recommendation:** LightRAG for simplicity and speed. Microsoft GraphRAG for completeness. nano-graphrag for hacking/learning.

---

## Paradigm 4: Agentic/Tool RAG (Iterative Retrieve-Read-Refine)

| Engine | Stars | License | GitHub | Key Differentiator |
|--------|-------|---------|--------|-------------------|
| **AutoGen** (Microsoft) | 50.4K | CC-BY-4.0 | [microsoft/autogen](https://github.com/microsoft/autogen) | Multi-agent conversations. Migrating to Microsoft Agent Framework. |
| **CrewAI** | 43.4K | MIT | [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | Role-based multi-agent crews. Built-in memory. Used by IBM, Microsoft, Walmart. |
| **smolagents** (HuggingFace) | 25K | Apache 2.0 | [huggingface/smolagents](https://github.com/huggingface/smolagents) | ~1K lines. Code-first agents (writes Python, not JSON). 30% fewer steps. Sandboxed. |
| **LangGraph** | 23.9K | MIT | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | Graph-based stateful workflows. Lowest latency/token usage in benchmarks. Used by Klarna, Replit. |
| **Strands Agents** (AWS) | 3K+ | Apache 2.0 | [strands-agents/sdk-python](https://github.com/strands-agents/sdk-python) | Model-driven. Built-in MCP. Powers Amazon Q Developer. 1M+ downloads in <4 months. |
| **LlamaIndex Agents** | 46K (framework) | MIT | [run-llama/llama_index](https://github.com/run-llama/llama_index) | ReAct agents, tool use. Tight retriever integration. |

**Recommendation:** LangGraph for complex stateful workflows. smolagents for lightweight code-first agents. CrewAI for multi-agent teams.

---

## Paradigm 5: Long-Context / Contextual RAG

| Approach | Source | GitHub / Link | Key Differentiator |
|----------|--------|---------------|-------------------|
| **Anthropic Contextual Retrieval** | Anthropic | [anthropics/anthropic-retrieval-demo](https://github.com/anthropics/anthropic-retrieval-demo) | Prepends chunk-specific context before embedding. Reduces failed retrievals by 49% (67% with reranking). |
| **Late Chunking** (Jina AI) | Paper | [arxiv.org/abs/2409.04701](https://arxiv.org/abs/2409.04701) | Embeds full doc first, then chunks embeddings. Preserves cross-boundary context. ~30 lines to implement. |
| **LongRAG** | TIGER-AI-Lab | [TIGER-AI-Lab/LongRAG](https://github.com/TIGER-AI-Lab/LongRAG) | 4K-token units (30x longer). Recall@1=71% on NQ (up from 52%). Research code. |
| **RAGFlow** (long-context mode) | InfiniFlow | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | Auto TOC generation + parent-child chunking + RAPTOR. Most production-ready. |
| **LLMLingua** (Microsoft) | Microsoft | [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua) | Prompt compression to reduce tokens while preserving key info. |

**Recommendation:** Anthropic contextual retrieval for chunk enrichment (easy to implement). RAGFlow for production. Late chunking for efficiency.

---

## Evaluation Frameworks

| Framework | Stars | License | GitHub | Best For |
|-----------|-------|---------|--------|----------|
| **RAGAS** | 12.4K | Apache 2.0 | [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | Research, experimentation. Reference-free. DSPy optimizer. Mentioned by OpenAI. |
| **Arize Phoenix** | 7.8K | Elastic v2 | [Arize-ai/phoenix](https://github.com/Arize-ai/phoenix) | Observability + tracing. OpenTelemetry. Self-hostable. Framework agnostic. |
| **DeepEval** | 5K+ | Apache 2.0 | [confident-ai/deepeval](https://github.com/confident-ai/deepeval) | CI/CD testing. 50+ metrics. PyTest integration. Red teaming. |
| **TruLens** | 3K | MIT | [truera/trulens](https://github.com/truera/trulens) | Production monitoring. Runtime feedback functions. Snowflake-backed. |
| **BERGEN** | N/A | CC-BY-NC-SA | [naver/bergen](https://github.com/naver/bergen) | Academic benchmarking. 20+ retrievers, YAML config. EMNLP 2024. |
| **FlashRAG** | N/A | N/A | [RUC-NLPIR/FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | Reproduction. 36 datasets, 23 algorithms. WWW2025. |

**Recommendation:** RAGAS for eval development. Phoenix for observability. DeepEval for CI/CD.

---

## Existing RAG Router/Orchestrator Projects

| Project | Stars | GitHub | Key Insight |
|---------|-------|--------|-------------|
| **RAGFlow** | 72K | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | Most complete all-in-one. Not a paradigm router but has configurable retrieval strategies. |
| **UltraRAG** (OpenBMB) | N/A | [OpenBMB/UltraRAG](https://github.com/OpenBMB/UltraRAG) | MCP-native architecture. YAML-configured inference orchestration (sequential, loop, conditional). |
| **OpenMined RAG Router** | N/A | [OpenMined/rag-router-demo](https://github.com/OpenMined/rag-router-demo) | Event-driven RAG routing with vector search, embedding management. Demo-level. |
| **LlamaIndex RouterQueryEngine** | (part of 46K) | [run-llama/llama_index](https://github.com/run-llama/llama_index) | Routes to different query engines via LLM selector. Closest to paradigm routing but not ensemble. |

**Gap confirmed:** No widely-adopted project dynamically routes between entire RAG paradigms (TOC vs vector vs graph vs agentic) and ensembles results. This is the opportunity.

---

## Recommended Stack for RAGRouter Adapters

| Paradigm | Primary Engine | GitHub | Alternative | GitHub | Adapter Complexity |
|----------|---------------|--------|-------------|--------|-------------------|
| TOC/Hierarchical | **PageIndex** | [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) | LlamaIndex TreeIndex | [run-llama/llama_index](https://github.com/run-llama/llama_index) | Low (~150 lines) |
| Hybrid Vector | **Haystack** + Qdrant + FlashRank | [deepset-ai/haystack](https://github.com/deepset-ai/haystack) | LlamaIndex + Qdrant | [qdrant/qdrant](https://github.com/qdrant/qdrant) | Medium (~200 lines) |
| GraphRAG | **LightRAG** | [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) | Microsoft GraphRAG | [microsoft/graphrag](https://github.com/microsoft/graphrag) | Medium (~200 lines) |
| Agentic/Tool | **LangGraph** | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | smolagents | [huggingface/smolagents](https://github.com/huggingface/smolagents) | Medium (~200 lines) |
| Long-Context | **Direct LLM** + contextual retrieval | [anthropics/anthropic-retrieval-demo](https://github.com/anthropics/anthropic-retrieval-demo) | RAGFlow | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | Low (~100 lines) |
| **Evaluation** | **RAGAS** + Phoenix | [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | DeepEval | [confident-ai/deepeval](https://github.com/confident-ai/deepeval) | Shared across all |

Total adapter code: ~850 lines. The rest is router, arbiter, evaluation -- that's what you build.
