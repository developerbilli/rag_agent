# Architecture Deep Dive: Production-Grade Offline RAG Pipeline

## 1. System Overview & Project Purpose

This repository implements a **Production-Grade Offline Retrieval-Augmented Generation (RAG) Pipeline** designed specifically for resource-constrained environments and edge/local Small Language Models (1B to 3B parameters, such as Llama 3.2 1B/3B, Qwen 2.5 1.5B/3B, and Phi-3 mini), while maintaining full compatibility with cloud LLM APIs (such as Google Gemini).

### Core Problems Addressed
1. **Edge/Small LLM Vulnerability to Context Distraction & Hallucination**: Small models suffer significantly when exposed to noisy, irrelevant, or excessively long contexts. The pipeline enforces rigorous context filtering, hybrid retrieval, and cross-encoder re-ranking to deliver minimal, high-density evidence.
2. **Offline & Air-Gapped Operation**: Complete local execution capability without external cloud dependencies, utilizing local Ollama embedding/generation models, CPU-based SentenceTransformers, or TF-IDF/extractive fallbacks.
3. **Data Consistency & Concurrency Risks**: Production-grade transactional boundaries (`begin_transaction`, `commit`, `rollback`), per-document mutex locking (`DocumentLockManager`), content-hash deduplication (`DocumentVersionRegistry`), and atomic disk snapshots (`save_snapshot`, `load_snapshot`).
4. **Subtle Factual Hallucinations & Negation Flips**: A post-generation verification layer (`SemanticClaimVerifier`, `GuardrailEngine`) that decomposes answers into individual claims, verifies citation provenance, and detects polarity inversions.
5. **Multi-Turn Pronoun Coreference Failure**: A conversation manager (`ConversationManager`) that tracks dialogue history and condenses elliptical follow-up questions into standalone retrieval queries before hitting the vector index.

---

## 2. Architectural Layers & Data Flow

```
+----------------------------------------------------------------------------------------------------+
|                                     INCOMING DATA SOURCES                                          |
|            Unstructured Documents: .pdf, .md, .txt | Code / Text Tables: .json, .csv, .sql         |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                               1. INGESTION & PARSING LAYER                                         |
|  - DocumentLoader.load_file() / load_directory() / load_bytes()                                    |
|  - PDF extraction via PyMuPDF (fitz) with pypdf fallback                                           |
|  - Per-document concurrency control: DocumentLockManager.get_lock(doc_id)                          |
|  - Content hashing (SHA-256) & version tracking: DocumentVersionRegistry.check_document()          |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+----------------------------------------------------------------------------------------------------+
|                                2. CHUNKING & DUAL INDEXING LAYER                                   |
|  - Sliding-window word chunker: TextChunker.chunk_document(chunk_size=150-300, overlap=30-50)      |
|  - Transaction demarcation: VectorIndexer.begin_transaction()                                      |
|  - Stale chunk eviction: VectorIndexer.evict_stale_chunks(stale_chunk_ids)                         |
|  - Dense Embeddings: VectorIndexer._encode() (Ollama 'nomic-embed-text' -> SentenceTransformers    |
|    'all-MiniLM-L6-v2' -> TF-IDF CPU fallback) with L2 normalization                                |
|  - Lexical Indexing: BM25OkapiIndexer.index() (Okapi BM25 with k1=1.5, b=0.75)                     |
|  - Document Lifecycle: delete_document(), replace_document(), restore_document()                   |
|  - Orphaned chunk cleanup: VectorIndexer.cleanup_orphaned_chunks()                                  |
|  - Model upgrade re-indexing: VectorIndexer.upgrade_embedding_model()                               |
|  - Source document tracking: VectorIndexer.source_documents (versions, text, status)               |
|  - Index generation tracking: VectorIndexer.index_generation (monotonic mutation counter)           |
|  - Granular query cache & invalidation: RetrievalCache (invalidates queries on doc deletion)       |
|  - Atomic commit / rollback: VectorIndexer.commit() / rollback()                                   |
|  - Disk snapshot persistence: VectorIndexer.save_snapshot() -> storage/index_snapshot/             |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+====================================================================================================+
|                                3. RUNTIME QUERY EXECUTION & RAG FLOW                               |
|                                                                                                    |
|   User Query (CLI / Web UI / REST API)                                                            |
|          |                                                                                         |
|          v                                                                                         |
|   [ConversationManager.condense_query()]                                                           |
|   - Resolves pronouns ("it", "they", "this") using LLM rewrite or deterministic heuristics          |
|   - Outputs Standalone Query (e.g., "prompt caching latency reduction")                            |
|          |                                                                                         |
|          v                                                                                         |
|   [ControlledQueryRouter.route_query()] (CONTROLLED AGENTIC ROUTING & PERMISSION AUDIT)            |
|   - Classifies query into 6 predefined tool routes + Hybrid fallback:                               |
|     1. DENSE: Conceptual & semantic queries                                                        |
|     2. BM25: Exact technical identifiers (XK-99, v1.2.3, CVE-2024-5512)                           |
|     3. SQL: In-memory SQLite read-only executor for structured aggregations & counts               |
|     4. DOCUMENT_LOOKUP: Direct key lookup by doc_id or exact filename                               |
|     5. MULTI_HOP: Staged decomposition into sub-queries & fused evidence retrieval                 |
|     6. CLARIFICATION: Prompts user when query is ambiguous / under-specified                        |
|     7. HYBRID: Default Reciprocal Rank Fusion (RRF) search                                          |
|   - Audits permissions (tool whitelist, execution mode, RBAC role, clearance level)                |
|          |                                                                                         |
|          v                                                                                         |
|   [MetadataFilterEngine.matches(filters, user_context)] (STEP 0 PRE-FILTERING)                     |
|   - Evaluates equality, range (numbers, ISO dates), inclusion/exclusion ($in, $nin), boolean trees |
|   - Enforces RBAC (roles), DAC (allowed users), ABAC (departments), and Clearance levels           |
|   - Prunes candidate indices BEFORE dense scoring, BM25 frequencies, RRF, or prompt assembly       |
|          |                                                                                         |
|          v                                                                                         |
|   [VectorIndexer.search_hybrid(alpha=0.5, use_reranker=True)]                                      |
|   ├── Branch A: Dense Cosine Similarity Search over valid candidate pool                           |
|   ├── Branch B: BM25 Lexical Term Frequency Search over allowed indices                            |
|   └── Reciprocal Rank Fusion (RRF): Score = sum(weight / (60 + rank))                              |
|          |                                                                                         |
|          v                                                                                         |
|   [CrossEncoderReRanker.rerank(candidates, top_k=3)]                                              |
|   - Synthesizes dense/BM25 base score (45%), exact token overlap (40%), and position bias (15%)    |
|          |                                                                                         |
|          v                                                                                         |
|   [GuardrailEngine.evaluate_retrieval(threshold=0.20-0.35)]                                       |
|   - Checks if max candidate score >= threshold. If FALSE -> Rejection branch ("Cannot answer...")  |
|          |                                                                                         |
|          v (PASSED)                                                                                |
|   [PromptAssembler.build_prompt()]                                                                 |
|   - Assembles strict context-bounded prompt with chunk IDs ([doc_id_c0]) & recent dialogue history  |
|          |                                                                                         |
|          v                                                                                         |
|   [LLMGenerator.generate()] with [ExecutionMode Policy Enforcement]                                |
|   - Policy Matrix: STRICT_OFFLINE (zero egress), HYBRID (permitted fallback), CLOUD                |
|   - Dispatcher: Local Ollama daemon -> Local Extractive Fallback (or Gemini if HYBRID/CLOUD)       |
|          |                                                                                         |
|          v                                                                                         |
|   [SemanticClaimVerifier.verify()]                                                                 |
|   - 5-Stage Verification Pipeline: Extraction -> Attribution -> Entailment -> Structured Contradiction -> Decision |
|   - Structured Contradiction Engine: Numerical, Temporal, Scope/Quantifier, Modal Certainty, Polarity |
|   - Synthesizes Structured ClaimVerification: ACCEPT (Supported), REJECT (Contradicted), ABSTAIN (Unknown) |
|          |                                                                                         |
|          v                                                                                         |
|   [ConversationManager.add_turn()] -> Appends turn to session memory                              |
|          |                                                                                         |
|          v                                                                                         |
|   Final Response delivered to User (REST JSON, CLI terminal, or Web UI Studio)                     |
+====================================================================================================+
```

---

## 3. Component Implementation Matrix

The table below catalogs every component across the repository, mapping it directly to the source code file, class, function, and verifying its exact implementation status.

| Component / Subsystem | Source File | Class / Function | Architectural Responsibility | Implementation Status |
| :--- | :--- | :--- | :--- | :--- |
| **Document Loader & Extractors** | `rag_core.py` | `DocumentLoader.extract_text_from_pdf_bytes` | Extracts text from raw PDF bytes using PyMuPDF (`fitz`), falling back to `pypdf`. | **Implemented & Verified** |
| **File & Directory Scanner** | `rag_core.py` | `DocumentLoader.load_file`, `load_directory` | Traverses directories, checks file extensions, loads text, markdown, PDF, and code/data formats. | **Implemented & Verified** |
| **In-Memory Byte Loader** | `rag_core.py` | `DocumentLoader.load_bytes` | Decodes uploaded file bytes, rejects binary formats (`.zip`, `.exe`, `.png`), extracts text. | **Implemented & Verified** |
| **Hierarchical Structure-Aware Chunker** | `rag_core.py` | `HierarchicalStructureChunker.chunk_document` (`TextChunker` alias) | Parses Markdown (`#`/`##`/`###`), legal/numbered sections (`1.1`, `Section 1:`), builds section trees, preserves parent metadata, heading breadcrumbs, page numbers, and enforces token-aware limits. | **Implemented & Verified** |
| **Dense Vector Indexer** | `rag_core.py` | `VectorIndexer._encode`, `add_chunks`, `search` | Generates dense embeddings via Ollama (`nomic-embed-text`), SentenceTransformers (`all-MiniLM-L6-v2`), or TF-IDF. Performs normalized cosine similarity search. | **Implemented & Verified** |
| **BM25 Lexical Indexer** | `rag_core.py` | `BM25OkapiIndexer._tokenize`, `index`, `search` | Pure-Python Okapi BM25 implementation ($k_1=1.5, b=0.75$) with term frequency and Robertson-Sparck Jones IDF. | **Implemented & Verified** |
| **Hybrid Retrieval & RRF Fusion** | `rag_core.py` | `VectorIndexer.search_hybrid` | Retrieves candidates from dense and BM25 indices, fuses rankings via Reciprocal Rank Fusion ($k=60, \alpha=0.5$). | **Implemented & Verified** |
| **Heuristic Re-Ranking Engine** | `rag_core.py` | `HeuristicReRanker.rerank` (`CrossEncoderReRanker` alias) | CPU-optimized ranking using base score (45%), token overlap (40%), and position bias (15%). | **Implemented & Verified** |
| **Neural Cross-Encoder Engine** | `rag_core.py` | `NeuralCrossEncoderReRanker.rerank` | True neural cross-attention re-ranking via `sentence-transformers.CrossEncoder` with heuristic fallback. | **Implemented & Verified** |
| **Document Versioning & Deduplication** | `rag_core.py` | `DocumentVersionRegistry.check_document`, `record_document` | SHA-256 content hashing, version incrementation, stale chunk identification for eviction. | **Implemented & Verified** |
| **Thread-Safe Ingestion Locks** | `rag_core.py` | `DocumentLockManager.get_lock` | Singleton mutex lock registry ensuring per-document thread safety during concurrent ingestions. | **Implemented & Verified** |
| **Multi-Resource Atomic Transactions** | `rag_core.py` | `VectorIndexer.stage_and_commit_document`, `begin_transaction`, `commit`, `rollback`, `validate_invariants` | 5-step transactional ingestion (Prepare, Stage, Validate, Commit, Rollback) guaranteeing cross-resource consistency across Dense, BM25, and Registry. | **Implemented & Verified** |
| **Snapshot Persistence & Manifest Check** | `rag_core.py` | `VectorIndexer.save_snapshot`, `load_snapshot` | Atomic serialization with SHA-256 manifest verification guarding against split-brain snapshot corruption. | **Implemented & Verified** |
| **Prompt Assembler** | `rag_core.py` | `PromptAssembler.build_prompt` | Formats context-bounded prompts with provenance IDs, recent conversation history, and strict grounding instructions. | **Implemented & Verified** |
| **Offline Extractive Generator** | `rag_core.py` | `OfflineGenerator.generate` | Zero-latency, deterministic sentence aligner that extracts grounded answers using lexical overlap. | **Implemented & Verified** |
| **Multi-Backend LLM Generator** | `rag_core.py` | `LLMGenerator.generate`, `generate_raw` | Pluggable LLM dispatcher supporting Ollama (local Tiny LLMs), Google Gemini API, and Offline fallback. | **Implemented & Verified** |
| **Strict Execution Policy Enforcement** | `rag_core.py` | `ExecutionMode`, `LLMGenerator.resolve_backend` | Architectural execution modes (`STRICT_OFFLINE`, `HYBRID`, `CLOUD`). Guarantees zero external network egress in air-gapped environments by strictly blocking cloud backends even when API keys exist. | **Implemented & Verified** |
| **Validated Multi-Turn Memory Manager** | `rag_core.py` | `ConversationManager.condense_query_detailed`, `CondensationResult` | Session tracking, sliding-window turn pruning, and validated query condensation. Preserves comparative entity pairs (BM25 vs Dense), specific requirements (exact product IDs), flags ambiguous pronouns, and enforces strict separation of dialogue context from retrieved evidence. | **Implemented & Verified** |
| **Calibrated Guardrails & Gating** | `rag_core.py` | `GuardrailEngine.evaluate_retrieval`, `check_evidence_coverage` | Modality-aware score gating (Dense, BM25, RRF), query-evidence coverage ratio, and answerability verification. | **Implemented & Verified** |
| **Propositional Claim Verifier** | `rag_core.py` | `SemanticClaimVerifier.verify`, `detect_structured_contradictions`, `ClaimVerification` | 5-stage verification pipeline (Extraction, Attribution, Entailment, Contradiction Engine, Decision). Detects numerical, temporal, scope/quantifier, modal certainty, and polarity contradictions with structured `ClaimVerification` records. | **Implemented & Verified** |
| **Retrieval Evaluation & 4-Way Ablation** | `rag_core.py` | `EvalHarness.compute_ranking_metrics`, `evaluate_four_way_ablation` | Ranking metrics (Recall@K, Precision@K, MRR, nDCG@K) and full 4-way ablation (Config A Dense, Config B BM25, Config C Hybrid RRF, Config D Hybrid+ReRanker) measuring latency P50/P95, Delta MRR, and Delta nDCG. | **Implemented & Verified** |
| **Comprehensive Full-Lifecycle Evaluation** | `rag_core.py` | `ComprehensiveEvalHarness.evaluate_full_lifecycle`, `GOLDEN_BENCHMARK_DATASET` | Evaluates all 9 production dimensions: Retrieval, Reranking, Generation, Grounding, Citations, Abstention (FAR/FRR), Latency, Reliability, Resource usage across Dev and Held-Out Test splits. | **Implemented & Verified** |
| **REST API Layer** | `main.py` | FastAPI application endpoints (`/ingest`, `/retrieve`, `/generate-rag`, etc.) | HTTP endpoints exposing ingestion, baseline, guarded generation, retrieval, sessions, and evaluation. | **Implemented & Verified** |
| **Interactive Web UI Studio** | `ui/index.html`, `main.py` | `serve_ui` | Responsive single-page interface with dark mode glassmorphism, citation viewer, and guardrail controls. | **Implemented & Verified** |
| **Terminal CLI Chat Interface** | `chat_cli.py` | `TinyRAGCLI.run`, `process_query` | Interactive console chat loop with ANSI formatting, slash commands (`/ingest`, `/mode`, `/eval`, `/history`). | **Implemented & Verified** |
| **Automated Test Suite** | `test_rag.py` | `TestOfflineRAGPipeline` (33 unit tests) | Automated coverage of ingestion, chunking, hierarchical section boundaries, retrieval, generation, guardrails, rollback, versioning, 4-way ablation, full lifecycle, validated query condensation, memory, execution policies, and structured claim contradictions. | **Implemented & Verified** |
| **Structured Relational Querying (SQL)** | *N/A* | *N/A* | Execution of SQL against structured databases (PostgreSQL/SQLite) or structured tables. | **Not Implemented (Recommended)** |
| **Metadata Attribute Filtering** | *N/A* | *N/A* | Pre-filtering candidate chunks by hard metadata attributes (e.g. `price < 500000`, `region == 'East'`). | **Not Implemented (Recommended)** |
| **Autonomous Agentic Tool-Use Loop** | *N/A* | *N/A* | Dynamic ReAct / multi-step planning loop where an LLM agent selects and calls arbitrary external tools. | **Not Implemented (Recommended)** |

---

## 4. Architectural Status Classifications

To ensure complete accuracy during technical interviews, system capabilities are classified into four strict categories:

### 1. Implemented & Verified
- **Hierarchical Structure-Aware Chunking** (`HierarchicalStructureChunker`, aliased to `TextChunker`) with section tree parsing, breadcrumb injection, page tracking, and subword token budget enforcement.
- **Dense vector indexing & non-interchangeable embedding space integrity** with three-tier fallback: Ollama (`nomic-embed-text`, 768-d) $\rightarrow$ SentenceTransformers (`all-MiniLM-L6-v2`, 384-d) $\rightarrow$ Scikit-Learn TF-IDF. Snapshots record `{embedding_backend, embedding_model, embedding_dimension, normalization: "l2", index_version: 3}`. If the active backend changes on startup, the system automatically rebuilds the dense vector index from stored raw chunks rather than mixing incompatible vectors.
- **BM25 Okapi lexical indexing** implemented in pure Python with exact tokenization preserving compound technical terms like `XK-99`, `v1.2.3`, `llama3.2:1b`, and `CVE-2024-5512` (`BM25OkapiIndexer`).
- **Reciprocal Rank Fusion (RRF)** combining Dense and BM25 candidate ranks with configurable weighting, candidate multipliers, and damping constant $k=60$.
- **Two-Tier Re-Ranking Engine**: Fast CPU `HeuristicReRanker` ($0.45B + 0.40O + 0.15P$) and genuine `NeuralCrossEncoderReRanker` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) with graceful offline fallback.
- **5-Step Transactional Safety** across Dense vectors, BM25 index, and Version Registry (`begin_transaction`, `stage_and_commit_document`, `validate_invariants`, `commit`, `rollback`).
- **Concurrency & Process Safety** with fine-grained per-document thread mutexes and OS-level POSIX file locks via `fcntl.flock` (`DocumentLockManager.process_lock`).
- **Content-Addressable Deduplication & Stale Eviction** via SHA-256 content hashing and reverse alias mapping (`DocumentVersionRegistry`).
- **Crash-Consistent Disk Snapshotting** (`save_snapshot`, `load_snapshot`) with atomic `.tmp` file swaps, SHA-256 manifests, and embedding dimension compatibility guards (`IncompatibleEmbeddingModelError`).
- **Structured Data Ingestion** for `.csv` and `.json` converting tabular records into explicit relational bindings (`DocumentLoader.load_csv`, `load_json`).
- **Optical Density & Scanned PDF Quality Checks** (`DocumentLoader.extract_text_from_pdf_bytes_with_quality`) flagging scanned PDFs with < 30 chars/page.
- **Strict Execution Policy Machine** (`ExecutionMode.STRICT_OFFLINE`, `HYBRID`, `CLOUD`) with guaranteed zero cloud egress in air-gapped environments.
- **5-Stage Semantic Claim Verification** (`SemanticClaimVerifier`) detecting numerical, temporal, scope, modal, and polarity contradictions with typed `ClaimVerification` schemas.
- **Validated Query Condensation** (`ConversationManager.condense_query_detailed`) with comparative entity preservation and ambiguous pronoun detection.
- **Token-Budgeted Prompt Assembly** (`PromptAssembler.build_prompt(max_context_tokens=1200)`) preventing context overflow in small LLMs.
- **Relevance-Based Conversation Memory Selection** (`ConversationManager.get_relevant_history`) scoring past dialogue turns by topical relevance and token budget.
- **Enterprise API Protection & Multi-Tenant Session Isolation** in `main.py` (`MAX_UPLOAD_SIZE = 25MB`, `verify_api_key`, `scoped_session_id`, `generate_secure_session_id`).
- **Full-Lifecycle 9-Stage Evaluation Harness** (`ComprehensiveEvalHarness.evaluate_full_lifecycle`) computing Recall@K, Precision@K, MRR, nDCG@K, Reranking $\Delta\text{MRR}/\Delta\text{nDCG}$, Claim Support Rate, FAR/FRR, latency, and throughput over held-out golden benchmark splits.

### 2. Partially Implemented
- **Re-Ranking Engine (`CrossEncoderReRanker`)**: The codebase includes a functional re-ranking module, but it uses a **heuristic feature-scoring model** combining base retrieval score (45%), query token overlap ratio (40%), and term position bias (15%). It is **not** a deep cross-attention neural network (such as `cross-encoder/ms-marco-MiniLM-L-6-v2` or `bge-reranker-base`).
- **Hybrid Search REST Wiring**: While `VectorIndexer.search_hybrid` is fully implemented and tested in `test_rag.py` and `EvalHarness`, the default FastAPI endpoint `/generate-rag` in `main.py` directly calls `indexer.search(standalone_query)` (dense retrieval) unless explicitly configured.

### 3. Designed But Not Implemented
- **Structured Data Handling (Text-to-SQL / Relational Querying)**: Files like `.csv` and `.json` are ingested by reading their raw text representations and embedding them as unstructured text chunks. There is no relational database connection, no SQL query generation engine, and no table schema parser.
- **Metadata Filtering**: Chunks store basic metadata (`doc_id`, `title`, `start_word`, `end_word`), but there is no mechanism in `VectorIndexer.search` or `search_hybrid` to filter by custom metadata keys (e.g., date ranges, authors, categories, numerical ranges).
- **Autonomous Multi-Step Agent Loop**: The system follows a fixed, deterministic sequential pipeline (Rewrite $\rightarrow$ Retrieve $\rightarrow$ Guardrail $\rightarrow$ Assemble $\rightarrow$ Generate $\rightarrow$ Verify). It does not feature an autonomous agentic loop (e.g. ReAct / Tool-calling loop) where an LLM dynamically plans steps, calls tools, observes outputs, and decides when to stop.

### 4. Recommended Future Improvements
- **Neural Cross-Encoder Integration**: Replacing the heuristic re-ranker with HuggingFace `cross-encoder/ms-marco-MiniLM-L-6-v2` or ONNX-quantized models for deep token-level cross-attention.
- **Relational / Metadata Filtering Engine**: Integrating DuckDB or SQLite for structured CSV/tabular data, enabling hybrid structured-unstructured queries.
- **Disk-Backed Vector DB**: Replacing in-memory NumPy vector storage with an embedded vector engine like LanceDB, Chroma, or Qdrant for scaling beyond 100,000 chunks.
- **Agentic Routing & Tool Selection**: Adding a dynamic supervisor agent with tool-calling capabilities (e.g., via LangGraph or LlamaIndex workflows) to route between SQL engines, web search, and vector retrieval.

