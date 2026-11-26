# Production-Grade Offline RAG Pipeline | Hybrid Search, RRF, Cross-Encoder Reranking

A high-performance, offline **Retrieval-Augmented Generation (RAG)** pipeline optimized for local/edge small language models (1B to 3B parameters such as Llama 3.2 1B/3B, Qwen 2.5 1.5B/3B, and Phi-3 mini) as well as cloud models like Gemini.

### Key Highlights
- **Architected an offline RAG pipeline** combining BM25 keyword retrieval and dense-vector retrieval with Reciprocal Rank Fusion (RRF), followed by cross-encoder reranking.
- **Implemented transactional rollback and thread-safe ingestion** to improve reliability and protect index consistency across concurrent uploads and partial failures.
- **Added propositional claim verification** to check generated claims against retrieved evidence and reduce unsupported answers and polarity contradictions.
- **Integrated multi-turn conversational memory** with contextual query condensation to resolve pronoun coreference failures on resource-constrained edge devices.
- **Built Document Security & Indirect Prompt Injection Defenses** (`PromptInjectionGuard`) with XML quarantine boundaries (`<untrusted_evidence>`), delimiter breakout escaping, tool execution firewalls, and pre-retrieval access control.

### Architecture Phases
- **Phase 1: Baseline** (Naive generation without retrieval)
- **Phase 2: Ingestion & Dual Indexing** (Sliding-window chunker, dense vector embeddings, Okapi BM25, `.pdf`, `.md`, `.txt` file/folder loader)
- **Phase 3: Hybrid Retrieval & Generation** (BM25 + Dense RRF fusion, Cross-Encoder reranking, context-bounded prompt assembler, Recall@K eval harness)
- **Phase 4: Guardrails & Semantic Verification** (Confidence threshold filtering, propositional claim verification & polarity contradiction checks)
- **Phase 5: Conversation Memory & Query Condensation** (Multi-turn session tracking, coreference resolution, sliding window eviction)
- **Phase 6: Security & Controlled Query Routing** (Indirect prompt injection quarantine, delimiter escaping, pre-retrieval RBAC/ABAC, 6-route deterministic query router)
- **Interactive Web UI Studio** (Dark mode glassmorphism UI served directly by FastAPI)
- **Terminal CLI Chat Loop** (`chat_cli.py` with colored ANSI output and slash commands)

### 📚 Technical Documentation & Interview Guides
- **[20 Core Production RAG Interview Q&A](file:///Users/anusurya/Desktop/projects/RAG%20pipeline%20for%20tiny%20LLM/20_CORE_RAG_INTERVIEW_QNA.md)**: Standalone master guide answering the 20 foundational Staff/Senior interview questions on offline RAG, RRF calculations, cross-encoder latency, transactional rollbacks, thread safety, polarity inversions, and debugging.
- **[RAG Architectural Deep Dive (7 Core Questions)](file:///Users/anusurya/Desktop/projects/RAG%20pipeline%20for%20tiny%20LLM/RAG_ARCHITECTURAL_DEEP_DIVE_QNA.md)**: In-depth production answers on RAG vs. Fine-tuning, chunk size dynamics, hybrid retrieval, retrieval evaluation, hallucination shields, fallback cascades, and debugging runbooks.
- **[Master Interview Preparation Q&A Guide](file:///Users/anusurya/Desktop/projects/RAG%20pipeline%20for%20tiny%20LLM/INTERVIEW_PREP_QNA.md)**: Master interview preparation document containing 50+ end-to-end questions, architectural deep dives, and production troubleshooting scenarios.
- **[Production RAG Guide](file:///Users/anusurya/Desktop/projects/RAG%20pipeline%20for%20tiny%20LLM/PRODUCTION_RAG_GUIDE.md)**: Architectural blueprints and implementation details across the 10 production RAG questions.

---

## ⚡ Quick Start

### 1. Launch the Web UI Studio
```bash
uvicorn main:app --reload --port 8000
```
Open **[http://localhost:8000/](http://localhost:8000/)** in your browser.
From the Web UI, you can:
- **Chat** with your documents in real-time.
- **Inspect Citations** by clicking on any source tag `[doc_id_c0]`.
- **Drag and Drop** `.pdf`, `.md`, or `.txt` files to ingest them instantly.
- **Toggle Guardrails** and tune similarity thresholds.
- **Compare side-by-side** Naive Baseline vs Guarded RAG.
- **Run Recall@K Benchmark** with a single click.

### 2. Launch the Terminal CLI Chat Loop
Prefer working inside the terminal? Run:
```bash
python3 chat_cli.py
```
Commands inside the CLI:
- Ask any question directly (e.g. `What is quantization?`)
- `/ingest sample_data` or `/ingest /path/to/my_notes.pdf`
- `/mode ollama` or `/mode auto` or `/mode offline`
- `/topk 3`
- `/threshold 0.25`
- `/eval`
- `/status`
- `/help`

### 3. Run Automated Unit Tests
```bash
python3 -m unittest test_rag.py -v
```

---

## 🧩 Architecture Overview

```
                      ┌──────────────────────────────────────────────┐
                      │             Raw Document Inputs              │
                      │       (.pdf, .md, .txt, or API JSON)         │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │          DocumentLoader & TextChunker        │
                      │  - Sliding window (size=150, overlap=30)     │
                      │  - Generates chunk IDs (e.g. doc1_c0)        │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │                VectorIndexer                 │
                      │  - Dense vectors via SentenceTransformer     │
                      │  - Automatic CPU/TF-IDF fallback             │
                      │  - Normalized Cosine Similarity Dot-Product   │
                      └──────────────────────┬───────────────────────┘
                                             │
      User Query ────────────────────────────┤
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │          Top-K Vector Search Engine          │
                      │       Retrieves Top-K candidate chunks       │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │           Phase 4: Guardrail Check           │
                      │  - Max similarity < threshold (e.g. 0.20)?   │
                      │  - YES ──> "Cannot answer based on context"  │
                      └──────────────────────┬───────────────────────┘
                                             │ (Passed)
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │               PromptAssembler                │
                      │  - Injects strict context boundary rules     │
                      │  - Embeds source chunk citation tags         │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │                 LLMGenerator                 │
                      │  - Ollama (Llama 3.2 1B, Qwen 2.5, Phi-3)    │
                      │  - Google Gemini 1.5/2.0 Flash (Cloud API)   │
                      │  - Built-in Offline Extractive Fallback      │
                      └──────────────────────────────────────────────┘
```

---

## 🧠 Connecting a Real LLM

### Option A: Local LLM via Ollama (Recommended for Offline/Privacy)
Ollama runs locally on your Mac with Apple Silicon / Metal acceleration:
```bash
# 1. Meta's Llama 3.2 3B (Fast general assistant, ~2.0 GB)
ollama pull llama3.2:3b

# 2. Alibaba's Qwen 2.5 Coder 7B (Premier coding, reasoning & structured JSON, ~4.7 GB)
ollama pull qwen2.5-coder:7b

# (Or lightweight variants for tiny RAM footprints)
ollama pull llama3.2:1b          # 1.3 GB
ollama pull qwen2.5-coder:1.5b    # 1.0 GB
ollama pull qwen2.5-coder:3b      # 1.9 GB
```
`TinyRAG` automatically detects active Ollama models on startup, lists them in the Web UI dropdown and CLI, and connects to `http://localhost:11434`.

### Option B: Cloud API via Google Gemini
Set your API key in your terminal or pass it in the Web UI:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```
The pipeline will call `gemini-1.5-flash` with strictly bounded context.

### Option C: Built-in Offline Extractive Generator
If neither Ollama nor a cloud key is present, TinyRAG automatically uses its zero-dependency extractive aligner. It runs with zero latency directly on your CPU.

---

## 📁 Project Structure

```
.
├── main.py              # FastAPI REST service & UI host (17 endpoints)
├── rag_core.py          # Core engine (Chunker, Indexer, Loader, Generator, Guardrails)
├── chat_cli.py          # Interactive terminal CLI chat loop
├── test_rag.py          # Complete unit test suite (6 automated tests)
├── ui/
│   └── index.html       # Modern dark-mode web studio dashboard
└── sample_data/         # Ready-to-use sample documents
    ├── edge_ai_guide.pdf       # Sample PDF
    ├── quantization_primer.md  # Sample Markdown document
    └── prompt_caching.txt      # Sample Text document
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` or `/ui` | `GET` | Interactive Web UI Dashboard |
| `/models-status` | `GET` | Health check for Ollama, Gemini, and vector index size |
| `/generate-naive` | `POST` | Phase 1: Un-grounded baseline simulation |
| `/ingest` | `POST` | Phase 2: Ingest JSON document list |
| `/upload` | `POST` | Phase 2: Upload `.pdf`, `.md`, `.txt` files directly |
| `/ingest-folder` | `POST` | Phase 2: Scan and ingest all files in a folder path |
| `/index-status` | `GET` | Inspect all chunks currently in memory |
| `/clear-index` | `POST` | Reset in-memory vector index |
| `/retrieve` | `POST` | Phase 3: Retrieve top-k context chunks (hybrid BM25+Dense+RRF, dense, bm25) |
| `/generate-rag` | `POST` | Phase 3: Retrieve (hybrid/dense/bm25) and generate answer with provenance & session memory |
| `/eval-recall` | `POST` | Phase 3: Compute Recall@K retrieval accuracy |
| `/generate-guarded` | `POST` | Phase 4: Retrieve (hybrid/dense/bm25) with confidence guardrails & groundedness checks |
| `/sessions` | `GET` | Phase 5: List all active conversation sessions and turn counts |
| `/sessions/{session_id}` | `GET` | Phase 5: Inspect full conversation dialogue history for a session |
| `/sessions/{session_id}` | `DELETE` | Phase 5: Clear history buffer for a specific session |

---

## 🛡️ Upload Security & DoS Safeguards

The file upload endpoint (`POST /upload`) enforces defense-in-depth protections against memory exhaustion and Denial-of-Service (DoS) attacks:

- **Bounded Streaming**: Incoming files are read in bounded 64 KB chunks, aborting early if limits are reached and freeing memory without reading full payloads.
- **Request-Level File Count**: Blocks requests with excessive files upfront before reading content.
- **Cumulative Request Budget**: Limits total upload volume across all files in a single request.
- **Content-Length Middleware**: Rejects oversized requests with HTTP 413 before request parsing.

### Configurable Environment Variables:
| Variable | Default | Description |
|---|---|---|
| `RAG_MAX_UPLOAD_SIZE` | `26214400` (25 MB) | Maximum permitted size for any individual uploaded file |
| `RAG_MAX_UPLOAD_FILE_COUNT` | `20` | Maximum number of files permitted in a single upload request |
| `RAG_MAX_REQUEST_UPLOAD_SIZE` | `52428800` (50 MB) | Maximum cumulative uploaded file size per request |
| `RAG_MAX_REQUEST_BODY_SIZE` | `57671680` (55 MB) | Maximum HTTP request body limit (Content-Length check) |
| `RAG_UPLOAD_CHUNK_SIZE` | `65536` (64 KB) | Streaming chunk size for bounded asynchronous file reading |

---

## 🔐 Authentication & Transport Security

The pipeline provides configurable authentication modes via `RAG_AUTH_MODE=development|production`:

- **Production Mode (`RAG_AUTH_MODE=production` or `RAG_ENV=production`)**:
  - **Fail-Fast Startup**: If `RAG_API_KEY` is not configured, the service fails startup immediately with `RuntimeError`, preventing accidental unauthenticated production deployments.
  - **Zero Open Access**: Unauthenticated requests are rejected with HTTP 401 Unauthorized.
  - **Built-in Token Blocking**: Development credentials (`admin-token`, `user-token`) are strictly blocked.
  - **Strict Header Transport**: Credentials must be supplied via headers (`Authorization: Bearer <token>` or `X-API-Key: <token>`). Credentials in query strings or JSON bodies are rejected with HTTP 400 Bad Request.
- **Development Mode (`RAG_AUTH_MODE=development`)**:
  - Local developers can explore APIs without configuring an API key, assigned a scoped non-privileged `dev_user` principal.
- **Automated Log Redaction**:
  - `SensitiveHeaderRedactionFilter` automatically masks Bearer tokens, `X-API-Key` headers, and credential query strings across all system loggers, replacing sensitive tokens with `[REDACTED]`.
