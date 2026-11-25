# Question 1: Handling Structured and Unstructured Data in RAG

## 1. Short Interview Answer (60–90 Seconds)

> "In a production RAG system, **unstructured data** consists of free-form text like PDF research papers, technical manuals, or markdown notes, where meaning is distributed across paragraphs. **Structured data** consists of schema-bound entities like SQL tables, CSV records, or JSON objects with strict typed fields like prices, dates, IDs, and geographic coordinates.
>
> In my project, unstructured documents (.pdf, .md, .txt) are parsed via `DocumentLoader`, chunked using a sliding-window `TextChunker` (150–300 words with 30–50 word overlap), and indexed into an in-memory dense vector store (`VectorIndexer`) alongside a BM25 lexical index (`BM25OkapiIndexer`).
>
> For structured files like `.csv` and `.json`, my current implementation ingests them as raw text representations, chunking them into the vector index. However, in an enterprise architecture, vector similarity fails on exact mathematical constraints like *'units under \$500,000'*. The production-grade approach is a **Dual-Engine Hybrid Pipeline**: structured records are loaded into a relational engine like PostgreSQL or DuckDB, while unstructured narratives live in the vector index. An orchestrator uses query understanding to route filter constraints to SQL or metadata filters, and semantic search to the vector store, synthesizing both into the final LLM prompt context."

---

## 2. Detailed Technical Explanation

### 2.1 Definitions & Semantic Divergence
- **Unstructured Data**: Dense narrative prose with arbitrary syntax, idioms, and paragraph structures.
  - *Characteristics*: High semantic density, irregular length, absence of fixed schema.
  - *Optimal Retrieval*: Dense embeddings (bi-encoders capturing semantic similarity) combined with sparse lexical search (BM25 for exact terms).
- **Structured Data**: Rigidly typed attributes organized into rows, columns, or key-value structures.
  - *Characteristics*: Strict types (integers, floats, dates, enums), exact equality or range operations.
  - *Optimal Retrieval*: Deterministic relational algebra (SQL `WHERE`, `JOIN`, `GROUP BY`, `BETWEEN`) or inverted metadata indices.

### 2.2 Why Vector Search Fails on Structured Data
Vector embeddings project text into continuous high-dimensional vector spaces ($\mathbb{R}^{384}$ or $\mathbb{R}^{768}$). While excellent at finding thematic similarity (e.g., *"modern apartment with open floor plan"* $\approx$ *"contemporary flat with spacious living area"*), vector similarity fundamentally struggles with:
1. **Arithmetic & Range Comparisons**: A vector dot product cannot reliably evaluate whether $\$495,000 < \$500,000$.
2. **Exact Alphanumeric Filtering**: Model IDs, SKU numbers, or postal codes (e.g., `POL-089` or `560066`) are frequently collapsed into similar subword tokens or lost in dense space.
3. **Aggregations & Grouping**: Calculating `AVG(price_per_sqft)` or `COUNT(units)` cannot be computed across vector representations without retrieving every chunk and risking severe LLM hallucination.

### 2.3 Comprehensive Data Flow for Ingestion & Retrieval

```
                                  [DATA INGESTION]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
       [Structured Records]                           [Unstructured Documents]
     (.csv, .json, SQL tables)                         (.pdf, .md, .txt files)
                 │                                               │
                 ├───────────────────────────────┐               │
                 ▼ (Recommended)                 ▼ (Current)     ▼
      [Relational Database]            [Structured-to-Text] [DocumentLoader]
       (PostgreSQL / DuckDB)            (JSON / CSV as text) (fitz / pypdf)
                 │                               │               │
                 │                               └───────┬───────┘
                 │                                       ▼
                 │                              [TextChunker (Sliding Window)]
                 │                               (chunk_size=150, overlap=30)
                 │                                       │
                 │                                       ▼
                 │                              [VectorIndexer & BM25Okapi]
                 │                              (Dense Embeddings + Lexical)
                 │                                       │
─────────────────┼───────────────────────────────────────┼────────────────────
                 │          [QUERY ORCHESTRATION]        │
                 ▼                                       ▼
      User Query: "Show me residential projects in Whitefield under $150k with high energy ratings"
                 │
                 ▼
      [Query Understanding / Decomposition]
      ├── Structured Filters : Location="Whitefield", Category="Residential", Price < $150,000
      └── Semantic Concept   : "high energy efficiency rating green architectural review"
                 │
        ┌────────┴───────────────────────────────┐
        ▼                                        ▼
 [Structured Query Execution]             [Vector & Lexical Search]
 SELECT project_id, name, price           VectorIndexer.search_hybrid(
 FROM projects WHERE location='Whitefield'  query="high energy efficiency",
   AND price < 150000;                     candidate_k=15
        │                                        │
        └────────────────┬───────────────────────┘
                         ▼
             [Context Fusion & Gating]
   - Merge tabular metrics with retrieved narrative chunks
   - Enforce Guardrail threshold validation
                         ▼
             [PromptAssembler.build_prompt()]
                         ▼
             [LLM Generation & Claim Verification]
```

---

## 3. Project Implementation vs. Codebase Audit

### What is Actually Implemented in the Codebase:
1. **Document Loading (`DocumentLoader` in `rag_core.py`, lines 30–185)**:
   - Supports `.pdf` (via `extract_text_from_pdf_bytes` using `fitz` / `pypdf`), `.txt`, and `.md`.
   - Also accepts `.json`, `.csv`, `.tsv`, `.sql` in `TEXT_EXTENSIONS`.
   - **Crucial Technical Detail**: The file is read entirely as a plain-text string using Python's standard `open(file_path, 'r', encoding='utf-8')`. It is **not** parsed into tabular DataFrames or SQL tables.
2. **Chunking (`TextChunker` in `rag_core.py`, lines 186–224)**:
   - Normalizes whitespace with `re.sub(r'\s+', ' ', text)`.
   - Splits on whitespace into words: `words = text.split()`.
   - Slices words into sliding windows of length `chunk_size=150` with `overlap=30`.
   - Emits chunk dictionaries with `chunk_id`, `doc_id`, `title`, `text`, `start_word`, and `end_word`.
3. **Indexing & Retrieval (`VectorIndexer` in `rag_core.py`, lines 226–450)**:
   - Encodes text chunks into dense vectors via Ollama (`nomic-embed-text`) or SentenceTransformers (`all-MiniLM-L6-v2`), with TF-IDF fallback.
   - Calculates dot product on normalized vectors for cosine similarity.
   - Dual-indexes into `BM25OkapiIndexer` for exact keyword scoring.

### What is Missing / Not Implemented:
- **No Relational Database or SQL Engine**: There is no PostgreSQL, SQLite, DuckDB, or SQLAlchemy integration in the project.
- **No Metadata Attribute Indexing / Filtering**: Chunks do not have indexed metadata tags (like `price`, `date`, `category`), and `VectorIndexer.search` has no `filter={...}` parameter.
- **No Table-Aware Chunking**: Markdown or CSV tables are treated as raw text strings. A sliding word window will arbitrarily split across table rows and column headers, separating numeric values from their associated column headers.

---

## 4. Realistic Real-Estate Example & End-to-End Trace

### User Query:
> *"Show me the residential and commercial projects available in Whitefield, their inventory, average prices, and summarize the architectural insulation quality from the engineering reports."*

### Step-by-Step System Trace:

#### Step 1: Query Decomposition (Recommended Agent / Orchestrator)
The query contains two distinct information modalities:
- **Tabular / Analytical Constraints**:
  - `location`: "Whitefield"
  - `project_types`: `["Residential", "Commercial"]`
  - `metrics_needed`: `inventory`, `avg_price`, `price_per_sqft`
- **Semantic / Narrative Need**:
  - `semantic_query`: "architectural insulation quality building envelope energy efficiency"

#### Step 2: Structured Retrieval (Recommended Relational Engine)
```sql
SELECT 
    project_id, 
    project_name, 
    project_type, 
    available_units AS inventory, 
    avg_price, 
    price_per_sqft 
FROM project_inventory 
WHERE location = 'Whitefield' 
  AND project_type IN ('Residential', 'Commercial');
```
*Result Table:*
| project_id | project_name | project_type | inventory | avg_price | price_per_sqft |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `PRJ_042` | Palm Meadows East | Residential | 14 | \$220,000 | \$110/sqft |
| `PRJ_109` | Sigma Tech Plaza | Commercial | 4 | \$850,000 | \$185/sqft |

#### Step 3: Unstructured Retrieval (Implemented `VectorIndexer.search_hybrid`)
The semantic query `"architectural insulation quality building envelope energy efficiency"` is dispatched to `search_hybrid` with candidate parameter $K=10$:
- Dense vector similarity retrieves chunks matching building performance.
- BM25 searches for "insulation", "envelope", "quality".
- RRF combines candidates; `CrossEncoderReRanker` reranks top 3 chunks.
*Retrieved Chunks:*
- `[doc_palm_eng_c2]`: *"Palm Meadows East utilizes double-glazed low-E argon windows and R-21 fiberglass batt insulation, reducing heating loads by 28%."*
- `[doc_sigma_specs_c1]`: *"Sigma Tech Plaza features a curtain wall system with thermal break aluminum frames and insulated spandrel panels achieving U-0.24 rating."*

#### Step 4: Prompt Assembly (`PromptAssembler.build_prompt`)
```markdown
You are a factual, context-grounded AI assistant. Answer the user question based on the provided context blocks below.

=== RETRIEVED CONTEXT ===
[1] Document 'Palm Meadows East Inventory & Specs' (Chunk ID: PRJ_042_c0):
Inventory: 14 units | Avg Price: $220,000 ($110/sqft) | Type: Residential
Insulation: Utilizes double-glazed low-E argon windows and R-21 fiberglass batt insulation, reducing heating loads by 28%.

[2] Document 'Sigma Tech Plaza Inventory & Specs' (Chunk ID: PRJ_109_c1):
Inventory: 4 units | Avg Price: $850,000 ($185/sqft) | Type: Commercial
Insulation: Features a curtain wall system with thermal break aluminum frames and insulated spandrel panels achieving U-0.24 rating.
=========================

USER QUESTION: Show me the residential and commercial projects available in Whitefield, their inventory, average prices, and summarize the architectural insulation quality.

ANSWER (with citations):
```

#### Step 5: Answer Generation & Verification (`LLMGenerator` & `SemanticClaimVerifier`)
The LLM generates the factual answer citing `[PRJ_042_c0]` and `[PRJ_109_c1]`. `SemanticClaimVerifier` verifies each statement against source text to guarantee 100% citation precision and ensure no polarity inversions exist.

---

## 5. Design Trade-Offs

| Approach | Advantages | Disadvantages | When to Choose |
| :--- | :--- | :--- | :--- |
| **A. Convert All Structured Data to Text (Current Project Baseline)** | Simple, unified vector pipeline. No SQL database or dual-index infrastructure required. Single retrieval interface. | Complete failure on arithmetic filters, range queries (`price < X`), and exact aggregations. High token consumption. | Rapid prototyping, small knowledge bases with purely descriptive metadata. |
| **B. Metadata Filtering in Vector DB (Chroma, Qdrant, Pinecone)** | Single vector database. Allows filtering by simple scalar fields (`doc_type == 'specs'`) during vector search. | Limited relational joins. Inverted index performance degrades under complex multi-table joins or dynamic aggregations. | Document collections where metadata consists of flat categories, dates, and simple tags. |
| **C. Dual-Engine Architecture: Relational DB + Vector Search (Recommended)** | 100% precision on mathematical, relational, and aggregation queries. Optimal vector search for semantic narrative. | Higher system complexity. Requires query decomposition, schema management, and context fusion logic. | Enterprise systems combining tabular business data (prices, inventory, CRM) with unstructured reports. |

---

## 6. Interviewer Follow-Up Questions & Technical Answers

### Q1: "What happens if a CSV table has 50 columns and 50,000 rows? How would you chunk and index it?"
**Answer:**
"You should never feed raw 50-column CSV rows into a standard text chunker. If chunked arbitrarily, rows get split mid-line, column headers are separated from data values, and embedding a 50-column row creates an undifferentiated semantic blur.
In production, I implement a **hybrid strategy**:
1. **Relational Ingestion**: The raw CSV is loaded directly into a columnar SQL engine (e.g. DuckDB or Snowflake) to support exact SQL queries, aggregations, and range filters.
2. **Schema-Aware Text Summarization**: For vector discovery, we generate row-level or entity-level documents where each row is converted into a structured natural language template (e.g., *'Project: {name}, Location: {loc}, Price: \${price}. Summary: {desc}'*).
3. **Table Column Summaries**: We embed table-level data dictionaries describing column semantics so the LLM agent knows which table and columns to query via Text-to-SQL."

### Q2: "What is the difference between pre-filtering and post-filtering when using metadata in vector databases?"
**Answer:**
"**Pre-filtering** applies the metadata constraint (e.g., `location == 'Whitefield'`) *before* executing the vector similarity search, pruning the vector search space down to only matching vectors. This guarantees that all $K$ retrieved vectors satisfy the filter, but requires vector indices that support single-stage filtered HNSW or payload filtering (like Qdrant or Milvus).
**Post-filtering** performs vector search across the entire index first to retrieve Top-$M$ candidates, and then discards any candidates that do not match the metadata criteria. The major failure mode of post-filtering is **candidate starvation**: if the metadata filter is highly selective (e.g., selecting 1% of the corpus), retrieving Top-50 vectors might yield 0 matching candidates after the filter is applied. Therefore, pre-filtering is strictly preferred in production."

### Q3: "How does your current implementation handle a document update if the user uploads an amended PDF?"
**Answer:**
"My codebase addresses this via `DocumentVersionRegistry` in `rag_core.py` (lines 647–681). When a document is re-ingested, `compute_hash` calculates the SHA-256 hash of the content:
1. If the hash matches the existing registry entry, `check_document` returns `needs_update = False`, preventing redundant re-embedding.
2. If the hash differs, `check_document` increments the version number and returns the list of prior chunk IDs (`stale_chunk_ids`).
3. `VectorIndexer.evict_stale_chunks` then removes the old chunk dictionaries and slices the embedding matrix using NumPy index masking, preventing stale chunk 'ghosting' before new chunks are inserted."

### Q4: "If you have structured data in SQL and unstructured text in vectors, how do you handle security and Row-Level Security (RLS) across both?"
**Answer:**
"In enterprise systems, security must be enforced at retrieval time, not during prompt generation.
1. **Structured RLS**: When querying PostgreSQL or Snowflake, the user's JWT session or organization ID is passed directly into the SQL session, enforcing native database RLS policies.
2. **Vector Access Control**: Every chunk in the vector database stores access control lists in its metadata payload (e.g., `allowed_roles: ['finance_admin']` or `tenant_id: 'tenant_42'`).
3. **Pre-Filter Enforcement**: The orchestrator injects mandatory tenant and role predicates into the vector pre-filter before similarity scoring, ensuring unauthorized context blocks are mathematically excluded from the candidate pool."

### Q5: "How does sliding-window chunking in your project prevent semantic truncation at chunk boundaries?"
**Answer:**
"In `rag_core.py`, `TextChunker` uses an overlapping sliding window ($chunk\_size=150$, $overlap=30$). If a critical sentence or fact spans across words 130 to 170, the first chunk captures words 0–150 and the second chunk captures words 120–270. This 30-word overlap guarantees that any concept spanning the boundary is preserved intact in at least one chunk. In future iterations, I recommend moving from word-count sliding windows to **recursive semantic chunking**, which splits on markdown headings, paragraphs, and sentence boundaries before resorting to word-level splits."

---

## 7. Honest Implementation Status: What to Claim vs. What NOT to Claim

### What You CAN Confidently Claim:
- "I implemented a full file and byte ingestion pipeline supporting `.pdf`, `.md`, and `.txt` with PyMuPDF and pypdf."
- "I implemented a sliding-window chunker with configurable word counts and overlap."
- "I built a SHA-256 document versioning registry (`DocumentVersionRegistry`) that evicts stale chunks upon file modification."
- "I implemented thread-safe document ingestion using per-document mutexes (`DocumentLockManager`)."
- "I implemented dual indexing with dense vector embeddings and pure-Python BM25 Okapi lexical indexing."

### What You Must NOT Claim:
- **Do NOT claim** that your system currently connects to a SQL database or runs Text-to-SQL. (State clearly: *"In my current implementation, structured formats like CSV and JSON are read as raw text and chunked into the vector store. For production enterprise deployments, I recommend a dual-store architecture with PostgreSQL/DuckDB."*)
- **Do NOT claim** that your vector store performs metadata pre-filtering on structured attributes like price ranges or dates. (State clearly: *"Currently, metadata filtering is not implemented in `VectorIndexer`; the index relies on dense cosine similarity and BM25 text matching. In my architecture plan, I recommend integrating metadata payload filtering via Qdrant or LanceDB."*)

