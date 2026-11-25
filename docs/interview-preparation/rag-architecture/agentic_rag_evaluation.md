# Question 3: Evaluating an Agentic RAG System

## 1. Short Interview Answer (60–90 Seconds)

> "Evaluating an agentic RAG system requires separating the pipeline into three distinct evaluation planes: **Retrieval Quality**, **Generation Quality**, and **Agentic Workflow Execution**.
>
> 1. **Retrieval Metrics** evaluate whether the right context was retrieved: **Recall@K** measures if the ground-truth document was captured in the top $K$, while **MRR** (Mean Reciprocal Rank) and **nDCG@K** penalize the system if relevant documents appear further down the ranking.
> 2. **Generation Metrics** evaluate answer integrity: **Faithfulness** (or groundedness) measures whether every generated claim is directly supported by the retrieved context, preventing hallucination; **Answer Relevance** measures if the response directly addresses the user's intent; and **Citation Precision** verifies that cited document tags actually contain the attributed facts.
> 3. **Agentic Workflow Metrics** evaluate multi-step decision-making: **Task Completion Rate**, **Tool Selection Accuracy**, **Multi-Step Reasoning Effectiveness**, and **Error Recovery** (how the agent handles empty retrievals or malformed inputs), alongside operational metrics like **Latency** and **Cost per Task**.
>
> In my project, I implemented `EvalHarness` for Recall@K and A/B retrieval benchmarking, and `ComprehensiveEvalHarness` which pairs with `SemanticClaimVerifier` to automate claim-level citation precision and polarity contradiction detection."

---

## 2. Comprehensive Metric Framework

### Plane 1: Retrieval Metrics

```
Retrieval Funnel: 15,000 Chunks ──> [Retrieval System] ──> Top-K Candidate Chunks (K=3)
Evaluates whether ground-truth evidence chunks exist within the Top-K retrieved set.
```

#### 1. Precision@K
- **What it measures**: The fraction of retrieved chunks in the Top-$K$ that are genuinely relevant to the query.
- **Why it matters**: Measures signal-to-noise ratio. High noise dilutes tiny LLM attention and triggers context distraction.
- **Formula**:
  $$\text{Precision@K} = \frac{|\text{Relevant Chunks} \cap \text{Top-K Retrieved Chunks}|}{K}$$
- **Example**: If $K=3$ and 2 chunks are relevant, $\text{Precision@3} = 2/3 = 0.67$.
- **Project Status**: **Implemented** in `ComprehensiveEvalHarness.evaluate` via citation-to-retrieval intersection.

#### 2. Recall@K
- **What it measures**: The fraction of all relevant ground-truth chunks in the corpus that are successfully retrieved in the Top-$K$.
- **Why it matters**: If relevant evidence is missed, the LLM will hallucinate or refuse to answer.
- **Formula**:
  $$\text{Recall@K} = \frac{|\text{Relevant Chunks} \cap \text{Top-K Retrieved Chunks}|}{|\text{Total Ground-Truth Relevant Chunks}|}$$
- **Example**: If a query has 2 relevant chunks and both appear in the Top-3, $\text{Recall@3} = 2/2 = 1.0$.
- **Project Status**: **Implemented & Verified** in `EvalHarness.calculate_recall_at_k` (`rag_core.py`, lines 1327–1369).

#### 3. Mean Reciprocal Rank (MRR)
- **What it measures**: Evaluates how high the *first* relevant document appears in the ranked results.
- **Why it matters**: In question-answering, placing the most decisive document at Rank 1 maximizes tiny LLM extraction accuracy.
- **Formula**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
  *(where $\text{rank}_i$ is the rank position of the first relevant document for query $i$)*
- **Example**: If query 1 has its first relevant chunk at rank 1, and query 2 at rank 2: $\text{MRR} = \frac{1}{2} (1/1 + 1/2) = 0.75$.
- **Project Status**: **Partially Implemented** (Computed inside `ComprehensiveEvalHarness`, but not exported as a standalone REST field).

#### 4. Normalized Discounted Cumulative Gain (nDCG@K)
- **What it measures**: Evaluates multi-grade relevance rankings, heavily penalizing relevant documents ranked lower down.
- **Why it matters**: Critical when documents have degrees of relevance (e.g. 3 = Perfect Answer, 2 = Contextual, 0 = Irrelevant).
- **Formula**:
  $$\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$
- **Project Status**: **Recommended** (Current evaluation treats relevance as binary 0 or 1).

#### 5. Hit Rate@K
- **What it measures**: The percentage of test queries where *at least one* relevant document is retrieved in the Top-$K$.
- **Formula**:
  $$\text{Hit Rate@K} = \frac{\sum_{q \in Q} \mathbb{I}(|\text{Retrieved}_q \cap \text{Relevant}_q| > 0)}{|Q|}$$
- **Project Status**: **Implemented & Verified** in `EvalHarness.calculate_recall_at_k` (variable `hits / total_queries`).

---

### Plane 2: Answer Quality & Groundedness Metrics

```
Generation Funnel: Retrieved Chunks + Query ──> [LLM Generator] ──> Generated Answer
Evaluates factual entailment, absence of hallucination, and prompt adherence.
```

#### 1. Faithfulness (Groundedness)
- **What it measures**: The percentage of factual claims in the generated response that can be mathematically inferred from the context.
- **Why it matters**: Eliminates confabulation and unverified hallucination.
- **Formula**:
  $$\text{Faithfulness} = \frac{|\text{Context-Supported Claims}|}{|\text{Total Claims in Answer}|}$$
- **Example**: If an answer contains 4 sentences and 1 contains an unmentioned number, $\text{Faithfulness} = 3/4 = 0.75$.
- **Project Status**: **Implemented & Verified** in `SemanticClaimVerifier.verify` (`rag_core.py`, lines 1250–1320).

#### 2. Polarity Contradiction Detection
- **What it measures**: Whether the generated response asserts an affirmative statement when the source context states a negation, or vice versa.
- **Why it matters**: Tiny LLMs often drop words like *"not"*, *"unapproved"*, or *"without"*, reversing contractual or medical meaning.
- **Formula**: Exact token-level negation tracking across claim and source sentences using negation lexicons:
  $$\text{Contradiction} = \mathbb{I}(\text{claim has neg}) \neq \mathbb{I}(\text{source has neg})$$
- **Project Status**: **Implemented & Verified** in `SemanticClaimVerifier` (`NEGATION_WORDS = {"not", "no", "never", "cannot", ...}`).

#### 3. Citation Precision & Recall
- **What it measures**:
  - **Citation Precision**: The percentage of citations in the generated text (e.g. `[doc_1_c0]`) that actually contain the claimed fact.
  - **Citation Recall**: The percentage of retrieved chunks used to formulate the answer that are explicitly cited.
- **Project Status**: **Implemented & Verified** in `ComprehensiveEvalHarness.evaluate` (`overall_citation_precision`).

#### 4. Context Relevance
- **What it measures**: Whether the retrieved context contains *only* information pertinent to the question, without bloat.
- **Project Status**: **Recommended** (Evaluated in production via LLM-as-a-judge frameworks like Ragas).

---

### Plane 3: Agentic Workflow Metrics

```
Agentic Loop: User Goal ──> [Planner] ──> [Tool Selection] ──> [Observation] ──> [Correction]
Evaluates dynamic reasoning, multi-step execution, and autonomous error recovery.
```

| Metric | What It Measures | Why It Matters | Calculation Method | Project Status |
| :--- | :--- | :--- | :--- | :--- |
| **Task Completion Rate** | % of user requests where the agent achieves the full objective without human intervention. | Ultimate North-Star metric for autonomous agents. | $\frac{\text{Successful Tasks}}{\text{Total Tasks Evaluated}}$ | **Recommended** *(Current project is sequential pipeline)* |
| **Tool Selection Accuracy** | % of steps where the agent chooses the correct tool and parameters from the schema. | Incorrect tool selection wastes tokens and risks erroneous state mutation. | $\frac{\text{Correct Tool Invocations}}{\text{Total Tool Invocations}}$ | **Recommended** |
| **Tool Execution Success** | % of tool calls that execute without API, schema, or runtime exceptions. | Measures schema robustness and parameter formatting. | $\frac{\text{Successful Invocations}}{\text{Total Attempted Invocations}}$ | **Recommended** |
| **Multi-Step Reasoning Effectiveness** | Quality and efficiency of intermediate planning steps (no redundant loops). | Prevents agents from looping infinitely between tools. | Ratio of optimal path length to actual path length. | **Recommended** |
| **Error Recovery Rate** | Ability of the agent to recover after a tool failure or empty retrieval. | Measures system resilience under partial failure. | $\frac{\text{Recovered Failures}}{\text{Total Injected Failures}}$ | **Implemented in Guardrails** *(Refusal branch)* |
| **Latency per Task** | Wall-clock execution time from user query to completed response. | Critical for user experience (edge SLA: $< 500\text{ms}$). | Median ($P_{50}$) and $P_{95}$ latency timestamps. | **Implemented in Tests** (`test_rag.py`) |
| **Cost per Task** | Financial cost in tokens (or cloud API credits) per completed task. | Directly drives unit economics in production. | $\sum (\text{Input Tokens} \times \text{Rate}_{\text{in}} + \text{Output} \times \text{Rate}_{\text{out}})$ | **Recommended** |

---

## 3. Codebase Evaluation Inspection

### 1. `EvalHarness.calculate_recall_at_k` (`rag_core.py`, lines 1327–1369)
```python
@staticmethod
def calculate_recall_at_k(eval_dataset: List[Dict[str, Any]], indexer: VectorIndexer, k: int = 3) -> Dict[str, Any]:
    total_queries = len(eval_dataset)
    hits = 0
    for item in eval_dataset:
        query = item["query"]
        relevant_targets = set(item.get("relevant_doc_ids", item.get("relevant_chunk_ids", [])))
        retrieved = indexer.search(query, top_k=k)
        retrieved_ids = set([c["doc_id"] for c in retrieved] + [c["chunk_id"] for c in retrieved])
        hit = len(relevant_targets.intersection(retrieved_ids)) > 0
        if hit:
            hits += 1
    return {"recall_at_k": round(hits / total_queries, 4), "k": k, "hits": hits}
```
*Verification*: Fully functional and tested via `test_phase3_eval_harness` in `test_rag.py`.

### 2. `EvalHarness.compare_configurations` (`rag_core.py`, lines 1370–1437)
Compares two retrieval configurations side-by-side on the same golden dataset:
- **Config A**: Baseline Dense Cosine Similarity (Top-$K$).
- **Config B**: Hybrid (Dense + BM25 via RRF) + Cross-Encoder Reranking.
- Outputs `recall_delta` and determines the `winner` configuration.
*Verification*: Validated by `test_ab_eval_configuration_comparison` in `test_rag.py`.

### 3. `SemanticClaimVerifier.verify` (`rag_core.py`, lines 1250–1321)
Implements automated factual verification without requiring expensive cloud LLM-as-a-judge APIs:
1. Decomposes answer into sentences using regex boundary matching.
2. Extracts cited chunk IDs `[doc_id_c0]`.
3. Verifies that the cited chunk ID is present in `retrieved_chunks` and exhibits $>35\%$ token overlap with the claim.
4. Detects polarity contradictions by scanning `NEGATION_WORDS` against the closest matching context sentence.
*Verification*: Validated by `test_semantic_claim_verifier_catches_polarity_contradiction`.

---

## 4. Recommended Production Framework & Golden Dataset Schema

### Recommended Enterprise Evaluation Stack:
1. **Ragas / TruLens / DeepEval**: Production evaluation libraries for computing LLM-as-a-judge metrics (Faithfulness, Answer Relevance, Context Recall).
2. **Synthetic Data Generation**: Using models like Claude 3.5 Sonnet or GPT-4o to automatically generate query-context-answer triples from unannotated PDFs to build golden evaluation benchmarks.

### Production Golden Dataset Schema:
```json
[
  {
    "eval_id": "eval_001",
    "query": "What is the penalty for early termination of the retail lease?",
    "session_context": [
      {"role": "user", "content": "I am looking at Greenwood Phase 2 retail units."}
    ],
    "ground_truth_answer": "Early termination requires 90 days written notice and a penalty equal to 3 months base rent.",
    "expected_retrieval_chunk_ids": ["doc_gw_c14"],
    "expected_metadata_filters": {"project_id": "PRJ_GREENWOOD", "clause_type": "lease_penalty"},
    "expected_tool_call": "vector_search",
    "required_factual_entities": ["90 days", "3 months base rent"]
  }
]
```

---

## 5. Design Trade-Offs

| Evaluation Approach | Advantages | Disadvantages | When to Use |
| :--- | :--- | :--- | :--- |
| **Deterministic Rule-Based (Current Project)** | Instant execution ($< 5\text{ms}$), zero token cost, runs 100% offline, deterministic, perfect for CI/CD gates. | Misses abstract semantic paraphrasing and complex reasoning that doesn't share lexical tokens. | Pull request regression checks, unit test pipelines (`test_rag.py`), edge runtimes. |
| **LLM-as-a-Judge (Ragas / GPT-4o)** | Captures semantic subtlety, nuance, tone, and logical entailment without exact keyword matching. | Expensive, high latency ($\sim 1-3\text{s}$ per evaluation), non-deterministic, can suffer from self-preference bias. | Periodic offline benchmark evaluations, prompt tuning experiments, weekly quality audits. |
| **Human Expert Evaluation** | Absolute ground truth. Essential for high-stakes compliance (medical, legal). | Extremely expensive, slow, non-scalable, subject to human annotator fatigue and inconsistency. | Golden test dataset curation (initial 100–500 test cases) and quarterly calibration audits. |

---

## 6. Interviewer Follow-Up Questions & Technical Answers

### Q1: "What is 'Self-Preference Bias' in LLM-as-a-judge evaluation, and how do you mitigate it?"
**Answer:**
"Self-preference bias occurs when an LLM judge systematically awards higher quality and faithfulness scores to outputs generated by itself or models of the same family (e.g. GPT-4 preferring GPT-4 outputs over Claude 3.5 Sonnet).
To mitigate this in production:
1. **Cross-Model Evaluation**: Use a different model family for evaluation than for generation (e.g., evaluate local Mistral/Llama models using Gemini or Claude).
2. **Deconstructive Few-Shot Rubrics**: Instead of asking for a subjective 1–5 score, use structured rubrics that force the judge to extract claims first and output a binary matrix of claim-by-claim support.
3. **Reference-Guided Evaluation**: Provide the judge with human-verified ground-truth answers rather than evaluating outputs in isolation."

### Q2: "How do you calculate nDCG@K mathematically, and why is the logarithmic discount factor used?"
**Answer:**
"nDCG@K calculates the Discounted Cumulative Gain normalized by the Ideal Discounted Cumulative Gain:
$$\text{DCG@K} = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}$$
The numerator $2^{\text{rel}_i} - 1$ gives exponential weight to highly relevant documents ($\text{rel}=3 \rightarrow 7$, $\text{rel}=2 \rightarrow 3$).
The denominator $\log_2(i + 1)$ represents the logarithmic position discount. It models user attention decay: finding a critical document at Rank 1 has discount $\log_2(2) = 1.0$, whereas finding it at Rank 8 has discount $\log_2(9) = 3.17$, reducing its score contribution by over 68%. Dividing by IDCG (the DCG of a perfectly sorted list) normalizes the metric to $[0.0, 1.0]$."

### Q3: "How does your project catch polarity contradictions without using a cloud LLM judge?"
**Answer:**
"In `rag_core.py`, `SemanticClaimVerifier` implements a specialized polarity contradiction check (lines 1287–1303).
1. It scans the candidate answer claim for words from a predefined negation set (`NEGATION_WORDS = {'not', 'no', 'never', 'cannot', 'without', 'failed', 'denied'}`).
2. It finds the highest-overlap sentence in the retrieved evidence chunk.
3. It checks whether the evidence sentence contains a negation.
If one sentence has a negation and the other does not (e.g. claim asserts *'permit was approved'* while context states *'permit was not approved'*), a contradiction is flagged and `grounded` is set to `False`, suppressing the response."

### Q4: "Why is Precision@K often misleading in RAG retrieval when chunk overlap is used?"
**Answer:**
"When documents are chunked with overlap (e.g., $chunk\_size=150, overlap=30$), a single factual answer often spans across two adjacent chunks (e.g. Chunk 0 and Chunk 1). If $K=3$ and Chunk 0 and Chunk 1 both contain the answer while Chunk 2 provides surrounding context, the system has provided 100% of the required evidence. However, if the ground-truth benchmark labels only Chunk 0 as the target, Chunk 1 and Chunk 2 are marked as false positives, artificially lowering Precision@3.
For RAG systems, **Recall@K** and **Citation Precision** are much more indicative of operational success than raw chunk Precision@K."

### Q5: "What is 'Lost in the Middle' phenomenon, and what metric measures an LLM's susceptibility to it?"
**Answer:**
"The 'Lost in the Middle' phenomenon (proven by Liu et al., Stanford) shows that language models retrieve information with high accuracy when relevant evidence appears at the very beginning or very end of the prompt context, but accuracy degrades severely when evidence is positioned in the middle of a multi-document context window.
We evaluate this using **Positional Accuracy Curves**: we run test datasets while systematically varying the position of the golden evidence chunk from index 1 to index $K$, measuring the generation accuracy at each rank position."

---

## 7. Honest Implementation Status: What to Claim vs. What NOT to Claim

### What You CAN Confidently Claim:
- "I implemented a quantitative retrieval evaluation harness (`EvalHarness.calculate_recall_at_k`) measuring Recall@K and Hit Rate."
- "I built an A/B evaluation comparator (`EvalHarness.compare_configurations`) that objectively benchmarks Dense vs. Hybrid + Cross-Encoder retrieval."
- "I built an offline propositional claim verifier (`SemanticClaimVerifier`) that automatically computes citation precision and flags polarity contradictions."
- "I created automated unit tests covering evaluation harnesses and citation verification in `test_rag.py`."

### What You Must NOT Claim:
- **Do NOT claim** that your project currently integrates cloud-based LLM-as-a-judge frameworks like Ragas, TruLens, or DeepEval. (State clearly: *"Currently, evaluation is conducted using my built-in deterministic offline harnesses (`EvalHarness` and `SemanticClaimVerifier`), which run without cloud dependencies. In my architecture roadmap, I recommend integrating Ragas for periodic offline LLM-as-a-judge benchmarking."*)
- **Do NOT claim** that your project has an automated multi-step agent benchmarking harness measuring tool selection accuracy. (State clearly: *"The current evaluation harnesses test retrieval accuracy, citation precision, and groundedness across the RAG pipeline. Autonomous agent tool-calling evaluation is a recommended extension."*)

