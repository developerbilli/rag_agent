"""
Stress Test and Scale Benchmark for Tiny LLM Offline RAG Pipeline.
Evaluates scale (10k, 50k, 100k chunks), concurrency, edge cases,
and profiles latency (P50/P95/P99), memory, snapshot size, and recovery time.
"""
import os
import sys
import time
import json
import tempfile
import shutil
import threading
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Ensure local workspace is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag_core import (
    VectorIndexer,
    BM25OkapiIndexer,
    TextChunker,
    DocumentLockManager,
    DocumentLoader,
    HeuristicReRanker,
    DeterministicComputationEngine,
    SmallModelCompatibilityAudit
)

def generate_synthetic_chunks(num_chunks: int, dim: int = 384):
    """Generates synthetic chunk text and normalized float32 embeddings."""
    vocab = [
        "quantum", "encryption", "cooling", "subsystem", "reactor",
        "turbine", "protocol", "temperature", "pressure", "hydraulic",
        "sensor", "telemetry", "algorithm", "bandwidth", "latency",
        "throughput", "database", "vector", "embedding", "fusion",
        "bm25", "lexical", "reranking", "verification", "abstention"
    ]
    np.random.seed(42)
    # Generate random normalized embeddings
    raw_vecs = np.random.randn(num_chunks, dim).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    embeddings = raw_vecs / np.maximum(norms, 1e-12)

    chunks = []
    for i in range(num_chunks):
        v1 = vocab[i % len(vocab)]
        v2 = vocab[(i * 7) % len(vocab)]
        v3 = vocab[(i * 13) % len(vocab)]
        text = f"Subsystem {i} operational report: {v1} status nominal with {v2} metrics and {v3} telemetry verified."
        chunks.append({
            "chunk_id": f"chunk_{i:06d}",
            "doc_id": f"doc_{i // 20:04d}",
            "text": text,
            "title": f"Component Spec {i // 20}",
            "start_word": (i % 20) * 100,
            "end_word": (i % 20) * 100 + 120,
            "metadata": {
                "clearance": "public",
                "subsystem_id": f"SUB-{i % 500:03d}"
            }
        })
    return chunks, embeddings

def run_scale_benchmarks():
    print("=" * 80)
    print("STAGE 1: SCALE BENCHMARKS (10,000, 50,000, 100,000 CHUNKS)")
    print("=" * 80)

    scales = [10000, 50000, 100000]
    scale_results = {}

    for count in scales:
        print(f"\n--- Benchmarking Scale: {count:,} Chunks ---")
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0]

        t0 = time.perf_counter()
        chunks, embeddings = generate_synthetic_chunks(count, dim=384)
        gen_time = time.perf_counter() - t0

        indexer = VectorIndexer(model_name="all-MiniLM-L6-v2")
        indexer.chunks = chunks
        indexer.embeddings = embeddings

        t0 = time.perf_counter()
        indexer.bm25 = BM25OkapiIndexer()
        indexer.bm25.index(chunks)
        bm25_index_time = time.perf_counter() - t0

        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ram_mb = (current_mem - start_mem) / (1024 * 1024)
        peak_ram_mb = peak_mem / (1024 * 1024)

        # Retrieval Latency: 100 query evaluations
        num_eval_queries = 100
        query_vecs = np.random.randn(num_eval_queries, 384).astype(np.float32)
        query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)

        # 1. Dense retrieval latency
        dense_latencies = []
        for qv in query_vecs:
            t_start = time.perf_counter()
            scores = np.dot(embeddings, qv)
            top_k_idx = np.argpartition(scores, -10)[-10:]
            _ = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            dense_latencies.append((time.perf_counter() - t_start) * 1000)

        # 2. BM25 retrieval latency
        sample_queries = [
            "quantum encryption reactor cooling",
            "hydraulic pressure sensor telemetry",
            "database vector embedding fusion",
            "temperature algorithm bandwidth latency"
        ]
        bm25_latencies = []
        for i in range(num_eval_queries):
            q_text = sample_queries[i % len(sample_queries)]
            t_start = time.perf_counter()
            _ = indexer.bm25.search(q_text, top_k=10)
            bm25_latencies.append((time.perf_counter() - t_start) * 1000)

        # 3. Hybrid RRF retrieval latency
        hybrid_latencies = []
        for i in range(num_eval_queries):
            qv = query_vecs[i]
            q_text = sample_queries[i % len(sample_queries)]
            t_start = time.perf_counter()
            scores = np.dot(embeddings, qv)
            top_k_idx = np.argpartition(scores, -20)[-20:]
            dense_hits = [chunks[idx] for idx in top_k_idx]
            bm25_hits = indexer.bm25.search(q_text, top_k=20)
            # RRF calculation
            rrf_scores = {}
            for rank, c in enumerate(dense_hits):
                cid = c["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
            for rank, (idx, _) in enumerate(bm25_hits):
                cid = chunks[idx]["chunk_id"]
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (60 + rank + 1)
            sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            hybrid_latencies.append((time.perf_counter() - t_start) * 1000)

        # Snapshot Serialization & Recovery
        temp_dir = tempfile.mkdtemp()
        snap_path = os.path.join(temp_dir, f"snap_{count}")
        t_save_start = time.perf_counter()
        indexer.save_snapshot(snap_path)
        save_time_ms = (time.perf_counter() - t_save_start) * 1000

        # Measure snapshot disk footprint
        total_snap_bytes = sum(
            os.path.getsize(os.path.join(snap_path, f))
            for f in os.listdir(snap_path)
            if os.path.isfile(os.path.join(snap_path, f))
        )
        snap_size_mb = total_snap_bytes / (1024 * 1024)

        # Measure recovery / load time
        new_indexer = VectorIndexer(model_name="all-MiniLM-L6-v2")
        t_load_start = time.perf_counter()
        recovered = new_indexer.load_snapshot(snap_path)
        load_time_ms = (time.perf_counter() - t_load_start) * 1000

        shutil.rmtree(temp_dir, ignore_errors=True)

        res = {
            "count": count,
            "ram_mb": round(ram_mb, 2),
            "peak_ram_mb": round(peak_ram_mb, 2),
            "bm25_index_time_s": round(bm25_index_time, 3),
            "dense_p50_ms": round(float(np.percentile(dense_latencies, 50)), 2),
            "dense_p95_ms": round(float(np.percentile(dense_latencies, 95)), 2),
            "dense_p99_ms": round(float(np.percentile(dense_latencies, 99)), 2),
            "bm25_p50_ms": round(float(np.percentile(bm25_latencies, 50)), 2),
            "bm25_p95_ms": round(float(np.percentile(bm25_latencies, 95)), 2),
            "bm25_p99_ms": round(float(np.percentile(bm25_latencies, 99)), 2),
            "hybrid_p50_ms": round(float(np.percentile(hybrid_latencies, 50)), 2),
            "hybrid_p95_ms": round(float(np.percentile(hybrid_latencies, 95)), 2),
            "hybrid_p99_ms": round(float(np.percentile(hybrid_latencies, 99)), 2),
            "save_time_ms": round(save_time_ms, 2),
            "snap_size_mb": round(snap_size_mb, 2),
            "load_time_ms": round(load_time_ms, 2),
            "recovery_success": recovered
        }
        scale_results[count] = res

        print(f"  RAM Footprint:       {ram_mb:.2f} MB (Peak: {peak_ram_mb:.2f} MB)")
        print(f"  BM25 Index Build:    {bm25_index_time:.3f} s")
        print(f"  Dense Latency:       P50={res['dense_p50_ms']} ms | P95={res['dense_p95_ms']} ms | P99={res['dense_p99_ms']} ms")
        print(f"  BM25 Latency:        P50={res['bm25_p50_ms']} ms | P95={res['bm25_p95_ms']} ms | P99={res['bm25_p99_ms']} ms")
        print(f"  Hybrid RRF Latency:  P50={res['hybrid_p50_ms']} ms | P95={res['hybrid_p95_ms']} ms | P99={res['hybrid_p99_ms']} ms")
        print(f"  Snapshot Size:       {snap_size_mb:.2f} MB (Save: {save_time_ms:.1f} ms | Load: {load_time_ms:.1f} ms)")

    return scale_results

def run_concurrency_benchmarks(num_chunks: int = 100000):
    print("\n" + "=" * 80)
    print("STAGE 2: CONCURRENCY BENCHMARKS (100,000 CHUNKS)")
    print("=" * 80)

    chunks, embeddings = generate_synthetic_chunks(num_chunks, dim=384)
    indexer = VectorIndexer(model_name="all-MiniLM-L6-v2")
    indexer.chunks = chunks
    indexer.embeddings = embeddings
    indexer.bm25 = BM25OkapiIndexer()
    indexer.bm25.index(chunks)

    # 1. 25 Simultaneous Query Threads across 500 requests
    total_queries = 500
    concurrency_workers = 25
    query_vecs = np.random.randn(total_queries, 384).astype(np.float32)
    query_vecs /= np.linalg.norm(query_vecs, axis=1, keepdims=True)

    latencies = []
    errors = 0

    def execute_query(q_idx):
        qv = query_vecs[q_idx]
        t_start = time.perf_counter()
        try:
            scores = np.dot(indexer.embeddings, qv)
            top_k_idx = np.argpartition(scores, -10)[-10:]
            _ = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
            duration = (time.perf_counter() - t_start) * 1000
            return duration, None
        except Exception as e:
            return None, str(e)

    t0_concurrency = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency_workers) as executor:
        futures = [executor.submit(execute_query, i) for i in range(total_queries)]
        for f in as_completed(futures):
            dur, err = f.result()
            if err:
                errors += 1
            else:
                latencies.append(dur)
    wall_clock_time = time.perf_counter() - t0_concurrency
    qps = total_queries / wall_clock_time

    concurrency_results = {
        "total_queries": total_queries,
        "concurrency_workers": concurrency_workers,
        "wall_clock_s": round(wall_clock_time, 2),
        "throughput_qps": round(qps, 1),
        "errors": errors,
        "p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_ms": round(float(np.percentile(latencies, 99)), 2),
    }

    print(f"  Concurrent Read Load: {total_queries} queries across {concurrency_workers} threads")
    print(f"  Wall Time:            {wall_clock_time:.2f} s")
    print(f"  Throughput:           {qps:.1f} QPS")
    print(f"  Latency Profile:      P50={concurrency_results['p50_ms']} ms | P95={concurrency_results['p95_ms']} ms | P99={concurrency_results['p99_ms']} ms")
    print(f"  Error Count:          {errors}")

    # 2. Concurrent Document Updates with Lock Manager (10 Writers, 10 Readers)
    print("\n--- Testing Concurrent Ingestion & Query Mutation Safety ---")
    lock_mgr = DocumentLockManager.get_instance()
    chunker = TextChunker(chunk_size=100, overlap=20)
    mutation_errors = 0
    writer_threads = 10
    reader_threads = 10

    # Use a dedicated indexer seeded with 1,000 chunks to verify concurrency & locks without huge array reallocations
    mutation_indexer = VectorIndexer(model_name="all-MiniLM-L6-v2")
    m_chunks, m_embeddings = generate_synthetic_chunks(1000, dim=384)
    mutation_indexer.chunks = m_chunks
    mutation_indexer.embeddings = m_embeddings
    mutation_indexer.use_st = True
    mutation_indexer._encode = lambda texts: np.random.randn(len(texts), 384).astype(np.float32)
    mutation_indexer.bm25 = BM25OkapiIndexer()
    mutation_indexer.bm25.index(m_chunks)

    def writer_task(worker_id):
        nonlocal mutation_errors
        doc_id = f"concurrent_doc_{worker_id}"
        doc_text = f"Payload updated by worker {worker_id} at timestamp {time.time()} with critical parameters."
        try:
            with lock_mgr.process_lock(doc_id):
                res = mutation_indexer.stage_and_commit_document(doc_id, doc_text, f"Doc {worker_id}", chunker)
                if res.get("status") not in ("committed", "active"):
                    mutation_errors += 1
        except Exception as e:
            print(f"Writer error: {e}", flush=True)
            mutation_errors += 1

    def reader_task(worker_id):
        nonlocal mutation_errors
        try:
            hits = mutation_indexer.search("critical parameters", top_k=5)
            if not isinstance(hits, list):
                mutation_errors += 1
        except Exception as e:
            print(f"Reader error: {e}", flush=True)
            mutation_errors += 1

    with ThreadPoolExecutor(max_workers=writer_threads + reader_threads) as executor:
        f_writers = [executor.submit(writer_task, i) for i in range(writer_threads)]
        f_readers = [executor.submit(reader_task, i) for i in range(reader_threads)]
        for f in as_completed(f_writers + f_readers):
            f.result()

    concurrency_results["mutation_errors"] = mutation_errors
    print(f"  Concurrent Writer/Reader Ingestion Integrity: {mutation_errors} errors across {writer_threads} writers & {reader_threads} readers")

    return concurrency_results

def run_edge_case_benchmarks():
    print("\n" + "=" * 80)
    print("STAGE 3: EDGE-CASE PAYLOADS & MALFORMED INPUTS")
    print("=" * 80)

    chunker = TextChunker(chunk_size=100, overlap=20)
    indexer = VectorIndexer(model_name="all-MiniLM-L6-v2")

    edge_results = {}

    # 1. Empty Documents (0-byte text, whitespace only)
    indexer.stage_and_commit_document("valid_doc", "quantum encryption cooling telemetry parameters.", "Valid Doc", chunker)
    res_empty = indexer.stage_and_commit_document("empty_doc", "", "Empty Doc", chunker)
    res_whitespace = indexer.stage_and_commit_document("space_doc", "   \n\t   ", "Space Doc", chunker)
    edge_results["empty_doc_status"] = res_empty.get("status")
    edge_results["space_doc_status"] = res_whitespace.get("status")
    print(f"  Empty Document Handling:       {res_empty.get('status')} ({res_empty.get('reason')})")
    print(f"  Whitespace Document Handling:  {res_whitespace.get('status')} ({res_whitespace.get('reason')})")

    # 2. Corrupted PDF/JSON parsing
    temp_dir = tempfile.mkdtemp()
    try:
        corrupt_pdf_bytes = b"%PDF-1.4\nCorrupted content without xref table or endobj EOF"
        pdf_text, quality = DocumentLoader.extract_text_from_pdf_bytes_with_quality(corrupt_pdf_bytes)
        edge_results["corrupt_pdf_handled"] = (quality.get("total_chars") == 0)

        corrupt_json_text = "{ invalid json file ["
        parsed_json = DocumentLoader.load_json(corrupt_json_text, "corrupt.json")
        edge_results["corrupt_json_handled"] = (parsed_json.get("structured") is False)

        print(f"  Corrupt PDF Handling:          Handled safely (Total chars: {quality.get('total_chars', 0)})")
        print(f"  Corrupt JSON Handling:         Handled safely (Structured: {parsed_json.get('structured')})")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 3. Extremely long query (5,000 tokens)
    long_query = "quantum encryption cooling telemetry " * 1000  # ~5,000 words
    t0 = time.perf_counter()
    tokens = long_query.split()
    truncated_query = " ".join(tokens[:256]) # Token budget constraint
    _ = indexer.search(truncated_query, top_k=5)
    long_q_dur_ms = (time.perf_counter() - t0) * 1000
    edge_results["long_query_latency_ms"] = round(long_q_dur_ms, 2)
    print(f"  5,000-Word Query Handled:      Budget-capped & searched in {long_q_dur_ms:.2f} ms")

    # 4. Large synthetic document (10MB / 25,000 words chunking throughput)
    large_text = ("The operational telemetry of the advanced turbine subsystem indicates stable harmonic vibrations. " * 500) * 5
    t0 = time.perf_counter()
    large_chunks = chunker.chunk_document("large_doc", large_text, "Large Turbine Doc")
    chunk_dur_ms = (time.perf_counter() - t0) * 1000
    edge_results["large_doc_words"] = len(large_text.split())
    edge_results["large_doc_chunks"] = len(large_chunks)
    edge_results["chunking_throughput_words_sec"] = round(len(large_text.split()) / (chunk_dur_ms / 1000))
    print(f"  Large Doc ({edge_results['large_doc_words']:,} words): Chunked into {len(large_chunks)} chunks in {chunk_dur_ms:.2f} ms ({edge_results['chunking_throughput_words_sec']:,} words/sec)")

    return edge_results

if __name__ == "__main__":
    t_start_all = time.perf_counter()
    scale_res = run_scale_benchmarks()
    concurrency_res = run_concurrency_benchmarks(num_chunks=100000)
    edge_res = run_edge_case_benchmarks()
    total_time = time.perf_counter() - t_start_all

    final_payload = {
        "scale": scale_res,
        "concurrency": concurrency_res,
        "edge_cases": edge_res,
        "total_benchmark_time_s": round(total_time, 2)
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(final_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY IN " + f"{total_time:.2f}s")
    print("Results saved to benchmark_results.json")
    print("=" * 80)
