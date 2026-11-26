#!/usr/bin/env python3
"""
Interactive Terminal CLI Chat Loop for Tiny RAG Pipeline.
Allows chatting with indexed documents, inspecting citations, adjusting guardrails,
and ingesting local files (.pdf, .txt, .md) directly in the console.
"""

import os
import sys
import readline  # Enable arrow key navigation & input history
from typing import List, Dict, Any

from rag_core import (
    TextChunker,
    VectorIndexer,
    PromptAssembler,
    LLMGenerator,
    GuardrailEngine,
    EvalHarness,
    DocumentLoader,
    ConversationManager,
    ExecutionMode
)

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class TinyRAGCLI:
    def __init__(self):
        self.chunker = TextChunker(chunk_size=150, overlap=30)
        self.indexer = VectorIndexer(model_name="all-MiniLM-L6-v2")
        self.guardrails = GuardrailEngine(min_similarity_threshold=0.35 if self.indexer.use_ollama else 0.20)
        self.execution_mode = os.environ.get("RAG_EXECUTION_MODE", ExecutionMode.STRICT_OFFLINE)
        self.backend = "auto"
        self.model = None
        self.api_key = None
        self.top_k = 3
        self.retrieval_mode = os.environ.get("RAG_DEFAULT_RETRIEVAL_MODE", "hybrid").lower()
        self.enforce_guardrails = True
        self.conversation_manager = ConversationManager(max_history_turns=5)
        self.session_id = "cli_default"

        self._seed_default_corpus()

    def _seed_default_corpus(self):
        # Auto-ingest sample_data if exists
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
        if os.path.isdir(sample_dir):
            docs = DocumentLoader.load_directory(sample_dir, recursive=False)
            all_chunks = []
            for doc in docs:
                c = self.chunker.chunk_document(doc["doc_id"], doc["text"], doc["title"])
                all_chunks.extend(c)
            self.indexer.add_chunks(all_chunks)

    def print_banner(self):
        print(f"\n{MAGENTA}{BOLD}========================================================================{RESET}")
        print(f"{CYAN}{BOLD}          ✦ PRODUCTION-GRADE OFFLINE RAG PIPELINE ✦          {RESET}")
        print(f"{DIM}    Hybrid Search, RRF, Cross-Encoder Reranking & Claim Verification    {RESET}")
        print(f"{MAGENTA}{BOLD}========================================================================{RESET}")
        ollama_status = LLMGenerator.check_ollama_status()
        ollama_str = f"{GREEN}Online{RESET} ({len(ollama_status['models'])} models)" if ollama_status['running'] else f"{YELLOW}Offline{RESET}"
        print(f" • Generator Mode : {BOLD}{self.backend.upper()}{RESET} (Default: Auto)")
        policy_label = "Air-Gapped Safe (Zero Egress)" if self.execution_mode == ExecutionMode.STRICT_OFFLINE else "Permitted Egress"
        print(f" • Execution Policy: {BOLD}{self.execution_mode.upper()}{RESET} ({policy_label})")
        print(f" • Ollama Daemon  : {ollama_str}")
        print(f" • Indexed Chunks : {GREEN}{BOLD}{len(self.indexer.chunks)}{RESET} chunks")
        print(f" • Type {CYAN}/help{RESET} for commands or ask any question directly.\n")

    def print_help(self):
        print(f"\n{BOLD}Available CLI Commands:{RESET}")
        print(f"  {CYAN}/ingest <path>{RESET}      Ingest a file (.pdf, .txt, .md) or whole folder")
        print(f"  {CYAN}/policy <name>{RESET}      Set execution policy ({YELLOW}strict_offline{RESET}, {YELLOW}hybrid{RESET}, {YELLOW}cloud{RESET})")
        print(f"  {CYAN}/retriever <name>{RESET}   Set retrieval mode ({YELLOW}hybrid{RESET}, {YELLOW}dense{RESET}, {YELLOW}bm25{RESET}, current: {self.retrieval_mode})")
        print(f"  {CYAN}/mode <name>{RESET}        Set backend ({YELLOW}auto{RESET}, {YELLOW}ollama{RESET}, {YELLOW}gemini{RESET}, {YELLOW}offline{RESET})")
        print(f"  {CYAN}/model <name>{RESET}       Set model (e.g. {YELLOW}llama3.2:1b{RESET}, {YELLOW}qwen2.5:1.5b{RESET})")
        print(f"  {CYAN}/topk <int>{RESET}         Set Top-K retrieved chunks (current: {self.top_k})")
        print(f"  {CYAN}/threshold <val>{RESET}    Set Guardrail similarity threshold (current: {self.guardrails.min_similarity_threshold})")
        print(f"  {CYAN}/guardrails{RESET}         Toggle guardrails ON/OFF (current: {self.enforce_guardrails})")
        print(f"  {CYAN}/history{RESET}            Show active multi-turn conversation memory buffer")
        print(f"  {CYAN}/new{RESET}                Start a new conversation session (clear history)")
        print(f"  {CYAN}/status{RESET}             Show vector store size and backend health")
        print(f"  {CYAN}/eval{RESET}               Run Recall@K quantitative evaluation benchmark")
        print(f"  {CYAN}/clear{RESET}              Clear terminal screen")
        print(f"  {CYAN}/reset{RESET}              Clear all indexed chunks from memory")
        print(f"  {CYAN}/exit{RESET} or {CYAN}/quit{RESET}    Exit the CLI session\n")

    def ingest_path(self, path_str: str):
        path_str = path_str.strip().strip("'\"")
        if not os.path.exists(path_str):
            print(f"{RED}Error: Path '{path_str}' not found.{RESET}")
            return

        print(f"{DIM}Ingesting '{path_str}'...{RESET}")
        docs = []
        if os.path.isdir(path_str):
            docs = DocumentLoader.load_directory(path_str, recursive=True)
        else:
            try:
                docs = [DocumentLoader.load_file(path_str)]
            except Exception as e:
                print(f"{RED}Error loading file: {e}{RESET}")
                return

        new_chunks = []
        for d in docs:
            c = self.chunker.chunk_document(d["doc_id"], d["text"], d["title"])
            new_chunks.extend(c)

        self.indexer.add_chunks(new_chunks)
        print(f"{GREEN}✓ Successfully ingested {len(docs)} document(s) ({len(new_chunks)} chunks). Total index: {len(self.indexer.chunks)} chunks.{RESET}")

    def run_eval(self):
        print(f"{DIM}Running Recall@K benchmark on test queries...{RESET}")
        eval_dataset = [
            {"query": "What is quantization for tiny models?", "relevant_doc_ids": ["quantization_primer"]},
            {"query": "Tell me about prompt caching and context windows", "relevant_doc_ids": ["prompt_caching"]},
            {"query": "How do on-device tiny LLMs work?", "relevant_doc_ids": ["edge_ai_guide"]}
        ]
        metrics = EvalHarness.calculate_recall_at_k(eval_dataset, self.indexer, k=self.top_k)
        print(f"\n{BOLD}Evaluation Results:{RESET}")
        print(f" • Recall@{self.top_k} : {CYAN}{BOLD}{metrics['recall_at_k'] * 100:.1f}%{RESET}")
        print(f" • Hits        : {GREEN}{metrics['hits']}{RESET} / {metrics['total_queries']}")
        for d in metrics["details"]:
            status_icon = f"{GREEN}✓ HIT {RESET}" if d["hit"] else f"{RED}✗ MISS{RESET}"
            print(f"   [{status_icon}] Query: {d['query']} -> Retrieved: {d['retrieved']}")
        print()

    def process_query(self, query: str):
        # 1. Memory & Contextual Query Condensation
        history = self.conversation_manager.get_history(self.session_id)
        standalone_query = query
        if history:
            standalone_query = self.conversation_manager.condense_query(
                session_id=self.session_id,
                query=query,
                backend=self.backend,
                model=self.model,
                api_key=self.api_key
            )
            if standalone_query.strip().lower() != query.strip().lower():
                print(f"{DIM}🔍 Rewritten retrieval query: {CYAN}{standalone_query}{RESET}")

        # 2. Search using standalone_query with configured retrieval mode (hybrid, dense, bm25)
        mode = getattr(self, "retrieval_mode", "hybrid").lower()
        if mode == "hybrid":
            retrieved = self.indexer.search_hybrid(standalone_query, top_k=self.top_k)
        elif mode == "bm25":
            if not hasattr(self.indexer, "bm25") or self.indexer.bm25 is None:
                from rag_core import BM25OkapiIndexer
                self.indexer.bm25 = BM25OkapiIndexer()
                self.indexer.bm25.index(self.indexer.chunks)
            bm25_scored = self.indexer.bm25.search(standalone_query, top_k=self.top_k)
            retrieved = [dict(self.indexer.chunks[idx], score=sc, modality="bm25") for idx, sc in bm25_scored]
        else:
            retrieved = self.indexer.search(standalone_query, top_k=self.top_k)

        # 3. Guardrails Check
        if self.enforce_guardrails:
            passed, reason = self.guardrails.evaluate_retrieval(retrieved, query=standalone_query)
            if not passed:
                print(f"\n{RED}{BOLD}[🚫 BLOCKED BY GUARDRAIL]{RESET}")
                print(f"{DIM}Reason: {reason}{RESET}")
                print(f"{BOLD}Answer:{RESET} I cannot answer this based on the provided context.\n")
                self.conversation_manager.add_turn(
                    session_id=self.session_id,
                    user_query=query,
                    assistant_answer="I cannot answer this based on the provided context.",
                    citations=[],
                    rewritten_query=standalone_query if standalone_query != query else None
                )
                return

        # 4. Prompt Assembly with history + context
        prompt = PromptAssembler.build_prompt(
            query=query,
            retrieved_chunks=retrieved,
            conversation_history=history
        )

        # 5. Generate
        print(f"{DIM}Generating answer...{RESET}")
        output = LLMGenerator.generate(
            prompt=prompt,
            retrieved_chunks=retrieved,
            query=standalone_query,
            backend=self.backend,
            model=self.model,
            api_key=self.api_key,
            execution_mode=self.execution_mode
        )

        # 6. Record Turn in Conversation Memory
        self.conversation_manager.add_turn(
            session_id=self.session_id,
            user_query=query,
            assistant_answer=output["answer"],
            citations=output.get("provenance", []),
            rewritten_query=standalone_query if standalone_query != query else None
        )

        # 7. Output
        guard_badge = f"{GREEN}[🛡️ Guardrail PASSED]{RESET}" if self.enforce_guardrails else f"{YELLOW}[Guardrails OFF]{RESET}"
        backend_badge = f"{MAGENTA}[⚡ Backend: {output.get('backend_used', self.backend)} | Model: {output.get('model_used', 'none')}]{RESET}"
        history_badge = f"{CYAN}[💬 Turns: {len(history) + 1}]{RESET}"
        
        print(f"\n{guard_badge} {backend_badge} {history_badge}")
        print(f"{BOLD}Answer:{RESET}\n{output['answer']}\n")

        # Provenance breakdown
        print(f"{DIM}--- Provenance Citations ---{RESET}")
        for i, chunk in enumerate(retrieved, 1):
            score = chunk.get('score', 0.0)
            print(f" [{i}] {CYAN}{chunk['chunk_id']}{RESET} | Doc: {chunk.get('title', chunk['doc_id'])} | Score: {GREEN}{score:.4f}{RESET}")
        print()

    def run(self):
        self.print_banner()
        while True:
            try:
                line = input(f"{BOLD}{CYAN}RAG>{RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{DIM}Exiting session. Goodbye!{RESET}")
                break

            if not line:
                continue

            if line.startswith("/"):
                parts = line.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/exit", "/quit", "/q"):
                    print(f"{DIM}Exiting session. Goodbye!{RESET}")
                    break
                elif cmd == "/help":
                    self.print_help()
                elif cmd == "/clear":
                    os.system("clear" if os.name != "nt" else "cls")
                    self.print_banner()
                elif cmd == "/ingest":
                    if not arg:
                        print(f"{YELLOW}Usage: /ingest <folder_or_file_path>{RESET}")
                    else:
                        self.ingest_path(arg)
                elif cmd == "/policy":
                    p = arg.lower().strip()
                    if p in (ExecutionMode.STRICT_OFFLINE, ExecutionMode.HYBRID, ExecutionMode.CLOUD):
                        self.execution_mode = p
                        print(f"{GREEN}✓ Execution policy set to: {self.execution_mode.upper()}{RESET}")
                    else:
                        print(f"{YELLOW}Valid execution policies: strict_offline, hybrid, cloud{RESET}")
                elif cmd in ("/retriever", "/retrieval"):
                    r_mode = arg.lower().strip()
                    if r_mode in ("hybrid", "dense", "bm25"):
                        self.retrieval_mode = r_mode
                        print(f"{GREEN}✓ Retrieval mode set to: {self.retrieval_mode.upper()}{RESET}")
                    else:
                        print(f"{YELLOW}Valid retrieval modes: hybrid, dense, bm25{RESET}")
                elif cmd == "/mode":
                    if arg.lower() in ("auto", "ollama", "gemini", "offline"):
                        self.backend = arg.lower()
                        print(f"{GREEN}✓ Generator mode set to: {self.backend.upper()}{RESET}")
                    else:
                        print(f"{YELLOW}Valid modes: auto, ollama, gemini, offline{RESET}")
                elif cmd == "/model":
                    self.model = arg.strip()
                    print(f"{GREEN}✓ Target model set to: {self.model}{RESET}")
                elif cmd == "/topk":
                    try:
                        self.top_k = int(arg)
                        print(f"{GREEN}✓ Top-K set to: {self.top_k}{RESET}")
                    except ValueError:
                        print(f"{RED}Please provide an integer for Top-K.{RESET}")
                elif cmd == "/threshold":
                    try:
                        self.guardrails.min_similarity_threshold = float(arg)
                        print(f"{GREEN}✓ Guardrail threshold set to: {self.guardrails.min_similarity_threshold}{RESET}")
                    except ValueError:
                        print(f"{RED}Please provide a float between 0.0 and 1.0.{RESET}")
                elif cmd == "/guardrails":
                    self.enforce_guardrails = not self.enforce_guardrails
                    state = f"{GREEN}ON{RESET}" if self.enforce_guardrails else f"{RED}OFF{RESET}"
                    print(f"Guardrail enforcement is now: {state}")
                elif cmd == "/status":
                    ollama = LLMGenerator.check_ollama_status()
                    print(f"\n{BOLD}System Status:{RESET}")
                    print(f" • Indexed Chunks : {len(self.indexer.chunks)}")
                    print(f" • Execution Policy: {self.execution_mode.upper()}")
                    print(f" • Backend Mode   : {self.backend}")
                    print(f" • Ollama Running : {ollama['running']} (Host: {ollama['host']})")
                    print(f" • Ollama Models  : {ollama['models']}")
                    print(f" • Top-K          : {self.top_k}")
                    print(f" • Min Threshold  : {self.guardrails.min_similarity_threshold}\n")
                elif cmd == "/eval":
                    self.run_eval()
                elif cmd == "/history":
                    hist = self.conversation_manager.get_history(self.session_id)
                    if not hist:
                        print(f"{YELLOW}No conversation turns recorded yet in this session.{RESET}")
                    else:
                        print(f"\n{BOLD}Conversation History ({len(hist)} turns):{RESET}")
                        for i, t in enumerate(hist, 1):
                            rw = f" {DIM}(query rewritten as: '{t['rewritten_query']}'){RESET}" if t.get('rewritten_query') else ""
                            print(f" {CYAN}[Turn {i}] User:{RESET} {t['user']}{rw}")
                            print(f"          {GREEN}AI:{RESET}   {t['assistant'][:120]}...\n")
                elif cmd in ("/new", "/clearchat"):
                    self.conversation_manager.clear_session(self.session_id)
                    print(f"{GREEN}✓ Started fresh conversation session. Memory cleared.{RESET}")
                elif cmd == "/reset":
                    self.indexer.clear()
                    print(f"{YELLOW}✓ Vector index cleared. (0 chunks remaining){RESET}")
                else:
                    print(f"{YELLOW}Unknown command '{cmd}'. Type /help for options.{RESET}")
            else:
                self.process_query(line)


if __name__ == "__main__":
    cli = TinyRAGCLI()
    cli.run()
