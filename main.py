"""
FastAPI REST Service exposing all 4 Phases of the Offline RAG Pipeline + Web UI:
Phase 1: Naive baseline (/generate-naive)
Phase 2: Ingest pipeline (/ingest, /upload, /ingest-folder, /index-status, /clear-index)
Phase 3: Retrieve & Generate & Eval (/retrieve, /generate-rag, /eval-recall)
Phase 4: Guardrails (/generate-guarded)
System & Models Status: (/models-status)
Interactive Web UI: (/ and /ui)
"""

import os
import time
import secrets
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Header, Query, Depends, Security, APIRouter, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("rag_api.auth")

from rag_core import (
    TextChunker,
    VectorIndexer,
    PromptAssembler,
    OfflineGenerator,
    LLMGenerator,
    GuardrailEngine,
    EvalHarness,
    DocumentLoader,
    ConversationManager,
    ControlledQueryRouter,
    RouteTarget,
    RoutingDecision,
    ToolPermission,
    PromptInjectionGuard,
    PromptInjectionAssessment,
    DeterministicComputationEngine,
    ComputationResult,
    ProgrammaticCitationAssigner,
    StructuredOutputValidator,
    SmallModelCompatibilityAudit,
    OperationalObservabilityDashboard,
    StrictOfflineGuard,
    AuthenticatedPrincipal,
    PrincipalRegistry
)

app = FastAPI(
    title="Production-Grade Offline RAG Pipeline",
    description="Production-grade offline RAG pipeline combining BM25 keyword retrieval and dense-vector retrieval with Reciprocal Rank Fusion (RRF), cross-encoder reranking, transactional rollback, propositional claim verification, and multi-turn memory.",
    version="2.0.0"
)

# Authentication Schemes & Server-Side Identity Layer
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

principal_registry = PrincipalRegistry()

async def get_current_principal(
    request: Request,
    header_key: Optional[str] = Security(api_key_header_scheme),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> AuthenticatedPrincipal:
    """
    Derives caller identity and authorization claims server-side from validated authentication.
    Privilege-bearing claims (roles, clearance_level, user_id, department) are NEVER accepted
    from arbitrary caller request fields.
    Checks:
    1. X-API-Key header (via Security scheme)
    2. Authorization: Bearer <key> header (via Security scheme)
    3. api_key query parameter
    4. JSON body 'api_key' field (if application/json)
    """
    expected = os.environ.get("RAG_API_KEY")

    provided = header_key or request.headers.get("X-API-Key")
    if not provided and bearer_token:
        provided = bearer_token.credentials
    if not provided:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            provided = auth_header.replace("Bearer ", "").strip()

    if not provided:
        provided = request.query_params.get("api_key")

    if not provided and request.headers.get("content-type", "").startswith("application/json"):
        try:
            body = await request.json()
            if isinstance(body, dict):
                provided = body.get("api_key")
        except Exception:
            pass

    # 1. Check if token maps directly to an authoritative principal in the server-side registry
    if provided:
        registered_principal = principal_registry.resolve(provided)
        if registered_principal:
            return registered_principal

    # 2. When RAG_API_KEY is configured in the environment:
    if expected:
        if not provided or not secrets.compare_digest(str(provided), str(expected)):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid or missing API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Primary service API key maps to authoritative configured service principal
        service_roles = [r.strip().lower() for r in os.environ.get("RAG_SERVICE_ROLES", "admin,user").split(",") if r.strip()]
        service_clearance = os.environ.get("RAG_SERVICE_CLEARANCE", "top_secret").strip().lower()
        return AuthenticatedPrincipal(
            principal_id=os.environ.get("RAG_SERVICE_PRINCIPAL_ID", "primary_service_account"),
            roles=service_roles,
            clearance_level=service_clearance,
            department="system",
            auth_source="primary_rag_api_key"
        )

    # 3. Open Developer Mode: When RAG_API_KEY is unset, use default developer principal
    return principal_registry.get_default_developer_principal()


async def verify_api_key(
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
) -> AuthenticatedPrincipal:
    """Enforces authentication dependency on protected router."""
    return principal


def build_authoritative_user_context(
    principal: AuthenticatedPrincipal,
    client_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Derives authoritative user_context strictly from the authenticated principal.
    Privilege-bearing claims (roles, clearance_level, user_id, department)
    are derived EXCLUSIVELY from the validated server principal.
    Any caller-supplied privilege-bearing claims in client_context are stripped
    to prevent privilege escalation.
    """
    auth_ctx = principal.to_user_context()
    if not client_context:
        return auth_ctx

    tampered = []
    for key in ("roles", "clearance_level", "user_id", "department"):
        if key in client_context and client_context[key] != auth_ctx.get(key):
            tampered.append(f"{key}={client_context[key]}")
    if tampered:
        logger.warning(
            f"[Security Audit] Blocked caller-supplied privilege escalation attempt: {tampered}. "
            f"Enforcing validated principal claims: principal_id='{principal.principal_id}', roles={principal.roles}."
        )

    # Retain harmless non-privilege contextual parameters, but strictly overwrite privilege claims
    safe_ctx = {k: v for k, v in client_context.items() if k not in ("roles", "clearance_level", "user_id", "department")}
    safe_ctx.update(auth_ctx)
    return safe_ctx


def resolve_scoped_session(principal: AuthenticatedPrincipal, requested_session_id: Optional[str]) -> Tuple[str, str]:
    """
    Resolves client-requested session identifier to an authenticated-user scoped session ID.
    Returns (client_visible_session_id, internal_scoped_session_id).
    Prevents cross-caller conversation history leaks when session_id is omitted or set to 'default'.
    """
    raw_session = (requested_session_id or "default").strip()
    scoped_session = ConversationManager.scoped_session_id(principal.principal_id, raw_session)
    return raw_session, scoped_session


# ==============================================================================
# ROUTER ARCHITECTURE: Explicit Separation of Public vs. Protected Endpoints
# ==============================================================================
public_router = APIRouter(tags=["Public & System"])
protected_router = APIRouter(dependencies=[Depends(verify_api_key)], tags=["Data & Operations"])

# Global in-memory state
chunker = TextChunker(chunk_size=200, overlap=40)
indexer = VectorIndexer(model_name="all-MiniLM-L6-v2", ollama_embed_model="nomic-embed-text")
guardrails = GuardrailEngine(min_similarity_threshold=0.35 if indexer.use_ollama else 0.20)
conversation_manager = ConversationManager(max_history_turns=5)
query_router = ControlledQueryRouter()
dashboard = OperationalObservabilityDashboard.get_instance()

# Sample benchmark corpus for demonstration
SAMPLE_DOCUMENTS = [
    {
        "doc_id": "doc_rag_overview",
        "title": "Retrieval-Augmented Generation",
        "text": "Retrieval-Augmented Generation (RAG) optimizes the output of a language model by referencing an authoritative knowledge base outside its training data sources before generating a response. RAG reduces hallucinations and grounds responses in factual domain knowledge."
    },
    {
        "doc_id": "doc_vector_index",
        "title": "Dense Vector Indexing & Cosine Similarity",
        "text": "Vector databases convert document chunks into high-dimensional numerical dense embeddings. Cosine similarity calculates the dot product between normalized query vectors and document vectors to identify top-k relevant context items."
    },
    {
        "doc_id": "doc_guardrails",
        "title": "LLM Guardrails and Faithfulness",
        "text": "Guardrails evaluate input queries and retrieved contexts before sending them to the generator. A confidence score threshold filters out weak context matches, instructing the model to reply 'I cannot answer this based on the provided context' when evidence is lacking."
    }
]

# Request / Response Schemas
class DocumentItem(BaseModel):
    doc_id: str
    title: str = ""
    text: str
    metadata: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    documents: List[DocumentItem]
    chunk_size: Optional[int] = 150
    overlap: Optional[int] = 30

class IngestFolderRequest(BaseModel):
    folder_path: str
    recursive: Optional[bool] = True
    chunk_size: Optional[int] = 150
    overlap: Optional[int] = 30

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    backend: Optional[str] = "auto"
    execution_mode: Optional[str] = "strict_offline"
    model: Optional[str] = None
    api_key: Optional[str] = None
    threshold: Optional[float] = None
    session_id: Optional[str] = "default"
    user_id: Optional[str] = "default"
    use_memory: Optional[bool] = True
    filters: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    auto_route: Optional[bool] = False
    small_model_mode: Optional[bool] = False
    is_untrusted_source: Optional[bool] = False

MAX_UPLOAD_SIZE = 25 * 1024 * 1024  # 25 MB enterprise upload guard (Item 12)

class EvalItem(BaseModel):
    query: str
    relevant_doc_ids: List[str]

class EvalRequest(BaseModel):
    dataset: List[EvalItem]
    k: Optional[int] = 3

class InjectionScanRequest(BaseModel):
    text: str

class FactualComputeRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    filters: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None


@app.on_event("startup")
def auto_ingest_sample_corpus():
    """Seed index on startup with sample corpus and sample_data files if present."""
    all_chunks = []
    for doc in SAMPLE_DOCUMENTS:
        c = chunker.chunk_document(doc["doc_id"], doc["text"], doc["title"])
        all_chunks.extend(c)
    
    # Also load sample_data directory if it exists
    if os.path.isdir("sample_data"):
        try:
            extra_docs = DocumentLoader.load_directory("sample_data", recursive=False)
            for doc in extra_docs:
                c = chunker.chunk_document(doc["doc_id"], doc["text"], doc["title"])
                all_chunks.extend(c)
        except Exception as e:
            print(f"[Startup] Error auto-loading sample_data: {e}")

    indexer.add_chunks(all_chunks)


# ==========================================
# PUBLIC ENDPOINTS: UI DASHBOARD & SYSTEM HEALTH
# (Unauthenticated: accessible by browsers, Prometheus, Kubernetes probes)
# ==========================================
@public_router.get("/", response_class=HTMLResponse)
@public_router.get("/ui", response_class=HTMLResponse)
def serve_ui(request: Request):
    """Serves the interactive web UI dashboard."""
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return HTMLResponse("<h2>Tiny RAG Studio UI file not found.</h2>", status_code=404)


@public_router.get("/observability")
def get_observability_metrics():
    """Returns operational dashboard metrics including P50/P90/P95 latencies, QPS, cache hit rates, guardrails, and document states."""
    return dashboard.get_metrics(indexer)


@public_router.get("/health")
def get_health_status():
    """Returns system health check status, offline policy adherence, index generation, and uptime."""
    return dashboard.get_health_status(indexer)


@public_router.get("/models-status")
def models_status():
    """Returns status of local Ollama, Gemini API, and index stats."""
    ollama_status = LLMGenerator.check_ollama_status()
    gemini_status = LLMGenerator.check_gemini_status()
    active_docs = 0
    if hasattr(indexer, 'source_documents') and indexer.source_documents:
        active_docs = sum(1 for m in indexer.source_documents.values() if m.get("status") != "deleted")
    elif hasattr(indexer, 'version_registry') and indexer.version_registry:
        active_docs = sum(1 for m in indexer.version_registry.registry.values() if m.get("status") != "deleted")

    return {
        "ollama": ollama_status,
        "gemini": gemini_status,
        "strict_offline": StrictOfflineGuard.is_strict_offline(),
        "index": {
            "total_chunks": len(indexer.chunks),
            "documents_indexed": len(set(c["doc_id"] for c in indexer.chunks)),
            "active_documents": active_docs,
            "index_generation": getattr(indexer, 'index_generation', 1),
            "embedding_model": indexer.embedding_model,
            "embedding_backend": indexer.embedding_backend,
            "storage_backend": "sqlite_disk" if getattr(indexer, 'disk_store', None) else "numpy_memory"
        }
    }


# ==========================================
# PHASE 1 — Naive Baseline (No Retrieval)
# ==========================================
@protected_router.post("/generate-naive")
def generate_naive(req: QueryRequest):
    """
    Phase 1: Demonstrates response generation WITHOUT retrieval.
    The model lacks external context and will fail or provide generic answers.
    """
    prompt = f"USER QUESTION: {req.query}\nANSWER:"
    answer = f"[Naive Baseline Model Output]: I do not have external document context loaded. Question '{req.query}' cannot be verified against internal memory."
    return {
        "phase": "Phase 1: Naive Baseline",
        "query": req.query,
        "prompt_used": prompt,
        "answer": answer,
        "retrieval_used": False,
        "provenance": []
    }


# ==========================================
# PHASE 2 — Ingest Pipeline & File Loading
# ==========================================
@protected_router.post("/ingest")
def ingest_documents(req: IngestRequest):
    """
    Phase 2: Ingests documents, chunks them, embeds vectors, and indexes them in memory.
    """
    chunker.chunk_size = req.chunk_size or 150
    chunker.overlap = req.overlap or 30

    created_chunks = []
    for doc in req.documents:
        chunks = chunker.chunk_document(doc.doc_id, doc.text, doc.title, metadata=doc.metadata)
        created_chunks.extend(chunks)

    indexer.add_chunks(created_chunks)

    return {
        "phase": "Phase 2: Ingest Pipeline",
        "documents_processed": len(req.documents),
        "total_chunks_created": len(created_chunks),
        "total_index_size": len(indexer.chunks)
    }


@protected_router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Phase 2: Uploads one or more files (.pdf, .txt, .md, and code formats), chunks them, and adds to index.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    processed_count = 0
    new_chunks = []
    errors = []

    for file in files:
        filename = os.path.basename(file.filename or "uploaded_file")
        try:
            content = await file.read()
            if len(content) > MAX_UPLOAD_SIZE:
                errors.append(f"'{filename}': File exceeds maximum allowed upload limit of 25MB ({len(content)} bytes)")
                continue

            doc = DocumentLoader.load_bytes(filename, content)
            raw_text = doc.get("text", "").strip()
            if not raw_text:
                errors.append(f"'{filename}': File is empty or contains unreadable/scanned content")
                continue

            chunks = chunker.chunk_document(doc["doc_id"], raw_text, doc["title"])
            if not chunks:
                errors.append(f"'{filename}': Produced 0 chunks (insufficient text)")
                continue

            new_chunks.extend(chunks)
            processed_count += 1
        except Exception as e:
            err_msg = f"'{filename}': {str(e)}"
            errors.append(err_msg)
            print(f"[Upload Error] {err_msg}", flush=True)

    if new_chunks:
        indexer.add_chunks(new_chunks)

    if processed_count == 0 or len(new_chunks) == 0:
        detail_msg = "; ".join(errors) if errors else "No readable text chunks could be extracted from uploaded files."
        raise HTTPException(status_code=400, detail=detail_msg)

    return {
        "status": "success",
        "files_processed": processed_count,
        "chunks_created": len(new_chunks),
        "total_index_size": len(indexer.chunks),
        "warnings": errors if errors else None
    }


@protected_router.post("/ingest-folder")
def ingest_folder(req: IngestFolderRequest):
    """
    Phase 2: Scans a local folder on disk for .pdf, .txt, and .md files and indexes them.
    """
    if not os.path.isdir(req.folder_path):
        raise HTTPException(status_code=404, detail=f"Directory '{req.folder_path}' does not exist.")

    docs = DocumentLoader.load_directory(req.folder_path, recursive=req.recursive)
    new_chunks = []
    for doc in docs:
        c = chunker.chunk_document(doc["doc_id"], doc["text"], doc["title"])
        new_chunks.extend(c)

    if new_chunks:
        indexer.add_chunks(new_chunks)

    return {
        "status": "success",
        "folder_path": req.folder_path,
        "documents_processed": len(docs),
        "chunks_created": len(new_chunks),
        "total_index_size": len(indexer.chunks)
    }


@protected_router.post("/clear-index")
def clear_index(principal: AuthenticatedPrincipal = Depends(get_current_principal)):
    """
    Clears all indexed chunks from memory.
    Destructive mutation strictly restricted to administrators.
    """
    if not principal.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrative privileges required to clear index."
        )
    indexer.clear()
    return {
        "status": "cleared",
        "total_indexed_chunks": 0
    }


@protected_router.get("/index-status")
def index_status(
    include_text: bool = Query(False),
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Operational index status and health diagnostics.
    Omits full document text to prevent sensitive data leakage.
    Full text export requires administrator privileges and explicit include_text=true.
    """
    reg = getattr(indexer, 'version_registry', None)
    allow_full_text = include_text and principal.is_admin()

    chunks_telemetry = []
    for c in indexer.chunks:
        entry = {
            "chunk_id": c.get("chunk_id"),
            "doc_id": c.get("doc_id"),
            "char_count": len(c.get("text", "")),
            "metadata": c.get("metadata", {})
        }
        if allow_full_text:
            entry["text"] = c.get("text", "")
        chunks_telemetry.append(entry)

    return {
        "total_indexed_chunks": len(indexer.chunks),
        "index_generation": getattr(indexer, 'index_generation', 1),
        "embedding_model": indexer.embedding_model,
        "embedding_backend": indexer.embedding_backend,
        "storage_backend": "sqlite_disk" if getattr(indexer, 'disk_store', None) else "numpy_memory",
        "registry": reg.registry if reg else {},
        "source_documents_count": len(getattr(indexer, 'source_documents', {})),
        "chunks": chunks_telemetry
    }


# ==========================================
# PHASE 2B — Document Lifecycle Management
# ==========================================
class ReplaceDocumentRequest(BaseModel):
    text: str
    title: Optional[str] = ""
    metadata: Optional[Dict[str, Any]] = None

@protected_router.delete("/documents/{doc_id}")
def delete_document_endpoint(
    doc_id: str,
    hard_delete: Optional[bool] = False,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Deletes a document by ID, evicting chunks from dense/BM25 indices and invalidating cache.
    Destructive mutation strictly restricted to administrators.
    """
    if not principal.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrative privileges required to delete documents."
        )
    res = indexer.delete_document(doc_id, hard_delete=hard_delete)
    if res.get("status") == "not_found":
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return res

@protected_router.put("/documents/{doc_id}")
def replace_document_endpoint(
    doc_id: str,
    req: ReplaceDocumentRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Atomically replaces a document with new content and re-indexes dense/BM25 representations.
    Destructive mutation strictly restricted to administrators.
    """
    if not principal.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrative privileges required to replace documents."
        )
    try:
        res = indexer.replace_document(
            doc_id=doc_id,
            text=req.text,
            title=req.title or doc_id,
            chunker=chunker,
            metadata=req.metadata
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@protected_router.post("/documents/{doc_id}/restore")
def restore_document_endpoint(
    doc_id: str,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Restores a soft-deleted document from source document tracking and re-indexes it.
    Index restoration strictly restricted to administrators.
    """
    if not principal.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrative privileges required to restore documents."
        )
    try:
        res = indexer.restore_document(doc_id=doc_id, chunker=chunker)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@protected_router.post("/documents/cleanup-orphans")
def cleanup_orphans_endpoint(
    quarantine_malformed: bool = Query(True),
    reconcile_stale: bool = Query(True),
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Cleans up chunk lifecycle anomalies across 5 distinct states.
    Restricted to administrators.
    """
    if not principal.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrative privileges required to run orphan cleanup."
        )
    return indexer.cleanup_orphaned_chunks(
        quarantine_malformed=quarantine_malformed,
        reconcile_stale=reconcile_stale
    )

@protected_router.get("/documents/quarantine")
def list_quarantined_chunks_endpoint(
    include_content: bool = Query(False),
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Returns quarantined chunk diagnostics.
    Restricted to administrators to protect unvalidated or sensitive payloads.
    Omits full content text to prevent sensitive data exposure.
    """
    if not principal.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrator authorization required to view quarantined chunks."
        )

    quarantined = getattr(indexer, 'quarantined_chunks', [])
    sanitized = []
    for c in quarantined:
        entry = {
            "chunk_id": c.get("chunk_id", "unknown"),
            "doc_id": c.get("doc_id", "missing"),
            "reason": c.get("quarantine_reason", c.get("reason", "malformed_metadata")),
            "char_count": len(c.get("text", "")) if "text" in c else 0
        }
        if include_content:
            entry["content_preview"] = (c.get("text", "")[:100] + "...") if len(c.get("text", "")) > 100 else c.get("text", "")
        sanitized.append(entry)

    return {
        "status": "success",
        "total_quarantined": len(quarantined),
        "quarantined_chunks": sanitized
    }

@protected_router.get("/documents")
def list_documents_endpoint(
    include_deleted: Optional[bool] = Query(False),
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
    x_user_role: Optional[str] = Header(None, alias="X-User-Role"),
    x_clearance_level: Optional[str] = Header(None, alias="X-Clearance-Level"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    role: Optional[str] = Query(None),
    clearance_level: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None)
):
    """
    Lists all source documents with their versions, status, and chunk counts.
    Enforces authorization and lifecycle policy:
    - Identity and permissions are derived strictly server-side from the authenticated principal.
    - Untrusted caller-supplied headers and query parameters are ignored for authorization.
    - Ordinary users only see active documents they are authorized to view.
    - Administrators (verified via authenticated principal) can pass include_deleted=true
      to review deleted documents for audit & restoration.
    """
    if x_user_role or role or x_clearance_level or clearance_level:
        logger.warning(
            f"[Security Audit] Blocked caller-supplied privilege escalation headers/query parameters on /documents. "
            f"Enforcing authenticated principal claims: principal_id='{principal.principal_id}', roles={principal.roles}."
        )

    user_ctx = principal.to_user_context()
    filtered_docs = indexer.list_source_documents(user_context=user_ctx, include_deleted=include_deleted)
    return {
        "index_generation": getattr(indexer, 'index_generation', 1),
        "total_source_documents": len(filtered_docs),
        "documents": filtered_docs
    }

@protected_router.get("/documents/{doc_id}")
def get_document_endpoint(
    doc_id: str,
    include_deleted: Optional[bool] = Query(False),
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
    x_user_role: Optional[str] = Header(None, alias="X-User-Role"),
    x_clearance_level: Optional[str] = Header(None, alias="X-Clearance-Level"),
    x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
    role: Optional[str] = Query(None),
    clearance_level: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None)
):
    """
    Retrieves full source document metadata and lifecycle status.
    Enforces authorization and lifecycle policy:
    - Identity and permissions are derived strictly server-side from the authenticated principal.
    - Untrusted caller-supplied headers and query parameters are ignored for authorization.
    - If document does not exist: returns 404 Not Found.
    - If document is soft-deleted:
        * Administrators (verified via authenticated principal) passing include_deleted=true
          may access deleted content for restoration or auditing.
        * Ordinary users are denied access (returns 403 Forbidden with lifecycle policy explanation).
    - If document is active:
        * Enforces document-level clearance and role access control (RBAC/ABAC).
    """
    if x_user_role or role or x_clearance_level or clearance_level:
        logger.warning(
            f"[Security Audit] Blocked caller-supplied privilege escalation headers/query parameters on /documents/{doc_id}. "
            f"Enforcing authenticated principal claims: principal_id='{principal.principal_id}', roles={principal.roles}."
        )

    user_ctx = principal.to_user_context()

    raw_doc = None
    if hasattr(indexer, 'source_documents') and doc_id in indexer.source_documents:
        raw_doc = indexer.source_documents[doc_id]
    elif hasattr(indexer, 'version_registry') and doc_id in indexer.version_registry.registry:
        raw_doc = indexer.version_registry.registry[doc_id]

    if not raw_doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    is_deleted = raw_doc.get("status") == "deleted" or bool(raw_doc.get("deleted"))
    is_admin = principal.is_admin()

    if is_deleted:
        if not is_admin:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied: Document '{doc_id}' has been soft-deleted. Administrator authorization required."
            )
        if not include_deleted:
            raise HTTPException(
                status_code=403,
                detail=f"Document '{doc_id}' is soft-deleted. Set include_deleted=true to retrieve deleted content for restoration."
            )

    doc = indexer.get_document_status(doc_id, user_context=user_ctx, include_deleted=include_deleted)
    if not doc:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: Insufficient role or clearance level for document '{doc_id}'."
        )

    return doc


# ==========================================
# PHASE 3 — Retrieve, Generate & Evaluate
# ==========================================
@protected_router.post("/route-query")
def route_query_endpoint(
    req: QueryRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Classifies the user query into one of predefined routes (Dense, BM25, SQL, Document Lookup, Multi-Hop, Clarification, Hybrid Structured)
    with explicit execution permission checks.
    Authoritative identity and permissions are derived strictly server-side from the authenticated principal.
    """
    t0 = time.perf_counter()
    trusted_user_context = build_authoritative_user_context(principal, req.user_context)
    known_docs = list(set(c.get("doc_id", "") for c in indexer.chunks))
    history = conversation_manager.get_history(req.session_id or "default") if req.use_memory else None
    decision = query_router.route_query(
        query=req.query,
        history=history,
        user_context=trusted_user_context,
        execution_mode=req.execution_mode or "strict_offline",
        known_doc_ids=known_docs
    )
    lat_ms = (time.perf_counter() - t0) * 1000.0
    dashboard.record_request("/route-query", route=decision.route, latency_ms=lat_ms)
    return {
        "query": req.query,
        "routing_decision": decision.to_dict()
    }


@protected_router.post("/retrieve")
def retrieve_context(
    req: QueryRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Retrieves Top-K relevant context chunks for a given query.
    Supports optional auto-routing through ControlledQueryRouter.
    Authoritative user_context is derived server-side from authenticated principal.
    """
    t0 = time.perf_counter()
    trusted_user_context = build_authoritative_user_context(principal, req.user_context)
    if req.auto_route:
        known_docs = list(set(c.get("doc_id", "") for c in indexer.chunks))
        history = conversation_manager.get_history(req.session_id or "default") if req.use_memory else None
        decision = query_router.route_query(
            query=req.query,
            history=history,
            user_context=trusted_user_context,
            execution_mode=req.execution_mode or "strict_offline",
            known_doc_ids=known_docs
        )
        route_exec = query_router.execute_route(
            decision=decision,
            indexer=indexer,
            query=req.query,
            top_k=req.top_k or 3,
            user_context=trusted_user_context,
            filters=req.filters,
            execution_mode=req.execution_mode or "strict_offline",
            is_untrusted_source=bool(req.is_untrusted_source)
        )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        dashboard.record_request("/retrieve", route=route_exec.get("route_used"), latency_ms=lat_ms)
        return {
            "query": req.query,
            "top_k": req.top_k,
            "route_used": route_exec.get("route_used"),
            "routing_decision": decision.to_dict(),
            "retrieved_chunks": route_exec.get("chunks", []),
            "tool_output": route_exec.get("tool_output")
        }

    results = indexer.search(
        req.query,
        top_k=req.top_k or 3,
        filters=req.filters,
        user_context=trusted_user_context
    )
    lat_ms = (time.perf_counter() - t0) * 1000.0
    dashboard.record_request("/retrieve", route="dense", latency_ms=lat_ms)
    return {
        "query": req.query,
        "top_k": req.top_k,
        "retrieved_chunks": results
    }


@protected_router.post("/generate-rag")
def generate_rag(
    req: QueryRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Phase 3: Retrieves context, builds context-bounded prompt, and generates grounded response.
    Supports multi-backend generation (Ollama, Gemini, Offline) and multi-turn memory.
    Authoritative user_context and conversation session are strictly scoped server-side to the authenticated principal.
    """
    client_sess_id, scoped_sess_id = resolve_scoped_session(principal, req.session_id)
    standalone_query = req.query
    history = []
    trusted_user_context = build_authoritative_user_context(principal, req.user_context)

    # 1. Memory / Query Condensation
    if req.use_memory:
        history = conversation_manager.get_history(scoped_sess_id)
        if history:
            standalone_query = conversation_manager.condense_query(
                session_id=scoped_sess_id,
                query=req.query,
                backend=req.backend or "auto",
                model=req.model,
                api_key=req.api_key
            )

    # 2. Retrieval using standalone condensed query & trusted user_context
    retrieved = indexer.search(
        standalone_query,
        top_k=req.top_k or 3,
        filters=req.filters,
        user_context=trusted_user_context
    )

    # 3. Prompt Assembly with conversation history & retrieved context
    prompt = PromptAssembler.build_prompt(
        query=req.query,
        retrieved_chunks=retrieved,
        conversation_history=history if req.use_memory else None,
        small_model_mode=bool(req.small_model_mode)
    )
    
    # 4. Generate
    output = LLMGenerator.generate(
        prompt=prompt,
        retrieved_chunks=retrieved,
        query=standalone_query,
        backend=req.backend or "auto",
        model=req.model,
        api_key=req.api_key,
        execution_mode=req.execution_mode
    )

    # 5. Injection telemetry audit
    injection_audit = PromptInjectionGuard.scan_retrieved_chunks(retrieved)

    # 6. Record Turn in Conversation Memory (scoped to authenticated user)
    if req.use_memory:
        conversation_manager.add_turn(
            session_id=scoped_sess_id,
            user_query=req.query,
            assistant_answer=output["answer"],
            citations=output.get("provenance", []),
            rewritten_query=standalone_query if standalone_query != req.query else None
        )

    return {
        "phase": "Phase 3: Retrieve & Generate",
        "query": req.query,
        "standalone_query": standalone_query,
        "session_id": client_sess_id,
        "scoped_session_id": scoped_sess_id,
        "history_turn_count": len(history),
        "prompt": prompt,
        "retrieved_chunks": retrieved,
        "answer": output["answer"],
        "provenance": output["provenance"],
        "grounded": output["grounded"],
        "backend_used": output.get("backend_used", "offline"),
        "model_used": output.get("model_used", "none"),
        "injection_audit": injection_audit
    }


@protected_router.post("/eval-recall")
def run_eval(req: EvalRequest):
    """
    Phase 3 Eval Harness: Measures Recall@K retrieval accuracy over a test dataset.
    """
    dataset_dict = [item.dict() for item in req.dataset]
    metrics = EvalHarness.calculate_recall_at_k(dataset_dict, indexer, k=req.k or 3)
    return {
        "phase": "Phase 3: Retrieval Quality Eval",
        "metrics": metrics
    }


# ==========================================
# PHASE 4 — Guardrails & Faithfulness
# ==========================================
@protected_router.post("/generate-guarded")
def generate_guarded(
    req: QueryRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Phase 4: Integrates confidence thresholds & faithfulness guardrails before generation.
    Supports multi-turn memory and query condensation.
    Authoritative user_context and conversation sessions are strictly scoped server-side to the authenticated principal.
    """
    t0 = time.perf_counter()
    client_sess_id, scoped_sess_id = resolve_scoped_session(principal, req.session_id)
    standalone_query = req.query
    history = []
    trusted_user_context = build_authoritative_user_context(principal, req.user_context)

    # 1. Memory / Query Condensation
    if req.use_memory:
        history = conversation_manager.get_history(scoped_sess_id)
        if history:
            standalone_query = conversation_manager.condense_query(
                session_id=scoped_sess_id,
                query=req.query,
                backend=req.backend or "auto",
                model=req.model,
                api_key=req.api_key
            )

    # 2. Retrieval using standalone query & trusted user_context
    retrieved = indexer.search(
        standalone_query,
        top_k=req.top_k or 3,
        filters=req.filters,
        user_context=trusted_user_context
    )
    
    # Evaluate retrieval confidence
    threshold = req.threshold if req.threshold is not None else guardrails.min_similarity_threshold
    orig_threshold = guardrails.min_similarity_threshold
    guardrails.min_similarity_threshold = threshold
    passed, reason = guardrails.evaluate_retrieval(retrieved, query=standalone_query)
    guardrails.min_similarity_threshold = orig_threshold

    if not passed:
        answer = "I cannot answer this based on the provided context."
        if req.use_memory:
            conversation_manager.add_turn(
                session_id=scoped_sess_id,
                user_query=req.query,
                assistant_answer=answer,
                citations=[],
                rewritten_query=standalone_query if standalone_query != req.query else None
            )
        lat_ms = (time.perf_counter() - t0) * 1000.0
        dashboard.record_guardrail_event(False, reason, injection_detected=False)
        dashboard.record_request("/generate-guarded", latency_ms=lat_ms, status="GUARDRAIL_BLOCKED")
        return {
            "phase": "Phase 4: Guardrails",
            "query": req.query,
            "standalone_query": standalone_query,
            "session_id": client_sess_id,
            "scoped_session_id": scoped_sess_id,
            "history_turn_count": len(history),
            "status": "BLOCKED_BY_GUARDRAIL",
            "reason": reason,
            "answer": answer,
            "provenance": [],
            "backend_used": "guardrail-block",
            "model_used": "none",
            "injection_audit": guardrails.check_prompt_injection(retrieved)
        }

    # 3. Prompt Assembly with history + context
    prompt = PromptAssembler.build_prompt(
        query=req.query,
        retrieved_chunks=retrieved,
        conversation_history=history if req.use_memory else None,
        small_model_mode=bool(req.small_model_mode)
    )

    output = LLMGenerator.generate(
        prompt=prompt,
        retrieved_chunks=retrieved,
        query=standalone_query,
        backend=req.backend or "auto",
        model=req.model,
        api_key=req.api_key,
        execution_mode=req.execution_mode
    )
    
    # Groundedness reflection check
    is_grounded = guardrails.verify_groundedness(output["answer"], retrieved)
    injection_audit = guardrails.check_prompt_injection(retrieved)

    lat_ms = (time.perf_counter() - t0) * 1000.0
    dashboard.record_guardrail_event(True, reason, injection_detected=bool(injection_audit.get("injection_detected")))
    dashboard.record_request("/generate-guarded", latency_ms=lat_ms, status="SUCCESS")

    # 4. Record Turn in Memory (scoped to authenticated user)
    if req.use_memory:
        conversation_manager.add_turn(
            session_id=scoped_sess_id,
            user_query=req.query,
            assistant_answer=output["answer"],
            citations=output.get("provenance", []),
            rewritten_query=standalone_query if standalone_query != req.query else None
        )

    return {
        "phase": "Phase 4: Guardrails",
        "query": req.query,
        "standalone_query": standalone_query,
        "session_id": client_sess_id,
        "scoped_session_id": scoped_sess_id,
        "history_turn_count": len(history),
        "status": "PASSED",
        "guardrail_reason": reason,
        "is_grounded_verified": is_grounded,
        "answer": output["answer"],
        "provenance": output["provenance"],
        "retrieved_chunks": retrieved,
        "backend_used": output.get("backend_used", "offline"),
        "model_used": output.get("model_used", "none"),
        "prompt": prompt,
        "injection_audit": injection_audit
    }


@protected_router.post("/scan-injection")
def scan_prompt_injection(req: InjectionScanRequest):
    """
    Security Layer: Scans input text for indirect prompt injection and document security threats.
    """
    assessment = PromptInjectionGuard.scan_text(req.text)
    return assessment.to_dict()


@protected_router.get("/small-model-audit")
def get_small_model_compatibility_audit():
    """
    Small LLM Compatibility Audit:
    Audits the pipeline across the 8 Small-Model (1B-3B) operational areas:
    Query Rewriting, Context Understanding, Claim Extraction, Citation Generation,
    Numerical Reasoning, Long Responses, JSON Generation, and Multi-Hop Questions.
    """
    return SmallModelCompatibilityAudit.get_audit_report()


@protected_router.post("/compute-factual")
def execute_factual_computation_endpoint(
    req: FactualComputeRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Decoupled Factual & Mathematical Computation Pipeline (Small LLM Optimization):
    1. Retrieve project/document records
    2. Extract numeric values deterministically with units and provenance
    3. Calculate aggregations (average, sum, min, max, difference) using Python/SQL
    4. Validate mathematical integrity
    5. Return verified computation payload for prompt assembly.
    Authoritative user_context is derived server-side from authenticated principal.
    """
    trusted_user_context = build_authoritative_user_context(principal, req.user_context)
    retrieved = indexer.search(
        req.query,
        top_k=req.top_k or 5,
        filters=req.filters,
        user_context=trusted_user_context
    )
    result = DeterministicComputationEngine.execute_factual_computation(req.query, retrieved)
    if not result:
        return {
            "status": "NO_COMPUTATION_DETECTED",
            "query": req.query,
            "retrieved_count": len(retrieved),
            "result": None,
            "prompt_block": None
        }
    return {
        "status": "SUCCESS" if result.is_valid else "VALIDATION_FAILED",
        "query": req.query,
        "retrieved_count": len(retrieved),
        "computation": result.to_dict(),
        "prompt_block": DeterministicComputationEngine.format_for_prompt(result)
    }


# ==========================================
# PHASE 5 — Conversation Memory & Session API
# ==========================================
@protected_router.get("/sessions")
def list_sessions(principal: AuthenticatedPrincipal = Depends(get_current_principal)):
    """
    Returns active conversation session summaries scoped to the authenticated user.
    Administrators can view all system sessions; regular users only see their own.
    """
    all_sessions = conversation_manager.list_sessions()
    if principal.is_admin():
        return {"sessions": all_sessions}

    user_prefix = f"{principal.principal_id}:"
    user_sessions = []
    for s in all_sessions:
        sid = s.get("session_id", "")
        if sid.startswith(user_prefix):
            user_sessions.append({
                "session_id": sid[len(user_prefix):],
                "turn_count": s.get("turn_count", 0),
                "last_active": s.get("last_active", 0.0),
                "last_query": s.get("last_query")
            })
    return {"sessions": user_sessions}


@protected_router.get("/sessions/{session_id}")
def get_session(
    session_id: str,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Returns conversation history for a specific session scoped to the authenticated user.
    Prevents unauthorized access to other callers' conversation logs.
    """
    # 1. Look up principal-scoped session
    scoped_id = ConversationManager.scoped_session_id(principal.principal_id, session_id)
    history = conversation_manager.get_history(scoped_id, max_turns=50)

    # 2. If not found and caller is admin, allow reading raw session_id
    if not history and principal.is_admin():
        history = conversation_manager.get_history(session_id, max_turns=50)

    # 3. Cross-user access prevention check
    if not history:
        if ":" in session_id and not principal.is_admin():
            owner = session_id.split(":", 1)[0]
            if owner != principal.principal_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: Cannot access conversation session belonging to another user."
                )

    return {
        "session_id": session_id,
        "history": history
    }


@protected_router.delete("/sessions/{session_id}")
def clear_session(
    session_id: str,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Clears conversation history for a specific session scoped to the authenticated user.
    """
    scoped_id = ConversationManager.scoped_session_id(principal.principal_id, session_id)
    cleared = conversation_manager.clear_session(scoped_id)

    # If caller is admin and raw session_id exists, allow clearing
    if not cleared and principal.is_admin():
        cleared = conversation_manager.clear_session(session_id)

    return {
        "session_id": session_id,
        "cleared": cleared
    }


# ==============================================================================
# MOUNT PUBLIC AND PROTECTED ROUTERS ONTO FASTAPI APP
# ==============================================================================
app.include_router(public_router)
app.include_router(protected_router)
