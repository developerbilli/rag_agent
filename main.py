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
import re
import time
import secrets
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Header, Query, Depends, Security, APIRouter, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple, Union, Set

logger = logging.getLogger("rag_api.auth")

# ==============================================================================
# Sensitive Header Redaction Filter & Log Masking
# ==============================================================================
class SensitiveHeaderRedactionFilter(logging.Filter):
    """
    Log filter that intercepts log records across all loggers (including root, uvicorn,
    uvicorn.access, and rag_api.auth) and redacts authorization headers, Bearer tokens,
    and API key query parameters to prevent secret leakage in logs, traces, and metrics.
    """
    PATTERNS = [
        (re.compile(r'(?i)(bearer\s+)[A-Za-z0-9_\-\.]+'), r'\1[REDACTED]'),
        (re.compile(r'(?i)(authorization["\']?\s*[:=]\s*["\']?)(?:Bearer\s+)?[^"\'\s,;}]+'), r'\1[REDACTED]'),
        (re.compile(r'(?i)(x-api-key["\']?\s*[:=]\s*["\']?)[^"\'\s,;}]+'), r'\1[REDACTED]'),
        (re.compile(r'(?i)(api[_-]?key["\']?\s*[:=]\s*["\']?)[^"\'\s,;}]+'), r'\1[REDACTED]'),
        (re.compile(r'(?i)(\?|&)api_key=[^&\s]+'), r'\1api_key=[REDACTED]'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern, repl in self.PATTERNS:
                record.msg = pattern.sub(repl, record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: (self._sanitize(v) if isinstance(v, str) else v) for k, v in record.args.items()}
            elif isinstance(record.args, (list, tuple)):
                record.args = tuple(self._sanitize(a) if isinstance(a, str) else a for a in record.args)
        return True

    def _sanitize(self, val: str) -> str:
        for pattern, repl in self.PATTERNS:
            val = pattern.sub(repl, val)
        return val


def redact_headers(headers: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes a dictionary of HTTP headers, replacing sensitive credentials with [REDACTED].
    """
    sensitive_keys = {"authorization", "x-api-key", "api-key", "cookie", "set-cookie", "token"}
    redacted = {}
    for k, v in headers.items():
        if k.lower() in sensitive_keys:
            redacted[k] = "[REDACTED]"
        else:
            redacted[k] = v
    return redacted


# Attach redaction filter to auth logger, root logger, and ASGI/Uvicorn loggers
_redaction_filter = SensitiveHeaderRedactionFilter()
logger.addFilter(_redaction_filter)
logging.getLogger().addFilter(_redaction_filter)
for _logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi"):
    logging.getLogger(_logger_name).addFilter(_redaction_filter)

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
    PrincipalRegistry,
    MetadataFilterEngine,
    BM25OkapiIndexer
)

app = FastAPI(
    title="Production-Grade Offline RAG Pipeline",
    description="Production-grade offline RAG pipeline combining BM25 keyword retrieval and dense-vector retrieval with Reciprocal Rank Fusion (RRF), cross-encoder reranking, transactional rollback, propositional claim verification, and multi-turn memory.",
    version="2.0.0"
)

# ==============================================================================
# File Upload & DoS Protection Safeguards
# ==============================================================================
def get_upload_limits() -> Dict[str, int]:
    """
    Retrieve current upload and DoS limits from environment variables with safe enterprise defaults:
    - RAG_MAX_UPLOAD_SIZE: max size per individual file (default: 25 MB)
    - RAG_MAX_UPLOAD_FILE_COUNT: max files per upload request (default: 20)
    - RAG_MAX_REQUEST_UPLOAD_SIZE: max cumulative upload bytes per request (default: 50 MB)
    - RAG_MAX_REQUEST_BODY_SIZE: max HTTP request body size (default: 55 MB)
    - RAG_UPLOAD_CHUNK_SIZE: chunk size for bounded streaming reads (default: 64 KB)
    """
    try:
        max_file_size = int(os.environ.get("RAG_MAX_UPLOAD_SIZE", str(25 * 1024 * 1024)))
    except ValueError:
        max_file_size = 25 * 1024 * 1024

    try:
        max_file_count = int(os.environ.get("RAG_MAX_UPLOAD_FILE_COUNT", "20"))
    except ValueError:
        max_file_count = 20

    try:
        max_request_size = int(os.environ.get("RAG_MAX_REQUEST_UPLOAD_SIZE", str(50 * 1024 * 1024)))
    except ValueError:
        max_request_size = 50 * 1024 * 1024

    try:
        max_body_size = int(os.environ.get("RAG_MAX_REQUEST_BODY_SIZE", str(55 * 1024 * 1024)))
    except ValueError:
        max_body_size = 55 * 1024 * 1024

    try:
        chunk_size = int(os.environ.get("RAG_UPLOAD_CHUNK_SIZE", str(64 * 1024)))
    except ValueError:
        chunk_size = 64 * 1024

    return {
        "max_file_size": max_file_size,
        "max_file_count": max_file_count,
        "max_request_size": max_request_size,
        "max_body_size": max_body_size,
        "chunk_size": chunk_size,
    }

MAX_UPLOAD_SIZE = 25 * 1024 * 1024
MAX_UPLOAD_FILE_COUNT = 20
MAX_REQUEST_UPLOAD_SIZE = 50 * 1024 * 1024
MAX_REQUEST_BODY_SIZE = 55 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 64 * 1024


@app.middleware("http")
async def enforce_request_body_size_limit(request: Request, call_next):
    """
    Early rejection of requests that exceed server request-body limits.
    Enforces limits on both Content-Length headers and chunked/streaming request bodies.
    Protects reverse proxy, application memory, and multipart parsers before buffer spooling.
    """
    limits = get_upload_limits()
    max_body = limits["max_body_size"]

    content_length_str = request.headers.get("content-length")
    if content_length_str:
        try:
            content_length = int(content_length_str)
            if content_length > max_body:
                return JSONResponse(
                    status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                    content={"detail": f"Request entity too large: Content-Length ({content_length} bytes) exceeds limit of {max_body} bytes."}
                )
        except ValueError:
            pass

    # For chunked transfer or requests without Content-Length, bound incoming streaming body
    orig_receive = request.receive
    bytes_received = 0

    async def bounded_receive():
        nonlocal bytes_received
        msg = await orig_receive()
        if msg.get("type") == "http.request":
            body = msg.get("body", b"")
            bytes_received += len(body)
            if bytes_received > max_body:
                raise HTTPException(
                    status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                    detail=f"Request entity too large: streaming body exceeds limit of {max_body} bytes."
                )
        return msg

    request._receive = bounded_receive

    try:
        return await call_next(request)
    except HTTPException as exc:
        if exc.status_code == getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413):
            return JSONResponse(
                status_code=getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413),
                content={"detail": exc.detail}
            )
        raise exc

# Authentication Schemes & Server-Side Identity Layer
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

principal_registry = PrincipalRegistry()

def get_auth_mode() -> str:
    """
    Resolves the active authentication mode ('development' or 'production').
    Configured explicitly via RAG_AUTH_MODE=development|production.
    
    Hard Security Boundary:
    If RAG_ENV or ENVIRONMENT is set to 'production' or 'prod', production mode
    is strictly enforced and CANNOT be downgraded to 'development' by RAG_AUTH_MODE.
    """
    is_prod_env = os.environ.get("RAG_ENV", "").strip().lower() in ("production", "prod") or \
                  os.environ.get("ENVIRONMENT", "").strip().lower() in ("production", "prod")
    if is_prod_env:
        return "production"

    raw_mode = os.environ.get("RAG_AUTH_MODE", "").strip().lower()
    if raw_mode in ("production", "prod"):
        return "production"
    if raw_mode in ("development", "dev", "local", "test"):
        return "development"
    if raw_mode:
        raise ValueError(f"Invalid RAG_AUTH_MODE '{raw_mode}'. Expected 'development' or 'production'.")

    return "development"


def has_auth_configuration() -> bool:
    """
    Returns True if authoritative production authentication is configured
    (RAG_API_KEY is set and non-empty).
    """
    return bool(os.environ.get("RAG_API_KEY", "").strip())


def is_production_auth_configured() -> bool:
    """Checks whether production authentication policy is active or configured."""
    return get_auth_mode() == "production" or bool(os.environ.get("RAG_API_KEY", "").strip())


def validate_auth_configuration() -> None:
    """
    Startup authentication validator.
    In production mode (RAG_AUTH_MODE=production or production environment):
    1. Fails startup if attempting to downgrade a production environment to development mode.
    2. Fails startup if no authentication configuration exists (e.g. RAG_API_KEY is absent).
    3. Fails startup if development-only bypass flags are enabled:
       - RAG_ALLOW_BUILTIN_TOKENS=true
       - RAG_ALLOW_INSECURE_CREDENTIAL_TRANSPORT=true
    """
    is_prod_env = os.environ.get("RAG_ENV", "").strip().lower() in ("production", "prod") or \
                  os.environ.get("ENVIRONMENT", "").strip().lower() in ("production", "prod")
    raw_auth_mode = os.environ.get("RAG_AUTH_MODE", "").strip().lower()

    # 1. Hard Security Boundary: Prohibit downgrading a production environment
    if is_prod_env and raw_auth_mode in ("development", "dev", "local", "test"):
        err_msg = (
            f"Production startup failed: Attempted to downgrade production environment "
            f"(RAG_ENV/ENVIRONMENT='{os.environ.get('RAG_ENV') or os.environ.get('ENVIRONMENT')}') "
            f"to development mode (RAG_AUTH_MODE='{raw_auth_mode}'). "
            f"Production environment is a hard security boundary that cannot be downgraded."
        )
        logger.critical(f"[Security Startup Violation] {err_msg}")
        raise RuntimeError(err_msg)

    auth_mode = get_auth_mode()
    is_hosted = os.environ.get("RAG_HOSTED_DEPLOYMENT", "").strip().lower() in ("true", "1", "yes")

    if auth_mode == "production" or is_hosted:
        # 2. Require authoritative authentication configuration
        if not has_auth_configuration():
            mode_desc = "RAG_HOSTED_DEPLOYMENT is active" if is_hosted and auth_mode != "production" else "RAG_AUTH_MODE is set to 'production' (or production environment detected)"
            err_msg = (
                f"Production startup failed: {mode_desc}, but no authentication configuration exists "
                f"(RAG_API_KEY is absent or empty). "
                f"Open developer mode without authentication is prohibited in production/hosted deployments. "
                f"Set RAG_API_KEY to a secure secret, or set RAG_AUTH_MODE=development for local testing."
            )
            logger.critical(f"[Security Startup Violation] {err_msg}")
            raise RuntimeError(err_msg)

        # 3. Development overrides cannot weaken production security
        if os.environ.get("RAG_ALLOW_BUILTIN_TOKENS", "").strip().lower() in ("true", "1", "yes"):
            err_msg = (
                "Production startup failed: RAG_ALLOW_BUILTIN_TOKENS is enabled while in production mode. "
                "Built-in development tokens (admin-token, user-token) cannot be permitted in production environments."
            )
            logger.critical(f"[Security Startup Violation] {err_msg}")
            raise RuntimeError(err_msg)

        if os.environ.get("RAG_ALLOW_INSECURE_CREDENTIAL_TRANSPORT", "").strip().lower() in ("true", "1", "yes"):
            err_msg = (
                "Production startup failed: RAG_ALLOW_INSECURE_CREDENTIAL_TRANSPORT is enabled while in production mode. "
                "Insecure credential transport (query parameters / JSON body) cannot be permitted in production environments."
            )
            logger.critical(f"[Security Startup Violation] {err_msg}")
            raise RuntimeError(err_msg)

    if is_operational_metrics_protection_enabled() and not has_auth_configuration():
        logger.warning(
            "[Security Warning] RAG_PROTECT_OPERATIONAL_METRICS is enabled but RAG_API_KEY is not configured. "
            "Operational endpoints (/observability, /models-status) will strictly reject unauthenticated access "
            "and developer mode fallbacks."
        )


def are_builtin_tokens_allowed() -> bool:
    """
    Built-in development tokens (admin-token, user-token, etc.) are strictly disabled
    in production mode or whenever RAG_API_KEY is configured.
    Production security always takes absolute precedence over development-only overrides.
    """
    # Production always wins over development overrides
    if get_auth_mode() == "production":
        return False
    if bool(os.environ.get("RAG_API_KEY", "").strip()):
        return False

    # In development mode without an API key:
    val = os.environ.get("RAG_ALLOW_BUILTIN_TOKENS", "").strip().lower()
    if val in ("false", "0", "no"):
        return False
    return True


def is_strict_header_auth_enforced() -> bool:
    """
    In production mode or when RAG_API_KEY is configured, strict header-only
    authentication is ALWAYS enforced and cannot be weakened by development overrides.
    Query parameter and body credentials are systematically rejected.
    """
    # Production always wins over development overrides
    if get_auth_mode() == "production":
        return True
    if bool(os.environ.get("RAG_API_KEY", "").strip()):
        return True

    # In development mode without an API key:
    if os.environ.get("RAG_ALLOW_INSECURE_CREDENTIAL_TRANSPORT", "").strip().lower() in ("true", "1", "yes"):
        return False
    return os.environ.get("RAG_ENFORCE_HEADER_AUTH", "").strip().lower() in ("true", "1", "yes")


async def get_current_principal(
    request: Request,
    header_key: Optional[str] = Security(api_key_header_scheme),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> AuthenticatedPrincipal:
    """
    Derives caller identity and authorization claims server-side from validated authentication.
    Privilege-bearing claims (roles, clearance_level, user_id, department) are NEVER accepted
    from arbitrary caller request fields.

    In production:
    - Accepts credentials EXCLUSIVELY via standard headers:
      1. X-API-Key header (via Security scheme)
      2. Authorization: Bearer <key> header (via Security scheme)
    - Rejects credentials submitted in URL query parameters (?api_key=...) or JSON bodies ({"api_key": ...})
      with HTTP 400 Bad Request to prevent credential leakage in access logs, browser history, and traces.
    """
    expected = os.environ.get("RAG_API_KEY", "").strip() or None
    allow_builtin = are_builtin_tokens_allowed()
    strict_header_auth = is_strict_header_auth_enforced()

    # 1. Enforce strict transport security in production: reject query parameters and body credentials
    if strict_header_auth:
        if bool(request.query_params.get("api_key")):
            logger.warning(
                "[Security Audit] Rejected request with credentials in URL query parameter under strict production auth policy."
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insecure credential transmission: Passing API credentials in URL query parameters is prohibited in production. Provide credentials exclusively via 'Authorization: Bearer <key>' or 'X-API-Key' headers."
            )

        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.json()
                if isinstance(body, dict) and bool(body.get("api_key")):
                    logger.warning(
                        "[Security Audit] Rejected request with credentials in JSON request body under strict production auth policy."
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Insecure credential transmission: Passing API credentials in request body is prohibited in production. Provide credentials exclusively via 'Authorization: Bearer <key>' or 'X-API-Key' headers."
                    )
            except HTTPException:
                raise
            except Exception:
                pass

    # 2. Extract credential exclusively from headers (or fallback to legacy transport if non-prod/relaxed)
    provided = (header_key or request.headers.get("X-API-Key") or "").strip() or None
    if not provided and bearer_token and bearer_token.credentials:
        provided = bearer_token.credentials.strip() or None
    if not provided:
        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header.startswith("Bearer "):
            provided = auth_header[7:].strip() or None

    if not provided and not strict_header_auth:
        provided = request.query_params.get("api_key")
        if not provided and request.headers.get("content-type", "").startswith("application/json"):
            try:
                body = await request.json()
                if isinstance(body, dict):
                    provided = body.get("api_key")
            except Exception:
                pass

    # 3. If a credential was provided:
    if provided:
        # Block built-in development credentials when production authentication is enforced
        if principal_registry.is_builtin_token(provided):
            if not allow_builtin:
                logger.warning(
                    "[Security Audit] Blocked attempt to authenticate with built-in development token '[REDACTED]' "
                    "while production authentication is enforced."
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized: Built-in development credentials are disabled when production authentication is configured.",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # Check server-side credential registry (for dynamic / IDP-registered credentials)
        registered_principal = principal_registry.resolve(provided, allow_builtin=allow_builtin)
        if registered_principal:
            return registered_principal

        # Check dedicated administrative API key (RAG_ADMIN_API_KEY) for role separation
        admin_key = os.environ.get("RAG_ADMIN_API_KEY", "").strip()
        if admin_key and secrets.compare_digest(str(provided), str(admin_key)):
            return AuthenticatedPrincipal(
                principal_id=os.environ.get("RAG_ADMIN_PRINCIPAL_ID", "admin_service_account"),
                roles=["admin", "superuser", "data_admin", "sysadmin"],
                clearance_level="top_secret",
                department="security",
                auth_source="admin_rag_api_key"
            )

        # Check primary configured RAG_API_KEY
        if expected:
            if secrets.compare_digest(str(provided), str(expected)):
                # If dedicated RAG_ADMIN_API_KEY is configured, primary key defaults to standard 'user'.
                # Otherwise, in single-key mode, defaults to 'admin,user' unless overridden by RAG_SERVICE_ROLES.
                default_roles = "user" if admin_key else "admin,user"
                default_clearance = "internal" if admin_key else "top_secret"
                service_roles = [r.strip().lower() for r in os.environ.get("RAG_SERVICE_ROLES", default_roles).split(",") if r.strip()]
                service_clearance = os.environ.get("RAG_SERVICE_CLEARANCE", default_clearance).strip().lower()
                return AuthenticatedPrincipal(
                    principal_id=os.environ.get("RAG_SERVICE_PRINCIPAL_ID", "primary_service_account"),
                    roles=service_roles,
                    clearance_level=service_clearance,
                    department="system",
                    auth_source="primary_rag_api_key"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unauthorized: Invalid or missing API key.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        else:
            # A credential was provided, but didn't match any registered token and no RAG_API_KEY was configured
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid API key or credential.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # 4. When RAG_API_KEY is configured in the environment and no credential was provided:
    if expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid or missing API key. Provide credentials via 'Authorization: Bearer <key>' or 'X-API-Key' headers.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # In production auth mode, unauthenticated access is strictly forbidden
    if get_auth_mode() == "production":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Production authentication is enforced (RAG_AUTH_MODE=production). Provide credentials via 'Authorization: Bearer <key>' or 'X-API-Key' headers.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 5. Open Developer Mode: When RAG_API_KEY is unset and no valid credential was provided (development mode only)
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
    retrieval_mode: Optional[str] = None  # "hybrid" (default in production), "dense", "bm25"
    alpha: Optional[float] = 0.5
    use_reranker: Optional[Union[bool, str]] = False
    rrf_k: Optional[int] = 60
    weight_dense: Optional[float] = None
    weight_bm25: Optional[float] = None
    candidate_multiplier: Optional[int] = 5
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


VALID_RETRIEVAL_MODES = {
    "hybrid": "hybrid",
    "rrf": "hybrid",
    "dense_bm25": "hybrid",
    "default": "hybrid",
    "dense": "dense",
    "vector": "dense",
    "semantic": "dense",
    "cosine": "dense",
    "bm25": "bm25",
    "lexical": "bm25",
    "keyword": "bm25",
}


def resolve_retrieval_mode(req_mode: Optional[str] = None) -> str:
    """
    Resolves the authoritative retrieval mode from request or environment configuration.
    Allowed modes:
    - 'hybrid' (default): BM25 lexical + Dense vector + Reciprocal Rank Fusion (RRF)
    - 'dense': Dense vector cosine similarity search
    - 'bm25': BM25 Okapi lexical keyword search

    Validation:
    Raises HTTPException 400 if an unknown or invalid retrieval mode is supplied,
    preventing silent unintended fallbacks.
    """
    if req_mode is not None and req_mode.strip():
        raw = req_mode.strip().lower()
        if raw not in VALID_RETRIEVAL_MODES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid retrieval_mode '{req_mode}'. Allowed modes are 'hybrid', 'dense', 'bm25'."
            )
        return VALID_RETRIEVAL_MODES[raw]

    configured = os.environ.get("RAG_DEFAULT_RETRIEVAL_MODE", "hybrid").strip().lower()
    if configured not in VALID_RETRIEVAL_MODES:
        logger.warning(f"Invalid RAG_DEFAULT_RETRIEVAL_MODE '{configured}'. Falling back to 'hybrid'.")
        return "hybrid"
    return VALID_RETRIEVAL_MODES[configured]


def execute_configured_retrieval(
    indexer: VectorIndexer,
    query: str,
    top_k: int = 3,
    retrieval_mode: Optional[str] = None,
    alpha: float = 0.5,
    use_reranker: Union[bool, str] = False,
    rrf_k: int = 60,
    weight_dense: Optional[float] = None,
    weight_bm25: Optional[float] = None,
    candidate_multiplier: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    user_context: Optional[Dict[str, Any]] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Executes context retrieval according to the resolved retrieval mode:
    - 'hybrid': Routes to indexer.search_hybrid(...) via BM25 + Dense + RRF (+ optional re-ranking)
    - 'dense': Routes to indexer.search(...) via Dense cosine similarity
    - 'bm25': Routes to indexer.bm25 via Okapi lexical search with RBAC/ABAC filtering
    Returns:
        (retrieved_chunks, mode_used)
    """
    mode = resolve_retrieval_mode(retrieval_mode)

    if mode == "hybrid":
        results = indexer.search_hybrid(
            query=query,
            top_k=top_k,
            alpha=alpha,
            use_reranker=use_reranker,
            rrf_k=rrf_k,
            weight_dense=weight_dense,
            weight_bm25=weight_bm25,
            candidate_multiplier=candidate_multiplier,
            filters=filters,
            user_context=user_context
        )
        return results, "hybrid"

    elif mode == "bm25":
        deleted_docs = {
            k for k, v in getattr(indexer, 'source_documents', {}).items()
            if v.get("status") == "deleted" or bool(v.get("deleted"))
        }
        valid_set = {
            i for i, c in enumerate(indexer.chunks)
            if c.get("doc_id") not in deleted_docs and MetadataFilterEngine.matches(c, filters, user_context)
        }
        if not hasattr(indexer, 'bm25') or indexer.bm25 is None:
            indexer.bm25 = BM25OkapiIndexer()
            indexer.bm25.index(indexer.chunks)
        bm25_scored = indexer.bm25.search(query, top_k=top_k, allowed_indices=valid_set)
        results = []
        for idx, sc in bm25_scored:
            item = dict(indexer.chunks[idx])
            item["score"] = sc
            item["modality"] = "bm25"
            results.append(item)
        return results, "bm25"

    else:
        # Dense vector search
        results = indexer.search(
            query=query,
            top_k=top_k,
            filters=filters,
            user_context=user_context
        )
        return results, "dense"


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
def startup_auth_validation():
    """Enforce production authentication requirements on server startup."""
    validate_auth_configuration()


@app.on_event("startup")
def auto_ingest_sample_corpus():
    """Seed index on startup through canonical lifecycle API."""
    for doc in SAMPLE_DOCUMENTS:
        try:
            doc_meta = dict(doc.get("metadata", {}))
            doc_meta.setdefault("owner_id", "system")
            doc_meta.setdefault("created_by", "system")
            indexer.stage_and_commit_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                title=doc["title"],
                chunker=chunker,
                metadata=doc_meta
            )
        except Exception as e:
            print(f"[Startup] Error seeding {doc.get('doc_id')}: {e}")
    
    # Also load sample_data directory if it exists
    if os.path.isdir("sample_data"):
        try:
            extra_docs = DocumentLoader.load_directory("sample_data", recursive=False)
            for doc in extra_docs:
                raw_text = doc.get("text", "").strip()
                if raw_text:
                    extra_meta = dict(doc.get("metadata", {}))
                    extra_meta.setdefault("owner_id", "system")
                    extra_meta.setdefault("created_by", "system")
                    indexer.stage_and_commit_document(
                        doc_id=doc["doc_id"],
                        text=raw_text,
                        title=doc.get("title", doc["doc_id"]),
                        chunker=chunker,
                        metadata=extra_meta
                    )
        except Exception as e:
            print(f"[Startup] Error auto-loading sample_data: {e}")


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


def is_operational_metrics_protection_enabled() -> bool:
    """
    Returns whether operational telemetry and model status endpoints are protected.
    Configured via RAG_PROTECT_OPERATIONAL_METRICS=true or RAG_HOSTED_DEPLOYMENT=true.
    When enabled, /observability and /models-status require authenticated principal credentials.
    """
    return (
        os.environ.get("RAG_PROTECT_OPERATIONAL_METRICS", "").strip().lower() in ("true", "1", "yes") or
        os.environ.get("RAG_HOSTED_DEPLOYMENT", "").strip().lower() in ("true", "1", "yes")
    )


async def require_operational_metrics_access(
    request: Request,
    header_key: Optional[str] = Security(api_key_header_scheme),
    bearer_token: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
) -> Optional[AuthenticatedPrincipal]:
    """
    Enforces authentication for operational endpoints (/observability, /models-status).
    When operational metrics protection is disabled, permits unauthenticated public access.
    When operational metrics protection is enabled (RAG_PROTECT_OPERATIONAL_METRICS=true or RAG_HOSTED_DEPLOYMENT=true):
    - Requires an explicitly authenticated principal with validated credentials.
    - Header presence alone is never accepted as proof of authentication.
    - Rejects open developer mode fallbacks (auth_source == "developer_open_mode" or dev_user).
    - Rejects missing, invalid, unrecognized, or unsupported credentials with HTTP 401 Unauthorized.
    """
    if not is_operational_metrics_protection_enabled():
        return None

    auth_header = request.headers.get("Authorization", "").strip()
    x_api_key = (header_key or request.headers.get("X-API-Key") or "").strip()

    # 1. Require a non-empty, supported credential header
    if not x_api_key and not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Access to operational observability metrics requires authentication in hosted deployments. Provide credentials via 'Authorization: Bearer <key>' or 'X-API-Key' headers.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # 2. Reject unsupported authorization schemes (e.g., Basic, Token, Custom)
    if auth_header and not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Unsupported authorization scheme. Provide credentials via 'Authorization: Bearer <key>' or 'X-API-Key' headers.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # 3. Reject empty Bearer token if X-API-Key is not provided
    if auth_header and auth_header.startswith("Bearer "):
        token_part = auth_header[7:].strip()
        if not token_part and not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Empty or missing Bearer token.",
                headers={"WWW-Authenticate": "Bearer"}
            )
        if token_part and not bearer_token:
            bearer_token = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token_part)

    # 4. Authenticate caller against authoritative server-side identity layer
    principal = await get_current_principal(
        request=request,
        header_key=x_api_key or None,
        bearer_token=bearer_token
    )

    # 5. Open developer mode fallback cannot access protected operational metrics
    if not principal or getattr(principal, "auth_source", "") == "developer_open_mode" or principal.principal_id == "dev_user":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Valid operational credentials required. Open developer mode cannot access protected operational endpoints.",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return principal


@public_router.get("/observability")
async def get_observability_metrics(
    principal: Optional[AuthenticatedPrincipal] = Depends(require_operational_metrics_access)
):
    """
    Returns operational dashboard metrics including P50/P90/P95 latencies, QPS, cache hit rates, guardrails, and document states.
    In hosted deployments (RAG_PROTECT_OPERATIONAL_METRICS=true), enforces authentication.
    """
    return dashboard.get_metrics(indexer)


@public_router.get("/health")
def get_health_status():
    """Returns system health check status, offline policy adherence, index generation, uptime, and auth mode."""
    res = dashboard.get_health_status(indexer)
    res["auth_mode"] = get_auth_mode()
    return res


@public_router.get("/models-status")
async def models_status(
    principal: Optional[AuthenticatedPrincipal] = Depends(require_operational_metrics_access)
):
    """
    Returns status of local Ollama, Gemini API, and index stats.
    In hosted deployments (RAG_PROTECT_OPERATIONAL_METRICS=true), enforces authentication.
    """
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
def ingest_documents(
    req: IngestRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Phase 2: Ingests documents, chunks them, embeds vectors, and indexes them in memory.
    Routes every document through canonical lifecycle management (stage_and_commit_document).
    Guarantees revision tracking, atomic publication, and chunk ID uniqueness.

    Security & Authorization:
    - Strictly restricted to administrative and data_admin roles (admin, superuser, data_admin, sysadmin).
    - Ordinary users (role: user) are denied with HTTP 403 Forbidden to prevent direct document tampering.
    - Records comprehensive audit logs for all attempts (principal_id, doc_id, operation, result).
    """
    # 1. Administrative or data_admin authorization required
    if not (principal.is_admin() or principal.has_role("data_admin")):
        target_ids = [d.doc_id for d in req.documents] if req.documents else []
        dashboard.record_document_audit_event(
            principal_id=principal.principal_id,
            doc_id=",".join(target_ids) if target_ids else "empty",
            operation="ingest",
            result="FORBIDDEN",
            details={"roles": principal.roles, "reason": "Insufficient administrative privileges for raw JSON ingestion"}
        )
        logger.warning(
            f"[Security Audit] Ingestion denied: principal_id='{principal.principal_id}', "
            f"roles={principal.roles}, target_docs={target_ids}. Operation: 'ingest', Result: 'FORBIDDEN'"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrative or data_admin privileges required to ingest documents directly."
        )

    # Request-local chunker prevents concurrency race condition across requests
    request_chunker = TextChunker(
        chunk_size=req.chunk_size or 150,
        overlap=req.overlap or 30
    )

    results = []
    total_chunks_created = 0

    for doc in req.documents:
        # Authoritative existence and lifecycle check on internal document store
        raw_existing_doc = indexer.get_raw_document(doc.doc_id)
        if raw_existing_doc:
            is_deleted = raw_existing_doc.get("status") == "deleted" or bool(raw_existing_doc.get("deleted"))
            if is_deleted:
                if not principal.is_admin():
                    dashboard.record_document_audit_event(
                        principal_id=principal.principal_id,
                        doc_id=doc.doc_id,
                        operation="resurrection_rejected",
                        result="FORBIDDEN",
                        details={"reason": "Non-admin cannot resurrect deleted document via ingest"}
                    )
                    logger.warning(
                        f"[Security Audit] Ingestion resurrection rejected: principal_id='{principal.principal_id}' "
                        f"attempted to ingest/resurrect soft-deleted document '{doc.doc_id}'."
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Access denied: Document '{doc.doc_id}' has been deleted. "
                               f"Restoration requires administrator authorization."
                    )
                op_type = "replace"
            else:
                op_type = "replace"
        else:
            op_type = "create"

        doc_meta = dict(doc.metadata or {})
        doc_meta.setdefault("owner_id", principal.principal_id)
        doc_meta.setdefault("created_by", principal.principal_id)

        res = indexer.stage_and_commit_document(
            doc_id=doc.doc_id,
            text=doc.text,
            title=doc.title,
            chunker=request_chunker,
            metadata=doc_meta
        )
        results.append(res)
        total_chunks_created += res.get("chunks_indexed", 0)

        audit_result = "SUCCESS" if res.get("status") in ("committed", "skipped") else "FAILED"
        dashboard.record_document_audit_event(
            principal_id=principal.principal_id,
            doc_id=doc.doc_id,
            operation=op_type,
            result=audit_result,
            details={
                "version": res.get("version"),
                "status": res.get("status"),
                "chunks_indexed": res.get("chunks_indexed", 0)
            }
        )
        logger.info(
            f"[Security Audit] Document {op_type}: principal_id='{principal.principal_id}', "
            f"doc_id='{doc.doc_id}', version={res.get('version')}, result='{audit_result}'"
        )

    return {
        "phase": "Phase 2: Ingest Pipeline",
        "documents_processed": len(req.documents),
        "total_chunks_created": total_chunks_created,
        "total_index_size": len(indexer.chunks),
        "results": results
    }


async def read_bounded_upload_file(
    file: UploadFile,
    max_file_size: int,
    chunk_size: int = 64 * 1024,
    remaining_request_budget: Optional[int] = None,
    stream_to_disk: bool = True,
    suffix: str = ""
) -> Tuple[Optional[Union[str, bytes]], int, Optional[str]]:
    """
    Streams and reads an UploadFile in bounded chunks.
    When stream_to_disk=True (default), writes directly to a temporary file on disk,
    bounding peak memory strictly to chunk_size (64 KB) instead of duplicating the entire file buffer in RAM.
    Aborts immediately as soon as max_file_size or remaining_request_budget is exceeded.
    Returns:
        (content_path_or_bytes, bytes_read, error_message)
    If a limit is exceeded, content is None and temporary resources are immediately cleaned up.
    """
    total_bytes_read = 0

    if stream_to_disk:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(prefix="rag_upload_", suffix=suffix, delete=False)
        tmp_path = tmp.name
        try:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break

                chunk_len = len(chunk)
                total_bytes_read += chunk_len

                # 1. Enforce individual file size limit
                if total_bytes_read > max_file_size:
                    tmp.close()
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    try:
                        await file.close()
                    except Exception:
                        pass
                    return (
                        None,
                        total_bytes_read,
                        f"File exceeds maximum allowed upload limit of {max_file_size} bytes (aborted early after reading {total_bytes_read} bytes)"
                    )

                # 2. Enforce cumulative request upload budget
                if remaining_request_budget is not None and total_bytes_read > remaining_request_budget:
                    tmp.close()
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    try:
                        await file.close()
                    except Exception:
                        pass
                    return (
                        None,
                        total_bytes_read,
                        f"Cumulative request upload size exceeds maximum allowed limit of {remaining_request_budget} bytes (aborted early after reading {total_bytes_read} bytes)"
                    )

                tmp.write(chunk)

            tmp.flush()
            tmp.close()

            try:
                await file.close()
            except Exception:
                pass

            return tmp_path, total_bytes_read, None
        except Exception as e:
            tmp.close()
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            try:
                await file.close()
            except Exception:
                pass
            return None, total_bytes_read, f"Streaming read error: {str(e)}"

    # In-memory streaming fallback (when stream_to_disk=False)
    chunks: List[bytes] = []
    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break

            chunk_len = len(chunk)
            total_bytes_read += chunk_len

            # 1. Enforce individual file size limit
            if total_bytes_read > max_file_size:
                chunks.clear()
                try:
                    await file.close()
                except Exception:
                    pass
                return (
                    None,
                    total_bytes_read,
                    f"File exceeds maximum allowed upload limit of {max_file_size} bytes (aborted early after reading {total_bytes_read} bytes)"
                )

            # 2. Enforce cumulative request upload budget
            if remaining_request_budget is not None and total_bytes_read > remaining_request_budget:
                chunks.clear()
                try:
                    await file.close()
                except Exception:
                    pass
                return (
                    None,
                    total_bytes_read,
                    f"Cumulative request upload size exceeds maximum allowed limit of {remaining_request_budget} bytes (aborted early after reading {total_bytes_read} bytes)"
                )

            chunks.append(chunk)

        try:
            await file.close()
        except Exception:
            pass

        return b"".join(chunks), total_bytes_read, None
    except Exception as e:
        chunks.clear()
        try:
            await file.close()
        except Exception:
            pass
        return None, total_bytes_read, f"Streaming read error: {str(e)}"


@protected_router.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Phase 2: Uploads one or more files (.pdf, .txt, .md, and code formats).
    Streams uploads in bounded chunks to prevent denial-of-service (DoS) memory exhaustion.
    Enforces request-level file count and cumulative size limits.
    Routes every document through canonical lifecycle management (stage_and_commit_document).
    Guarantees revision tracking, idempotent re-upload handling, and chunk ID uniqueness.

    Security & Authorization:
    - Enforces Document Ownership: Ordinary users CANNOT overwrite/replace documents they do not own.
    - Separates document creation permission from document replacement permission.
    - Records comprehensive audit logs with principal ID, document ID, operation, and result.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files provided.")

    limits = get_upload_limits()
    max_file_size = limits["max_file_size"]
    max_file_count = limits["max_file_count"]
    max_request_size = limits["max_request_size"]
    chunk_size = limits["chunk_size"]

    # 1. Request-level file-count limit
    if len(files) > max_file_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exceeded maximum file count limit: received {len(files)} files, maximum allowed is {max_file_count} files per request."
        )

    processed_count = 0
    total_chunks_created = 0
    cumulative_bytes = 0
    errors = []
    results = []

    for file in files:
        filename = os.path.basename(file.filename or "uploaded_file")
        try:
            # Check if cumulative request budget is already exhausted
            remaining_budget = max_request_size - cumulative_bytes
            if remaining_budget <= 0:
                err_msg = f"'{filename}': Request cumulative upload size limit of {max_request_size} bytes reached; skipping remaining files."
                errors.append(err_msg)
                break

            ext = os.path.splitext(filename)[1].lower()
            tmp_path = None
            try:
                content_or_path, bytes_read, read_err = await read_bounded_upload_file(
                    file=file,
                    max_file_size=max_file_size,
                    chunk_size=chunk_size,
                    remaining_request_budget=remaining_budget,
                    stream_to_disk=True,
                    suffix=ext
                )
                cumulative_bytes += bytes_read

                if read_err:
                    errors.append(f"'{filename}': {read_err}")
                    continue

                if not content_or_path:
                    errors.append(f"'{filename}': File is empty or contains no readable content")
                    continue

                if isinstance(content_or_path, str):
                    tmp_path = content_or_path
                    if os.path.getsize(tmp_path) == 0:
                        errors.append(f"'{filename}': File is empty or contains no readable content")
                        continue
                    doc = DocumentLoader.load_file(tmp_path)
                    doc["doc_id"] = re.sub(r'[^a-zA-Z0-9_\-]', '_', os.path.splitext(filename)[0])
                    doc["title"] = filename
                    doc["source_path"] = filename
                else:
                    doc = DocumentLoader.load_bytes(filename, content_or_path)

                raw_text = doc.get("text", "").strip()
                if not raw_text:
                    errors.append(f"'{filename}': File is empty or contains unreadable/scanned content")
                    continue

                doc_id = doc["doc_id"]

                # Authoritative Existence & Lifecycle Check (directly on internal document store, without authorization filtering)
                raw_existing_doc = indexer.get_raw_document(doc_id)
                if raw_existing_doc:
                    is_deleted = raw_existing_doc.get("status") == "deleted" or bool(raw_existing_doc.get("deleted"))
                    existing_owner = raw_existing_doc.get("owner_id") or raw_existing_doc.get("metadata", {}).get("owner_id") or "system"

                    # Rule 1: Existing deleted document -> reject ordinary user upload, resurrection is administrator-only
                    if is_deleted:
                        if not principal.is_admin():
                            dashboard.record_document_audit_event(
                                principal_id=principal.principal_id,
                                doc_id=doc_id,
                                operation="resurrection_rejected",
                                result="FORBIDDEN",
                                details={
                                    "existing_owner": existing_owner,
                                    "filename": filename,
                                    "reason": "Regular user cannot resurrect soft-deleted document"
                                }
                            )
                            logger.warning(
                                f"[Security Audit] Resurrection rejected: principal_id='{principal.principal_id}' "
                                f"attempted to upload/resurrect soft-deleted document '{doc_id}'. "
                                f"Operation: 'upload_resurrection', Result: 'FORBIDDEN'"
                            )
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail=f"Access denied: Document '{doc_id}' has been deleted. "
                                       f"Restoration or replacement of deleted documents requires administrator authorization."
                            )
                        op_type = "replace"
                    else:
                        # Rule 2: Existing active document -> enforce ownership and replacement permission
                        op_type = "replace"
                        if not principal.is_admin() and existing_owner != principal.principal_id:
                            dashboard.record_document_audit_event(
                                principal_id=principal.principal_id,
                                doc_id=doc_id,
                                operation="overwrite_rejected",
                                result="FORBIDDEN",
                                details={
                                    "existing_owner": existing_owner,
                                    "filename": filename,
                                    "reason": "Caller does not own existing document"
                                }
                            )
                            logger.warning(
                                f"[Security Audit] Overwrite rejected: principal_id='{principal.principal_id}' "
                                f"attempted to overwrite document '{doc_id}' owned by '{existing_owner}'. "
                                f"Operation: 'upload_overwrite', Result: 'FORBIDDEN'"
                            )
                            raise HTTPException(
                                status_code=status.HTTP_403_FORBIDDEN,
                                detail=f"Access denied: Document '{doc_id}' already exists and is owned by '{existing_owner}'. "
                                       f"You do not have permission to overwrite documents you do not own."
                            )

                        # Verify caller replacement privileges
                        if not principal.is_admin():
                            allow_own = os.environ.get("RAG_ALLOW_USER_DOCUMENT_REPLACEMENT", "true").lower() in ("true", "1", "yes")
                            if not (allow_own or any(r in ("editor", "doc_editor", "data_admin") for r in principal.roles)):
                                dashboard.record_document_audit_event(
                                    principal_id=principal.principal_id,
                                    doc_id=doc_id,
                                    operation="overwrite_rejected",
                                    result="FORBIDDEN",
                                    details={"reason": "User lacks document replacement permission"}
                                )
                                raise HTTPException(
                                    status_code=status.HTTP_403_FORBIDDEN,
                                    detail=f"Access denied: Principal '{principal.principal_id}' lacks document replacement permission."
                                )
                else:
                    # Rule 3: Brand-new document -> assign ownership from authenticated principal
                    op_type = "create"

                doc_meta = dict(doc.get("metadata", {}))
                doc_meta.setdefault("owner_id", principal.principal_id)
                doc_meta.setdefault("created_by", principal.principal_id)

                # Request-local chunker prevents concurrency race
                upload_chunker = TextChunker(chunk_size=200, overlap=40)
                res = indexer.stage_and_commit_document(
                    doc_id=doc["doc_id"],
                    text=raw_text,
                    title=doc["title"],
                    chunker=upload_chunker,
                    metadata=doc_meta
                )
                results.append(res)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            audit_result = "SUCCESS" if res.get("status") in ("committed", "skipped") else "FAILED"
            dashboard.record_document_audit_event(
                principal_id=principal.principal_id,
                doc_id=doc_id,
                operation=op_type,
                result=audit_result,
                details={
                    "filename": filename,
                    "version": res.get("version"),
                    "status": res.get("status"),
                    "chunks_indexed": res.get("chunks_indexed", 0)
                }
            )
            logger.info(
                f"[Security Audit] Upload {op_type}: principal_id='{principal.principal_id}', "
                f"doc_id='{doc_id}', version={res.get('version')}, result='{audit_result}'"
            )

            if res.get("status") == "skipped" and res.get("reason") == "no_text_chunks":
                errors.append(f"'{filename}': Produced 0 chunks (insufficient text)")
                continue

            total_chunks_created += res.get("chunks_indexed", 0)
            processed_count += 1
        except HTTPException:
            raise
        except Exception as e:
            err_msg = f"'{filename}': {str(e)}"
            errors.append(err_msg)
            print(f"[Upload Error] {err_msg}", flush=True)

    if processed_count == 0:
        detail_msg = "; ".join(errors) if errors else "No readable text chunks could be extracted from uploaded files."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail_msg)

    return {
        "status": "success",
        "files_processed": processed_count,
        "chunks_created": total_chunks_created,
        "total_index_size": len(indexer.chunks),
        "results": results,
        "warnings": errors if errors else None
    }


@protected_router.post("/ingest-folder")
def ingest_folder(
    req: IngestFolderRequest,
    principal: AuthenticatedPrincipal = Depends(get_current_principal)
):
    """
    Phase 2: Scans a local folder on disk within the allowlisted ingestion root for supported files.
    Protected against arbitrary server path traversal, symlink escapes, DoS file/size exhaustion,
    and can be completely disabled in hosted environments via RAG_ENABLE_FOLDER_INGESTION=false.
    Strictly restricted to administrators.
    """
    # 1. Disable in hosted deployments if configured
    if os.environ.get("RAG_ENABLE_FOLDER_INGESTION", "true").lower() in ("false", "0", "no", "disabled", "off"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Folder ingestion is disabled in hosted environments."
        )

    # 2. Administrative authorization required
    if not principal.is_admin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: Administrator authorization required for folder ingestion."
        )

    if not req.folder_path or not req.folder_path.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Folder path cannot be empty.")

    # 3. Resolve and validate allowlisted root directory
    config_root = os.environ.get("RAG_INGESTION_ROOT_DIR")
    if config_root:
        allowed_root = os.path.realpath(os.path.abspath(config_root))
    else:
        # Default allowlisted root is the project's sample_data directory
        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data")
        allowed_root = os.path.realpath(os.path.abspath(default_dir))

    raw_path = req.folder_path.strip()
    if os.path.isabs(raw_path):
        candidate_path = os.path.realpath(os.path.abspath(raw_path))
    else:
        cwd_candidate = os.path.realpath(os.path.abspath(raw_path))
        root_candidate = os.path.realpath(os.path.abspath(os.path.join(allowed_root, raw_path)))
        try:
            if os.path.commonpath([cwd_candidate, allowed_root]) == allowed_root:
                candidate_path = cwd_candidate
            else:
                candidate_path = root_candidate
        except ValueError:
            candidate_path = root_candidate

    # Verify candidate_path is strictly inside or equal to allowed_root
    try:
        if os.path.commonpath([candidate_path, allowed_root]) != allowed_root:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Path '{req.folder_path}' resolves outside allowlisted ingestion root."
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied: Path '{req.folder_path}' resolves outside allowlisted ingestion root."
        )

    # Verify directory exists
    if not os.path.exists(candidate_path) or not os.path.isdir(candidate_path):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Directory '{req.folder_path}' does not exist.")

    # 4. Limits for files and cumulative size
    try:
        max_files = int(os.environ.get("RAG_MAX_FOLDER_FILES", "100"))
    except ValueError:
        max_files = 100

    try:
        max_total_size = int(os.environ.get("RAG_MAX_FOLDER_SIZE_BYTES", str(50 * 1024 * 1024)))
    except ValueError:
        max_total_size = 50 * 1024 * 1024

    # 5. Load documents with root confinement, symlink escape rejection, and resource caps
    try:
        docs = DocumentLoader.load_directory(
            candidate_path,
            recursive=req.recursive,
            allowed_root=allowed_root,
            max_files=max_files,
            max_total_size=max_total_size,
            reject_symlink_escapes=True
        )
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to load folder documents: {e}")

    # 6. Canonical Document Lifecycle Ingestion with Request-Local Chunker
    request_chunker = TextChunker(
        chunk_size=req.chunk_size or 150,
        overlap=req.overlap or 30
    )

    total_chunks_created = 0
    results = []
    for doc in docs:
        raw_text = doc.get("text", "").strip()
        if not raw_text:
            continue
        res = indexer.stage_and_commit_document(
            doc_id=doc["doc_id"],
            text=raw_text,
            title=doc.get("title", doc["doc_id"]),
            chunker=request_chunker,
            metadata=doc.get("metadata", {})
        )
        results.append(res)
        total_chunks_created += res.get("chunks_indexed", 0)

    return {
        "status": "success",
        "folder_path": req.folder_path,
        "resolved_path": candidate_path,
        "documents_processed": len(docs),
        "chunks_created": total_chunks_created,
        "total_index_size": len(indexer.chunks),
        "results": results
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
    Operational index status and health diagnostics.
    Enforces Document-Level Access Control (RBAC/ABAC) and Lifecycle filtering:
    - Identity and permissions are derived strictly server-side from the authenticated principal.
    - Caller-supplied privilege escalation headers and query parameters are blocked/ignored.
    - Chunks and version registry entries are strictly filtered against the caller's access context.
    - Ordinary users only see chunks and metadata for documents they are authorized to access.
    - Full text export requires administrator privileges and explicit include_text=true.
    """
    if x_user_role or role or x_clearance_level or clearance_level or x_user_id or user_id:
        logger.warning(
            f"[Security Audit] Blocked caller-supplied privilege escalation headers/query parameters on /index-status. "
            f"Enforcing authenticated principal claims: principal_id='{principal.principal_id}', roles={principal.roles}."
        )

    reg = getattr(indexer, 'version_registry', None)
    is_admin = principal.is_admin()
    user_ctx = principal.to_user_context()
    allow_deleted = bool(include_deleted and is_admin)
    allow_full_text = bool(include_text and is_admin)

    # 1. Determine authorized document IDs across source documents and version registry
    all_known_doc_ids = set()
    if hasattr(indexer, 'source_documents') and indexer.source_documents:
        all_known_doc_ids.update(indexer.source_documents.keys())
    if reg and hasattr(reg, 'registry') and reg.registry:
        all_known_doc_ids.update(reg.registry.keys())

    if is_admin and allow_deleted:
        authorized_doc_ids = set(all_known_doc_ids)
    else:
        authorized_doc_ids = {
            doc_id for doc_id in all_known_doc_ids
            if indexer.get_document_status(doc_id, user_context=user_ctx, include_deleted=allow_deleted) is not None
        }

    # 2. Filter version registry entries using principal's access context
    if reg and hasattr(reg, 'registry') and reg.registry:
        if is_admin and allow_deleted:
            filtered_registry = dict(reg.registry)
        else:
            filtered_registry = {
                doc_id: v for doc_id, v in reg.registry.items()
                if doc_id in authorized_doc_ids
            }
    else:
        filtered_registry = {}

    # 3. Filter chunks based on document authorization and chunk access control / lifecycle
    chunks_telemetry = []
    for c in indexer.chunks:
        doc_id = c.get("doc_id")
        if not (is_admin and allow_deleted):
            # Enforce document-level authorization
            if doc_id and doc_id in all_known_doc_ids and doc_id not in authorized_doc_ids:
                continue
            # Enforce chunk-level access control & lifecycle
            if not MetadataFilterEngine.matches(c, user_context=user_ctx, include_deleted=allow_deleted):
                continue

        entry = {
            "chunk_id": c.get("chunk_id"),
            "doc_id": c.get("doc_id"),
            "char_count": len(c.get("text", "")),
            "metadata": c.get("metadata", {})
        }
        if allow_full_text:
            entry["text"] = c.get("text", "")
        chunks_telemetry.append(entry)

    source_docs = getattr(indexer, 'source_documents', {})
    if is_admin and allow_deleted:
        source_docs_count = len(source_docs)
    else:
        source_docs_count = sum(1 for d in source_docs.keys() if d in authorized_doc_ids)

    return {
        "total_indexed_chunks": len(chunks_telemetry),
        "index_generation": getattr(indexer, 'index_generation', 1),
        "embedding_model": indexer.embedding_model,
        "embedding_backend": indexer.embedding_backend,
        "storage_backend": "sqlite_disk" if getattr(indexer, 'disk_store', None) else "numpy_memory",
        "registry": filtered_registry,
        "source_documents_count": source_docs_count,
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
    _, scoped_session_id = resolve_scoped_session(principal, req.session_id)
    history = conversation_manager.get_history(scoped_session_id) if req.use_memory else None
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
        _, scoped_session_id = resolve_scoped_session(principal, req.session_id)
        history = conversation_manager.get_history(scoped_session_id) if req.use_memory else None
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

    results, mode_used = execute_configured_retrieval(
        indexer=indexer,
        query=req.query,
        top_k=req.top_k or 3,
        retrieval_mode=req.retrieval_mode,
        alpha=req.alpha if req.alpha is not None else 0.5,
        use_reranker=req.use_reranker if req.use_reranker is not None else False,
        rrf_k=req.rrf_k if req.rrf_k is not None else 60,
        weight_dense=req.weight_dense,
        weight_bm25=req.weight_bm25,
        candidate_multiplier=req.candidate_multiplier if req.candidate_multiplier is not None else 5,
        filters=req.filters,
        user_context=trusted_user_context
    )
    lat_ms = (time.perf_counter() - t0) * 1000.0
    dashboard.record_request("/retrieve", route=mode_used, latency_ms=lat_ms)
    return {
        "query": req.query,
        "top_k": req.top_k,
        "retrieval_mode": mode_used,
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
    retrieved, mode_used = execute_configured_retrieval(
        indexer=indexer,
        query=standalone_query,
        top_k=req.top_k or 3,
        retrieval_mode=req.retrieval_mode,
        alpha=req.alpha if req.alpha is not None else 0.5,
        use_reranker=req.use_reranker if req.use_reranker is not None else False,
        rrf_k=req.rrf_k if req.rrf_k is not None else 60,
        weight_dense=req.weight_dense,
        weight_bm25=req.weight_bm25,
        candidate_multiplier=req.candidate_multiplier if req.candidate_multiplier is not None else 5,
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
        "retrieval_mode": mode_used,
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
    retrieved, mode_used = execute_configured_retrieval(
        indexer=indexer,
        query=standalone_query,
        top_k=req.top_k or 3,
        retrieval_mode=req.retrieval_mode,
        alpha=req.alpha if req.alpha is not None else 0.5,
        use_reranker=req.use_reranker if req.use_reranker is not None else False,
        rrf_k=req.rrf_k if req.rrf_k is not None else 60,
        weight_dense=req.weight_dense,
        weight_bm25=req.weight_bm25,
        candidate_multiplier=req.candidate_multiplier if req.candidate_multiplier is not None else 5,
        filters=req.filters,
        user_context=trusted_user_context
    )
    
    # Evaluate retrieval confidence using request-local threshold (thread-safe, zero shared object mutation)
    passed, reason = guardrails.evaluate_retrieval(
        retrieved,
        query=standalone_query,
        modality=mode_used if mode_used != "hybrid" else "rrf",
        threshold=req.threshold
    )

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
            "retrieval_mode": mode_used,
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
        "retrieval_mode": mode_used,
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
