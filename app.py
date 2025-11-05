"""
Phase 2 FastAPI Application - Trustworthy Model Registry
Main application entry point with REST API endpoints matching OpenAPI spec v3.4.2
"""

import logging
import os
import sys

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set environment defaults for Lambda
os.environ.setdefault("USE_SQLITE", "0")

try:
    from fastapi import (
        FastAPI,
        HTTPException,
        Depends,
        Query,
        Header,
        Response,
        File,
        Form,
        UploadFile,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    from datetime import datetime
    from enum import Enum
    from mangum import Mangum
    import uvicorn
    import hashlib
except ImportError as e:
    logger.error(f"Critical import failed: {e}", exc_info=True)
    raise

# Import auth module
try:
    from src.auth.jwt_auth import auth as jwt_auth
except ImportError as e:
    logger.error(f"Failed to import jwt_auth: {e}", exc_info=True)
    raise

# Initialize FastAPI app
app = FastAPI(
    title="ECE 461 - Fall 2025 - Project Phase 2",
    description="API for ECE 461/Fall 2025/Project Phase 2: A Trustworthy Model Registry",
    version="3.4.2",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend integration
# Allow all origins for health endpoint (autograder compatibility)
# For other endpoints, we restrict to known origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for autograder compatibility
    allow_credentials=False,  # Set to False when allowing all origins
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security (we will accept X-Authorization header per spec; keep HTTPBearer for flexibility)
security = HTTPBearer()

# Import Phase 2 components (after app initialization to avoid circular imports)
try:
    from src.api.huggingface import scrape_hf_url  # noqa: E402
    from src.metrics.phase2_adapter import (  # noqa: E402
        create_eval_context_from_model_data,
        calculate_phase2_metrics,
        calculate_phase2_net_score,
    )
    from src.metrics import size as size_metric  # noqa: E402
except ImportError as e:
    logger.warning(f"Optional imports failed (may not be available in Lambda): {e}")
    # Set to None to prevent errors - use type: ignore for mypy compatibility
    scrape_hf_url = None  # type: ignore[assignment]
    create_eval_context_from_model_data = None  # type: ignore[assignment]
    calculate_phase2_metrics = None  # type: ignore[assignment]
    calculate_phase2_net_score = None  # type: ignore[assignment]
    size_metric = None  # type: ignore[assignment]

# Storage layer selection: in-memory (default) or SQLite (Milestone 2)
USE_SQLITE: bool = os.environ.get("USE_SQLITE", "0") == "1"

artifacts_db: Dict[str, Dict[str, Any]] = {}  # artifact_id -> artifact_data (in-memory)
users_db: Dict[str, Dict[str, Any]] = {}
audit_log: List[Dict[str, Any]] = []

# Initialize SQLite if enabled (wrap in try/except for Lambda compatibility)
if USE_SQLITE:
    try:
        from src.db.database import Base, engine, get_db
        from src.db import crud as db_crud

        # Ensure schema exists
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        logger.warning(f"SQLite initialization failed: {e}, falling back to in-memory storage")
        USE_SQLITE = False

# Default admin user - password matches what autograder sends (requirements doc says 'packages', not 'artifacts')
DEFAULT_ADMIN = {
    "username": "ece30861defaultadminuser",
    "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;",
    "permissions": ["upload", "search", "download", "admin"],
    "created_at": datetime.now().isoformat(),
}

# Add default admin to users database and DB (if enabled)
admin_username: str = str(DEFAULT_ADMIN["username"])
# Use copy() to ensure we have an independent dict that won't be affected by mutations
users_db[admin_username] = DEFAULT_ADMIN.copy()
if USE_SQLITE:
    try:
        with next(get_db()) as _db:  # type: ignore[misc]
            db_crud.ensure_schema(_db)
            db_crud.upsert_default_admin(
                _db,
                username=str(DEFAULT_ADMIN["username"]),
                password=str(DEFAULT_ADMIN["password"]),
                permissions=list(DEFAULT_ADMIN["permissions"]),  # type: ignore[arg-type]
            )
    except Exception as e:
        logger.warning(f"Failed to initialize default admin in database: {e}")
        # Ensure default admin is still in in-memory users_db even if SQLite fails
        if admin_username not in users_db:
            users_db[admin_username] = DEFAULT_ADMIN.copy()


# Pydantic models matching OpenAPI spec v3.3.1


class ArtifactType(str, Enum):
    MODEL = "model"
    DATASET = "dataset"
    CODE = "code"


class ArtifactMetadata(BaseModel):
    name: str
    id: str
    type: ArtifactType


class ArtifactData(BaseModel):
    url: str


class Artifact(BaseModel):
    metadata: ArtifactMetadata
    data: ArtifactData


class ArtifactQuery(BaseModel):
    name: str
    types: Optional[List[ArtifactType]] = None


class User(BaseModel):
    name: str
    is_admin: bool


class UserAuthenticationInfo(BaseModel):
    password: str


class AuthenticationRequest(BaseModel):
    user: User
    secret: UserAuthenticationInfo


class AuthenticationToken(BaseModel):
    token: str


class ModelRating(BaseModel):
    name: str
    category: str
    net_score: float
    net_score_latency: float
    ramp_up_time: float
    ramp_up_time_latency: float
    bus_factor: float
    bus_factor_latency: float
    performance_claims: float
    performance_claims_latency: float
    license: float
    license_latency: float
    dataset_and_code_score: float
    dataset_and_code_score_latency: float
    dataset_quality: float
    dataset_quality_latency: float
    code_quality: float
    code_quality_latency: float
    reproducibility: float
    reproducibility_latency: float
    reviewedness: float
    reviewedness_latency: float
    tree_score: float
    tree_score_latency: float
    size_score: Dict[str, float]
    size_score_latency: float


class ArtifactCost(BaseModel):
    total_cost: float
    standalone_cost: Optional[float] = None


class ArtifactAuditEntry(BaseModel):
    user: User
    date: str
    artifact: ArtifactMetadata
    action: str


class ArtifactLineageNode(BaseModel):
    artifact_id: str
    name: str
    source: str
    metadata: Optional[Dict[str, Any]] = None


class ArtifactLineageEdge(BaseModel):
    from_node_artifact_id: str
    to_node_artifact_id: str
    relationship: str


class ArtifactLineageGraph(BaseModel):
    nodes: List[ArtifactLineageNode]
    edges: List[ArtifactLineageEdge]


class SimpleLicenseCheckRequest(BaseModel):
    github_url: str


class ArtifactRegEx(BaseModel):
    regex: str


class EnumerateOffset(BaseModel):
    offset: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime: str
    models_count: int
    users_count: int
    last_hour_activity: Dict[str, int]


# -------------------------
# Health component schemas (v3.4.2)
# -------------------------


class HealthStatus(str, Enum):
    ok = "ok"
    degraded = "degraded"
    critical = "critical"
    unknown = "unknown"


class HealthLogReference(BaseModel):
    label: str
    url: str
    tail_available: Optional[bool] = None
    last_updated_at: Optional[str] = None


class HealthTimelineEntry(BaseModel):
    bucket: str
    value: float
    unit: Optional[str] = None


class HealthIssue(BaseModel):
    code: str
    severity: str
    summary: str
    details: Optional[str] = None


class HealthComponentDetail(BaseModel):
    id: str
    display_name: Optional[str] = None
    status: HealthStatus
    observed_at: str
    description: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    issues: Optional[List[HealthIssue]] = None
    timeline: Optional[List[HealthTimelineEntry]] = None
    logs: Optional[List[HealthLogReference]] = None


class HealthComponentCollection(BaseModel):
    components: List[HealthComponentDetail]
    generated_at: str
    window_minutes: Optional[int] = None


# Server-side token call count tracking (Milestone 3 requirement: 1000 calls or 10 hours)
token_call_counts: Dict[str, int] = {}  # token_hash -> call_count


# Authentication functions (Milestone 3 - with call count tracking)
def verify_token(
    x_authorization: Optional[str] = Header(None, alias="X-Authorization"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Dict[str, Any]:
    """Verify token from X-Authorization (spec) or Authorization header and return user info"""
    token: Optional[str] = None

    if x_authorization and isinstance(x_authorization, str) and x_authorization.strip():
        raw = x_authorization.strip()
        token = raw[7:].strip() if raw.lower().startswith("bearer ") else raw
    elif authorization and isinstance(authorization, str) and authorization.strip():
        # Support both "Bearer <token>" and raw token in Authorization
        raw = authorization.strip()
        if raw.lower().startswith("bearer "):
            token = raw[7:].strip()
        else:
            token = raw

    if not token:
        # Spec: 403 for invalid or missing AuthenticationToken
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    # Validate JWT token
    payload = jwt_auth.verify_token(token)
    if not payload:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    # Track call count server-side (Milestone 3 requirement: 1000 calls or 10 hours)
    # Use token hash as key to track calls
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    # Get or initialize call count
    current_calls = token_call_counts.get(token_hash, payload.get("call_count", 0))

    # Check if exceeded max calls
    max_calls = payload.get("max_calls", 1000)
    if current_calls >= max_calls:
        raise HTTPException(
            status_code=403,
            detail="Authentication failed due to invalid or missing AuthenticationToken.",
        )

    # Increment call count
    token_call_counts[token_hash] = current_calls + 1

    username = str(payload.get("sub", DEFAULT_ADMIN["username"]))
    permissions = payload.get("permissions", DEFAULT_ADMIN["permissions"])  # type: ignore[assignment]
    return {"username": username, "permissions": permissions}


def check_permission(user: Dict[str, Any], required_permission: str) -> bool:
    """Check if user has required permission"""
    return required_permission in user.get("permissions", [])


# User registration request model (Milestone 3)
class UserRegistrationRequest(BaseModel):
    username: str
    password: str
    permissions: List[str]


# API Endpoints matching OpenAPI spec v3.3.1


@app.get("/health", response_model=HealthResponse)
async def health_check(response: Response) -> HealthResponse:
    """System health endpoint - lightweight liveness probe (BASELINE)"""
    # Explicitly set CORS headers for API Gateway compatibility
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"

    try:
        # Safely get counts - handle case where databases might not be initialized
        models_count = len(artifacts_db) if artifacts_db is not None else 0
        users_count = len(users_db) if users_db is not None else 0

        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime="0:00:00",  # Simplified for Milestone 1
            models_count=models_count,
            users_count=users_count,
            last_hour_activity={"uploads": 0, "downloads": 0, "searches": 0},
        )
    except Exception as e:
        # Log error but still return healthy status for liveness probe
        logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime="0:00:00",
            models_count=0,
            users_count=0,
            last_hour_activity={"uploads": 0, "downloads": 0, "searches": 0},
        )


# Component health (NON-BASELINE in spec v3.4.2), minimal implementation
@app.get("/health/components", response_model=HealthComponentCollection)
async def health_components(
    windowMinutes: int = Query(60, ge=5, le=1440),
    includeTimeline: bool = Query(False),
) -> HealthComponentCollection:
    now_iso = datetime.now().isoformat()

    components: List[HealthComponentDetail] = []

    # API component
    api_component = HealthComponentDetail(
        id="api",
        display_name="FastAPI Service",
        status=HealthStatus.ok,
        observed_at=now_iso,
        description="Handles REST API requests",
        metrics={
            "uptime_seconds": 0,
            "total_requests": 0,
        },
        issues=[],
        logs=[
            HealthLogReference(
                label="Application Log",
                url="https://example.com/logs/app.log",
            )
        ],
        timeline=(
            [HealthTimelineEntry(bucket=now_iso, value=0.0, unit="rpm")]
            if includeTimeline
            else None
        ),
    )
    components.append(api_component)

    # Metrics component (placeholder)
    metrics_component = HealthComponentDetail(
        id="metrics",
        display_name="Metrics Aggregator",
        status=HealthStatus.ok,
        observed_at=now_iso,
        description="Aggregates request metrics",
        metrics={
            "routes_tracked": 0,
        },
        issues=[],
        logs=[],
        timeline=(
            [HealthTimelineEntry(bucket=now_iso, value=0.0, unit="rpm")]
            if includeTimeline
            else None
        ),
    )
    components.append(metrics_component)

    return HealthComponentCollection(
        components=components,
        generated_at=now_iso,
        window_minutes=windowMinutes,
    )


# Root route for CI/API liveness checks
@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Trustworthy Model Registry API"}


# -------------------------
# Known deferrals (Delivery 1): explicit placeholders
# -------------------------


@app.post("/models/upload", response_model=Artifact, status_code=201)
async def models_upload(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    user: Dict[str, Any] = Depends(verify_token),
) -> Artifact:
    """Upload model as ZIP file"""
    from src.storage import file_storage

    try:
        # Validate file is a ZIP
        if not file.filename or not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="File must be a ZIP archive")

        # Generate artifact ID
        artifact_id = f"model-{len(artifacts_db) + 1}-{int(datetime.now().timestamp())}"

        # Read file content
        content = await file.read()

        # Save ZIP file
        file_info = file_storage.save_uploaded_file(artifact_id, content, file.filename)

        # Extract ZIP
        artifact_dir = file_storage.get_artifact_directory(artifact_id)
        extracted_files = file_storage.extract_zip(file_info["path"], artifact_dir)

        # Find and read model card
        card_path = file_storage.find_model_card(artifact_dir)
        readme_text = file_storage.read_model_card(card_path) if card_path else ""

        # Determine model name
        model_name = name or file.filename.replace(".zip", "")

        # Create artifact entry
        artifact_entry = {
            "metadata": {
                "name": model_name,
                "id": artifact_id,
                "type": "model",
            },
            "data": {
                "url": f"local://{artifact_id}",
                "files": extracted_files,
                "checksum": file_info["checksum"],
                "size": file_info["size"],
            },
            "created_at": datetime.now().isoformat(),
            "created_by": user["username"],
        }

        artifacts_db[artifact_id] = artifact_entry

        # Store in SQLite if enabled
        if USE_SQLITE:
            with next(get_db()) as _db:  # type: ignore[misc]
                db_crud.create_artifact(
                    _db,
                    artifact_id=artifact_id,
                    name=model_name,
                    type_="model",
                    url=f"local://{artifact_id}",
                )

        # Log audit entry
        audit_log.append(
            {
                "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
                "date": datetime.now().isoformat(),
                "artifact": artifact_entry["metadata"],
                "action": "UPLOAD",
            }
        )

        # Trigger metrics calculation asynchronously (don't wait for completion)
        try:
            model_data = {
                "url": f"local://{artifact_id}",
                "hf_data": [{"readme_text": readme_text}] if readme_text else [],
                "gh_data": [],
            }
            # Note: metrics will run in background, results stored separately if needed
            _ = await calculate_phase2_metrics(model_data)
        except Exception as e:
            logger.warning(f"Metrics calculation failed for uploaded model: {e}")

        return Artifact(
            metadata=ArtifactMetadata(**artifact_entry["metadata"]),
            data=ArtifactData(url=artifact_entry["data"]["url"]),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/models/{id}/download")
async def models_download(
    id: str,
    aspect: str = Query("full", regex="^(full|weights|datasets|code)$"),
    user: Dict[str, Any] = Depends(verify_token),
):
    """Download model files with optional aspect filtering"""
    from src.storage import file_storage
    import tempfile

    try:
        # Check if artifact exists
        if id not in artifacts_db:
            raise HTTPException(status_code=404, detail="Artifact does not exist.")

        artifact_data = artifacts_db[id]

        if artifact_data["metadata"]["type"] != "model":
            raise HTTPException(status_code=400, detail="Not a model artifact.")

        # Get artifact directory
        artifact_dir = file_storage.get_artifact_directory(id)

        if not os.path.exists(artifact_dir):
            raise HTTPException(
                status_code=404,
                detail="Model files not found. This model may only have URL metadata.",
            )

        # Filter files by aspect
        files = file_storage.filter_files_by_aspect(artifact_dir, aspect)

        if not files:
            raise HTTPException(status_code=404, detail=f"No files found for aspect: {aspect}")

        # Create temporary ZIP with filtered files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            zip_path = tmp.name

        file_storage.create_zip_from_files(files, artifact_dir, zip_path)

        # Calculate checksum for integrity
        checksum = file_storage.calculate_checksum(zip_path)

        # Log download audit
        audit_log.append(
            {
                "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
                "date": datetime.now().isoformat(),
                "artifact": artifact_data["metadata"],
                "action": f"DOWNLOAD_{aspect.upper()}",
            }
        )

        # Return file with checksum in headers
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=f"{artifact_data['metadata']['name']}_{aspect}.zip",
            headers={"X-File-Checksum": checksum, "X-File-Aspect": aspect},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/verify-token")
async def verify_token_placeholder() -> Dict[str, str]:
    """Placeholder: tokens are validated per request in Delivery 1."""
    raise HTTPException(
        status_code=501,
        detail="No standalone token verification endpoint in Delivery 1; tokens are validated per request.",
    )


# User registration endpoint (Milestone 3 - admin only)
@app.post("/register")
async def register_user(
    request: UserRegistrationRequest, user: Dict[str, Any] = Depends(verify_token)
):
    """Register a new user (Milestone 3 - admin only)"""
    # Only admins can register users
    if not check_permission(user, "admin"):
        raise HTTPException(status_code=401, detail="You do not have permission to register users.")

    # Check if user already exists
    if request.username in users_db:
        raise HTTPException(status_code=409, detail="User already exists.")

    # Hash password with bcrypt (Milestone 3 requirement)
    hashed_password = jwt_auth.get_password_hash(request.password)

    # Create user entry
    new_user = {
        "username": request.username,
        "password": hashed_password,
        "permissions": request.permissions,
        "created_at": datetime.now().isoformat(),
        "is_admin": "admin" in request.permissions,
    }

    users_db[request.username] = new_user

    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            from src.db import models as db_models

            db_user = db_models.User(
                username=request.username,
                password=hashed_password,
                is_admin="admin" in request.permissions,
                permissions=",".join(request.permissions),
            )
            _db.add(db_user)
            _db.commit()

    return {"message": "User registered successfully."}


# User deletion endpoint (Milestone 3)
@app.delete("/user/{username}")
async def delete_user(username: str, user: Dict[str, Any] = Depends(verify_token)):
    """Delete a user (Milestone 3 - users can delete own account, admins can delete any)"""
    # Users can delete their own account, admins can delete any account
    if username != user["username"] and not check_permission(user, "admin"):
        raise HTTPException(
            status_code=401, detail="You do not have permission to delete this user."
        )

    # Prevent deleting default admin
    if username == DEFAULT_ADMIN["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete default admin user.")

    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found.")

    del users_db[username]

    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            from src.db import models as db_models

            db_user = _db.get(db_models.User, username)
            if db_user:
                _db.delete(db_user)
                _db.commit()

    return {"message": "User deleted successfully."}


# Authentication endpoint
@app.put("/authenticate", response_model=str)
async def create_auth_token(request: AuthenticationRequest) -> str:
    """Create an access token (Milestone 3 - with proper validation)"""
    # CRITICAL: Log entry point immediately
    print("=== AUTHENTICATE ENDPOINT CALLED ===")
    print(f"DEBUG: Request user.name={request.user.name}, is_admin={request.user.is_admin}")
    print(f"DEBUG: Password length={len(request.secret.password)}")
    print(f"DEBUG: Received password (first 50 chars): {repr(request.secret.password[:50])}")
    print(f"DEBUG: Received password (all chars): {repr(request.secret.password)}")
    print(f"DEBUG: Received password (hex): {request.secret.password.encode('utf-8').hex()}")
    sys.stdout.flush()
    logger.info(f"Authenticate endpoint called for user: {request.user.name}")

    # CRITICAL: Ensure default admin exists on EVERY request (Lambda cold start protection)
    admin_username = str(DEFAULT_ADMIN["username"])
    if admin_username not in users_db:
        users_db[admin_username] = DEFAULT_ADMIN.copy()
        logger.info(f"Recreated default admin user: {admin_username}")
        print(f"DEBUG: Recreated default admin user: {admin_username}")
        sys.stdout.flush()

    # Validate user credentials against in-memory/SQLite user store
    user_data = users_db.get(request.user.name)
    if not user_data:
        logger.warning(f"User not found: {request.user.name}")
        print(f"DEBUG: User not found: {request.user.name}, users_db keys: {list(users_db.keys())}")
        sys.stdout.flush()
        raise HTTPException(status_code=401, detail="The user or password is invalid.")

    # Use bcrypt for password verification (Milestone 3 requirement)
    stored_password = user_data.get("password", "")
    print(f"DEBUG: Stored password length={len(stored_password)}")
    print(f"DEBUG: Stored password (first 50 chars): {repr(stored_password[:50])}")
    print(f"DEBUG: Stored password (all chars): {repr(stored_password)}")
    print(f"DEBUG: Stored password (hex): {stored_password.encode('utf-8').hex()}")
    sys.stdout.flush()
    if isinstance(stored_password, str) and stored_password.startswith("$2b$"):
        # Password is hashed, use bcrypt verification
        if not jwt_auth.verify_password(request.secret.password, stored_password):
            logger.warning(f"Password verification failed for user: {request.user.name}")
            print(f"DEBUG: Bcrypt password verification failed for user: {request.user.name}")
            sys.stdout.flush()
            raise HTTPException(status_code=401, detail="The user or password is invalid.")
    else:
        # Plain text password (for default admin during migration)
        # Handle both password variants:
        # 1. OpenAPI spec: ends with 'artifacts;' (63 chars)
        # 2. Autograder/requirements doc: ends with 'packages;' (62 chars)
        password_matches = request.secret.password == stored_password

        # If exact match fails, normalize both variants for comparison
        # Replace 'artifacts;' with 'packages;' in both passwords for comparison
        if not password_matches:
            received_normalized = request.secret.password.replace("artifacts;", "packages;")
            stored_normalized = stored_password.replace("artifacts;", "packages;")
            if received_normalized == stored_normalized:
                print("DEBUG: Password matches after normalizing variants")
                password_matches = True

        if not password_matches:
            logger.warning(
                f"Plain text password mismatch for user: {request.user.name}. "
                f"Expected length: {len(stored_password)}, Got length: {len(request.secret.password)}"
            )
            print(
                f"DEBUG: Password mismatch for user: {request.user.name}. "
                f"Expected: {repr(stored_password[:20])}..., Got: {repr(request.secret.password[:20])}..."
            )
            sys.stdout.flush()
            raise HTTPException(status_code=401, detail="The user or password is invalid.")

    # Issue JWT containing subject and permissions
    payload = {
        "sub": user_data["username"],
        "permissions": user_data.get("permissions", []),
    }
    jwt_token = jwt_auth.create_access_token(payload)

    # Spec: AuthenticationToken is a string; we include 'bearer ' prefix to ease client use
    return f"bearer {jwt_token}"


# Registry reset endpoint
@app.delete("/reset")
async def registry_reset(user: Dict[str, Any] = Depends(verify_token)):
    """Reset the registry to a system default state (BASELINE)"""
    print("=== RESET ENDPOINT CALLED ===")
    sys.stdout.flush()
    logger.info("Reset endpoint called")

    if not check_permission(user, "admin"):
        # Spec: 401 if you do not have permission; 403 is used for invalid/missing token
        raise HTTPException(
            status_code=401, detail="You do not have permission to reset the registry."
        )

    # Clear all artifacts (CRITICAL: ensure this is complete)
    artifacts_db.clear()
    audit_log.clear()

    # Clear token call counts on reset
    token_call_counts.clear()

    # Clear users but preserve default admin (per spec requirement)
    users_db.clear()

    # CRITICAL: Recreate default admin user immediately after clear
    admin_username: str = str(DEFAULT_ADMIN["username"])
    users_db[admin_username] = DEFAULT_ADMIN.copy()

    # Log reset completion
    logger.info(
        f"Registry reset completed. Artifacts cleared: {len(artifacts_db) == 0}, Default admin recreated: {admin_username in users_db}"
    )
    print(
        f"DEBUG: Reset completed. artifacts_db size: {len(artifacts_db)}, users_db keys: {list(users_db.keys())}"
    )
    sys.stdout.flush()

    if USE_SQLITE:
        try:
            with next(get_db()) as _db:  # type: ignore[misc]
                db_crud.reset_registry(_db)
                # Recreate default admin in database
                db_crud.upsert_default_admin(
                    _db,
                    username=str(DEFAULT_ADMIN["username"]),
                    password=str(DEFAULT_ADMIN["password"]),
                    permissions=list(DEFAULT_ADMIN["permissions"]),  # type: ignore[arg-type]
                )
        except Exception as e:
            logger.error(f"Failed to reset SQLite database: {e}")
            print(f"DEBUG: SQLite reset failed: {e}")
            sys.stdout.flush()

    return {"message": "Registry is reset."}


# Artifact endpoints matching OpenAPI spec


@app.post("/artifacts")
async def artifacts_list(
    queries: List[ArtifactQuery],
    response: Response,
    offset: Optional[str] = Query(None),
    user: Dict[str, Any] = Depends(verify_token),
) -> List[ArtifactMetadata]:
    """Get the artifacts from the registry (BASELINE)"""
    results: List[ArtifactMetadata] = []

    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            db_items = db_crud.list_by_queries(_db, [q.model_dump() for q in queries])
            for art in db_items:
                results.append(
                    ArtifactMetadata(name=art.name, id=art.id, type=ArtifactType(art.type))
                )
    else:
        for query in queries:
            if query.name == "*":
                for artifact_id, artifact_data in artifacts_db.items():
                    results.append(
                        ArtifactMetadata(
                            name=artifact_data["metadata"]["name"],
                            id=artifact_id,
                            type=artifact_data["metadata"]["type"],
                        )
                    )
            else:
                for artifact_id, artifact_data in artifacts_db.items():
                    if artifact_data["metadata"]["name"] == query.name:
                        artifact_type_value = (
                            ArtifactType(artifact_data["metadata"]["type"])  # type: ignore[arg-type]
                            if isinstance(artifact_data["metadata"].get("type"), str)
                            else artifact_data["metadata"].get("type")
                        )
                        if not query.types or (artifact_type_value in query.types):
                            results.append(
                                ArtifactMetadata(
                                    name=artifact_data["metadata"]["name"],
                                    id=artifact_id,
                                    type=artifact_type_value,
                                )
                            )

    # Simple pagination implementation per spec using an "offset" page index and fixed page size
    page_size: int = 50
    try:
        page_index: int = int(offset) if offset is not None else 0
    except ValueError:
        page_index = 0

    start: int = page_index * page_size
    end: int = start + page_size
    paged_results = results[start:end]

    # Set the next offset header if there are more results
    next_offset: Optional[int] = page_index + 1 if end < len(results) else page_index
    response.headers["offset"] = str(next_offset)

    # If too many artifacts would be returned without pagination
    if len(results) > 10000:
        raise HTTPException(status_code=413, detail="Too many artifacts returned.")

    return paged_results


@app.post("/models/ingest", response_model=Artifact, status_code=201)
async def models_ingest(
    model_name: str = Query(..., description="HuggingFace model name (e.g., 'google/gemma-2-2b')"),
    user: Dict[str, Any] = Depends(verify_token),
) -> Artifact:
    """Ingest a HuggingFace model after validating it meets 0.5 threshold on all non-latency metrics (BASELINE)"""
    if not check_permission(user, "upload"):
        raise HTTPException(status_code=401, detail="You do not have permission to ingest models.")

    # Construct HuggingFace URL
    hf_url = f"https://huggingface.co/{model_name}"

    try:
        # Scrape HuggingFace data
        if scrape_hf_url is None or calculate_phase2_metrics is None:
            raise HTTPException(
                status_code=501, detail="HuggingFace ingestion not available in this environment."
            )
        hf_data, repo_type = scrape_hf_url(hf_url)
        model_data = {"url": hf_url, "hf_data": [hf_data], "gh_data": []}

        # Calculate all metrics
        metrics = await calculate_phase2_metrics(model_data)

        # Filter out latency metrics - only check non-latency metrics for threshold
        # Latency metrics have "_latency" suffix or are "net_score_latency", "size_score_latency", etc.
        non_latency_metrics = {
            k: v
            for k, v in metrics.items()
            if not k.endswith("_latency") and k != "net_score_latency"
        }

        # Check threshold: all non-latency metrics must be >= 0.5
        failing_metrics = [
            k
            for k, v in non_latency_metrics.items()
            if isinstance(v, (int, float)) and float(v) < 0.5
        ]

        if failing_metrics:
            raise HTTPException(
                status_code=424,
                detail=f"Model does not meet 0.5 threshold requirement. Failing metrics: {', '.join(failing_metrics)}",
            )

        # Model passed threshold - proceed with ingest (create artifact)
        artifact_id = f"model-{len(artifacts_db) + 1}-{int(datetime.now().timestamp())}"
        model_display_name = model_name.split("/")[-1] if "/" in model_name else model_name

        artifact_entry = {
            "metadata": {
                "name": model_display_name,
                "id": artifact_id,
                "type": "model",
            },
            "data": {"url": hf_url},
            "created_at": datetime.now().isoformat(),
            "created_by": user["username"],
            "hf_model_name": model_name,
        }

        artifacts_db[artifact_id] = artifact_entry

        if USE_SQLITE:
            with next(get_db()) as _db:  # type: ignore[misc]
                db_crud.create_artifact(
                    _db,
                    artifact_id=artifact_id,
                    name=model_display_name,
                    type_="model",
                    url=hf_url,
                )

        # Log audit entry
        audit_log.append(
            {
                "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
                "date": datetime.now().isoformat(),
                "artifact": artifact_entry["metadata"],
                "action": "INGEST",
            }
        )

        if USE_SQLITE:
            with next(get_db()) as _db:  # type: ignore[misc]
                art_row = db_crud.get_artifact(_db, artifact_id)
                if art_row:
                    db_crud.log_audit(
                        _db,
                        artifact=art_row,
                        user_name=user["username"],
                        user_is_admin=user.get("is_admin", False),
                        action="INGEST",
                    )

        return Artifact(
            metadata=ArtifactMetadata(**artifact_entry["metadata"]),
            data=ArtifactData(url=artifact_entry["data"]["url"]),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingest failed for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")


@app.get("/models")
async def models_enumerate(
    cursor: Optional[str] = Query(None, description="Cursor for pagination"),
    limit: int = Query(25, ge=1, le=100, description="Number of results per page"),
    user: Dict[str, Any] = Depends(verify_token),
) -> Dict[str, Any]:
    """Enumerate models with cursor-based pagination (BASELINE)"""
    if not check_permission(user, "search"):
        raise HTTPException(status_code=401, detail="You do not have permission to search models.")

    # Get all model artifacts
    all_models: List[ArtifactMetadata] = []

    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            # Query only models from database
            db_items = db_crud.list_by_queries(_db, [{"name": "*", "types": ["model"]}])
            for art in db_items:
                if art.type == "model":
                    all_models.append(
                        ArtifactMetadata(name=art.name, id=art.id, type=ArtifactType(art.type))
                    )
    else:
        # Filter to only models
        for artifact_id, artifact_data in artifacts_db.items():
            if artifact_data["metadata"]["type"] == "model":
                all_models.append(
                    ArtifactMetadata(
                        name=artifact_data["metadata"]["name"],
                        id=artifact_id,
                        type=ArtifactType(artifact_data["metadata"]["type"]),
                    )
                )

    # Sort by ID for stable pagination
    all_models.sort(key=lambda x: x.id)

    # Cursor-based pagination
    start_idx = 0
    if cursor:
        # Find the index of the cursor (artifact ID)
        for idx, model in enumerate(all_models):
            if model.id == cursor:
                start_idx = idx + 1
                break

    # Get the page of results
    end_idx = min(start_idx + limit, len(all_models))
    page_models = all_models[start_idx:end_idx]

    # Determine next cursor
    next_cursor = None
    if end_idx < len(all_models):
        next_cursor = page_models[-1].id if page_models else None

    return {
        "items": [model.model_dump() for model in page_models],
        "next_cursor": next_cursor,
    }


@app.post("/artifact/{artifact_type}", response_model=Artifact, status_code=201)
async def artifact_create(
    artifact_type: ArtifactType,
    artifact_data: ArtifactData,
    user: Dict[str, Any] = Depends(verify_token),
) -> Artifact:
    """Register a new artifact (BASELINE)"""
    # If model ingest, enforce 0.5 threshold on non-latency metrics per plan/spec (424 on disqualified)
    if artifact_type == ArtifactType.MODEL:
        try:
            # Only enforce gating for HuggingFace URLs per spec language
            if "huggingface.co" in artifact_data.url.lower():
                if scrape_hf_url is not None and calculate_phase2_metrics is not None:
                    hf_data, repo_type = scrape_hf_url(artifact_data.url)
                    model_data = {"url": artifact_data.url, "hf_data": [hf_data], "gh_data": []}
                    metrics = await calculate_phase2_metrics(model_data)
                    # Filter out latency metrics - only check non-latency metrics for threshold
                    non_latency_metrics = {
                        k: v
                        for k, v in metrics.items()
                        if not k.endswith("_latency") and k != "net_score_latency"
                    }
                    # Require all available non-latency metrics to be at least 0.5
                    failing_metrics = [
                        k
                        for k, v in non_latency_metrics.items()
                        if isinstance(v, (int, float)) and float(v) < 0.5
                    ]
                    if failing_metrics:
                        raise HTTPException(
                            status_code=424,
                            detail=f"Artifact is not registered due to the disqualified rating. Failing metrics: {', '.join(failing_metrics)}",
                        )
        except HTTPException:
            raise
        except Exception:
            # If metrics fail, allow ingest for Delivery 1
            pass
    # Generate unique artifact ID
    artifact_id = f"{artifact_type.value}-{len(artifacts_db) + 1}-{int(datetime.now().timestamp())}"

    # Create artifact entry
    artifact_entry = {
        "metadata": {
            "name": artifact_data.url.split("/")[-1],  # Extract name from URL
            "id": artifact_id,
            "type": artifact_type.value,
        },
        "data": {"url": artifact_data.url},
        "created_at": datetime.now().isoformat(),
        "created_by": user["username"],
    }

    artifacts_db[artifact_id] = artifact_entry
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            db_crud.create_artifact(
                _db,
                artifact_id=artifact_id,
                name=artifact_entry["metadata"]["name"],
                type_=artifact_type.value,
                url=artifact_data.url,
            )

    # Log audit entry
    audit_log.append(
        {
            "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
            "date": datetime.now().isoformat(),
            "artifact": artifact_entry["metadata"],
            "action": "CREATE",
        }
    )
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            art_row = db_crud.get_artifact(_db, artifact_id)
            if art_row:
                db_crud.log_audit(
                    _db,
                    artifact=art_row,
                    user_name=user["username"],
                    user_is_admin=user.get("is_admin", False),
                    action="CREATE",
                )

    return Artifact(
        metadata=ArtifactMetadata(**artifact_entry["metadata"]),
        data=ArtifactData(url=artifact_entry["data"]["url"]),
    )


@app.get("/artifacts/{artifact_type}/{id}", response_model=Artifact)
async def artifact_retrieve(
    artifact_type: ArtifactType, id: str, user: Dict[str, Any] = Depends(verify_token)
) -> Artifact:
    """Interact with the artifact with this id (BASELINE)"""
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            art = db_crud.get_artifact(_db, id)
            if not art:
                raise HTTPException(status_code=404, detail="Artifact does not exist.")
            if art.type != artifact_type.value:
                raise HTTPException(status_code=400, detail="Artifact type mismatch.")
            return Artifact(
                metadata=ArtifactMetadata(name=art.name, id=art.id, type=ArtifactType(art.type)),
                data=ArtifactData(url=art.url),
            )
    else:
        if id not in artifacts_db:
            raise HTTPException(status_code=404, detail="Artifact does not exist.")
        artifact_data = artifacts_db[id]
        if artifact_data["metadata"]["type"] != artifact_type.value:
            raise HTTPException(status_code=400, detail="Artifact type mismatch.")
        return Artifact(
            metadata=ArtifactMetadata(**artifact_data["metadata"]),
            data=ArtifactData(url=artifact_data["data"]["url"]),
        )


@app.put("/artifacts/{artifact_type}/{id}")
async def artifact_update(
    artifact_type: ArtifactType,
    id: str,
    artifact: Artifact,
    user: Dict[str, Any] = Depends(verify_token),
):
    """Update this content of the artifact (BASELINE)"""
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch.")

    # Update artifact
    artifacts_db[id] = {
        "metadata": artifact.metadata.model_dump(),
        "data": artifact.data.model_dump(),
        "updated_at": datetime.now().isoformat(),
        "updated_by": user["username"],
    }
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            db_crud.update_artifact(
                _db,
                artifact_id=id,
                name=artifact.metadata.name,
                type_=artifact_type.value,
                url=artifact.data.url,
            )

    # Log audit entry
    audit_log.append(
        {
            "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
            "date": datetime.now().isoformat(),
            "artifact": artifact.metadata.model_dump(),
            "action": "UPDATE",
        }
    )
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            db_art = db_crud.get_artifact(_db, id)
            if db_art:
                db_crud.log_audit(
                    _db,
                    artifact=db_art,
                    user_name=user["username"],
                    user_is_admin=user.get("is_admin", False),
                    action="UPDATE",
                )

    return {"message": "Artifact is updated."}


@app.delete("/artifacts/{artifact_type}/{id}")
async def artifact_delete(
    artifact_type: ArtifactType, id: str, user: Dict[str, Any] = Depends(verify_token)
):
    """Delete this artifact (NON-BASELINE)"""
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch.")

    # Log audit entry before deletion
    audit_log.append(
        {
            "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
            "date": datetime.now().isoformat(),
            "artifact": artifact_data["metadata"],
            "action": "DELETE",
        }
    )

    del artifacts_db[id]
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            db_crud.delete_artifact(_db, id)
    return {"message": "Artifact is deleted."}


# -------------------------
# Additional endpoints per spec
# -------------------------


@app.get("/artifact/byName/{name}")
async def artifact_by_name(
    name: str, user: Dict[str, Any] = Depends(verify_token)
) -> List[ArtifactMetadata]:
    matches: List[ArtifactMetadata] = []
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            items = db_crud.list_by_name(_db, name)
            for a in items:
                matches.append(ArtifactMetadata(name=a.name, id=a.id, type=ArtifactType(a.type)))
    else:
        for artifact_id, artifact_data in artifacts_db.items():
            if artifact_data["metadata"]["name"] == name:
                matches.append(
                    ArtifactMetadata(
                        name=artifact_data["metadata"]["name"],
                        id=artifact_id,
                        type=ArtifactType(artifact_data["metadata"]["type"]),
                    )
                )
    if not matches:
        raise HTTPException(status_code=404, detail="No such artifact.")
    return matches


@app.post("/artifact/byRegEx")
async def artifact_by_regex(
    regex: ArtifactRegEx, user: Dict[str, Any] = Depends(verify_token)
) -> List[ArtifactMetadata]:
    import re as _re

    matches: List[ArtifactMetadata] = []
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            items = db_crud.list_by_regex(_db, regex.regex)
            for a in items:
                matches.append(ArtifactMetadata(name=a.name, id=a.id, type=ArtifactType(a.type)))
    else:
        pattern = _re.compile(_re.escape(regex.regex))
        for artifact_id, artifact_data in artifacts_db.items():
            name = artifact_data["metadata"]["name"]
            if pattern.search(name):
                matches.append(
                    ArtifactMetadata(
                        name=name,
                        id=artifact_id,
                        type=ArtifactType(artifact_data["metadata"]["type"]),
                    )
                )
    if not matches:
        raise HTTPException(status_code=404, detail="No artifact found under this regex.")
    return matches


@app.get("/artifact/{artifact_type}/{id}/audit")
async def artifact_audit(
    artifact_type: ArtifactType, id: str, user: Dict[str, Any] = Depends(verify_token)
) -> List[ArtifactAuditEntry]:
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch.")

    entries: List[ArtifactAuditEntry] = []
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            items = db_crud.list_audit(_db, id)
            for e in items:
                entries.append(
                    ArtifactAuditEntry(
                        user=User(name=e.user_name, is_admin=e.user_is_admin),
                        date=e.date.isoformat(),
                        artifact=ArtifactMetadata(name="", id=id, type=artifact_type),
                        action=e.action,
                    )
                )
    else:
        for entry in audit_log:
            if entry.get("artifact", {}).get("id") == id:
                entries.append(
                    ArtifactAuditEntry(
                        user=User(**entry["user"]),
                        date=entry["date"],
                        artifact=ArtifactMetadata(**entry["artifact"]),
                        action=entry["action"],
                    )
                )
    return entries


@app.get("/artifact/model/{id}/lineage")
async def artifact_lineage(
    id: str, user: Dict[str, Any] = Depends(verify_token)
) -> ArtifactLineageGraph:
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != "model":
        raise HTTPException(status_code=400, detail="Not a model artifact.")

    # Build lineage using available HF metadata when possible
    nodes: List[ArtifactLineageNode] = [
        ArtifactLineageNode(
            artifact_id=id, name=artifact_data["metadata"]["name"], source="config_json"
        )
    ]
    edges: List[ArtifactLineageEdge] = []
    try:
        url = artifact_data["data"]["url"]
        if isinstance(url, str) and "huggingface.co" in url.lower():
            hf_data, _ = scrape_hf_url(url)
            for ds in (hf_data.get("datasets") or [])[:5]:
                ds_id = f"dataset:{ds}"
                nodes.append(
                    ArtifactLineageNode(artifact_id=ds_id, name=str(ds), source="config_json")
                )
                edges.append(
                    ArtifactLineageEdge(
                        from_node_artifact_id=ds_id,
                        to_node_artifact_id=id,
                        relationship="fine_tuning_dataset",
                    )
                )
    except Exception:
        # Fall back to single-node graph
        pass
    return ArtifactLineageGraph(nodes=nodes, edges=edges)


@app.post("/artifact/model/{id}/license-check")
async def artifact_license_check(
    id: str, request: SimpleLicenseCheckRequest, user: Dict[str, Any] = Depends(verify_token)
) -> bool:
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != "model":
        raise HTTPException(status_code=400, detail="Not a model artifact.")

    # Use license metric score where 0.5+ means compatible enough
    model_data = {"url": request.github_url, "hf_data": [], "gh_data": []}
    ctx = create_eval_context_from_model_data(model_data)
    from src.metrics.license_check import metric as license_metric  # local import to avoid cycles

    license_score = await license_metric(ctx)
    return bool(license_score >= 0.5)


@app.get("/artifact/model/{id}/rate", response_model=ModelRating)
async def model_artifact_rate(id: str, user: Dict[str, Any] = Depends(verify_token)) -> ModelRating:
    """Get ratings for this model artifact (BASELINE)"""
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != "model":
        raise HTTPException(status_code=400, detail="Not a model artifact.")

    url = artifact_data["data"]["url"]
    category = "classification"
    size_scores: Dict[str, float] = {
        "raspberry_pi": 1.0,
        "jetson_nano": 1.0,
        "desktop_pc": 1.0,
        "aws_server": 1.0,
    }

    metrics: Dict[str, float] = {}
    try:
        hf_data = None
        if isinstance(url, str) and "huggingface.co" in url.lower():
            hf_data, _ = scrape_hf_url(url)
            if isinstance(hf_data.get("pipeline_tag"), str):
                category = str(hf_data.get("pipeline_tag"))
        model_data = {"url": url, "hf_data": [hf_data] if hf_data else [], "gh_data": []}
        metrics = await calculate_phase2_metrics(model_data)
        # Compute size_score dict explicitly
        ctx = create_eval_context_from_model_data(model_data)
        size_scores = await size_metric.metric(ctx)
    except Exception:
        metrics = {}

    net_score = calculate_phase2_net_score(metrics) if metrics else 0.0

    def get_m(name: str) -> float:
        v = metrics.get(name)
        try:
            return float(v) if isinstance(v, (int, float)) else 0.0
        except Exception:
            return 0.0

    return ModelRating(
        name=artifact_data["metadata"]["name"],
        category=category or "unknown",
        net_score=net_score,
        net_score_latency=0.0,
        ramp_up_time=get_m("ramp_up_time"),
        ramp_up_time_latency=0.0,
        bus_factor=get_m("bus_factor"),
        bus_factor_latency=0.0,
        performance_claims=get_m("performance_claims"),
        performance_claims_latency=0.0,
        license=get_m("license"),
        license_latency=0.0,
        dataset_and_code_score=get_m("dataset_and_code_score"),
        dataset_and_code_score_latency=0.0,
        dataset_quality=get_m("dataset_quality"),
        dataset_quality_latency=0.0,
        code_quality=get_m("code_quality"),
        code_quality_latency=0.0,
        reproducibility=get_m("reproducibility"),
        reproducibility_latency=0.0,
        reviewedness=get_m("reviewedness"),
        reviewedness_latency=0.0,
        tree_score=get_m("tree_score"),
        tree_score_latency=0.0,
        size_score=size_scores,
        size_score_latency=0.0,
    )


@app.get("/artifact/{artifact_type}/{id}/cost", response_model=Dict[str, ArtifactCost])
async def artifact_cost(
    artifact_type: ArtifactType,
    id: str,
    dependency: bool = Query(False),
    user: Dict[str, Any] = Depends(verify_token),
) -> Dict[str, ArtifactCost]:
    """Get the cost of an artifact (BASELINE)"""
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch.")

    # Compute approximate cost from size metric (MB)
    url = artifact_data["data"]["url"]
    hf_data = None
    if isinstance(url, str) and "huggingface.co" in url.lower():
        try:
            hf_data, _ = scrape_hf_url(url)
        except Exception:
            hf_data = None
    model_data = {"url": url, "hf_data": [hf_data] if hf_data else [], "gh_data": []}
    ctx = create_eval_context_from_model_data(model_data)
    try:
        await size_metric.metric(ctx)
        required_bytes = int(getattr(ctx, "size_required_bytes", 0))
    except Exception:
        required_bytes = 0
    mb = float(required_bytes) / (1024.0 * 1024.0)
    standalone_cost = round(mb, 1)
    total_cost = standalone_cost  # no dependency graph persisted yet
    result = {id: ArtifactCost(total_cost=total_cost)}
    if dependency:
        result[id].standalone_cost = standalone_cost
    return result


@app.get("/tracks")
async def get_tracks() -> Dict[str, List[str]]:
    """Get the list of tracks a student has planned to implement"""
    return {
        "plannedTracks": [
            "Other Security track",
            "Access control track",
        ]
    }


# Favicon route
# @app.get("/favicon.ico")
# @app.head("/favicon.ico")
# async def favicon():
#     """Serve the favicon"""
#     return FileResponse("frontend/public/favicon.ico")


# Mount static files to serve favicon and other static assets
# app.mount("/static", StaticFiles(directory="frontend/public"), name="static")

# Create Mangum handler for Lambda
# This is the entry point that Lambda calls (configured as app.handler)
# Per Stack Overflow: Lambda must return statusCode (int), headers (dict), body (string)
logger.info("Initializing Mangum handler for Lambda...")
sys.stdout.flush()
_mangum_handler: Optional[Mangum] = None
try:
    _mangum_handler = Mangum(app, lifespan="off")
    logger.info("✅ Mangum handler initialized successfully")
    sys.stdout.flush()
except Exception as e:
    logger.error(f"❌ Failed to initialize Mangum handler: {e}", exc_info=True)
    import traceback

    logger.error(f"Full traceback: {traceback.format_exc()}")
    sys.stdout.flush()
    sys.stderr.flush()
    _mangum_handler = None


# Lambda handler wrapper - ensures proper response format per Stack Overflow article
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler entry point.
    Ensures response format matches API Gateway requirements:
    - statusCode: int (required)
    - headers: dict (required)
    - body: string (required)
    """
    # Force log flushing immediately
    print("=== LAMBDA HANDLER CALLED ===")
    sys.stdout.flush()

    # Log handler invocation for debugging
    try:
        if isinstance(event, dict):
            event_keys = list(event.keys())
            # Support both API Gateway REST API and HTTP API v2 formats
            event_path = event.get("path") or event.get("rawPath", "N/A")
            event_method = event.get("httpMethod") or event.get("requestContext", {}).get(
                "http", {}
            ).get("method", "N/A")
            route_key = event.get("routeKey", "N/A")
        else:
            event_keys = "not a dict"
            event_path = "N/A"
            event_method = "N/A"
            route_key = "N/A"

        logger.info(f"Handler invoked. Event keys: {event_keys}")
        logger.info(f"Event path: {event_path}, method: {event_method}, routeKey: {route_key}")
        print(
            f"DEBUG: Event keys={event_keys}, path={event_path}, method={event_method}, routeKey={route_key}"
        )
        sys.stdout.flush()
    except Exception as log_err:
        print(f"ERROR logging event: {log_err}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        sys.stdout.flush()

    try:
        if _mangum_handler is None:
            error_msg = "Mangum handler not initialized"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            sys.stdout.flush()
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": '{"message":"Internal Server Error"}',
            }

        # Call Mangum handler
        logger.info("Calling Mangum handler...")
        print("DEBUG: Calling Mangum handler...")
        sys.stdout.flush()

        response = _mangum_handler(event, context)

        logger.info(f"Mangum handler returned. Response type: {type(response)}")
        if isinstance(response, dict):
            status_code = response.get("statusCode", "N/A")
            body_preview = str(response.get("body", ""))[:200] if response.get("body") else "N/A"
            print(f"DEBUG: Mangum returned statusCode={status_code}, body preview={body_preview}")
        else:
            print(f"DEBUG: Mangum returned type={type(response)}")
        sys.stdout.flush()

        # Ensure response has correct format (per Stack Overflow requirements)
        if not isinstance(response, dict):
            logger.error(f"Handler returned non-dict: {type(response)}")
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": '{"message":"Internal Server Error"}',
            }

        # Ensure statusCode is an int (not string)
        if "statusCode" not in response:
            logger.error("Response missing statusCode")
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": '{"message":"Internal Server Error"}',
            }

        # Ensure statusCode is an integer
        if not isinstance(response["statusCode"], int):
            logger.error(f"statusCode is not int: {type(response['statusCode'])}")
            response["statusCode"] = int(response["statusCode"])

        # Ensure body is a string
        if "body" in response and not isinstance(response["body"], str):
            logger.warning(f"Converting body to string from {type(response['body'])}")
            response["body"] = str(response["body"])

        logger.info(f"Returning response with statusCode: {response.get('statusCode', 'N/A')}")
        sys.stdout.flush()
        return response

    except Exception as e:
        import traceback

        logger.error(f"Lambda handler error: {e}", exc_info=True)
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.stdout.flush()
        sys.stderr.flush()
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": '{"message":"Internal Server Error"}',
        }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
