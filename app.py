"""
Phase 2 FastAPI Application - Trustworthy Model Registry
Main application entry point with REST API endpoints matching OpenAPI spec v3.4.2
"""

from fastapi import FastAPI, HTTPException, Depends, Query, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.auth.jwt_auth import auth as jwt_auth
import os
import uvicorn
import logging
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ECE 461 - Fall 2025 - Project Phase 2",
    description="API for ECE 461/Fall 2025/Project Phase 2: A Trustworthy Model Registry",
    version="3.4.2",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Development
        "http://localhost:8000",  # Local development
        "https://*.execute-api.us-east-1.amazonaws.com",  # AWS API Gateway
        "http://*.s3-website-us-east-1.amazonaws.com",  # S3 static hosting
        "*",  # Allow all for development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Security (we will accept X-Authorization header per spec; keep HTTPBearer for flexibility)
security = HTTPBearer()

# Import Phase 2 components
from src.api.huggingface import scrape_hf_url
from src.metrics.phase2_adapter import (
    create_eval_context_from_model_data,
    calculate_phase2_metrics,
    calculate_phase2_net_score,
)
from src.metrics import size as size_metric

# Storage layer selection: in-memory (default) or SQLite (Milestone 2)
USE_SQLITE: bool = os.environ.get("USE_SQLITE", "0") == "1"

artifacts_db: Dict[str, Dict[str, Any]] = {}  # artifact_id -> artifact_data (in-memory)
users_db: Dict[str, Dict[str, Any]] = {}
audit_log: List[Dict[str, Any]] = []

if USE_SQLITE:
    from src.db.database import Base, engine, get_db
    from src.db import crud as db_crud

    # Ensure schema exists
    Base.metadata.create_all(bind=engine)

# Default admin user
DEFAULT_ADMIN = {
    "username": "ece30861defaultadminuser",
    "password": "correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages",
    "permissions": ["upload", "search", "download", "admin"],
    "created_at": datetime.now().isoformat(),
}

# Add default admin to users database and DB (if enabled)
admin_username: str = str(DEFAULT_ADMIN["username"])
users_db[admin_username] = DEFAULT_ADMIN
if USE_SQLITE:
    with next(get_db()) as _db:  # type: ignore[misc]
        db_crud.ensure_schema(_db)
        db_crud.upsert_default_admin(
            _db,
            username=str(DEFAULT_ADMIN["username"]),
            password=str(DEFAULT_ADMIN["password"]),
            permissions=list(DEFAULT_ADMIN["permissions"]),  # type: ignore[arg-type]
        )


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


# Authentication functions (simplified for Milestone 1)
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

    username = str(payload.get("sub", DEFAULT_ADMIN["username"]))
    permissions = payload.get("permissions", DEFAULT_ADMIN["permissions"])  # type: ignore[assignment]
    return {"username": username, "permissions": permissions}


def check_permission(user: Dict[str, Any], required_permission: str) -> bool:
    """Check if user has required permission"""
    return required_permission in user.get("permissions", [])


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    # For demo purposes, simple string comparison
    # In production, use proper password hashing
    return plain_password == hashed_password


# API Endpoints matching OpenAPI spec v3.3.1


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime="0:00:00",  # Simplified for Milestone 1
        models_count=len(artifacts_db),
        users_count=len(users_db),
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


# Authentication endpoint
@app.put("/authenticate", response_model=str)
async def create_auth_token(request: AuthenticationRequest) -> str:
    """Create an access token (Delivery 1)"""
    # Validate user credentials against in-memory/SQLite user store
    user_data = users_db.get(request.user.name)
    if not user_data or not verify_password(request.secret.password, user_data.get("password", "")):
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
    if not check_permission(user, "admin"):
        # Spec: 401 if you do not have permission; 403 is used for invalid/missing token
        raise HTTPException(
            status_code=401, detail="You do not have permission to reset the registry."
        )

    artifacts_db.clear()
    audit_log.clear()
    if USE_SQLITE:
        with next(get_db()) as _db:  # type: ignore[misc]
            db_crud.reset_registry(_db)

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
                hf_data, repo_type = scrape_hf_url(artifact_data.url)
                model_data = {"url": artifact_data.url, "hf_data": [hf_data], "gh_data": []}
                metrics = await calculate_phase2_metrics(model_data)
                # Require all available non-latency metrics to be at least 0.5
                if any((isinstance(v, (int, float)) and float(v) < 0.5) for v in metrics.values()):
                    raise HTTPException(
                        status_code=424,
                        detail="Artifact is not registered due to the disqualified rating.",
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
        pattern = _re.compile(regex.regex)
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
        reproducibility=0.0,
        reproducibility_latency=0.0,
        reviewedness=0.0,
        reviewedness_latency=0.0,
        tree_score=0.0,
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
            "Performance track",
            "Access control track",
            "High assurance track",
            "Other Security track",
        ]
    }


# Favicon route
@app.get("/favicon.ico")
@app.head("/favicon.ico")
async def favicon():
    """Serve the favicon"""
    return FileResponse("frontend/public/favicon.ico")


# Mount static files to serve favicon and other static assets
app.mount("/static", StaticFiles(directory="frontend/public"), name="static")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
