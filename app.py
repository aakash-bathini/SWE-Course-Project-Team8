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

# In-memory storage for demo (will be replaced with SQLite in Milestone 2)
artifacts_db: Dict[str, Dict[str, Any]] = {}  # artifact_id -> artifact_data
users_db: Dict[str, Dict[str, Any]] = {}
audit_log: List[Dict[str, Any]] = []

# Default admin user
DEFAULT_ADMIN = {
    "username": "ece30861defaultadminuser",
    "password": "correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages",
    "permissions": ["upload", "search", "download", "admin"],
    "created_at": datetime.now().isoformat(),
}

# Add default admin to users database
admin_username: str = str(DEFAULT_ADMIN["username"])
users_db[admin_username] = DEFAULT_ADMIN


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
        token = x_authorization.strip()
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

    # For Milestone 1, accept any non-empty token and treat as default admin
    return DEFAULT_ADMIN


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
        timeline=[
            HealthTimelineEntry(bucket=now_iso, value=0.0, unit="rpm")
        ]
        if includeTimeline
        else None,
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
        timeline=[
            HealthTimelineEntry(bucket=now_iso, value=0.0, unit="rpm")
        ]
        if includeTimeline
        else None,
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
    """Create an access token (NON-BASELINE)"""
    # Check if user exists and password is correct
    user_data = users_db.get(request.user.name)
    if not user_data or not verify_password(request.secret.password, user_data.get("password", "")):
        raise HTTPException(status_code=401, detail="The user or password is invalid.")

    # For now, return a simple token (will implement proper JWT in Phase 2)
    token = f"bearer demo_token_{request.user.name}"

    # Spec: AuthenticationToken is a string
    return token


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

    for query in queries:
        if query.name == "*":
            # Return all artifacts
            for artifact_id, artifact_data in artifacts_db.items():
                results.append(
                    ArtifactMetadata(
                        name=artifact_data["metadata"]["name"],
                        id=artifact_id,
                        type=artifact_data["metadata"]["type"],
                    )
                )
        else:
            # Filter by name and types
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
            hf_data, repo_type = scrape_hf_url(artifact_data.url)
            # Minimal heuristic for Delivery 1:
            # Accept if license is present OR size > 0; reject only if both are missing/zero
            has_license: bool = bool(hf_data.get("license"))
            size_ok: bool = bool(hf_data.get("size", 0) and hf_data.get("size", 0) > 0)
            if not has_license and not size_ok:
                raise HTTPException(
                    status_code=424,
                    detail="Artifact is not registered due to the disqualified rating.",
                )
        except HTTPException:
            raise
        except Exception:
            # Do not block Delivery 1 if HF fetch fails; proceed without gating
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

    # Log audit entry
    audit_log.append(
        {
            "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
            "date": datetime.now().isoformat(),
            "artifact": artifact_entry["metadata"],
            "action": "CREATE",
        }
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
        "metadata": artifact.metadata.dict(),
        "data": artifact.data.dict(),
        "updated_at": datetime.now().isoformat(),
        "updated_by": user["username"],
    }

    # Log audit entry
    audit_log.append(
        {
            "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
            "date": datetime.now().isoformat(),
            "artifact": artifact.metadata.dict(),
            "action": "UPDATE",
        }
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
    return {"message": "Artifact is deleted."}


# -------------------------
# Additional endpoints per spec
# -------------------------


@app.get("/artifact/byName/{name}")
async def artifact_by_name(
    name: str, user: Dict[str, Any] = Depends(verify_token)
) -> List[ArtifactMetadata]:
    matches: List[ArtifactMetadata] = []
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

    pattern = _re.compile(regex.regex)
    matches: List[ArtifactMetadata] = []
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

    # Minimal mock lineage graph per spec example
    nodes = [
        ArtifactLineageNode(
            artifact_id=id, name=artifact_data["metadata"]["name"], source="config_json"
        ),
    ]
    edges: List[ArtifactLineageEdge] = []
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
    # Minimal successful mock
    return True


@app.get("/artifact/model/{id}/rate", response_model=ModelRating)
async def model_artifact_rate(id: str, user: Dict[str, Any] = Depends(verify_token)) -> ModelRating:
    """Get ratings for this model artifact (BASELINE)"""
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")

    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != "model":
        raise HTTPException(status_code=400, detail="Not a model artifact.")

    # Generate mock rating data (will be replaced with actual metrics in Phase 2)
    return ModelRating(
        name=artifact_data["metadata"]["name"],
        category="classification",
        net_score=0.75,
        net_score_latency=1.2,
        ramp_up_time=0.8,
        ramp_up_time_latency=0.5,
        bus_factor=0.6,
        bus_factor_latency=2.1,
        performance_claims=0.9,
        performance_claims_latency=1.8,
        license=0.7,
        license_latency=0.3,
        dataset_and_code_score=0.8,
        dataset_and_code_score_latency=1.5,
        dataset_quality=0.75,
        dataset_quality_latency=1.0,
        code_quality=0.85,
        code_quality_latency=2.0,
        reproducibility=0.7,
        reproducibility_latency=1.5,
        reviewedness=0.6,
        reviewedness_latency=0.8,
        tree_score=0.8,
        tree_score_latency=1.2,
        size_score={"raspberry_pi": 0.3, "jetson_nano": 0.5, "desktop_pc": 0.9, "aws_server": 0.95},
        size_score_latency=0.5,
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

    # Mock cost calculation (will be replaced with actual cost calculation)
    standalone_cost = 100.0
    total_cost = standalone_cost + (50.0 if dependency else 0.0)

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
