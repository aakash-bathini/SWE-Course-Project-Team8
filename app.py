"""
Phase 2 FastAPI Application - Trustworthy Model Registry
Main application entry point with REST API endpoints matching OpenAPI spec v3.3.1
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from datetime import datetime
import json
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ECE 461 - Fall 2025 - Project Phase 2",
    description="API for ECE 461/Fall 2025/Project Phase 2: A Trustworthy Model Registry",
    version="3.3.1",
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

# Security
security = HTTPBearer()

# Import Phase 2 components
from src.orchestration.metric_orchestrator import orchestrate

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


# Authentication functions (simplified for Milestone 1)
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Verify JWT token and return user info"""
    # Simplified token verification for Milestone 1
    # In Milestone 3, this will use proper JWT validation
    token = credentials.credentials
    if token == "demo_token":
        return DEFAULT_ADMIN
    raise HTTPException(status_code=401, detail="Invalid token")


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

# Authentication endpoint
@app.put("/authenticate", response_model=AuthenticationToken)
async def create_auth_token(request: AuthenticationRequest) -> AuthenticationToken:
    """Create an access token (NON-BASELINE)"""
    # Check if user exists and password is correct
    user_data = users_db.get(request.user.name)
    if not user_data or not verify_password(request.secret.password, user_data.get("password", "")):
        raise HTTPException(status_code=401, detail="The user or password is invalid.")
    
    # Create JWT token
    token_data = {
        "sub": request.user.name,
        "is_admin": request.user.is_admin,
        "iat": datetime.utcnow()
    }
    
    # For now, return a simple token (will implement proper JWT in Phase 2)
    token = f"bearer demo_token_{request.user.name}"
    
    return AuthenticationToken(token=token)

# Registry reset endpoint
@app.delete("/reset")
async def registry_reset(user: Dict[str, Any] = Depends(verify_token)):
    """Reset the registry to a system default state (BASELINE)"""
    if not check_permission(user, "admin"):
        raise HTTPException(status_code=401, detail="You do not have permission to reset the registry.")
    
    artifacts_db.clear()
    audit_log.clear()
    
    return {"message": "Registry is reset."}

# Artifact endpoints matching OpenAPI spec

@app.post("/artifacts")
async def artifacts_list(
    queries: List[ArtifactQuery],
    offset: Optional[str] = Query(None),
    user: Dict[str, Any] = Depends(verify_token)
) -> List[ArtifactMetadata]:
    """Get the artifacts from the registry (BASELINE)"""
    results = []
    
    for query in queries:
        if query.name == "*":
            # Return all artifacts
            for artifact_id, artifact_data in artifacts_db.items():
                results.append(ArtifactMetadata(
                    name=artifact_data["metadata"]["name"],
                    id=artifact_id,
                    type=artifact_data["metadata"]["type"]
                ))
        else:
            # Filter by name and types
            for artifact_id, artifact_data in artifacts_db.items():
                if artifact_data["metadata"]["name"] == query.name:
                    if not query.types or artifact_data["metadata"]["type"] in query.types:
                        results.append(ArtifactMetadata(
                            name=artifact_data["metadata"]["name"],
                            id=artifact_id,
                            type=artifact_data["metadata"]["type"]
                        ))
    
    return results

@app.post("/artifact/{artifact_type}", response_model=Artifact)
async def artifact_create(
    artifact_type: ArtifactType,
    artifact_data: ArtifactData,
    user: Dict[str, Any] = Depends(verify_token)
) -> Artifact:
    """Register a new artifact (BASELINE)"""
    # Generate unique artifact ID
    artifact_id = f"{artifact_type}_{len(artifacts_db) + 1}_{int(datetime.now().timestamp())}"
    
    # Create artifact entry
    artifact_entry = {
        "metadata": {
            "name": artifact_data.url.split("/")[-1],  # Extract name from URL
            "id": artifact_id,
            "type": artifact_type.value
        },
        "data": {
            "url": artifact_data.url
        },
        "created_at": datetime.now().isoformat(),
        "created_by": user["username"]
    }
    
    artifacts_db[artifact_id] = artifact_entry
    
    # Log audit entry
    audit_log.append({
        "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
        "date": datetime.now().isoformat(),
        "artifact": artifact_entry["metadata"],
        "action": "CREATE"
    })
    
    return Artifact(
        metadata=ArtifactMetadata(**artifact_entry["metadata"]),
        data=ArtifactData(url=artifact_entry["data"]["url"])
    )

@app.get("/artifacts/{artifact_type}/{id}", response_model=Artifact)
async def artifact_retrieve(
    artifact_type: ArtifactType,
    id: str,
    user: Dict[str, Any] = Depends(verify_token)
) -> Artifact:
    """Interact with the artifact with this id (BASELINE)"""
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch.")
    
    return Artifact(
        metadata=ArtifactMetadata(**artifact_data["metadata"]),
        data=ArtifactData(url=artifact_data["data"]["url"])
    )

@app.put("/artifacts/{artifact_type}/{id}")
async def artifact_update(
    artifact_type: ArtifactType,
    id: str,
    artifact: Artifact,
    user: Dict[str, Any] = Depends(verify_token)
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
        "updated_by": user["username"]
    }
    
    # Log audit entry
    audit_log.append({
        "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
        "date": datetime.now().isoformat(),
        "artifact": artifact.metadata.dict(),
        "action": "UPDATE"
    })
    
    return {"message": "Artifact is updated."}

@app.delete("/artifacts/{artifact_type}/{id}")
async def artifact_delete(
    artifact_type: ArtifactType,
    id: str,
    user: Dict[str, Any] = Depends(verify_token)
):
    """Delete this artifact (NON-BASELINE)"""
    if id not in artifacts_db:
        raise HTTPException(status_code=404, detail="Artifact does not exist.")
    
    artifact_data = artifacts_db[id]
    if artifact_data["metadata"]["type"] != artifact_type.value:
        raise HTTPException(status_code=400, detail="Artifact type mismatch.")
    
    # Log audit entry before deletion
    audit_log.append({
        "user": {"name": user["username"], "is_admin": user.get("is_admin", False)},
        "date": datetime.now().isoformat(),
        "artifact": artifact_data["metadata"],
        "action": "DELETE"
    })
    
    del artifacts_db[id]
    return {"message": "Artifact is deleted."}

@app.get("/artifact/model/{id}/rate", response_model=ModelRating)
async def model_artifact_rate(
    id: str,
    user: Dict[str, Any] = Depends(verify_token)
) -> ModelRating:
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
        size_score={
            "raspberry_pi": 0.3,
            "jetson_nano": 0.5,
            "desktop_pc": 0.9,
            "aws_server": 0.95
        },
        size_score_latency=0.5
    )

@app.get("/artifact/{artifact_type}/{id}/cost", response_model=Dict[str, ArtifactCost])
async def artifact_cost(
    artifact_type: ArtifactType,
    id: str,
    dependency: bool = Query(False),
    user: Dict[str, Any] = Depends(verify_token)
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
            "Other Security track"
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
