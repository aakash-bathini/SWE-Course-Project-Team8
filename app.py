"""
Phase 2 FastAPI Application - Trustworthy Model Registry
Main application entry point with REST API endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trustworthy Model Registry",
    description="ACME Corporation's secure model registry with authentication and monitoring",
    version="2.0.0",
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
models_db: Dict[str, Dict[str, Any]] = {}
users_db: Dict[str, Dict[str, Any]] = {}

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


# Pydantic models
class ModelUpload(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None


class ModelResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    tags: Optional[List[str]]
    upload_date: str
    file_size: int
    metrics: Optional[Dict[str, Any]] = None


class UserRegistration(BaseModel):
    username: str
    password: str
    permissions: List[str] = ["search"]


class UserLogin(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


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


# API Endpoints


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint with API information"""
    return {
        "message": "Trustworthy Model Registry API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """System health endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime="0:00:00",  # Simplified for Milestone 1
        models_count=len(models_db),
        users_count=len(users_db),
        last_hour_activity={"uploads": 0, "downloads": 0, "searches": 0},
    )


@app.post("/models/upload", response_model=ModelResponse)
async def upload_model(
    file: UploadFile = File(...),
    model_data: str = Query(..., description="JSON string with model metadata"),
    current_user: Dict[str, Any] = Depends(verify_token),
) -> ModelResponse:
    """Upload a model ZIP file"""
    if not check_permission(current_user, "upload"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Parse model metadata
        metadata = json.loads(model_data)
        model_id = f"model_{len(models_db) + 1}"

        # Store model info (simplified for Milestone 1)
        model_info = {
            "id": model_id,
            "name": metadata.get("name", file.filename),
            "description": metadata.get("description", ""),
            "tags": metadata.get("tags", []),
            "upload_date": datetime.now().isoformat(),
            "file_size": file.size if file.size else 0,
            "filename": file.filename,
            "uploader": current_user["username"],
        }

        models_db[model_id] = model_info

        logger.info(f"Model uploaded: {model_id} by {current_user['username']}")

        return ModelResponse(**model_info)

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


@app.get("/models/{model_id}/rate", response_model=Dict[str, Any])
async def rate_model(
    model_id: str, current_user: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, Any]:
    """Rate a model using Phase 2 metrics"""
    if not check_permission(current_user, "search"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Use Phase 2 metrics
        model_info = models_db[model_id]

        # Calculate metrics using Phase 2 orchestration with original Phase 1 metrics
        metrics = await orchestrate(model_info)

        return {"model_id": model_id, "metrics": metrics, "timestamp": datetime.now().isoformat()}

    except Exception as e:
        logger.error(f"Rating error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Rating failed: {str(e)}")


@app.get("/models/{model_id}/download")
async def download_model(
    model_id: str,
    aspect: Optional[str] = Query(
        None, description="Download aspect: full, weights, datasets, code"
    ),
    current_user: Dict[str, Any] = Depends(verify_token),
) -> Dict[str, Any]:
    """Download a model or its components"""
    if not check_permission(current_user, "download"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    # Simplified download for Milestone 1
    model_info = models_db[model_id]

    logger.info(f"Download requested: {model_id} by {current_user['username']}")

    return {
        "message": f"Download initiated for {model_id}",
        "aspect": aspect or "full",
        "model_info": model_info,
    }


@app.delete("/models/{model_id}")
async def delete_model(
    model_id: str, current_user: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, str]:
    """Delete a model"""
    if not check_permission(current_user, "upload"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    if model_id not in models_db:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if user owns the model or is admin
    model_info = models_db[model_id]
    if model_info["uploader"] != current_user["username"] and not check_permission(
        current_user, "admin"
    ):
        raise HTTPException(status_code=403, detail="Can only delete your own models")

    del models_db[model_id]

    logger.info(f"Model deleted: {model_id} by {current_user['username']}")

    return {"message": f"Model {model_id} deleted successfully"}


@app.get("/models", response_model=List[ModelResponse])
async def list_models(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    current_user: Dict[str, Any] = Depends(verify_token),
) -> List[ModelResponse]:
    """List all models with pagination"""
    if not check_permission(current_user, "search"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Simplified pagination for Milestone 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    models = list(models_db.values())[start_idx:end_idx]

    return [ModelResponse(**model) for model in models]


@app.post("/models/ingest")
async def ingest_huggingface_model(
    model_name: str = Query(..., description="HuggingFace model name"),
    current_user: Dict[str, Any] = Depends(verify_token),
) -> Dict[str, Any]:
    """Ingest a model from HuggingFace"""
    if not check_permission(current_user, "upload"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        # Use Phase 1 HuggingFace API (simplified for Milestone 1)
        logger.info(f"Ingesting HuggingFace model: {model_name}")

        # Mock ingestion for demo
        model_id = f"hf_{len(models_db) + 1}"
        model_info = {
            "id": model_id,
            "name": model_name,
            "description": f"Imported from HuggingFace: {model_name}",
            "tags": ["huggingface", "imported"],
            "upload_date": datetime.now().isoformat(),
            "file_size": 0,  # Will be determined during actual download
            "source": "huggingface",
            "uploader": current_user["username"],
        }

        models_db[model_id] = model_info

        return {"message": f"Model {model_name} ingested successfully", "model_id": model_id}

    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {str(e)}")


@app.post("/register", response_model=Dict[str, str])
async def register_user(
    user_data: UserRegistration, current_user: Dict[str, Any] = Depends(verify_token)
) -> Dict[str, str]:
    """Register a new user (admin only)"""
    if not check_permission(current_user, "admin"):
        raise HTTPException(status_code=403, detail="Only administrators can register users")

    if user_data.username in users_db:
        raise HTTPException(status_code=409, detail="Username already exists")

    # Simplified user creation for Milestone 1
    new_user = {
        "username": user_data.username,
        "password": user_data.password,  # Will be hashed in Milestone 3
        "permissions": user_data.permissions,
        "created_at": datetime.now().isoformat(),
        "created_by": current_user["username"],
    }

    users_db[user_data.username] = new_user

    logger.info(f"User registered: {user_data.username} by {current_user['username']}")

    return {"message": f"User {user_data.username} registered successfully"}


@app.post("/authenticate", response_model=TokenResponse)
async def authenticate_user(user_data: UserLogin) -> TokenResponse:
    """Authenticate user and return token"""
    if user_data.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = users_db[user_data.username]
    if user["password"] != user_data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Simplified token for Milestone 1 (will use JWT in Milestone 3)
    token = "demo_token"

    logger.info(f"User authenticated: {user_data.username}")

    return TokenResponse(access_token=token, token_type="bearer", expires_in=36000)  # 10 hours


# AWS Lambda handler for deployment
def handler(event: Dict[str, Any], context: Any) -> Any:
    """AWS Lambda handler for FastAPI application"""
    try:
        from mangum import Mangum

        asgi_handler = Mangum(app)
        return asgi_handler(event, context)
    except ImportError:
        # Fallback for local development without mangum
        return {
            "statusCode": 500,
            "body": "Mangum not installed - use uvicorn for local development",
            "headers": {"Content-Type": "application/json"},
        }
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Internal server error"}),
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
