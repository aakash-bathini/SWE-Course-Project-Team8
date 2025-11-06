# Storage Backend Migration Guide

## Current Architecture

The codebase uses a storage abstraction pattern with:
- **SQLite** (`src/db/crud.py`) - Current production backend
- **In-memory** (`artifacts_db` dict) - Fallback/development
- **File Storage** (`src/storage/file_storage.py`) - Local filesystem

## Migration Path to S3

### Option 1: Create S3 Storage Adapter (Recommended)

Create `src/storage/s3_storage.py` that implements the same interface as `db_crud`:

```python
# src/storage/s3_storage.py
import boto3
import json
from typing import Optional, List, Dict
from datetime import datetime

class S3StorageAdapter:
    def __init__(self, bucket_name: str):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name
        
    def create_artifact(self, artifact_id: str, name: str, type_: str, url: str):
        """Create artifact metadata in S3"""
        key = f"artifacts/{artifact_id}/metadata.json"
        metadata = {
            "id": artifact_id,
            "name": name,
            "type": type_,
            "url": url,
            "created_at": datetime.now().isoformat()
        }
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(metadata),
            ContentType="application/json"
        )
        return metadata
    
    def get_artifact(self, artifact_id: str) -> Optional[Dict]:
        """Get artifact metadata from S3"""
        try:
            key = f"artifacts/{artifact_id}/metadata.json"
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response['Body'].read())
        except self.s3.exceptions.NoSuchKey:
            return None
    
    def list_by_queries(self, queries: List[Dict]) -> List[Dict]:
        """List artifacts matching queries"""
        # List all artifacts from S3 prefix
        # Filter by query criteria
        # Return matching artifacts
        pass
    
    # Implement other CRUD methods...
```

### Option 2: Use Environment Variable Pattern

Update `app.py` to use storage adapter:

```python
# In app.py initialization
USE_SQLITE = os.environ.get("USE_SQLITE", "0") == "1"
USE_S3 = os.environ.get("USE_S3", "0") == "1"

if USE_S3:
    from src.storage.s3_storage import S3StorageAdapter
    storage_adapter = S3StorageAdapter(
        bucket_name=os.environ.get("S3_BUCKET_NAME", "trustworthy-model-registry")
    )
elif USE_SQLITE:
    from src.db import crud as storage_adapter
else:
    storage_adapter = None  # Use in-memory
```

### Option 3: Refactor Endpoints to Use Adapter

Update endpoints to use adapter pattern:

```python
# Before:
if USE_SQLITE:
    with next(get_db()) as _db:
        db_crud.create_artifact(_db, ...)
else:
    artifacts_db[artifact_id] = ...

# After:
if storage_adapter:
    storage_adapter.create_artifact(...)
else:
    artifacts_db[artifact_id] = ...
```

## File Storage Migration

For actual file uploads/downloads, update `src/storage/file_storage.py`:

```python
# Add S3 support
USE_S3_STORAGE = os.environ.get("USE_S3_STORAGE", "0") == "1"
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")

def save_uploaded_file(artifact_id: str, file_content: bytes, filename: str):
    if USE_S3_STORAGE:
        # Upload to S3
        s3_key = f"artifacts/{artifact_id}/{filename}"
        s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=file_content)
        return {"path": f"s3://{S3_BUCKET}/{s3_key}", ...}
    else:
        # Current local storage logic
        ...
```

## Benefits of Current Architecture

✅ **Already abstracted**: Code uses `USE_SQLITE` flag pattern  
✅ **Modular**: CRUD operations in separate module  
✅ **Swappable**: Easy to add new storage backend  
✅ **Backward compatible**: Can keep SQLite as fallback  

## Migration Steps

1. **Create S3 adapter** (`src/storage/s3_storage.py`)
2. **Add environment variable** (`USE_S3=1`)
3. **Update initialization** in `app.py`
4. **Test locally** with S3 LocalStack or real S3
5. **Update CI/CD** to set `USE_S3=1` and `S3_BUCKET_NAME`
6. **Keep SQLite as fallback** for local development

## Environment Variables for S3

```bash
USE_S3=1
S3_BUCKET_NAME=trustworthy-model-registry-prod
AWS_REGION=us-east-1
# IAM role will provide credentials automatically in Lambda
```

## Notes

- **Lambda IAM Role**: Add S3 permissions to Lambda execution role
- **DynamoDB Alternative**: For metadata, DynamoDB might be better than S3
- **Cost**: S3 is cheaper for object storage than SQLite in `/tmp`
- **Scalability**: S3 scales better than SQLite
- **Persistence**: S3 persists across container recycles (unlike `/tmp`)

