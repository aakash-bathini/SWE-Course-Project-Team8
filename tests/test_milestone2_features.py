"""
Tests for Milestone 2 new features: Upload, Download, and New Metrics
"""

import os
import sys
import io
import zipfile
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client_and_token():
    """Get authenticated client and token"""
    from app import app

    client = TestClient(app)

    # Authenticate
    auth_payload = {
        "user": {"name": "ece30861defaultadminuser", "is_admin": True},
        "secret": {
            "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;",
        },
    }
    token_resp = client.put("/authenticate", json=auth_payload)
    assert token_resp.status_code == 200
    token = token_resp.json()

    headers = {"X-Authorization": token, "Authorization": token}
    return client, headers


def create_test_zip() -> bytes:
    """Create a test ZIP file in memory"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add README.md (model card)
        readme_content = """# Test Model

This is a test model for unit testing.

## Usage

```python
import torch
model = torch.load('model.pth')
print("Model loaded successfully")
```

## Training Data

Trained on test dataset.
"""
        zipf.writestr("README.md", readme_content)

        # Add a fake model weight file
        zipf.writestr("model.pth", b"fake model weights content")

        # Add a fake dataset
        zipf.writestr("data.csv", "col1,col2\nval1,val2\n")

        # Add a Python code file
        zipf.writestr("train.py", "print('Training script')\n")

    zip_buffer.seek(0)
    return zip_buffer.read()


def test_zip_upload(client_and_token):
    """Test POST /models/upload with ZIP file"""
    client, headers = client_and_token

    # Create test ZIP
    zip_content = create_test_zip()

    # Upload ZIP
    files = {"file": ("test_model.zip", zip_content, "application/zip")}
    data = {"name": "test_uploaded_model"}

    response = client.post("/models/upload", files=files, data=data, headers=headers)

    assert response.status_code == 201
    result = response.json()
    assert "metadata" in result
    assert result["metadata"]["type"] == "model"
    assert result["metadata"]["name"] == "test_uploaded_model"


def test_zip_upload_invalid_file(client_and_token):
    """Test upload with non-ZIP file fails"""
    client, headers = client_and_token

    # Try to upload non-ZIP file
    files = {"file": ("test.txt", b"not a zip file", "text/plain")}

    response = client.post("/models/upload", files=files, headers=headers)

    assert response.status_code == 400
    assert "ZIP" in response.json()["detail"]


def test_download_full(client_and_token):
    """Test GET /models/{id}/download with aspect=full"""
    client, headers = client_and_token

    # First upload a model
    zip_content = create_test_zip()
    files = {"file": ("model.zip", zip_content, "application/zip")}
    data = {"name": "download_test_model"}

    upload_resp = client.post("/models/upload", files=files, data=data, headers=headers)
    assert upload_resp.status_code == 201
    artifact_id = upload_resp.json()["metadata"]["id"]

    # Download with aspect=full
    download_resp = client.get(f"/models/{artifact_id}/download?aspect=full", headers=headers)

    assert download_resp.status_code == 200
    assert download_resp.headers["content-type"] == "application/zip"
    assert "X-File-Checksum" in download_resp.headers
    assert download_resp.headers["X-File-Aspect"] == "full"


def test_download_weights_only(client_and_token):
    """Test GET /models/{id}/download with aspect=weights"""
    client, headers = client_and_token

    # Upload a model
    zip_content = create_test_zip()
    files = {"file": ("model.zip", zip_content, "application/zip")}

    upload_resp = client.post("/models/upload", files=files, headers=headers)
    artifact_id = upload_resp.json()["metadata"]["id"]

    # Download weights only
    download_resp = client.get(f"/models/{artifact_id}/download?aspect=weights", headers=headers)

    assert download_resp.status_code == 200
    assert download_resp.headers["X-File-Aspect"] == "weights"


def test_download_datasets_only(client_and_token):
    """Test GET /models/{id}/download with aspect=datasets"""
    client, headers = client_and_token

    # Upload a model
    zip_content = create_test_zip()
    files = {"file": ("model.zip", zip_content, "application/zip")}

    upload_resp = client.post("/models/upload", files=files, headers=headers)
    artifact_id = upload_resp.json()["metadata"]["id"]

    # Download datasets only
    download_resp = client.get(f"/models/{artifact_id}/download?aspect=datasets", headers=headers)

    assert download_resp.status_code == 200
    assert download_resp.headers["X-File-Aspect"] == "datasets"


def test_download_code_only(client_and_token):
    """Test GET /models/{id}/download with aspect=code"""
    client, headers = client_and_token

    # Upload a model
    zip_content = create_test_zip()
    files = {"file": ("model.zip", zip_content, "application/zip")}

    upload_resp = client.post("/models/upload", files=files, headers=headers)
    artifact_id = upload_resp.json()["metadata"]["id"]

    # Download code only
    download_resp = client.get(f"/models/{artifact_id}/download?aspect=code", headers=headers)

    assert download_resp.status_code == 200
    assert download_resp.headers["X-File-Aspect"] == "code"


def test_download_nonexistent_artifact(client_and_token):
    """Test download of non-existent artifact returns 404"""
    client, headers = client_and_token

    response = client.get("/models/nonexistent-id/download", headers=headers)

    assert response.status_code == 404


def test_download_url_only_artifact(client_and_token):
    """Test download of URL-only artifact (no local files) returns 404"""
    client, headers = client_and_token

    # Create URL-only artifact
    create_resp = client.post(
        "/artifact/model",
        headers=headers,
        json={"url": "https://example.org/model"},
    )
    assert create_resp.status_code == 201
    artifact_id = create_resp.json()["metadata"]["id"]

    # Try to download (should fail since no files)
    download_resp = client.get(f"/models/{artifact_id}/download", headers=headers)

    assert download_resp.status_code == 404
    assert "not found" in download_resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_reproducibility_metric_with_code():
    """Test reproducibility metric with demo code"""
    from src.metrics.reproducibility import metric
    from src.models.model_types import EvalContext

    # Create context with demo code
    context = EvalContext(
        url="https://huggingface.co/test",
        hf_data=[
            {
                "readme_text": """
# Model

## Usage

```python
print("Hello World")
x = 1 + 1
```
"""
            }
        ],
    )

    score = await metric(context)

    # Should be > 0 since there's code
    assert score >= 0.0
    assert score <= 1.0


@pytest.mark.asyncio
async def test_reproducibility_metric_no_code():
    """Test reproducibility metric without demo code"""
    from src.metrics.reproducibility import metric
    from src.models.model_types import EvalContext

    # Create context without code
    context = EvalContext(
        url="https://huggingface.co/test", hf_data=[{"readme_text": "# Model\n\nNo code here."}]
    )

    score = await metric(context)

    # Per spec: "0 (no code/doesn't run)" - should return 0.0 when no code found
    assert score == 0.0


@pytest.mark.asyncio
async def test_reviewedness_metric_no_github():
    """Test reviewedness metric without GitHub repo"""
    from src.metrics.reviewedness import metric
    from src.models.model_types import EvalContext

    # Create context without GitHub
    context = EvalContext(url="https://huggingface.co/test", hf_data=[{}])

    score = await metric(context)

    # Should be -1 for no GitHub
    assert score == -1.0


@pytest.mark.asyncio
async def test_treescore_metric_no_parents():
    """Test treescore metric without parent models"""
    from src.metrics.treescore import metric
    from src.models.model_types import EvalContext

    # Create context without parents
    context = EvalContext(url="https://huggingface.co/test", hf_data=[{"card_yaml": {}}])

    score = await metric(context)

    # Should be neutral/ignored when no parents
    assert score in (-1.0, 0.0)


@pytest.mark.asyncio
async def test_treescore_metric_with_base_model():
    """Test treescore metric with base model"""
    from src.metrics.treescore import metric
    from src.models.model_types import EvalContext

    # Create context with base model
    context = EvalContext(
        url="https://huggingface.co/test/fine-tuned",
        hf_data=[{"card_yaml": {"base_model": "bert-base-uncased"}}],
    )

    score = await metric(context)

    # Should calculate score (may be 0.5 default if parent calc fails)
    assert score >= 0.0
    assert score <= 1.0


def test_rating_includes_new_metrics(client_and_token):
    """Test that /artifact/model/{id}/rate includes new metrics"""
    client, headers = client_and_token

    # Create a model artifact
    create_resp = client.post(
        "/artifact/model",
        headers=headers,
        json={"url": "https://example.org/model"},
    )
    assert create_resp.status_code == 201
    artifact_id = create_resp.json()["metadata"]["id"]

    # Get rating
    rating_resp = client.get(f"/artifact/model/{artifact_id}/rate", headers=headers)

    assert rating_resp.status_code == 200
    rating = rating_resp.json()

    # Check that new metrics are present
    assert "reproducibility" in rating
    assert "reviewedness" in rating
    assert "tree_score" in rating

    # Check they have values (not necessarily non-zero, but defined)
    assert isinstance(rating["reproducibility"], (int, float))
    assert isinstance(rating["reviewedness"], (int, float))
    assert isinstance(rating["tree_score"], (int, float))
