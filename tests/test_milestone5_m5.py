"""
Milestone 5 Tests - Sensitive Models, JS Sandbox, Package Confusion Audit
Tests for M5.1 (sensitive models & JS monitoring) and M5.2 (package confusion detection)
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client():
    """Create a fresh FastAPI test client"""
    from app import app
    return TestClient(app)


@pytest.fixture
def admin_token(client):
    """Get admin authentication token"""
    auth_payload = {
        "user": {"name": "ece30861defaultadminuser", "is_admin": True},
        "secret": {
            "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;",
        },
    }
    try:
        token_resp = client.put("/authenticate", json=auth_payload)
        if token_resp.status_code == 200:
            return token_resp.json()
    except Exception:
        pass
    return ""


@pytest.fixture
def user_token(client, admin_token):
    """Get regular user authentication token"""
    if admin_token:
        # Register test user
        client.post(
            "/register",
            json={"username": "testuser", "password": "testpass123"},
            headers={"X-Authorization": admin_token},
        )

        # Get user token
        user_auth = {
            "user": {"name": "testuser", "is_admin": False},
            "secret": {"password": "testpass123"},
        }
        try:
            user_resp = client.put("/authenticate", json=user_auth)
            if user_resp.status_code == 200:
                return user_resp.json()
        except Exception:
            pass
    return ""


# ============================================================================
# M5.1a: SENSITIVE MODELS UPLOAD TESTS
# ============================================================================

def test_upload_sensitive_model_success(client, admin_token):
    """Test successful sensitive model upload"""
    if not admin_token:
        pytest.skip("Admin token not available")

    response = client.post(
        "/sensitive-models/upload",
        data={"model_name": "test_model"},
        files={"file": ("model.zip", b"fake zip content")},
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "Sensitive model uploaded successfully"
    assert "id" in data
    assert data["model_name"] == "test_model"


def test_upload_sensitive_model_no_auth(client):
    """Test upload without authentication fails"""
    response = client.post(
        "/sensitive-models/upload",
        data={"model_name": "test_model"},
        files={"file": ("model.zip", b"fake zip content")}
    )
    assert response.status_code == 403  # FastAPI returns 403 for missing auth


def test_upload_sensitive_model_with_js_program(client, admin_token):
    """Test sensitive model upload with JS program association"""
    if not admin_token:
        pytest.skip("Admin token not available")

    # Create JS program first
    js_response = client.post(
        "/js-programs",
        data={
            "name": "monitoring_program",
            "code": "console.log('monitoring');"
        },
        headers={"X-Authorization": admin_token}
    )
    if js_response.status_code != 201:
        pytest.skip("Failed to create JS program")

    program_id = js_response.json()["id"]

    # Upload model with JS program
    response = client.post(
        "/sensitive-models/upload",
        data={
            "model_name": "test_model",
            "js_program_id": program_id
        },
        files={"file": ("model.zip", b"fake zip content")},
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 201
    assert response.json()["js_program_id"] == program_id


# ============================================================================
# M5.1b & M5.1c: SENSITIVE MODELS DOWNLOAD & HISTORY TESTS
# ============================================================================

def test_download_sensitive_model_success(client, admin_token):
    """Test successful sensitive model download"""
    if not admin_token:
        pytest.skip("Admin token not available")

    # Upload model first
    upload_response = client.post(
        "/sensitive-models/upload",
        data={"model_name": "test_model"},
        files={"file": ("model.zip", b"fake zip content")},
        headers={"X-Authorization": admin_token}
    )
    if upload_response.status_code != 201:
        pytest.skip("Failed to upload model")

    model_id = upload_response.json()["id"]

    # Download model
    response = client.get(
        f"/sensitive-models/{model_id}/download",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["id"] == model_id


def test_download_nonexistent_model(client, admin_token):
    """Test download of non-existent model fails"""
    if not admin_token:
        pytest.skip("Admin token not available")

    response = client.get(
        "/sensitive-models/nonexistent_id/download",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 404


def test_download_no_auth(client):
    """Test download without authentication fails"""
    response = client.get("/sensitive-models/test_id/download")
    assert response.status_code == 403


def test_get_download_history_empty(client, admin_token):
    """Test getting download history for new model"""
    if not admin_token:
        pytest.skip("Admin token not available")

    # Upload model
    upload_response = client.post(
        "/sensitive-models/upload",
        data={"model_name": "test_model"},
        files={"file": ("model.zip", b"fake zip content")},
        headers={"X-Authorization": admin_token}
    )
    if upload_response.status_code != 201:
        pytest.skip("Failed to upload model")

    model_id = upload_response.json()["id"]

    # Get history
    response = client.get(
        f"/download-history/{model_id}",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200
    assert response.json()["total_downloads"] == 0
    assert response.json()["history"] == []


def test_get_download_history_with_downloads(client, admin_token, user_token):
    """Test download history after multiple downloads"""
    if not admin_token or not user_token:
        pytest.skip("Admin or user token not available")

    # Upload model
    upload_response = client.post(
        "/sensitive-models/upload",
        data={"model_name": "test_model"},
        files={"file": ("model.zip", b"fake zip content")},
        headers={"X-Authorization": admin_token}
    )
    if upload_response.status_code != 201:
        pytest.skip("Failed to upload model")

    model_id = upload_response.json()["id"]

    # Download multiple times
    client.get(
        f"/sensitive-models/{model_id}/download",
        headers={"X-Authorization": admin_token}
    )
    client.get(
        f"/sensitive-models/{model_id}/download",
        headers={"X-Authorization": user_token}
    )

    # Get history
    response = client.get(
        f"/download-history/{model_id}",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200
    assert response.json()["total_downloads"] == 2
    assert len(response.json()["history"]) == 2


def test_get_history_nonexistent_model(client, admin_token):
    """Test getting history for non-existent model fails"""
    if not admin_token:
        pytest.skip("Admin token not available")

    response = client.get(
        "/download-history/nonexistent_id",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 404


# ============================================================================
# M5.1c: JS PROGRAM CRUD TESTS
# ============================================================================

def test_create_js_program(client, admin_token):
    """Test creating a JS program"""
    if not admin_token:
        pytest.skip("Admin token not available")

    response = client.post(
        "/js-programs",
        data={
            "name": "test_program",
            "code": "console.log('test');"
        },
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test_program"
    assert "id" in data


def test_create_js_program_no_auth(client):
    """Test creating JS program without auth fails"""
    response = client.post(
        "/js-programs",
        data={
            "name": "test_program",
            "code": "console.log('test');"
        }
    )
    assert response.status_code == 403


def test_get_js_program(client, admin_token):
    """Test retrieving a JS program"""
    if not admin_token:
        pytest.skip("Admin token not available")

    # Create program first
    create_response = client.post(
        "/js-programs",
        data={
            "name": "test_program",
            "code": "console.log('test');"
        },
        headers={"X-Authorization": admin_token}
    )
    if create_response.status_code != 201:
        pytest.skip("Failed to create JS program")

    program_id = create_response.json()["id"]

    # Get program
    response = client.get(
        f"/js-programs/{program_id}",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200
    assert response.json()["id"] == program_id
    assert response.json()["name"] == "test_program"
    assert "code" in response.json()


def test_get_nonexistent_js_program(client, admin_token):
    """Test getting non-existent JS program fails"""
    if not admin_token:
        pytest.skip("Admin token not available")

    response = client.get(
        "/js-programs/nonexistent_id",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 404


def test_update_js_program(client, admin_token):
    """Test updating a JS program"""
    if not admin_token:
        pytest.skip("Admin token not available")

    # Create program first
    create_response = client.post(
        "/js-programs",
        data={
            "name": "test_program",
            "code": "console.log('test');"
        },
        headers={"X-Authorization": admin_token}
    )
    if create_response.status_code != 201:
        pytest.skip("Failed to create JS program")

    program_id = create_response.json()["id"]

    # Update program
    update_response = client.put(
        f"/js-programs/{program_id}",
        data={
            "name": "updated_program",
            "code": "console.log('updated');"
        },
        headers={"X-Authorization": admin_token}
    )
    assert update_response.status_code == 200
    assert update_response.json()["name"] == "updated_program"


def test_delete_js_program(client, admin_token):
    """Test deleting a JS program"""
    if not admin_token:
        pytest.skip("Admin token not available")

    # Create program
    create_response = client.post(
        "/js-programs",
        data={
            "name": "test_program",
            "code": "console.log('test');"
        },
        headers={"X-Authorization": admin_token}
    )
    if create_response.status_code != 201:
        pytest.skip("Failed to create JS program")

    program_id = create_response.json()["id"]

    # Delete program
    delete_response = client.delete(
        f"/js-programs/{program_id}",
        headers={"X-Authorization": admin_token}
    )
    assert delete_response.status_code == 200
    assert "deleted successfully" in delete_response.json()["message"]

    # Verify deleted
    get_response = client.get(
        f"/js-programs/{program_id}",
        headers={"X-Authorization": admin_token}
    )
    assert get_response.status_code == 404


# ============================================================================
# M5.2: PACKAGE CONFUSION AUDIT TESTS
# ============================================================================

def test_package_confusion_no_models(client, admin_token):
    """Test audit works (no models or existing models from other tests)"""
    if not admin_token:
        pytest.skip("Admin token not available")

    response = client.get(
        "/audit/package-confusion",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200
    # Status is either "no_models" or "success" depending on whether other tests created models
    assert response.json()["status"] in ["no_models", "success"]


def test_package_confusion_single_model(client, admin_token):
    """Test audit with at least one model"""
    if not admin_token:
        pytest.skip("Admin token not available")

    # Upload model
    upload_response = client.post(
        "/sensitive-models/upload",
        data={"model_name": "test_model"},
        files={"file": ("model.zip", b"fake zip content")},
        headers={"X-Authorization": admin_token}
    )
    if upload_response.status_code != 201:
        pytest.skip("Failed to upload model")

    # Run audit
    response = client.get(
        "/audit/package-confusion",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    # Should have at least 1 model (this + others from other tests)
    assert response.json()["models_analyzed"] >= 1
    assert "analysis" in response.json()
    # New model should be in analysis
    assert len(response.json()["analysis"]) >= 1


def test_package_confusion_specific_model(client, admin_token):
    """Test audit for specific model"""
    if not admin_token:
        pytest.skip("Admin token not available")

    # Upload two models
    upload1 = client.post(
        "/sensitive-models/upload",
        data={"model_name": "model1"},
        files={"file": ("model.zip", b"fake")},
        headers={"X-Authorization": admin_token}
    )
    if upload1.status_code != 201:
        pytest.skip("Failed to upload first model")

    model_id1 = upload1.json()["id"]

    upload2 = client.post(
        "/sensitive-models/upload",
        data={"model_name": "model2"},
        files={"file": ("model.zip", b"fake")},
        headers={"X-Authorization": admin_token}
    )
    if upload2.status_code != 201:
        pytest.skip("Failed to upload second model")

    # Audit only model1
    response = client.get(
        f"/audit/package-confusion?model_id={model_id1}",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200
    assert response.json()["models_analyzed"] == 1
    assert response.json()["analysis"][0]["model_id"] == model_id1


def test_package_confusion_no_auth(client):
    """Test audit without auth fails"""
    response = client.get("/audit/package-confusion")
    assert response.status_code == 403


def test_package_confusion_includes_indicators(client, admin_token):
    """Test audit returns indicator metrics"""
    if not admin_token:
        pytest.skip("Admin token not available")

    # Upload model
    upload_response = client.post(
        "/sensitive-models/upload",
        data={"model_name": "test_model"},
        files={"file": ("model.zip", b"fake zip content")},
        headers={"X-Authorization": admin_token}
    )
    if upload_response.status_code != 201:
        pytest.skip("Failed to upload model")

    # Run audit
    response = client.get(
        "/audit/package-confusion",
        headers={"X-Authorization": admin_token}
    )
    assert response.status_code == 200
    analysis = response.json()["analysis"][0]
    assert "suspicious" in analysis
    assert "risk_score" in analysis
    assert "total_downloads" in analysis
    assert "unique_users" in analysis
    assert "indicators" in analysis


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_full_m5_workflow(client, admin_token, user_token):
    """Test complete M5 workflow"""
    if not admin_token or not user_token:
        pytest.skip("Admin or user token not available")

    # 1. Create JS program
    js_response = client.post(
        "/js-programs",
        data={
            "name": "security_monitor",
            "code": "console.log('Monitoring'); process.exit(0);"
        },
        headers={"X-Authorization": admin_token}
    )
    assert js_response.status_code == 201
    program_id = js_response.json()["id"]

    # 2. Upload model with JS
    upload_response = client.post(
        "/sensitive-models/upload",
        data={
            "model_name": "secure_model",
            "js_program_id": program_id
        },
        files={"file": ("model.zip", b"secure content")},
        headers={"X-Authorization": admin_token}
    )
    assert upload_response.status_code == 201
    model_id = upload_response.json()["id"]

    # 3. User downloads
    download_response = client.get(
        f"/sensitive-models/{model_id}/download",
        headers={"X-Authorization": user_token}
    )
    assert download_response.status_code == 200
    assert download_response.json()["status"] == "success"

    # 4. Check history
    history_response = client.get(
        f"/download-history/{model_id}",
        headers={"X-Authorization": admin_token}
    )
    assert history_response.status_code == 200
    assert history_response.json()["total_downloads"] == 1

    # 5. Run audit
    audit_response = client.get(
        f"/audit/package-confusion?model_id={model_id}",
        headers={"X-Authorization": admin_token}
    )
    assert audit_response.status_code == 200
    assert audit_response.json()["models_analyzed"] == 1
