"""
Tests for artifact CRUD operations to increase coverage
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _get_auth_token():
    """Helper to get auth token"""
    from src.auth.jwt_auth import auth

    token_data = {
        "sub": "ece30861defaultadminuser",
        "permissions": ["upload", "search", "download", "admin"],
    }
    return auth.create_access_token(token_data)


def _get_headers():
    """Helper to get auth headers"""
    token = _get_auth_token()
    return {"X-Authorization": f"bearer {token}"}


class TestArtifactCreate:
    """Test artifact creation endpoint"""

    def test_create_artifact_model(self):
        """Test creating a model artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        artifact_data = {"url": "https://huggingface.co/test/model"}
        response = client.post("/artifact/model", json=artifact_data, headers=headers)
        assert response.status_code in [201, 202, 409]
        if response.status_code == 201:
            data = response.json()
            assert "metadata" in data
            assert "data" in data

    def test_create_artifact_dataset(self):
        """Test creating a dataset artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        artifact_data = {"url": "https://example.com/dataset"}
        response = client.post("/artifact/dataset", json=artifact_data, headers=headers)
        assert response.status_code in [201, 202, 409]

    def test_create_artifact_code(self):
        """Test creating a code artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        artifact_data = {"url": "https://github.com/test/repo"}
        response = client.post("/artifact/code", json=artifact_data, headers=headers)
        assert response.status_code in [201, 202, 409]

    def test_create_artifact_no_permission(self):
        """Test creating artifact without upload permission"""
        from app import app
        from src.auth.jwt_auth import auth

        token_data = {
            "sub": "limiteduser",
            "permissions": ["search"],  # No upload permission
        }
        token = auth.create_access_token(token_data)
        headers = {"X-Authorization": f"bearer {token}"}

        client = TestClient(app)
        artifact_data = {"url": "https://example.com/model"}
        response = client.post("/artifact/model", json=artifact_data, headers=headers)
        assert response.status_code == 401

    def test_create_artifact_missing_url(self):
        """Test creating artifact with missing URL"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.post("/artifact/model", json={}, headers=headers)
        assert response.status_code == 422  # Validation error


class TestArtifactUpdate:
    """Test artifact update endpoint"""

    def test_update_artifact(self):
        """Test updating an artifact"""
        from app import app, artifacts_db

        headers = _get_headers()
        client = TestClient(app)

        # First create an artifact
        artifact_data = {"url": "https://example.com/test"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Update the artifact
            update_data = {
                "metadata": {"name": "updated_name", "id": artifact_id, "type": "model"},
                "data": {"url": "https://example.com/updated"},
            }
            response = client.put(
                f"/artifacts/model/{artifact_id}", json=update_data, headers=headers
            )
            assert response.status_code == 200

    def test_update_artifact_not_found(self):
        """Test updating non-existent artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        update_data = {
            "metadata": {"name": "test", "id": "nonexistent_999", "type": "model"},
            "data": {"url": "https://example.com/test"},
        }
        response = client.put("/artifacts/model/nonexistent_999", json=update_data, headers=headers)
        assert response.status_code == 404

    def test_update_artifact_type_mismatch(self):
        """Test updating artifact with type mismatch"""
        from app import app, artifacts_db

        headers = _get_headers()
        client = TestClient(app)

        # Create a model artifact
        artifact_data = {"url": "https://example.com/test"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Try to update as dataset (should fail)
            update_data = {
                "metadata": {"name": "test", "id": artifact_id, "type": "dataset"},
                "data": {"url": "https://example.com/test"},
            }
            response = client.put(
                f"/artifacts/dataset/{artifact_id}", json=update_data, headers=headers
            )
            assert response.status_code in [400, 404]


class TestArtifactDelete:
    """Test artifact delete endpoint"""

    def test_delete_artifact(self):
        """Test deleting an artifact"""
        from app import app, artifacts_db

        headers = _get_headers()
        client = TestClient(app)

        # First create an artifact
        artifact_data = {"url": "https://example.com/test"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Delete the artifact
            response = client.delete(f"/artifacts/model/{artifact_id}", headers=headers)
            assert response.status_code == 200

    def test_delete_artifact_not_found(self):
        """Test deleting non-existent artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.delete("/artifacts/model/nonexistent_999", headers=headers)
        assert response.status_code == 404


class TestArtifactRetrieve:
    """Test artifact retrieve endpoint"""

    def test_retrieve_artifact_singular(self):
        """Test retrieving artifact via singular endpoint"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # First create an artifact
        artifact_data = {"url": "https://example.com/test"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Retrieve via singular endpoint
            response = client.get(f"/artifact/model/{artifact_id}", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert "metadata" in data
            assert "data" in data

    def test_retrieve_artifact_plural(self):
        """Test retrieving artifact via plural endpoint"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # First create an artifact
        artifact_data = {"url": "https://example.com/test"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Retrieve via plural endpoint
            response = client.get(f"/artifacts/model/{artifact_id}", headers=headers)
            assert response.status_code == 200


class TestModelsEnumerate:
    """Test models enumerate endpoint"""

    def test_models_enumerate_no_cursor(self):
        """Test models enumerate without cursor"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.get("/models", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "next_cursor" in data

    def test_models_enumerate_with_cursor(self):
        """Test models enumerate with cursor"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # First get initial page
        response1 = client.get("/models?limit=1", headers=headers)
        if response1.status_code == 200:
            data1 = response1.json()
            if data1.get("next_cursor"):
                # Get next page
                response2 = client.get(
                    f"/models?cursor={data1['next_cursor']}&limit=1", headers=headers
                )
                assert response2.status_code == 200

    def test_models_enumerate_with_limit(self):
        """Test models enumerate with custom limit"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.get("/models?limit=10", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) <= 10


class TestArtifactList:
    """Test artifact list endpoint"""

    def test_artifacts_list_wildcard(self):
        """Test artifacts list with wildcard query"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.post("/artifacts", json=[{"name": "*"}], headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_artifacts_list_specific_name(self):
        """Test artifacts list with specific name"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.post("/artifacts", json=[{"name": "test_model"}], headers=headers)
        assert response.status_code == 200

    def test_artifacts_list_with_types_filter(self):
        """Test artifacts list with type filter"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.post(
            "/artifacts", json=[{"name": "*", "types": ["model"]}], headers=headers
        )
        assert response.status_code == 200

    def test_artifacts_list_with_offset(self):
        """Test artifacts list with offset"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.post("/artifacts?offset=0", json=[{"name": "*"}], headers=headers)
        assert response.status_code == 200


class TestArtifactLineage:
    """Test artifact lineage endpoint"""

    def test_artifact_lineage_nonexistent(self):
        """Test artifact lineage for non-existent artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.get("/artifact/model/nonexistent_999/lineage", headers=headers)
        assert response.status_code in [404, 400, 500]

    def test_artifact_lineage_non_model(self):
        """Test artifact lineage for non-model artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create a dataset artifact
        artifact_data = {"url": "https://example.com/dataset"}
        create_response = client.post("/artifact/dataset", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]
            # Try to get lineage (should fail for non-model)
            response = client.get(f"/artifact/model/{artifact_id}/lineage", headers=headers)
            assert response.status_code in [400, 404]


class TestArtifactLicenseCheck:
    """Test artifact license check endpoint"""

    def test_license_check_nonexistent(self):
        """Test license check for non-existent artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.post(
            "/artifact/model/nonexistent_999/license-check",
            json={"github_url": "https://github.com/test/repo"},
            headers=headers,
        )
        assert response.status_code in [404, 400, 500, 502]


class TestArtifactAudit:
    """Test artifact audit endpoint"""

    def test_artifact_audit_sqlite_path(self):
        """Test artifact audit with SQLite path"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create an artifact first
        artifact_data = {"url": "https://example.com/test"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Get audit trail
            response = client.get(f"/artifact/model/{artifact_id}/audit", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_artifact_audit_in_memory_path(self):
        """Test artifact audit with in-memory path"""
        from app import app, artifacts_db, audit_log

        headers = _get_headers()
        client = TestClient(app)

        # Create an artifact and add audit entry
        test_id = "test_audit_123"
        artifacts_db[test_id] = {
            "metadata": {"name": "test", "id": test_id, "type": "model"},
            "data": {"url": "https://example.com/test"},
        }
        audit_log.append(
            {
                "user": {"name": "testuser", "is_admin": False},
                "date": "2024-01-01T00:00:00Z",
                "artifact": {"name": "test", "id": test_id, "type": "model"},
                "action": "CREATE",
            }
        )

        try:
            response = client.get(f"/artifact/model/{test_id}/audit", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
        finally:
            # Cleanup
            if test_id in artifacts_db:
                del artifacts_db[test_id]
            audit_log[:] = [e for e in audit_log if e.get("artifact", {}).get("id") != test_id]
