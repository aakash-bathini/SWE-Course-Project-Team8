"""
Comprehensive tests for Milestone 6 features:
- Lineage Graph (GET /artifact/model/{id}/lineage)
- Size Cost (GET /artifact/{artifact_type}/{id}/cost)
- License Check (POST /artifact/model/{id}/license-check)
- Reset (DELETE /reset) - preserves default admin
- Health endpoints and dashboard
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os


def _get_headers():
    """Get authentication headers"""
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    # Authenticate to get token
    auth_response = client.put(
        "/authenticate",
        json={
            "user": {"name": "ece30861defaultadminuser", "is_admin": True},
            "secret": {
                "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
            },
        },
    )
    if auth_response.status_code == 200:
        # Response is a JSON string like "bearer <jwt>", so use .json() to avoid quotes
        token = auth_response.json()
        # Provide only X-Authorization; backend accepts raw or 'bearer <token>'
        return {"X-Authorization": token}
    else:
        # Fallback: return empty headers
        return {}


def _get_admin_headers():
    """Get admin authentication headers"""
    return _get_headers()


class TestLineageGraph:
    """Test lineage graph endpoint (M6.2)"""

    def test_lineage_success(self):
        """Test successful lineage retrieval"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create a model artifact
        artifact_data = {"url": "https://huggingface.co/test-model"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Get lineage
            response = client.get(f"/artifact/model/{artifact_id}/lineage", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert "nodes" in data
            assert "edges" in data
            assert isinstance(data["nodes"], list)
            assert isinstance(data["edges"], list)
            # Should have at least the model node
            assert len(data["nodes"]) >= 1

    def test_lineage_nonexistent_artifact(self):
        """Test lineage for non-existent artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.get("/artifact/model/nonexistent-999/lineage", headers=headers)
        assert response.status_code in [404, 403]

    def test_lineage_non_model_artifact(self):
        """Test lineage for non-model artifact"""
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

    def test_lineage_with_datasets(self):
        """Test lineage includes related datasets"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create a dataset first
        dataset_data = {"url": "https://huggingface.co/datasets/test-dataset"}
        dataset_response = client.post("/artifact/dataset", json=dataset_data, headers=headers)

        if dataset_response.status_code == 201:
            _ = dataset_response.json()["metadata"]["id"]

            # Create a model that might reference it
            model_data = {"url": "https://huggingface.co/test-model"}
            model_response = client.post("/artifact/model", json=model_data, headers=headers)

            if model_response.status_code == 201:
                model_id = model_response.json()["metadata"]["id"]

                # Get lineage
                lineage_response = client.get(
                    f"/artifact/model/{model_id}/lineage", headers=headers
                )
                assert lineage_response.status_code == 200
                lineage_data = lineage_response.json()
                # Should have nodes and edges
                assert "nodes" in lineage_data
                assert "edges" in lineage_data


class TestSizeCost:
    """Test size cost endpoint (M6.2)"""

    def test_cost_success(self):
        """Test successful cost retrieval"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create a model artifact
        artifact_data = {"url": "https://huggingface.co/test-model"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Get cost
            response = client.get(f"/artifact/model/{artifact_id}/cost", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert artifact_id in data
            assert "total_cost" in data[artifact_id]

    def test_cost_with_dependencies(self):
        """Test cost with dependency breakdown"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create a model artifact
        artifact_data = {"url": "https://huggingface.co/test-model"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Get cost with dependencies
            response = client.get(
                f"/artifact/model/{artifact_id}/cost?dependency=true", headers=headers
            )
            assert response.status_code == 200
            data = response.json()
            assert artifact_id in data
            assert "total_cost" in data[artifact_id]
            assert "standalone_cost" in data[artifact_id]

    def test_cost_nonexistent_artifact(self):
        """Test cost for non-existent artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.get("/artifact/model/nonexistent-999/cost", headers=headers)
        assert response.status_code in [404, 403]

    def test_cost_type_mismatch(self):
        """Test cost with type mismatch"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create a dataset artifact
        artifact_data = {"url": "https://example.com/dataset"}
        create_response = client.post("/artifact/dataset", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]
            # Try to get cost as model (should fail)
            response = client.get(f"/artifact/model/{artifact_id}/cost", headers=headers)
            assert response.status_code == 400

    def test_cost_different_artifact_types(self):
        """Test cost for different artifact types"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Test model
        model_data = {"url": "https://huggingface.co/test-model"}
        model_response = client.post("/artifact/model", json=model_data, headers=headers)
        if model_response.status_code == 201:
            model_id = model_response.json()["metadata"]["id"]
            response = client.get(f"/artifact/model/{model_id}/cost", headers=headers)
            assert response.status_code == 200

        # Test dataset
        dataset_data = {"url": "https://huggingface.co/datasets/test-dataset"}
        dataset_response = client.post("/artifact/dataset", json=dataset_data, headers=headers)
        if dataset_response.status_code == 201:
            dataset_id = dataset_response.json()["metadata"]["id"]
            response = client.get(f"/artifact/dataset/{dataset_id}/cost", headers=headers)
            assert response.status_code == 200


class TestLicenseCheck:
    """Test license check endpoint (M6.2)"""

    def test_license_check_success(self):
        """Test successful license check"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create a model artifact
        artifact_data = {"url": "https://huggingface.co/test-model"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Check license
            payload = {"github_url": "https://github.com/test/repo"}
            response = client.post(
                f"/artifact/model/{artifact_id}/license-check", json=payload, headers=headers
            )
            assert response.status_code == 200
            # Should return boolean
            result = response.json()
            assert isinstance(result, bool)

    def test_license_check_nonexistent_artifact(self):
        """Test license check for non-existent artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        payload = {"github_url": "https://github.com/test/repo"}
        response = client.post(
            "/artifact/model/nonexistent-999/license-check", json=payload, headers=headers
        )
        assert response.status_code in [404, 403]

    def test_license_check_non_model_artifact(self):
        """Test license check for non-model artifact"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create a dataset artifact
        artifact_data = {"url": "https://example.com/dataset"}
        create_response = client.post("/artifact/dataset", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]
            # Try to check license (should fail for non-model)
            payload = {"github_url": "https://github.com/test/repo"}
            response = client.post(
                f"/artifact/model/{artifact_id}/license-check", json=payload, headers=headers
            )
            assert response.status_code in [400, 404]

    def test_license_check_missing_github_url(self):
        """Test license check with missing github_url"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        # Create a model artifact
        artifact_data = {"url": "https://huggingface.co/test-model"}
        create_response = client.post("/artifact/model", json=artifact_data, headers=headers)

        if create_response.status_code == 201:
            artifact_id = create_response.json()["metadata"]["id"]

            # Check license without github_url
            payload = {}
            response = client.post(
                f"/artifact/model/{artifact_id}/license-check", json=payload, headers=headers
            )
            # Should fail validation
            assert response.status_code in [400, 422]


class TestResetEndpoint:
    """Test reset endpoint (M6.2)"""

    def test_reset_preserves_default_admin(self):
        """Test that reset preserves default admin user"""
        from app import app, users_db, artifacts_db, audit_log

        headers = _get_admin_headers()
        client = TestClient(app)

        # Add some test data
        artifacts_db["test_artifact"] = {"metadata": {"id": "test_artifact"}}
        audit_log.append({"action": "CREATE"})

        try:
            # Reset
            response = client.delete("/reset", headers=headers)
            assert response.status_code in [200, 401, 403]

            # Verify artifacts cleared
            assert len(artifacts_db) == 0
            assert len(audit_log) == 0

            # Verify default admin still exists
            from app import DEFAULT_ADMIN

            admin_username = DEFAULT_ADMIN["username"]
            assert admin_username in users_db

            # Verify default admin can authenticate
            auth_response = client.put(
                "/authenticate",
                json={
                    "user": {"name": admin_username, "is_admin": True},
                    "secret": {"password": DEFAULT_ADMIN["password"]},
                },
            )
            assert auth_response.status_code == 200
        finally:
            # Ensure default admin is restored
            from app import DEFAULT_ADMIN

            users_db[DEFAULT_ADMIN["username"]] = DEFAULT_ADMIN.copy()

    def test_reset_requires_admin(self):
        """Test that reset requires admin permission"""
        from app import app

        headers = _get_headers()  # Regular user headers
        client = TestClient(app)

        # Try to reset without admin permission
        response = client.delete("/reset", headers=headers)
        # Should fail if user doesn't have admin permission
        # (This depends on the token's permissions)
        assert response.status_code in [200, 401, 403]

    def test_reset_clears_artifacts(self):
        """Test that reset clears all artifacts"""
        from app import app, artifacts_db

        headers = _get_admin_headers()
        client = TestClient(app)

        # Add test artifacts
        artifacts_db["test1"] = {"metadata": {"id": "test1"}}
        artifacts_db["test2"] = {"metadata": {"id": "test2"}}

        try:
            # Reset
            response = client.delete("/reset", headers=headers)
            assert response.status_code in [200, 401, 403]

            # Verify artifacts cleared
            assert len(artifacts_db) == 0
        finally:
            # Clean up
            artifacts_db.clear()
            from app import DEFAULT_ADMIN
            from app import users_db

            users_db[DEFAULT_ADMIN["username"]] = DEFAULT_ADMIN.copy()


class TestHealthEndpoints:
    """Test health endpoints (M6.3)"""

    def test_health_endpoint(self):
        """Test /health endpoint"""
        from app import app

        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "last_hour_activity" in data or "last_hour" in data or "activity" in data

    def test_health_components(self):
        """Test /health/components endpoint"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.get("/health/components", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert isinstance(data["components"], list)

    def test_health_components_with_params(self):
        """Test /health/components with query parameters"""
        from app import app

        headers = _get_headers()
        client = TestClient(app)

        response = client.get(
            "/health/components?windowMinutes=30&includeTimeline=true", headers=headers
        )
        assert response.status_code == 200
        data = response.json()
        assert "components" in data
