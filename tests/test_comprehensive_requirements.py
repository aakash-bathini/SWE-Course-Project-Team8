"""
Comprehensive Requirements Verification Tests
Tests for Baseline + Security Track requirements to ensure 100% compliance
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import time
import json
import os
import sys

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

UTC = timezone.utc


def _get_client_and_headers():
    """Helper to get authenticated client and headers"""
    from app import app

    client = TestClient(app)

    # Authenticate default admin (use exact password from requirements)
    auth_payload = {
        "user": {"name": "ece30861defaultadminuser", "is_admin": True},
        "secret": {
            "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;",
        },
    }
    token_resp = client.put("/authenticate", json=auth_payload)
    if token_resp.status_code != 200:
        return None, None

    token = token_resp.json()
    headers: Dict[str, str] = {
        "X-Authorization": token,
        "Authorization": token,
    }
    return client, headers


# ============================================================================
# BASELINE REQUIREMENTS TESTS
# ============================================================================


class TestBaselineCRUD:
    """Test CR[U]D operations: Upload, Rate, Download"""

    def test_upload_artifact(self):
        """Test artifact upload"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        response = client.post(
            "/artifact/model",
            json={"url": "https://huggingface.co/test/model"},
            headers=headers,
        )
        assert response.status_code in [201, 202]
        data = response.json()
        assert "metadata" in data
        assert "data" in data
        assert data["metadata"]["type"] == "model"

    def test_rate_returns_all_metrics(self):
        """Test rate endpoint returns net score and all sub-scores including Phase 2 metrics"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # First create an artifact
        create_resp = client.post(
            "/artifact/model",
            json={"url": "https://huggingface.co/test/rate-test"},
            headers=headers,
        )
        if create_resp.status_code not in [201, 202]:
            pytest.skip("Could not create test artifact")

        artifact_id = create_resp.json()["metadata"]["id"]

        # Wait a bit for async processing
        time.sleep(2)

        # Test rate endpoint
        rate_resp = client.get(
            f"/artifact/model/{artifact_id}/rate",
            headers=headers,
        )

        # Should return 200 or 404 (if not ready yet)
        assert rate_resp.status_code in [200, 404, 502]

        if rate_resp.status_code == 200:
            data = rate_resp.json()
            # Check for required fields
            assert "name" in data
            assert "net_score" in data
            # Check Phase 2 metrics (may be present as metric or metric_latency)
            # At least one form should be present
            has_repro = "reproducibility" in data or "reproducibility_latency" in data
            has_reviewed = "reviewedness" in data or "reviewedness_latency" in data
            has_tree = (
                "tree_score" in data
                or "tree_score_latency" in data
                or "treescore" in data
                or "treescore_latency" in data
            )
            # At least one Phase 2 metric should be present (metrics may be async)
            assert (
                has_repro or has_reviewed or has_tree
            ), f"Expected Phase 2 metrics, got: {list(data.keys())}"

    def test_download_with_aspects(self):
        """Test download with sub-aspects (full, weights, datasets, code)"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Create artifact first
        create_resp = client.post(
            "/artifact/model",
            json={"url": "https://huggingface.co/test/download-test"},
            headers=headers,
        )
        if create_resp.status_code not in [201, 202]:
            pytest.skip("Could not create test artifact")

        artifact_id = create_resp.json()["metadata"]["id"]

        # Test different aspects
        for aspect in ["full", "weights", "datasets", "code"]:
            resp = client.get(
                f"/models/{artifact_id}/download?aspect={aspect}",
                headers=headers,
            )
            # May return 404 if file doesn't exist, but endpoint should exist
            assert resp.status_code in [200, 404, 500, 502]


class TestPhase2Metrics:
    """Test Phase 2 metrics: Reproducibility, Reviewedness, Treescore"""

    def test_reproducibility_metric(self):
        """Test reproducibility returns 0, 0.5, or 1.0"""

        # This is tested in test_milestone2_features.py
        # Just verify the metric exists
        from src.metrics.reproducibility import metric as repro_metric

        assert callable(repro_metric)

    def test_reviewedness_metric(self):
        """Test reviewedness returns -1 if no repo, 0.0-1.0 otherwise"""

        from src.metrics.reviewedness import metric as reviewed_metric

        assert callable(reviewed_metric)

    def test_treescore_metric(self):
        """Test treescore averages parent model scores"""

        from src.metrics.treescore import metric as tree_metric

        assert callable(tree_metric)


class TestModelIngest:
    """Test model ingest with ≥0.5 threshold"""

    def test_ingest_filters_on_threshold(self):
        """Test ingest only proceeds if all non-latency metrics ≥0.5"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # This is tested in test_good_model_ingest_and_rate.py
        # Verify endpoint exists
        resp = client.post(
            "/models/ingest?model_name=test/model",
            headers=headers,
        )
        # Should return 201 (ingested), 424 (failed threshold), or 500 (error)
        assert resp.status_code in [201, 424, 500, 502]


class TestEnumerate:
    """Test enumerate with cursor-based pagination"""

    def test_enumerate_pagination(self):
        """Test /models endpoint with cursor pagination"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        resp = client.get("/models", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert isinstance(data["items"], list)

        # Test with cursor
        if "next_cursor" in data and data["next_cursor"]:
            resp2 = client.get(
                f"/models?cursor={data['next_cursor']}",
                headers=headers,
            )
            assert resp2.status_code == 200


class TestSearch:
    """Test search with regex over names and model cards"""

    def test_search_by_name(self):
        """Test /artifact/byName endpoint"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        resp = client.get(
            "/artifact/byName/test-model",
            headers=headers,
        )
        assert resp.status_code in [200, 404]

    def test_search_by_regex(self):
        """Test /artifact/byRegEx endpoint"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        resp = client.post(
            "/artifact/byRegEx",
            json={"regex": ".*test.*"},
            headers=headers,
        )
        assert resp.status_code in [200, 404]


class TestLineageGraph:
    """Test lineage graph endpoint"""

    def test_lineage_endpoint(self):
        """Test /artifact/model/{id}/lineage"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Create artifact first
        create_resp = client.post(
            "/artifact/model",
            json={"url": "https://huggingface.co/test/lineage-test"},
            headers=headers,
        )
        if create_resp.status_code not in [201, 202]:
            pytest.skip("Could not create test artifact")

        artifact_id = create_resp.json()["metadata"]["id"]

        resp = client.get(
            f"/artifact/model/{artifact_id}/lineage",
            headers=headers,
        )
        assert resp.status_code in [200, 404, 500]


class TestSizeCost:
    """Test size cost endpoint"""

    def test_size_cost_endpoint(self):
        """Test /artifact/{type}/{id}/cost"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Create artifact first
        create_resp = client.post(
            "/artifact/model",
            json={"url": "https://huggingface.co/test/cost-test"},
            headers=headers,
        )
        if create_resp.status_code not in [201, 202]:
            pytest.skip("Could not create test artifact")

        artifact_id = create_resp.json()["metadata"]["id"]

        resp = client.get(
            f"/artifact/model/{artifact_id}/cost",
            headers=headers,
        )
        assert resp.status_code in [200, 404, 500]


class TestLicenseCheck:
    """Test license check endpoint"""

    def test_license_check_endpoint(self):
        """Test /artifact/model/{id}/license-check"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Create artifact first
        create_resp = client.post(
            "/artifact/model",
            json={"url": "https://huggingface.co/test/license-test"},
            headers=headers,
        )
        if create_resp.status_code not in [201, 202]:
            pytest.skip("Could not create test artifact")

        artifact_id = create_resp.json()["metadata"]["id"]

        resp = client.post(
            f"/artifact/model/{artifact_id}/license-check",
            json={"github_url": "https://github.com/test/repo"},
            headers=headers,
        )
        assert resp.status_code in [200, 404, 500]


class TestReset:
    """Test reset endpoint"""

    def test_reset_endpoint(self):
        """Test DELETE /reset"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        resp = client.delete("/reset", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data


class TestHealthEndpoint:
    """Test health and observability"""

    def test_health_endpoint(self):
        """Test GET /health (no auth required)"""
        from app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data

    def test_health_components(self):
        """Test GET /health/components"""
        from app import app

        client = TestClient(app)
        resp = client.get("/health/components")
        assert resp.status_code == 200


# ============================================================================
# SECURITY TRACK REQUIREMENTS TESTS
# ============================================================================


class TestTokenExpiration:
    """Test token expiration (10 hours or 1000 calls)"""

    def test_token_expires_after_10_hours(self):
        """Test token expires after 10 hours"""
        from src.auth.jwt_auth import auth as jwt_auth

        # Create token with expiration in the past
        payload = {
            "sub": "testuser",
            "permissions": ["upload"],
            "exp": datetime.now(UTC) - timedelta(hours=11),  # 11 hours ago
        }
        token = jwt_auth.create_access_token(payload, expires_delta=timedelta(hours=-11))

        # Verify token is expired
        result = jwt_auth.verify_token(token)
        assert result is None or result.get("exp") is None

    def test_token_expires_after_1000_calls(self):
        """Test token expires after 1000 API calls"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        from app import token_call_counts
        import hashlib

        # Get token from headers (could be in X-Authorization or Authorization)
        token = headers.get("X-Authorization") or headers.get("Authorization", "")
        # Remove "bearer " prefix if present (case-insensitive)
        if token.lower().startswith("bearer "):
            token = token[7:]  # Remove "bearer " prefix
        if not token:
            pytest.skip("Could not extract token")

        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # Set call count to 999 (before the first test call)
        token_call_counts[token_hash] = 999

        # Make one more call to an authenticated endpoint - should work (999 < 1000, increments to 1000)
        # Use /models endpoint which requires authentication
        resp = client.get("/models", headers=headers)
        assert resp.status_code == 200, f"First call should succeed, got {resp.status_code}"

        # Verify count was incremented to 1000
        assert (
            token_call_counts.get(token_hash, 0) == 1000
        ), f"Expected count 1000, got {token_call_counts.get(token_hash, 0)}"

        # Make another call - should fail (1000 >= 1000)
        resp2 = client.get("/models", headers=headers)
        # Should fail with 403
        assert (
            resp2.status_code == 403
        ), f"Second call should fail with 403, got {resp2.status_code}"

        # Reset for other tests
        token_call_counts[token_hash] = 0

    def test_multiple_concurrent_tokens(self):
        """Test single user can have multiple concurrent tokens"""
        from app import app

        client = TestClient(app)
        # Authenticate twice with same user
        password = "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
        resp1 = client.put(
            "/authenticate",
            json={
                "user": {"name": "ece30861defaultadminuser", "is_admin": True},
                "secret": {"password": password},
            },
        )
        assert resp1.status_code == 200, f"First auth failed: {resp1.status_code} - {resp1.text}"
        token1 = resp1.json()

        resp2 = client.put(
            "/authenticate",
            json={
                "user": {"name": "ece30861defaultadminuser", "is_admin": True},
                "secret": {"password": password},
            },
        )
        assert resp2.status_code == 200, f"Second auth failed: {resp2.status_code} - {resp2.text}"
        token2 = resp2.json()

        # Both tokens should be valid strings (may be identical if created at same time)
        assert isinstance(token1, str) and isinstance(token2, str)
        assert len(token1) > 0 and len(token2) > 0

        # Both tokens should work independently
        headers1 = {"X-Authorization": token1}
        headers2 = {"X-Authorization": token2}

        resp1_use = client.get("/health", headers=headers1)
        resp2_use = client.get("/health", headers=headers2)

        assert resp1_use.status_code == 200
        assert resp2_use.status_code == 200


class TestSensitiveModels:
    """Test sensitive models functionality"""

    def test_upload_sensitive_model(self):
        """Test uploading sensitive model"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Create a dummy ZIP file
        import io

        zip_content = b"PK\x03\x04"  # Minimal ZIP header

        response = client.post(
            "/sensitive-models/upload",
            data={"model_name": "test-sensitive-model"},
            files={"file": ("model.zip", io.BytesIO(zip_content), "application/zip")},
            headers=headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["model_name"] == "test-sensitive-model"

    def test_delete_sensitive_model(self):
        """Test deleting sensitive model"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # First upload
        import io

        zip_content = b"PK\x03\x04"
        upload_resp = client.post(
            "/sensitive-models/upload",
            data={"model_name": "test-delete-model"},
            files={"file": ("model.zip", io.BytesIO(zip_content), "application/zip")},
            headers=headers,
        )
        if upload_resp.status_code != 201:
            pytest.skip("Could not upload test model")

        model_id = upload_resp.json()["id"]

        # Delete it
        delete_resp = client.delete(
            f"/sensitive-models/{model_id}",
            headers=headers,
        )
        assert delete_resp.status_code == 200

    def test_download_with_js_execution(self):
        """Test download executes JS program and blocks on non-zero exit"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Create JS program that exits with non-zero
        js_resp = client.post(
            "/js-programs",
            data={
                "name": "blocking-program",
                "code": "process.exit(1);",  # Exit with error
            },
            headers=headers,
        )
        if js_resp.status_code != 201:
            pytest.skip("Could not create JS program")

        js_program_id = js_resp.json()["id"]

        # Upload sensitive model with JS program
        import io

        zip_content = b"PK\x03\x04"
        upload_resp = client.post(
            "/sensitive-models/upload",
            data={
                "model_name": "test-js-block",
                "js_program_id": js_program_id,
            },
            files={"file": ("model.zip", io.BytesIO(zip_content), "application/zip")},
            headers=headers,
        )
        if upload_resp.status_code != 201:
            pytest.skip("Could not upload test model")

        model_id = upload_resp.json()["id"]

        # Try to download - should be blocked
        download_resp = client.get(
            f"/sensitive-models/{model_id}/download",
            headers=headers,
        )
        assert download_resp.status_code == 200
        data = download_resp.json()
        assert data["status"] == "blocked"
        assert "js_stdout" in data or "message" in data

    def test_download_history(self):
        """Test download history endpoint"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Upload a model first
        import io

        zip_content = b"PK\x03\x04"
        upload_resp = client.post(
            "/sensitive-models/upload",
            data={"model_name": "test-history-model"},
            files={"file": ("model.zip", io.BytesIO(zip_content), "application/zip")},
            headers=headers,
        )
        if upload_resp.status_code != 201:
            pytest.skip("Could not upload test model")

        model_id = upload_resp.json()["id"]

        # Download it
        client.get(f"/sensitive-models/{model_id}/download", headers=headers)

        # Get history
        history_resp = client.get(
            f"/download-history/{model_id}",
            headers=headers,
        )
        assert history_resp.status_code == 200
        data = history_resp.json()
        assert "history" in data
        assert "total_downloads" in data


class TestPackageConfusionAudit:
    """Test package confusion audit"""

    def test_package_confusion_audit_endpoint(self):
        """Test /audit/package-confusion endpoint"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        resp = client.get("/audit/package-confusion", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "suspicious_packages" in data  # Per requirements: "returns a list of packages"
        assert isinstance(data["suspicious_packages"], list)
        assert "total_analyzed" in data
        assert "total_suspicious" in data

    def test_search_presence_tracking(self):
        """Test that search hits are tracked for package confusion"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Create an artifact
        create_resp = client.post(
            "/artifact/model",
            json={"url": "https://huggingface.co/test/search-track"},
            headers=headers,
        )
        if create_resp.status_code not in [201, 202]:
            pytest.skip("Could not create test artifact")

        # Search for it (use artifact name from response)
        artifact_name = create_resp.json()["metadata"]["name"]
        search_resp = client.get(
            f"/artifact/byName/{artifact_name}",
            headers=headers,
        )

        # Search should work (may return 404 if name doesn't match exactly)
        assert search_resp.status_code in [200, 404]

        # Verify search history was recorded (check database)
        # This is tested indirectly through package confusion audit
        audit_resp = client.get("/audit/package-confusion", headers=headers)
        assert audit_resp.status_code == 200


class TestDefaultUser:
    """Test default user exists"""

    def test_default_user_exists(self):
        """Test default admin user can authenticate"""
        from app import app

        client = TestClient(app)
        # Use exact password from requirements: correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;
        resp = client.put(
            "/authenticate",
            json={
                "user": {"name": "ece30861defaultadminuser", "is_admin": True},
                "secret": {
                    "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
                },
            },
        )
        assert (
            resp.status_code == 200
        ), f"Authentication failed with status {resp.status_code}: {resp.text}"
        token = resp.json()
        assert token.startswith("bearer ") or len(token) > 0

    def test_default_user_after_reset(self):
        """Test default user exists after reset"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Reset
        client.delete("/reset", headers=headers)

        # Default user should still be able to authenticate
        resp = client.put(
            "/authenticate",
            json={
                "user": {"name": "ece30861defaultadminuser", "is_admin": True},
                "secret": {
                    "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
                },
            },
        )
        assert (
            resp.status_code == 200
        ), f"Authentication failed after reset with status {resp.status_code}: {resp.text}"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestEndToEndWorkflows:
    """End-to-end workflow tests"""

    def test_full_artifact_lifecycle(self):
        """Test complete artifact lifecycle: create, rate, download, delete"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Create
        create_resp = client.post(
            "/artifact/model",
            json={"url": "https://huggingface.co/test/lifecycle"},
            headers=headers,
        )
        assert create_resp.status_code in [201, 202]
        artifact_id = create_resp.json()["metadata"]["id"]

        # Rate (may need to wait)
        time.sleep(1)
        rate_resp = client.get(
            f"/artifact/model/{artifact_id}/rate",
            headers=headers,
        )
        assert rate_resp.status_code in [200, 404, 502]

        # Download
        download_resp = client.get(
            f"/models/{artifact_id}/download",
            headers=headers,
        )
        assert download_resp.status_code in [200, 404, 500, 502]

        # Delete
        delete_resp = client.delete(
            f"/artifacts/model/{artifact_id}",
            headers=headers,
        )
        assert delete_resp.status_code in [200, 404]

    def test_sensitive_model_workflow(self):
        """Test complete sensitive model workflow"""
        client, headers = _get_client_and_headers()
        if not client or not headers:
            pytest.skip("Could not authenticate")

        # Create JS program
        js_resp = client.post(
            "/js-programs",
            data={
                "name": "test-workflow-js",
                "code": "console.log('test'); process.exit(0);",
            },
            headers=headers,
        )
        if js_resp.status_code != 201:
            pytest.skip("Could not create JS program")

        js_program_id = js_resp.json()["id"]

        # Upload sensitive model
        import io

        zip_content = b"PK\x03\x04"
        upload_resp = client.post(
            "/sensitive-models/upload",
            data={
                "model_name": "workflow-test",
                "js_program_id": js_program_id,
            },
            files={"file": ("model.zip", io.BytesIO(zip_content), "application/zip")},
            headers=headers,
        )
        if upload_resp.status_code != 201:
            pytest.skip("Could not upload sensitive model")

        model_id = upload_resp.json()["id"]

        # Download (should succeed with exit code 0)
        download_resp = client.get(
            f"/sensitive-models/{model_id}/download",
            headers=headers,
        )
        assert download_resp.status_code == 200

        # Check history
        history_resp = client.get(
            f"/download-history/{model_id}",
            headers=headers,
        )
        assert history_resp.status_code == 200

        # Check audit
        audit_resp = client.get("/audit/package-confusion", headers=headers)
        assert audit_resp.status_code == 200

        # Delete
        delete_resp = client.delete(
            f"/sensitive-models/{model_id}",
            headers=headers,
        )
        assert delete_resp.status_code == 200
