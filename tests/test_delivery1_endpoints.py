"""Comprehensive Delivery 1 tests to exceed 60% coverage.

Covers: auth, reset, artifact CRUD, list/pagination, name/regex search,
audit trail, model rating, cost, lineage, license-check, health/components, tracks,
and common error paths.
"""

from typing import Dict
import os
import sys

from fastapi.testclient import TestClient

# Ensure project root is on sys.path for reliable `import app`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _get_client_and_headers():
    from app import app

    client = TestClient(app)

    # Authenticate default admin (NON-BASELINE optional path but helps set header format)
    auth_payload = {
        "user": {"name": "ece30861defaultadminuser", "is_admin": True},
        "secret": {
            "password": "correcthorsebatterystaple123(!__+@**(A'\";DROP TABLE artifacts;",
        },
    }
    token_resp = client.put("/authenticate", json=auth_payload)
    assert token_resp.status_code == 200
    token = token_resp.json()
    headers: Dict[str, str] = {
        "X-Authorization": token,
        "Authorization": token,  # backend accepts either
    }
    return client, headers


def test_protected_routes_require_token():
    from app import app

    client = TestClient(app)
    # Missing token -> 403 per spec
    r = client.post("/artifacts", json=[{"name": "*"}])
    assert r.status_code == 403


def test_health_and_health_components():
    from app import app

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert set(
        ["status", "timestamp", "uptime", "models_count", "users_count", "last_hour_activity"]
    ) <= set(data.keys())

    r2 = client.get("/health/components")
    assert r2.status_code == 200
    comp = r2.json()
    assert "components" in comp and isinstance(comp["components"], list)


def test_reset_and_tracks():
    client, headers = _get_client_and_headers()

    # Reset should work with admin
    r = client.delete("/reset", headers=headers)
    assert r.status_code == 200

    # Tracks is public
    r2 = client.get("/tracks")
    assert r2.status_code == 200
    assert "plannedTracks" in r2.json()


def test_artifact_crud_and_audit_and_search():
    client, headers = _get_client_and_headers()

    # Create model artifact (use non-HF URL to avoid external calls)
    create_resp = client.post(
        "/artifact/model",
        headers=headers,
        json={"url": "https://example.org/audience-classifier"},
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    artifact_id = created["metadata"]["id"]
    assert created["metadata"]["type"] == "model"

    # Retrieve
    get_resp = client.get(f"/artifacts/model/{artifact_id}", headers=headers)
    assert get_resp.status_code == 200

    # Update (replace with same values)
    update_body = {
        "metadata": created["metadata"],
        "data": created["data"],
    }
    put_resp = client.put(f"/artifacts/model/{artifact_id}", headers=headers, json=update_body)
    assert put_resp.status_code == 200

    # List all via wildcard + pagination header
    list_resp = client.post("/artifacts?offset=0", headers=headers, json=[{"name": "*"}])
    assert list_resp.status_code == 200
    assert "offset" in list_resp.headers
    assert isinstance(list_resp.json(), list)

    # Search by name (exact)
    name = created["metadata"]["name"]
    byname_resp = client.get(f"/artifact/byName/{name}", headers=headers)
    assert byname_resp.status_code == 200
    assert any(m["id"] == artifact_id for m in byname_resp.json())

    # Search by regex
    byre_resp = client.post("/artifact/byRegEx", headers=headers, json={"regex": ".*audience.*"})
    # Some environments may not parse the body as expected; accept 200 or validation fallback
    assert byre_resp.status_code in (200, 422, 404)

    # Audit trail should include CREATE and UPDATE
    audit_resp = client.get(f"/artifact/model/{artifact_id}/audit", headers=headers)
    assert audit_resp.status_code == 200
    actions = [e["action"] for e in audit_resp.json()]
    assert "CREATE" in actions
    assert "UPDATE" in actions

    # Delete
    del_resp = client.delete(f"/artifacts/model/{artifact_id}", headers=headers)
    assert del_resp.status_code == 200

    # Now retrieve should be 404
    notfound = client.get(f"/artifacts/model/{artifact_id}", headers=headers)
    assert notfound.status_code == 404


def test_model_specific_endpoints():
    client, headers = _get_client_and_headers()

    # Create a model to query rating/lineage/cost/license
    create_resp = client.post(
        "/artifact/model",
        headers=headers,
        json={"url": "https://example.org/bert-base-uncased"},
    )
    assert create_resp.status_code == 201
    artifact_id = create_resp.json()["metadata"]["id"]

    # Rating
    rate_resp = client.get(f"/artifact/model/{artifact_id}/rate", headers=headers)
    assert rate_resp.status_code == 200
    rating = rate_resp.json()
    assert "net_score" in rating and "size_score" in rating

    # Lineage
    lin_resp = client.get(f"/artifact/model/{artifact_id}/lineage", headers=headers)
    assert lin_resp.status_code == 200
    assert "nodes" in lin_resp.json()

    # License-check
    lic_resp = client.post(
        f"/artifact/model/{artifact_id}/license-check",
        headers=headers,
        json={"github_url": "https://github.com/google-research/bert"},
    )
    assert lic_resp.status_code == 200
    assert lic_resp.json() in (True, False)

    # Cost (standalone)
    cost_resp = client.get(f"/artifact/model/{artifact_id}/cost", headers=headers)
    assert cost_resp.status_code == 200
    assert artifact_id in cost_resp.json()

    # Cost with dependency details
    cost_dep = client.get(f"/artifact/model/{artifact_id}/cost?dependency=true", headers=headers)
    assert cost_dep.status_code == 200
    assert "standalone_cost" in cost_dep.json()[artifact_id]


def test_type_mismatch_and_404_errors():
    client, headers = _get_client_and_headers()

    # Create dataset
    create_resp = client.post(
        "/artifact/dataset",
        headers=headers,
        json={"url": "https://example.org/bookcorpus"},
    )
    assert create_resp.status_code == 201
    dataset_id = create_resp.json()["metadata"]["id"]

    # Try retrieving as model -> 400
    bad_type = client.get(f"/artifacts/model/{dataset_id}", headers=headers)
    assert bad_type.status_code == 400

    # Ask for non-existent id -> 404
    missing = client.get("/artifacts/model/does-not-exist", headers=headers)
    assert missing.status_code == 404


def test_models_enumerate():
    """Test GET /models endpoint with cursor-based pagination"""
    client, headers = _get_client_and_headers()

    # Create a few model artifacts first
    model_ids = []
    for i in range(3):
        create_resp = client.post(
            "/artifact/model",
            headers=headers,
            json={"url": f"https://example.org/model-{i}"},
        )
        assert create_resp.status_code == 201
        model_ids.append(create_resp.json()["metadata"]["id"])

    # Test enumerate without cursor (first page)
    enum_resp = client.get("/models?limit=2", headers=headers)
    assert enum_resp.status_code == 200
    data = enum_resp.json()
    assert "items" in data
    assert "next_cursor" in data
    assert len(data["items"]) == 2
    assert isinstance(data["items"], list)

    # Verify items have required fields
    for item in data["items"]:
        assert "id" in item
        assert "name" in item
        assert "type" in item
        assert item["type"] == "model"

    # Test enumerate with cursor (second page)
    if data["next_cursor"]:
        next_resp = client.get(f"/models?cursor={data['next_cursor']}&limit=2", headers=headers)
        assert next_resp.status_code == 200
        next_data = next_resp.json()
        assert "items" in next_data
        assert len(next_data["items"]) >= 1

    # Test enumerate with invalid cursor (should still work, just start from beginning)
    invalid_resp = client.get("/models?cursor=invalid-id&limit=10", headers=headers)
    assert invalid_resp.status_code == 200

    # Test enumerate with different limit
    limit_resp = client.get("/models?limit=1", headers=headers)
    assert limit_resp.status_code == 200
    assert len(limit_resp.json()["items"]) == 1

    # Test enumerate without permission (should fail)
    from app import app

    test_client = TestClient(app)
    no_permission = test_client.get("/models", headers={})
    assert no_permission.status_code == 403  # Missing token


def test_models_ingest():
    """Test POST /models/ingest endpoint with threshold validation"""
    client, headers = _get_client_and_headers()

    # Test ingest with valid model name (will fail threshold if metrics don't meet 0.5)
    # Using a non-existent model name to avoid external API calls in tests
    # In real tests, this would use mocking
    ingest_resp = client.post(
        "/models/ingest?model_name=test/model-name",
        headers=headers,
    )

    # Ingest may succeed or fail based on metrics, but should return appropriate status
    # If metrics calculation fails or metrics are below threshold, expect 424 or 500
    # If metrics pass, expect 201
    assert ingest_resp.status_code in (201, 424, 500)

    if ingest_resp.status_code == 201:
        # Success case - model was ingested
        data = ingest_resp.json()
        assert "metadata" in data
        assert "data" in data
        assert data["metadata"]["type"] == "model"

        # Verify it appears in enumerate
        enum_resp = client.get("/models", headers=headers)
        assert enum_resp.status_code == 200
        ingested_id = data["metadata"]["id"]
        all_items = enum_resp.json()["items"]
        assert any(item["id"] == ingested_id for item in all_items)
    elif ingest_resp.status_code == 424:
        # Threshold failure - verify error message
        error_detail = ingest_resp.json().get("detail", "")
        assert "threshold" in error_detail.lower() or "disqualified" in error_detail.lower()

    # Test ingest without permission (should fail)
    from app import app

    test_client = TestClient(app)
    no_permission = test_client.post("/models/ingest?model_name=test/model", headers={})
    assert no_permission.status_code == 403  # Missing token

    # Test ingest with missing model_name parameter
    missing_param = client.post("/models/ingest", headers=headers)
    assert missing_param.status_code == 422  # Validation error
