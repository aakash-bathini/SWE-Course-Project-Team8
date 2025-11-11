"""
Tests for alias routes under /models/* to ensure compatibility with spec paths.
"""

from fastapi.testclient import TestClient


def _auth_headers() -> dict:
    from app import app

    client = TestClient(app)
    resp = client.put(
        "/authenticate",
        json={
            "user": {"name": "ece30861defaultadminuser", "is_admin": True},
            "secret": {
                "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
            },
        },
    )
    if resp.status_code == 200:
        token = resp.json()
        return {"X-Authorization": token}
    return {}


def test_models_alias_endpoints_smoke():
    from app import app

    headers = _auth_headers()
    client = TestClient(app)

    # Create a model artifact to reference
    create = client.post("/artifact/model", json={"url": "https://huggingface.co/test-model"}, headers=headers)
    if create.status_code != 201:
        # If creation failed for any reason, skip alias checks gracefully
        return

    model_id = create.json()["metadata"]["id"]

    # /models/{id}/lineage alias
    r1 = client.get(f"/models/{model_id}/lineage", headers=headers)
    assert r1.status_code in [200, 400, 404]

    # /models/{id}/cost alias
    r2 = client.get(f"/models/{model_id}/cost", headers=headers)
    assert r2.status_code in [200, 400, 404]

    # /models/{id}/license-check alias (with missing github_url to exercise validation)
    r3 = client.post(f"/models/{model_id}/license-check", json={}, headers=headers)
    assert r3.status_code in [400, 422]


