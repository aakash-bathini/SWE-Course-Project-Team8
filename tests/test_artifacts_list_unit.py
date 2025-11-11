"""
High-yield tests for artifact listing and search to increase coverage.
"""

from fastapi.testclient import TestClient


def _auth_headers(client: TestClient) -> dict:
    r = client.put(
        "/authenticate",
        json={
            "user": {"name": "ece30861defaultadminuser", "is_admin": True},
            "secret": {
                "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;"
            },
        },
    )
    return {"X-Authorization": r.json()} if r.status_code == 200 else {}


def test_artifacts_list_and_by_name_and_models_enumerate():
    from app import app

    client = TestClient(app)
    headers = _auth_headers(client)

    # Create a couple of artifacts
    m = client.post("/artifact/model", json={"url": "https://example.org/model-a"}, headers=headers)
    d = client.post(
        "/artifact/dataset", json={"url": "https://example.org/dataset-a"}, headers=headers
    )

    if m.status_code == 201 and d.status_code == 201:
        mid = m.json()["metadata"]["id"]
        mname = m.json()["metadata"]["name"]

        # List with wildcard via POST /artifacts (offset optional)
        lst = client.post("/artifacts", json=[{"name": "*"}], headers=headers)
        assert lst.status_code == 200
        assert isinstance(lst.json(), list)

        # Search by name
        byname = client.get(f"/artifact/byName/{mname}", headers=headers)
        assert byname.status_code == 200
        assert any(item.get("id") == mid for item in byname.json())

        # Enumerate models
        enum = client.get("/models?limit=10", headers=headers)
        assert enum.status_code == 200
        body = enum.json()
        assert "items" in body and isinstance(body["items"], list)
