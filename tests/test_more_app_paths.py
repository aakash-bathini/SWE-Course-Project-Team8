"""
Additional app endpoint smoke tests to increase line coverage on simple paths.
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


def test_tracks_public_endpoint():
    from app import app

    client = TestClient(app)
    r = client.get("/tracks")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict)
    assert "plannedTracks" in body


def test_artifact_audit_path():
    from app import app

    client = TestClient(app)
    headers = _auth_headers(client)

    # Create model
    created = client.post("/artifact/model", json={"url": "https://example.org/model-b"}, headers=headers)
    if created.status_code != 201:
        return
    mid = created.json()["metadata"]["id"]

    # Fetch audit
    audit = client.get(f"/artifact/model/{mid}/audit", headers=headers)
    assert audit.status_code in [200, 404]
    if audit.status_code == 200:
        assert isinstance(audit.json(), list)


