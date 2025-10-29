import os
import sys
from fastapi.testclient import TestClient

# Ensure project root on sys.path for 'app' import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import app


def test_delivery1_smoke_flow():
    client = TestClient(app)

    # Authenticate
    resp = client.put(
        "/authenticate",
        json={
            "user": {"name": "ece30861defaultadminuser", "is_admin": True},
            "secret": {"password": "correcthorsebatterystaple123(!__+@**(A;DROP TABLE packages"},
        },
    )
    assert resp.status_code == 200
    token = resp.json()
    headers = {"X-Authorization": token}

    # Create (model via URL ingest)
    resp = client.post(
        "/artifact/model",
        json={"url": "https://huggingface.co/google-bert/bert-base-uncased"},
        headers=headers,
    )
    assert resp.status_code == 201
    art = resp.json()
    art_id = art["metadata"]["id"]

    # Enumerate (wildcard)
    resp = client.post("/artifacts", json=[{"name": "*"}], headers=headers)
    assert resp.status_code == 200
    assert any(item["id"] == art_id for item in resp.json())

    # Rate
    assert client.get(f"/artifact/model/{art_id}/rate", headers=headers).status_code == 200

    # Cost
    assert client.get(f"/artifact/model/{art_id}/cost", headers=headers).status_code == 200

    # Lineage
    assert client.get(f"/artifact/model/{art_id}/lineage", headers=headers).status_code == 200

    # License Check
    resp = client.post(
        f"/artifact/model/{art_id}/license-check",
        json={"github_url": "https://github.com/google-research/bert"},
        headers=headers,
    )
    assert resp.status_code == 200

    # Delete
    assert (
        client.delete(f"/artifacts/model/{art_id}", headers=headers).status_code == 200
    )


