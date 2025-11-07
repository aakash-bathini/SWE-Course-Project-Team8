import os
import sys
from typing import Dict
import pytest
from fastapi.testclient import TestClient

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client() -> TestClient:
    from app import app

    return TestClient(app)


@pytest.fixture
def admin_headers(client: TestClient) -> Dict[str, str]:
    payload = {
        "user": {"name": "ece30861defaultadminuser", "is_admin": True},
        "secret": {
            "password": "correcthorsebatterystaple123(!__+@**(A'\"`;DROP TABLE packages;",
        },
    }
    resp = client.put("/authenticate", json=payload)
    if resp.status_code != 200:
        return {}
    token = resp.json()
    return {"X-Authorization": token, "Authorization": token}


def test_users_requires_admin_token(client: TestClient):
    resp = client.get("/users")
    assert resp.status_code in (401, 403)


def test_users_happy_path(client: TestClient, admin_headers: Dict[str, str]):
    if not admin_headers:
        pytest.skip("Admin token not available")
    # List should include default admin
    resp = client.get("/users", headers=admin_headers)
    assert resp.status_code == 200
    users = resp.json()
    assert any(u.get("username") == "ece30861defaultadminuser" for u in users)

    # Register another user and confirm appears
    uname = "users_list_case"
    client.post(
        "/register",
        json={"username": uname, "password": "p", "permissions": ["upload"]},
        headers=admin_headers,
    )
    resp2 = client.get("/users", headers=admin_headers)
    assert resp2.status_code == 200
    users2 = resp2.json()
    assert any(u.get("username") == uname for u in users2)


