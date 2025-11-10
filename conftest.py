"""
Pytest configuration and fixtures for the entire test suite.
This file provides shared fixtures including auth mocking for testing endpoints.
"""

import pytest
from unittest.mock import patch
from typing import Dict, Any


@pytest.fixture
def mock_verify_token():
    """Mock the verify_token dependency to allow endpoint testing without auth complications."""

    def _mock_verify(token: str = None) -> Dict[str, Any]:
        """Return a valid admin user for testing."""
        return {
            "sub": "test-user-123",
            "email": "test@example.com",
            "role": "admin",
            "is_admin": True,
            "name": "Test User",
        }

    return _mock_verify


@pytest.fixture
def client_with_auth(mock_verify_token):
    """FastAPI TestClient with mocked auth for testing protected endpoints."""
    from fastapi.testclient import TestClient
    from app import app

    # Patch verify_token in the app module
    with patch("app.verify_token", return_value=mock_verify_token()):
        client = TestClient(app)
        yield client


@pytest.fixture
def admin_headers():
    """Return admin auth headers for testing (used by endpoint tests)."""
    return {"Authorization": "Bearer test-token-admin", "X-User-Role": "admin"}


@pytest.fixture
def user_headers():
    """Return regular user auth headers for testing."""
    return {"Authorization": "Bearer test-token-user", "X-User-Role": "user"}
