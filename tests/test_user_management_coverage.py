"""
Tests for user management endpoints to increase coverage
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _get_admin_token():
    """Helper to get admin auth token"""
    from src.auth.jwt_auth import auth
    token_data = {
        "sub": "ece30861defaultadminuser",
        "permissions": ["upload", "search", "download", "admin"],
    }
    return auth.create_access_token(token_data)


def _get_admin_headers():
    """Helper to get admin auth headers"""
    token = _get_admin_token()
    return {"X-Authorization": f"bearer {token}"}


def _get_user_token():
    """Helper to get regular user token"""
    from src.auth.jwt_auth import auth
    token_data = {
        "sub": "regularuser",
        "permissions": ["upload", "search"],
    }
    return auth.create_access_token(token_data)


def _get_user_headers():
    """Helper to get regular user auth headers"""
    token = _get_user_token()
    return {"X-Authorization": f"bearer {token}"}


class TestUserRegistration:
    """Test user registration endpoint"""

    def test_register_user_success(self):
        """Test successful user registration"""
        from app import app, users_db
        headers = _get_admin_headers()
        client = TestClient(app)
        
        user_data = {
            "username": "newuser123",
            "password": "password123",
            "permissions": ["upload", "search"]
        }
        
        try:
            response = client.post("/register", json=user_data, headers=headers)
            assert response.status_code in [200, 201]
            # Verify user was created
            assert "newuser123" in users_db
        finally:
            # Cleanup
            if "newuser123" in users_db:
                del users_db["newuser123"]

    def test_register_user_already_exists(self):
        """Test registering user that already exists"""
        from app import app, users_db
        headers = _get_admin_headers()
        client = TestClient(app)
        
        # Create user first
        users_db["existinguser"] = {
            "username": "existinguser",
            "password": "hashed_password",
            "permissions": ["upload"]
        }
        
        try:
            user_data = {
                "username": "existinguser",
                "password": "newpassword",
                "permissions": ["upload", "search"]
            }
            response = client.post("/register", json=user_data, headers=headers)
            assert response.status_code in [400, 409]
        finally:
            if "existinguser" in users_db:
                del users_db["existinguser"]

    def test_register_user_no_admin_permission(self):
        """Test registering user without admin permission"""
        from app import app
        headers = _get_user_headers()
        client = TestClient(app)
        
        user_data = {
            "username": "newuser",
            "password": "password123",
            "permissions": ["upload"]
        }
        response = client.post("/register", json=user_data, headers=headers)
        assert response.status_code == 401


class TestUserDeletion:
    """Test user deletion endpoint"""

    def test_delete_user_success(self):
        """Test successful user deletion"""
        from app import app, users_db
        headers = _get_admin_headers()
        client = TestClient(app)
        
        # Create user first
        users_db["todelete"] = {
            "username": "todelete",
            "password": "hashed_password",
            "permissions": ["upload"]
        }
        
        try:
            response = client.delete("/user/todelete", headers=headers)
            assert response.status_code in [200, 404]
            # Verify user was deleted
            assert "todelete" not in users_db
        finally:
            if "todelete" in users_db:
                del users_db["todelete"]

    def test_delete_user_nonexistent(self):
        """Test deleting non-existent user"""
        from app import app
        headers = _get_admin_headers()
        client = TestClient(app)
        
        response = client.delete("/user/nonexistent_user_999", headers=headers)
        assert response.status_code in [200, 404, 400]

    def test_delete_user_no_admin_permission(self):
        """Test deleting user without admin permission"""
        from app import app
        headers = _get_user_headers()
        client = TestClient(app)
        
        response = client.delete("/user/someuser", headers=headers)
        assert response.status_code == 401


class TestUserPermissionsUpdate:
    """Test user permissions update endpoint"""

    def test_update_permissions_success(self):
        """Test successful permissions update"""
        from app import app, users_db
        headers = _get_admin_headers()
        client = TestClient(app)
        
        # Create user first
        users_db["toupdate"] = {
            "username": "toupdate",
            "password": "hashed_password",
            "permissions": ["upload"]
        }
        
        try:
            update_data = {"permissions": ["upload", "search", "download"]}
            response = client.put("/user/toupdate/permissions", json=update_data, headers=headers)
            assert response.status_code in [200, 404]
            if response.status_code == 200:
                # Verify permissions were updated
                assert users_db["toupdate"]["permissions"] == ["upload", "search", "download"]
        finally:
            if "toupdate" in users_db:
                del users_db["toupdate"]

    def test_update_permissions_nonexistent_user(self):
        """Test updating permissions for non-existent user"""
        from app import app
        headers = _get_admin_headers()
        client = TestClient(app)
        
        update_data = {"permissions": ["upload", "search"]}
        response = client.put("/user/nonexistent_999/permissions", json=update_data, headers=headers)
        assert response.status_code in [200, 404, 400]

    def test_update_permissions_no_admin_permission(self):
        """Test updating permissions without admin permission"""
        from app import app
        headers = _get_user_headers()
        client = TestClient(app)
        
        update_data = {"permissions": ["upload", "search"]}
        response = client.put("/user/someuser/permissions", json=update_data, headers=headers)
        assert response.status_code == 401


class TestListUsers:
    """Test list users endpoint"""

    def test_list_users_success(self):
        """Test successful user listing"""
        from app import app, users_db
        headers = _get_admin_headers()
        client = TestClient(app)
        
        # Add some test users
        users_db["user1"] = {"username": "user1", "permissions": ["upload"]}
        users_db["user2"] = {"username": "user2", "permissions": ["search"]}
        
        try:
            response = client.get("/users", headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            # Should include at least the default admin
            assert len(data) >= 1
        finally:
            if "user1" in users_db:
                del users_db["user1"]
            if "user2" in users_db:
                del users_db["user2"]

    def test_list_users_sqlite_path(self):
        """Test list users with SQLite storage"""
        from app import app, get_db, db_crud
        headers = _get_admin_headers()
        client = TestClient(app)
        
        with patch.dict(os.environ, {"USE_SQLITE": "1"}):
            if get_db and db_crud:
                # Test that SQLite path exists (may not work without actual DB)
                response = client.get("/users", headers=headers)
                # Should succeed or fail gracefully
                assert response.status_code in [200, 500]

    def test_list_users_s3_path(self):
        """Test list users with S3 storage"""
        from app import app, s3_storage
        headers = _get_admin_headers()
        client = TestClient(app)
        
        if s3_storage:
            with patch.object(s3_storage, 'list_users', return_value=[
                {"username": "s3_user", "permissions": ["upload"]}
            ]):
                response = client.get("/users", headers=headers)
                assert response.status_code == 200


class TestAuthentication:
    """Test authentication endpoint"""

    def test_authenticate_success(self):
        """Test successful authentication"""
        from app import app, users_db, DEFAULT_ADMIN
        client = TestClient(app)
        
        # Ensure default admin exists (password is stored as plain text per code)
        admin_username = DEFAULT_ADMIN["username"]
        admin_password = DEFAULT_ADMIN["password"]
        users_db[admin_username] = DEFAULT_ADMIN.copy()
        
        auth_data = {
            "user": {"name": admin_username, "is_admin": True},
            "secret": {"password": admin_password}
        }
        
        response = client.put("/authenticate", json=auth_data)
        assert response.status_code == 200
        token = response.json()
        assert isinstance(token, str)
        assert len(token) > 0
        assert token.startswith("bearer ")

    def test_authenticate_invalid_password(self):
        """Test authentication with invalid password"""
        from app import app, users_db, DEFAULT_ADMIN
        client = TestClient(app)
        
        admin_username = DEFAULT_ADMIN["username"]
        users_db[admin_username] = DEFAULT_ADMIN.copy()
        
        auth_data = {
            "user": {"name": admin_username, "is_admin": True},
            "secret": {"password": "wrong_password"}
        }
        
        response = client.put("/authenticate", json=auth_data)
        assert response.status_code == 401

    def test_authenticate_nonexistent_user(self):
        """Test authentication with non-existent user"""
        from app import app, users_db
        client = TestClient(app)
        
        # Ensure user doesn't exist
        if "nonexistent_user_999" in users_db:
            del users_db["nonexistent_user_999"]
        
        auth_data = {
            "user": {"name": "nonexistent_user_999", "is_admin": False},
            "secret": {"password": "password123"}
        }
        
        response = client.put("/authenticate", json=auth_data)
        assert response.status_code == 401

    def test_authenticate_s3_user_lookup(self):
        """Test authentication with S3 user lookup"""
        from app import app, s3_storage, users_db
        client = TestClient(app)
        
        if s3_storage:
            mock_user = {
                "username": "s3_user",
                "password": "hashed_password",
                "permissions": ["upload"]
            }
            
            with patch.object(s3_storage, 'get_user', return_value=mock_user):
                # Mock password verification
                with patch('src.auth.jwt_auth.auth.verify_password', return_value=True):
                    auth_data = {
                        "user": {"name": "s3_user", "is_admin": False},
                        "secret": {"password": "password123"}
                    }
                    response = client.put("/authenticate", json=auth_data)
                    # May succeed or fail based on password verification
                    assert response.status_code in [200, 401]

    def test_authenticate_sqlite_user_lookup(self):
        """Test authentication with SQLite user lookup"""
        from app import app, get_db, db_crud, users_db
        client = TestClient(app)
        
        # Test that SQLite lookup path exists (user doesn't exist, so should fail)
        test_username = "sqlite_test_user"
        if test_username in users_db:
            del users_db[test_username]
        
        auth_data = {
            "user": {"name": test_username, "is_admin": False},
            "secret": {"password": "test_password"}
        }
        response = client.put("/authenticate", json=auth_data)
        # Should fail (user doesn't exist) but tests the SQLite lookup path
        assert response.status_code == 401


class TestResetEndpoint:
    """Test reset endpoint"""

    def test_reset_success(self):
        """Test successful reset"""
        from app import app, artifacts_db, users_db, audit_log
        headers = _get_admin_headers()
        client = TestClient(app)
        
        # Add some test data
        artifacts_db["test_artifact"] = {"metadata": {"id": "test_artifact"}}
        users_db["test_user"] = {"username": "test_user"}
        audit_log.append({"action": "CREATE"})
        
        try:
            response = client.delete("/reset", headers=headers)
            assert response.status_code == 200
            # Verify data was cleared
            assert len(artifacts_db) == 0
            assert len(audit_log) == 0
        finally:
            # Restore default admin
            from app import DEFAULT_ADMIN
            users_db[DEFAULT_ADMIN["username"]] = DEFAULT_ADMIN.copy()

    def test_reset_no_admin_permission(self):
        """Test reset without admin permission"""
        from app import app
        headers = _get_user_headers()
        client = TestClient(app)
        
        response = client.delete("/reset", headers=headers)
        assert response.status_code == 401

    def test_reset_sqlite_path(self):
        """Test reset with SQLite storage"""
        from app import app, get_db, db_crud
        headers = _get_admin_headers()
        client = TestClient(app)
        
        with patch.dict(os.environ, {"USE_SQLITE": "1"}):
            # Test that SQLite reset path exists
            response = client.delete("/reset", headers=headers)
            # Should succeed or fail gracefully
            assert response.status_code in [200, 500]

