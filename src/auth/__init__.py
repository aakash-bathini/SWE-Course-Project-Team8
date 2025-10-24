"""
Authentication module initialization
"""

from .jwt_auth import (
    JWTAuth,
    auth,
    create_demo_token,
    verify_demo_token,
    authenticate_user,
    create_user_token,
    check_user_permission,
)

__all__ = [
    "JWTAuth",
    "auth",
    "create_demo_token",
    "verify_demo_token",
    "authenticate_user",
    "create_user_token",
    "check_user_permission",
]
