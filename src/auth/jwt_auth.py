"""
JWT Authentication Module for Phase 2
Handles token generation, validation, and user authentication
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import logging
import os

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 10
ACCESS_TOKEN_EXPIRE_CALLS = 1000

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class JWTAuth:
    """JWT Authentication handler"""

    def __init__(self, secret_key: str = SECRET_KEY, algorithm: str = ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        result = pwd_context.verify(plain_password, hashed_password)
        return bool(result)

    def get_password_hash(self, password: str) -> str:
        """Hash a password"""
        # Bcrypt has a 72-byte limit, so truncate if necessary
        # This is safe because bcrypt truncates internally anyway
        if len(password.encode('utf-8')) > 72:
            password = password[:72]
        result = pwd_context.hash(password)
        return str(result)

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

        to_encode.update(
            {
                "exp": expire,
                "iat": datetime.utcnow(),
                "call_count": 0,
                "max_calls": ACCESS_TOKEN_EXPIRE_CALLS,
            }
        )

        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Token created for user: {data.get('sub', 'unknown')}")
        return str(encoded_jwt)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Check expiration
            exp = payload.get("exp")
            if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
                logger.warning("Token expired")
                return None

            # Check call count
            call_count = payload.get("call_count", 0)
            max_calls = payload.get("max_calls", ACCESS_TOKEN_EXPIRE_CALLS)
            if call_count >= max_calls:
                logger.warning("Token exceeded maximum calls")
                return None

            return dict(payload)

        except JWTError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            return None

    def increment_token_calls(self, token: str) -> Optional[str]:
        """Increment the call count for a token and return updated token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            # Increment call count
            payload["call_count"] = payload.get("call_count", 0) + 1

            # Check if token should expire due to call count
            if payload["call_count"] >= payload.get("max_calls", ACCESS_TOKEN_EXPIRE_CALLS):
                logger.info("Token expired due to call count limit")
                return None

            # Re-encode token with updated call count
            updated_token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return str(updated_token)

        except JWTError as e:
            logger.error(f"Token update failed: {str(e)}")
            return None


# Global auth instance
auth = JWTAuth()


# Demo functions for Milestone 1
def create_demo_token(username: str, permissions: list[str]) -> str:
    """Create a demo token for Milestone 1"""
    data = {"sub": username, "permissions": permissions, "type": "demo"}
    return auth.create_access_token(data)


def verify_demo_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify a demo token for Milestone 1"""
    return auth.verify_token(token)


# User management functions
def authenticate_user(
    username: str, password: str, users_db: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Authenticate a user against the database"""
    if username not in users_db:
        return None

    user = users_db[username]

    # For Milestone 1, we'll use plain text passwords
    # In Milestone 3, we'll use proper bcrypt hashing
    if user["password"] != password:
        return None

    return dict(user)


def create_user_token(user: Dict[str, Any]) -> str:
    """Create a token for an authenticated user"""
    data = {
        "sub": user["username"],
        "permissions": user.get("permissions", []),
        "user_id": user["username"],
    }
    return auth.create_access_token(data)


# Permission checking
def check_user_permission(user_data: Dict[str, Any], required_permission: str) -> bool:
    """Check if user has required permission"""
    permissions = user_data.get("permissions", [])
    return required_permission in permissions


def require_permission(required_permission: str) -> Any:
    """Decorator to require specific permission"""

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # This will be implemented in Milestone 3 with proper dependency injection
            pass

        return wrapper

    return decorator
