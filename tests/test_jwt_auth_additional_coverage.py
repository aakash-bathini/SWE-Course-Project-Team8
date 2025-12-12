import os
import sys
from datetime import timedelta

import pytest
from jose import jwt


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.auth.jwt_auth import JWTAuth  # noqa: E402


def test_verify_password_fallback_hash_success_and_failure() -> None:
    auth = JWTAuth(secret_key="k")

    plain = "pw"
    salt = "abc123"
    import hashlib

    expected_hash = hashlib.sha256((plain + salt).encode()).hexdigest()[:40]
    fallback_hash = f"$2b$12$test_fallback_{salt}_{expected_hash}"
    assert auth.verify_password(plain, fallback_hash) is True

    # malformed fallback hash
    assert auth.verify_password(plain, "$2b$12$test_fallback_") is False


def test_create_and_verify_token_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_ISSUER", "iss")
    monkeypatch.setenv("JWT_AUDIENCE", "aud")

    auth = JWTAuth(secret_key="secret")
    token = auth.create_access_token({"sub": "u", "permissions": ["read"]}, expires_delta=timedelta(minutes=5))
    payload = auth.verify_token(token)

    assert payload is not None
    assert payload["sub"] == "u"
    assert payload["iss"] == "iss"
    assert payload["aud"] == "aud"


def test_verify_token_rejects_issuer_and_audience_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_ISSUER", "expected-iss")
    monkeypatch.setenv("JWT_AUDIENCE", "expected-aud")

    secret = "secret"
    # bad issuer
    token_bad_iss = jwt.encode(
        {
            "sub": "u",
            "exp": 32503680000,  # year 3000
            "iat": 0,
            "iss": "wrong-iss",
            "aud": "expected-aud",
            "call_count": 0,
            "max_calls": 1000,
        },
        secret,
        algorithm="HS256",
    )

    auth = JWTAuth(secret_key=secret)
    assert auth.verify_token(token_bad_iss) is None

    # bad audience as list
    token_bad_aud = jwt.encode(
        {
            "sub": "u",
            "exp": 32503680000,
            "iat": 0,
            "iss": "expected-iss",
            "aud": ["other-aud"],
            "call_count": 0,
            "max_calls": 1000,
        },
        secret,
        algorithm="HS256",
    )
    assert auth.verify_token(token_bad_aud) is None


def test_verify_token_rejects_call_count_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_ISSUER", "iss")
    monkeypatch.setenv("JWT_AUDIENCE", "aud")

    secret = "secret"
    token = jwt.encode(
        {
            "sub": "u",
            "exp": 32503680000,
            "iat": 0,
            "iss": "iss",
            "aud": "aud",
            "call_count": 5,
            "max_calls": 5,
        },
        secret,
        algorithm="HS256",
    )

    auth = JWTAuth(secret_key=secret)
    assert auth.verify_token(token) is None


def test_increment_token_calls_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_ISSUER", "iss")
    monkeypatch.setenv("JWT_AUDIENCE", "aud")

    auth = JWTAuth(secret_key="secret")
    token = jwt.encode(
        {
            "sub": "u",
            "exp": 32503680000,
            "iat": 0,
            "iss": "iss",
            "aud": "aud",
            "call_count": 0,
            "max_calls": 1,
        },
        "secret",
        algorithm="HS256",
    )

    assert auth.increment_token_calls(token) is None
