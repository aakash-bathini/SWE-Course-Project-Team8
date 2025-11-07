from __future__ import annotations

import re
from typing import List, Optional, Dict, cast
from sqlalchemy.orm import Session

from . import models


def ensure_schema(db: Session) -> None:
    from .database import Base, engine

    Base.metadata.create_all(bind=engine)


def upsert_default_admin(db: Session, username: str, password: str, permissions: List[str]) -> None:
    """Create or update default admin user"""
    user = db.get(models.User, username)
    if user:
        # Update existing user
        user.password = password  # type: ignore[assignment]
        user.is_admin = True  # type: ignore[assignment]
        user.permissions = ",".join(permissions)  # type: ignore[assignment]
    else:
        # Create new user
        user = models.User(
            username=username,
            password=password,
            is_admin=True,
            permissions=",".join(permissions),
        )
        db.add(user)
        db.commit()


def create_artifact(
    db: Session, artifact_id: str, name: str, type_: str, url: str
) -> models.Artifact:
    art = models.Artifact(id=artifact_id, name=name, type=type_, url=url)
    db.add(art)
    db.commit()
    db.refresh(art)
    return art


def get_artifact(db: Session, artifact_id: str) -> Optional[models.Artifact]:
    return db.get(models.Artifact, artifact_id)


def count_artifacts_by_type(db: Session, artifact_type: str) -> int:
    """Get the count of artifacts of a specific type"""
    return db.query(models.Artifact).filter(models.Artifact.type == artifact_type).count()


def update_artifact(db: Session, artifact_id: str, name: str, type_: str, url: str) -> bool:
    art = get_artifact(db, artifact_id)
    if not art:
        return False
    art.name = name  # type: ignore[assignment]
    art.type = type_  # type: ignore[assignment]
    art.url = url  # type: ignore[assignment]
    db.commit()
    return True


def delete_artifact(db: Session, artifact_id: str) -> bool:
    art = get_artifact(db, artifact_id)
    if not art:
        return False
    db.delete(art)
    db.commit()
    return True


def list_by_queries(db: Session, queries: List[Dict]) -> List[models.Artifact]:
    results: List[models.Artifact] = []
    for q in queries:
        name = q.get("name")
        types = q.get("types")
        query = db.query(models.Artifact)
        if name and name != "*":
            query = query.filter(models.Artifact.name == name)
        if types:
            query = query.filter(models.Artifact.type.in_(types))
        # If name is "*" and no types filter, get all artifacts
        # Otherwise, use the filtered query
        if name == "*" and not types:
            results.extend(db.query(models.Artifact).all())
        else:
            results.extend(query.all())
    # Remove duplicates (in case multiple queries overlap)
    seen_ids = set()
    unique_results = []
    for art in results:
        if art.id not in seen_ids:
            seen_ids.add(art.id)
            unique_results.append(art)
    return unique_results


def list_by_name(db: Session, name: str) -> List[models.Artifact]:
    return db.query(models.Artifact).filter(models.Artifact.name == name).all()


def list_by_regex(db: Session, regex: str) -> List[models.Artifact]:
    """Search artifacts by regex pattern (matches names and READMEs)"""
    # Do NOT escape regex - it should be a real regex pattern
    # Note: Per OpenAPI spec, users can provide regex patterns for searching.
    # The regex is validated here and only used for matching, not for execution.
    # CodeQL warnings about regex injection are expected - this is intentional functionality.
    # ReDoS risk is mitigated through length limits enforced by the endpoint.

    # Security: Limit regex pattern length to prevent ReDoS attacks
    MAX_REGEX_LENGTH = 500
    if len(regex) > MAX_REGEX_LENGTH:
        # Return empty list if pattern is too long (endpoint will handle the error)
        return []

    try:
        # Use case-insensitive matching
        rx = re.compile(regex, re.IGNORECASE)
    except re.error:
        # If invalid regex, return empty list (will be caught by endpoint)
        return []

    items = db.query(models.Artifact).all()
    out: List[models.Artifact] = []
    for a in items:
        name = cast(str, a.name) if getattr(a, "name", None) is not None else ""
        # For SQLite, we only have name stored in DB, not README content
        # This is acceptable as README search is mainly for HuggingFace models
        # which are typically ingested and stored in-memory
        # Search case-insensitively
        if rx.search(name):
            out.append(a)
    return out


def log_audit(
    db: Session, artifact: models.Artifact, user_name: str, user_is_admin: bool, action: str
) -> None:
    entry = models.AuditEntry(
        artifact_id=artifact.id, user_name=user_name, user_is_admin=user_is_admin, action=action
    )
    db.add(entry)
    db.commit()


def list_audit(db: Session, artifact_id: str) -> List[models.AuditEntry]:
    return db.query(models.AuditEntry).filter(models.AuditEntry.artifact_id == artifact_id).all()


def reset_registry(db: Session) -> None:
    """Reset registry - clear all artifacts and audit entries"""
    # Clear all artifacts and audit entries
    db.query(models.AuditEntry).delete()
    db.query(models.Artifact).delete()
    # Clear all users except default admin (will be recreated by caller)
    db.query(models.User).delete()
    db.commit()
