from __future__ import annotations

import re
from typing import List, Optional, Dict, cast
from sqlalchemy.orm import Session

from . import models


def ensure_schema(db: Session) -> None:
    from .database import Base, engine

    Base.metadata.create_all(bind=engine)


def upsert_default_admin(db: Session, username: str, password: str, permissions: List[str]) -> None:
    user = db.get(models.User, username)
    if not user:
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
        results.extend(query.all())
        if name == "*" and not types:
            results = db.query(models.Artifact).all()
    return results


def list_by_name(db: Session, name: str) -> List[models.Artifact]:
    return db.query(models.Artifact).filter(models.Artifact.name == name).all()


def list_by_regex(db: Session, regex: str) -> List[models.Artifact]:
    # simple in-Python filter for SQLite compatibility
    rx = re.compile(regex)
    items = db.query(models.Artifact).all()
    out: List[models.Artifact] = []
    for a in items:
        text = cast(str, a.name) if getattr(a, "name", None) is not None else ""
        if rx.search(text):
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
    db.query(models.AuditEntry).delete()
    db.query(models.Artifact).delete()
    db.commit()
