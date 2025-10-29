from __future__ import annotations

from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    username = Column(String, primary_key=True, index=True)
    password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    permissions = Column(String, default="upload,search,download")  # comma-separated
    created_at = Column(DateTime, default=datetime.utcnow)


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    type = Column(String, index=True)
    url = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    audits = relationship("AuditEntry", back_populates="artifact", cascade="all, delete-orphan")


class AuditEntry(Base):
    __tablename__ = "audits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    artifact_id = Column(String, ForeignKey("artifacts.id"), index=True)
    user_name = Column(String, nullable=False)
    user_is_admin = Column(Boolean, default=False)
    action = Column(String, nullable=False)
    date = Column(DateTime, default=datetime.utcnow)

    artifact = relationship("Artifact", back_populates="audits")


