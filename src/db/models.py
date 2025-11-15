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


class JSProgram(Base):
    """JavaScript monitoring program for sensitive models (M5.1)"""

    __tablename__ = "js_programs"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    code = Column(Text, nullable=False)
    created_by = Column(String, ForeignKey("users.username"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sensitive_models = relationship("SensitiveModel", back_populates="js_program")


class SensitiveModel(Base):
    """Sensitive model with optional JavaScript monitoring program (M5.1)"""

    __tablename__ = "sensitive_models"

    id = Column(String, primary_key=True, index=True)
    model_id = Column(String, ForeignKey("artifacts.id"), nullable=False, index=True)
    uploader_username = Column(String, ForeignKey("users.username"), nullable=False)
    js_program_id = Column(String, ForeignKey("js_programs.id"), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    js_program = relationship("JSProgram", back_populates="sensitive_models")
    download_history = relationship(
        "DownloadHistory", back_populates="sensitive_model", cascade="all, delete-orphan"
    )


class DownloadHistory(Base):
    """Download audit trail for sensitive models (M5.1)"""

    __tablename__ = "download_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sensitive_model_id = Column(
        String, ForeignKey("sensitive_models.id"), nullable=False, index=True
    )
    downloader_username = Column(String, ForeignKey("users.username"), nullable=False)
    downloaded_at = Column(DateTime, default=datetime.utcnow)
    js_exit_code = Column(Integer, nullable=True)
    js_stdout = Column(Text, nullable=True)
    js_stderr = Column(Text, nullable=True)

    sensitive_model = relationship("SensitiveModel", back_populates="download_history")


class SearchHistory(Base):
    """Search hit tracking for package confusion detection (M5.2)"""

    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    artifact_id = Column(String, ForeignKey("artifacts.id"), nullable=False, index=True)
    searched_at = Column(DateTime, default=datetime.utcnow, index=True)
    search_type = Column(String, nullable=False)  # 'byName' or 'byRegEx'
