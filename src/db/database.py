"""SQLite database setup for Milestone 2.

This module initializes the SQLAlchemy engine, session factory, and Base.
It is safe to import even when the database is disabled; the engine will still
point at a local SQLite file by default.

For Lambda deployments, the database file is stored in /tmp (the only writable directory).
For local development, it's stored in the current directory.
"""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session


# Determine database path based on environment
# Lambda: Use /tmp (only writable directory, ~512MB-10GB depending on config)
# Local: Use current directory
if os.environ.get("ENVIRONMENT") == "production" or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    # Lambda environment - use /tmp
    db_path = "/tmp/registry.db"
else:
    # Local development - use current directory
    db_path = "./registry.db"

# Allow override via environment variable
SQLALCHEMY_DATABASE_URL = os.environ.get("SQLALCHEMY_DATABASE_URL", f"sqlite:///{db_path}")

# Ensure /tmp directory exists in Lambda (creates parent directories if needed)
if SQLALCHEMY_DATABASE_URL.startswith("sqlite:///tmp"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

# For SQLite, check_same_thread must be False when used with FastAPI
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args=(
        {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
    ),
    pool_pre_ping=True,  # Verify connections before using them
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
