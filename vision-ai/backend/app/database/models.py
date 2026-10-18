"""
Database models and session management for VISION AI.
Uses SQLAlchemy with SQLite.
"""
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from app.core.config import settings

Base = declarative_base()
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class DBSession(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    input_type = Column(String, nullable=False)   # "live_camera" | "uploaded_video"
    source_name = Column(String, default="")
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    status = Column(String, default="active")     # "active" | "completed" | "error"
    configuration = Column(Text, default="{}")    # JSON blob


class DBEvent(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    timestamp_ms = Column(Integer, default=0)
    person_id = Column(String, nullable=True)
    category = Column(String, nullable=False)    # SYSTEM | DETECT | FACE | ACTION | POSTURE | GESTURE | HAND
    action = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    source = Column(String, default="pose_classifier")
    severity = Column(String, default="info")    # info | warning | alert
    created_at = Column(DateTime, default=datetime.utcnow)


class DBAnalysisSummary(Base):
    __tablename__ = "analysis_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False, index=True)
    summary_text = Column(Text, nullable=False)
    model_name = Column(String, default="rule_based")
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db_ctx():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db():
    """FastAPI dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
