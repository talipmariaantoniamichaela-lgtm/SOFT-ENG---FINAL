"""
FastAPI router: sessions, events, video upload, export, and health.
"""
import asyncio
import csv
import io
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import get_db, DBSession, DBEvent, init_db
from app.schemas.schemas import (
    SessionCreateRequest, SessionResponse, EventsResponse,
    HealthResponse, ModelsResponse, UploadResponse,
)
from app.services.session_manager import session_manager
from app.services.pipeline import AnalysisPipeline

router = APIRouter()

# Active pipelines keyed by session_id
_pipelines: dict[str, AnalysisPipeline] = {}
_start_time = time.time()


# ─── Health ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        device=settings.get_device(),
        models_loaded={"pose": True, "face": True, "hands": True},
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/models", tags=["System"])
async def list_models():
    models_dir = settings.MODELS_DIR
    available = []
    if models_dir.exists():
        for f in models_dir.iterdir():
            available.append({"name": f.name, "size_bytes": f.stat().st_size})
    return {"available": available, "device": settings.get_device()}


@router.get("/settings", tags=["System"])
async def get_settings():
    return {
        "max_upload_mb": settings.MAX_UPLOAD_SIZE_MB,
        "allowed_extensions": list(settings.ALLOWED_VIDEO_EXTENSIONS),
        "default_inference_size": [settings.DEFAULT_INFERENCE_WIDTH, settings.DEFAULT_INFERENCE_HEIGHT],
        "device": settings.get_device(),
    }


# ─── Session ─────────────────────────────────────────────────────────────────

@router.post("/session/start", tags=["Session"])
async def start_session(request: SessionCreateRequest, db: Session = Depends(get_db)):
    cfg = request.config.model_dump()
    session_id = session_manager.create_session(cfg)

    # Persist to DB
    db_session = DBSession(
        id=session_id,
        input_type=cfg["input_type"],
        source_name=cfg.get("source_name", ""),
        configuration=json.dumps(cfg),
        status="active",
        started_at=datetime.utcnow(),
    )
    db.add(db_session)
    db.commit()

    return {"session_id": session_id, "status": "created", "config": cfg}


@router.post("/session/{session_id}/stop", tags=["Session"])
async def stop_session(session_id: str, db: Session = Depends(get_db)):
    # Stop pipeline if running
    pipeline = _pipelines.pop(session_id, None)
    if pipeline:
        pipeline.stop()

    session_manager.end_session(session_id)

    # Update DB
    db_session = db.query(DBSession).filter(DBSession.id == session_id).first()
    if db_session:
        db_session.status = "completed"
        db_session.ended_at = datetime.utcnow()
        db.commit()

    return {"session_id": session_id, "status": "stopped"}


@router.post("/session/{session_id}/reset", tags=["Session"])
async def reset_session(session_id: str):
    pipeline = _pipelines.get(session_id)
    if pipeline:
        pipeline.stop()
        _pipelines.pop(session_id, None)
    session = session_manager.get_session(session_id)
    if session:
        session.events.clear()
        session.event_count = 0
    return {"session_id": session_id, "status": "reset"}


@router.get("/session/{session_id}", tags=["Session"])
async def get_session_info(session_id: str):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    return {
        "session_id": session.session_id,
        "input_type": session.input_type,
        "source_name": session.source_name,
        "status": session.status,
        "event_count": session.event_count,
        "started_at": session.started_at,
        "config": session.config,
    }


@router.get("/sessions", tags=["Session"])
async def list_sessions():
    sessions = session_manager.list_sessions()
    return [
        {
            "session_id": s.session_id,
            "input_type": s.input_type,
            "source_name": s.source_name,
            "status": s.status,
            "event_count": s.event_count,
            "started_at": s.started_at,
        }
        for s in sessions
    ]


@router.delete("/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    pipeline = _pipelines.pop(session_id, None)
    if pipeline:
        pipeline.stop()
    session_manager.delete_session(session_id)
    db.query(DBEvent).filter(DBEvent.session_id == session_id).delete()
    db.query(DBSession).filter(DBSession.id == session_id).delete()
    db.commit()
    return {"session_id": session_id, "status": "deleted"}


# ─── Events ──────────────────────────────────────────────────────────────────

@router.get("/session/{session_id}/events", tags=["Events"])
async def get_events(
    session_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    category: Optional[str] = Query(None),
):
    events, total = session_manager.get_events(session_id, page, page_size, category)
    return {
        "events": events,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


# ─── Export ──────────────────────────────────────────────────────────────────

@router.get("/session/{session_id}/export", tags=["Export"])
async def export_events(
    session_id: str,
    format: str = Query("json", pattern="^(json|csv)$"),
):
    events, total = session_manager.get_events(session_id, 1, 10000)

    if format == "json":
        content = json.dumps({
            "session_id": session_id,
            "exported_at": datetime.utcnow().isoformat(),
            "event_count": total,
            "events": events,
        }, indent=2)
        return StreamingResponse(
            io.StringIO(content),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=session_{session_id[:8]}_events.json"},
        )
    else:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "timestamp_ms", "category", "action", "description",
            "person_id", "confidence", "source", "severity",
        ])
        writer.writeheader()
        for e in events:
            writer.writerow({
                "timestamp_ms": e.get("timestamp_ms", ""),
                "category": e.get("category", ""),
                "action": e.get("action", ""),
                "description": e.get("description", ""),
                "person_id": e.get("person_id", ""),
                "confidence": e.get("confidence", ""),
                "source": e.get("source", ""),
                "severity": e.get("severity", ""),
            })
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=session_{session_id[:8]}_events.csv"},
        )


# ─── Video Upload ─────────────────────────────────────────────────────────────

@router.post("/video/upload", tags=["Video"])
async def upload_video(file: UploadFile = File(...)):
    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. Allowed: {settings.ALLOWED_VIDEO_EXTENSIONS}")

    # Save to uploads dir
    safe_name = f"{uuid.uuid4()}{suffix}"
    dest = settings.UPLOAD_DIR / safe_name

    size = 0
    try:
        async with aiofiles.open(dest, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                    dest.unlink(missing_ok=True)
                    raise HTTPException(413, "File exceeds maximum allowed size.")
                await f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

    # Get video duration using cv2
    duration_s = None
    try:
        import cv2
        cap = cv2.VideoCapture(str(dest))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration_s = round(frames / fps, 2) if frames and fps else None
            cap.release()
    except Exception:
        pass

    return UploadResponse(
        session_id=safe_name,
        filename=file.filename or safe_name,
        size_bytes=size,
        duration_seconds=duration_s,
        message="Upload successful",
    )


# ─── Pipeline control (called by WS handler) ──────────────────────────────────

def get_or_create_pipeline(session_id: str, config: dict, loop: asyncio.AbstractEventLoop) -> AnalysisPipeline:
    """Create and start a pipeline for a session, bridging thread → asyncio."""
    if session_id in _pipelines:
        return _pipelines[session_id]

    def on_event(event: dict):
        session_manager.record_event(session_id, event)
        asyncio.run_coroutine_threadsafe(
            session_manager.broadcast(session_id, event), loop
        )

    pipeline = AnalysisPipeline(session_id=session_id, config=config, event_callback=on_event)
    pipeline.start()
    _pipelines[session_id] = pipeline
    return pipeline


def get_pipeline(session_id: str) -> Optional[AnalysisPipeline]:
    return _pipelines.get(session_id)
