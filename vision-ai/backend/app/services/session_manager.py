"""
Session manager for VISION AI.
Manages analysis sessions, event accumulation, and WebSocket broadcasting.
"""
import asyncio
import json
import uuid
import time
from datetime import datetime
from typing import Dict, Set, Optional, List, Any
from dataclasses import dataclass, field

from fastapi import WebSocket

from app.schemas.schemas import WSMovementEvent, WSFrameStats, WSSessionStatus


@dataclass
class AnalysisSession:
    session_id: str
    input_type: str
    source_name: str
    config: Dict[str, Any]
    started_at: float = field(default_factory=time.time)
    status: str = "active"
    event_count: int = 0
    events: List[Dict] = field(default_factory=list)   # in-memory buffer


class SessionManager:
    """
    Manages active analysis sessions and WebSocket connections.
    Handles event broadcasting to all connected clients for a session.
    """

    def __init__(self):
        self._sessions: Dict[str, AnalysisSession] = {}
        self._connections: Dict[str, Set[WebSocket]] = {}   # session_id -> websockets
        self._lock = asyncio.Lock()

    def create_session(self, config: Dict[str, Any]) -> str:
        session_id = str(uuid.uuid4())
        session = AnalysisSession(
            session_id=session_id,
            input_type=config.get("input_type", "live_camera"),
            source_name=config.get("source_name", ""),
            config=config,
        )
        self._sessions[session_id] = session
        self._connections[session_id] = set()
        return session_id

    def get_session(self, session_id: str) -> Optional[AnalysisSession]:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session:
            session.status = "completed"
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._connections.pop(session_id, None)
            return True
        return False

    def list_sessions(self) -> List[AnalysisSession]:
        return list(self._sessions.values())

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        if session_id not in self._connections:
            self._connections[session_id] = set()
        self._connections[session_id].add(websocket)

    def disconnect(self, session_id: str, websocket: WebSocket):
        if session_id in self._connections:
            self._connections[session_id].discard(websocket)

    async def broadcast(self, session_id: str, message: Dict):
        """Send a message to all WebSocket clients for this session."""
        conns = self._connections.get(session_id, set()).copy()
        dead = set()
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections[session_id].discard(ws)

    def record_event(self, session_id: str, event: Dict):
        """Store event in memory and trim to max 10k."""
        session = self._sessions.get(session_id)
        if session:
            session.events.append(event)
            session.event_count += 1
            if len(session.events) > 10000:
                session.events = session.events[-10000:]

    def get_events(
        self,
        session_id: str,
        page: int = 1,
        page_size: int = 100,
        category: Optional[str] = None,
    ) -> tuple:
        session = self._sessions.get(session_id)
        if not session:
            return [], 0
        events = session.events
        if category:
            events = [e for e in events if e.get("category") == category.upper()]
        total = len(events)
        start = (page - 1) * page_size
        end = start + page_size
        return events[start:end], total


# Singleton
session_manager = SessionManager()
