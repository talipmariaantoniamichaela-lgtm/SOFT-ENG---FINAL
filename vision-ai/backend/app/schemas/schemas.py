"""
Pydantic schemas for request/response validation in VISION AI.
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class InputType(str, Enum):
    live_camera = "live_camera"
    uploaded_video = "uploaded_video"


class SessionStatus(str, Enum):
    active = "active"
    completed = "completed"
    error = "error"


class ProcessingMode(str, Enum):
    performance = "performance"
    balanced = "balanced"
    accuracy = "accuracy"


# ─── Session Schemas ────────────────────────────────────────────────────────

class SessionConfig(BaseModel):
    input_type: InputType = InputType.live_camera
    source_name: str = ""
    processing_mode: ProcessingMode = ProcessingMode.balanced
    enable_face: bool = True
    enable_pose: bool = True
    enable_hands: bool = True
    enable_ai_descriptions: bool = False
    camera_index: int = 0
    inference_width: int = 640
    inference_height: int = 480


class SessionCreateRequest(BaseModel):
    config: SessionConfig = Field(default_factory=SessionConfig)


class SessionResponse(BaseModel):
    id: str
    input_type: str
    source_name: str
    started_at: datetime
    ended_at: Optional[datetime]
    status: str
    configuration: str

    class Config:
        from_attributes = True


# ─── Event Schemas ───────────────────────────────────────────────────────────

class EventCategory(str, Enum):
    SYSTEM = "SYSTEM"
    DETECT = "DETECT"
    FACE = "FACE"
    ACTION = "ACTION"
    POSTURE = "POSTURE"
    GESTURE = "GESTURE"
    HAND = "HAND"
    AI = "AI"
    ERROR = "ERROR"


class EventCreate(BaseModel):
    session_id: str
    timestamp_ms: int = 0
    person_id: Optional[str] = None
    category: EventCategory = EventCategory.SYSTEM
    action: str
    description: str
    confidence: Optional[float] = None
    source: str = "pose_classifier"
    severity: str = "info"


class EventResponse(BaseModel):
    id: int
    session_id: str
    timestamp_ms: int
    person_id: Optional[str]
    category: str
    action: str
    description: str
    confidence: Optional[float]
    source: str
    severity: str
    created_at: datetime

    class Config:
        from_attributes = True


class EventsResponse(BaseModel):
    events: List[EventResponse]
    total: int
    page: int
    page_size: int


# ─── WebSocket Message Schemas ───────────────────────────────────────────────

class WSEventType(str, Enum):
    movement_event = "movement_event"
    frame_stats = "frame_stats"
    session_status = "session_status"
    error = "error"
    ping = "ping"
    pong = "pong"


class WSFrameStats(BaseModel):
    type: str = "frame_stats"
    session_id: str
    fps: float
    latency_ms: float
    person_count: int
    face_count: int
    hand_count: int
    processing_mode: str
    device: str
    timestamp_ms: int


class WSMovementEvent(BaseModel):
    type: str = "movement_event"
    session_id: str
    timestamp_ms: int
    person_id: Optional[str] = None
    category: str
    action: str
    description: str
    confidence: Optional[float] = None
    source: str = "pose_classifier"
    severity: str = "info"


class WSSessionStatus(BaseModel):
    type: str = "session_status"
    session_id: str
    status: str
    message: str


# ─── API Response wrappers ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    device: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float


class ModelsResponse(BaseModel):
    available: List[Dict[str, Any]]


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    size_bytes: int
    duration_seconds: Optional[float]
    message: str


class ExportResponse(BaseModel):
    session_id: str
    format: str
    event_count: int
    data: Any
