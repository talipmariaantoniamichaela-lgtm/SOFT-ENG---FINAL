"""
Motion Reader Web Application Server.
Hosts the web-based Motion Reader application and samples on localhost:8000.
Provides real-time AI computer vision inference endpoints for multi-person tracking,
dataset-based facial recognition, and movement detection.
"""
import os
import sys
import base64
import time
from typing import Optional, Dict, Any, List
from dataclasses import asdict

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.services.multi_person_tracker import MultiPersonTracker

app = FastAPI(title="Motion Reader Web Application")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        if request.url.path.endswith((".html", ".js", ".css")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheMiddleware)

base_dir = os.path.dirname(os.path.abspath(__file__))
web_dir = os.path.join(base_dir, "web")
samples_dir = os.path.join(base_dir, "tests", "samples")

# Initialize Multi-Person AI Tracker
tracker = MultiPersonTracker()


class AnalyzeRequest(BaseModel):
    image: str
    timestamp: Optional[float] = None
    gesture: Optional[Dict[str, Any]] = None


@app.get("/api/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "model_loaded": tracker.subject_engine.is_loaded,
        "classes": tracker.subject_engine.classes,
        "active_tracks": len(tracker.tracks)
    }


@app.post("/api/reset")
def reset_session():
    """Resets the tracking session state."""
    tracker.reset()
    startup_events = tracker.get_initialization_events()
    return {
        "status": "reset",
        "startup_events": [asdict(e) for e in startup_events]
    }


@app.post("/api/analyze")
def analyze_frame(payload: AnalyzeRequest):
    """
    Receives video frame, runs face detection, PyTorch dataset recognition,
    and returns detected individuals and natural language motion events.
    """
    try:
        raw_b64 = payload.image
        prefix = "base64,"
        idx = raw_b64.find(prefix)
        if idx != -1:
            raw_b64 = raw_b64[idx + len(prefix):]

        img_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            raise ValueError("Failed to decode image from payload")

        # Handle pending gesture event
        gesture_events = []
        if payload.gesture:
            g_type = payload.gesture.get("type")
            g_conf = payload.gesture.get("confidence", 0.92)
            p_id = payload.gesture.get("person_id", 1)
            g_evt = tracker.trigger_hand_gesture(
                person_id=p_id,
                gesture_type=g_type,
                confidence=g_conf,
                current_time=payload.timestamp
            )
            if g_evt:
                gesture_events.append(g_evt)

        # Process frame through multi-person tracker
        persons, events = tracker.process_frame(frame, current_time=payload.timestamp)

        all_events = [asdict(e) for e in (gesture_events + events)]

        return {
            "status": "ok",
            "persons": persons,
            "events": all_events
        }
    except Exception as e:
        print(f"Error in /api/analyze: {e}")
        return {
            "status": "error",
            "message": str(e),
            "persons": [],
            "events": []
        }


if os.path.exists(samples_dir):
    app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")

# Mount web directory (MUST be mounted after API routes)
app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")


def run():
    print("=======================================================")
    print(" Motion Reader Web Application is running at:")
    print(" http://localhost:8000")
    print(" http://127.0.0.1:8000")
    print("=======================================================")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    run()
