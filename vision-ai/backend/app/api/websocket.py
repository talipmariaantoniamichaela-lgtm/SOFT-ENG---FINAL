"""
WebSocket handler for VISION AI real-time analysis.
"""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.session_manager import session_manager
from app.api.routes import get_or_create_pipeline, get_pipeline

ws_router = APIRouter()


@ws_router.websocket("/ws/analysis/{session_id}")
async def analysis_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time analysis events.

    Expected client messages:
      {"cmd": "start", "config": {...}}
      {"cmd": "stop"}
      {"cmd": "pause"}
      {"cmd": "resume"}
      {"cmd": "seek", "progress": 0.0-1.0}
      {"cmd": "ping"}
    """
    session = session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return

    await session_manager.connect(session_id, websocket)
    loop = asyncio.get_event_loop()

    try:
        # Send welcome
        await websocket.send_json({
            "type": "session_status",
            "session_id": session_id,
            "status": "connected",
            "message": "WebSocket connected. Send {\"cmd\": \"start\"} to begin analysis.",
        })

        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Keep-alive ping
                await websocket.send_json({"type": "ping", "session_id": session_id})
                continue

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                continue

            cmd = msg.get("cmd", "")

            if cmd == "ping":
                await websocket.send_json({"type": "pong", "session_id": session_id})

            elif cmd == "start":
                config = session.config.copy()
                config.update(msg.get("config", {}))
                pipeline = get_or_create_pipeline(session_id, config, loop)
                await websocket.send_json({
                    "type": "session_status",
                    "session_id": session_id,
                    "status": "running",
                    "message": "Analysis pipeline started.",
                    "models_loaded": pipeline.models_loaded,
                })

            elif cmd == "stop":
                pipeline = get_pipeline(session_id)
                if pipeline:
                    pipeline.stop()
                await websocket.send_json({
                    "type": "session_status",
                    "session_id": session_id,
                    "status": "stopped",
                    "message": "Analysis stopped.",
                })

            elif cmd == "pause":
                pipeline = get_pipeline(session_id)
                if pipeline:
                    pipeline.pause()
                await websocket.send_json({
                    "type": "session_status",
                    "session_id": session_id,
                    "status": "paused",
                    "message": "Analysis paused.",
                })

            elif cmd == "resume":
                pipeline = get_pipeline(session_id)
                if pipeline:
                    pipeline.resume()
                await websocket.send_json({
                    "type": "session_status",
                    "session_id": session_id,
                    "status": "running",
                    "message": "Analysis resumed.",
                })

            elif cmd == "seek":
                progress = float(msg.get("progress", 0.0))
                pipeline = get_pipeline(session_id)
                if pipeline:
                    pipeline.seek(progress)

            else:
                await websocket.send_json({
                    "type": "error",
                    "session_id": session_id,
                    "message": f"Unknown command: {cmd}",
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        session_manager.disconnect(session_id, websocket)
