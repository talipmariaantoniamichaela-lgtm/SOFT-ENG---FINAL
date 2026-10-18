# VISION AI — Intelligent Video Analysis & Movement Detection System

A production-grade, real-time computer vision web application for face tracking, human pose estimation, gesture recognition, and natural-language action logging.

---

## 🌟 Key Features

- **Real-Time Visual Processing**: OpenCV + MediaPipe integration for 33-point body pose, 468-point face mesh, and 21-point hand tracking.
- **Dual Input Modes**: Seamless switching between live webcam feeds and uploaded local video files (MP4, AVI, MOV, WebM).
- **Temporal Movement Classification**: Stateful action detection with hysteresis and cooldowns:
  - Arm raises (left, right, both)
  - Waving gestures
  - Posture tracking (Standing, Sitting, Bending)
  - Head orientation (Left, Right, Up, Down, Forward)
  - Hand gestures (Thumbs up/down, Open palm, Pointing, Fist)
  - Blinks, mouth open, and eye closures
- **Chronological Terminal Event Log**: Terminal-style real-time event feed with category tags, confidence scores, keyword search, severity filtering, auto-scroll, and JSON/CSV export.
- **Real-Time Performance HUD**: Live FPS counter, processing latency (ms), detection counts, model status badges, and device indicators.
- **Modern Dark UI**: Engineered with React 19, TypeScript, Vite, Tailwind CSS, and Lucide icons.

---

## 🚀 Running the System

Both servers are currently running on your machine:
- **Web App (Frontend)**: [http://localhost:5173](http://localhost:5173)
- **API & WebSocket (Backend)**: [http://localhost:8080](http://localhost:8080)
- **API Documentation (Swagger UI)**: [http://localhost:8080/docs](http://localhost:8080/docs)

### Quick Start Scripts

To start both servers anytime:
```powershell
# From the root directory:
.\scripts\start.ps1
```
or on Windows Command Prompt:
```cmd
.\scripts\start.bat
```

---

## 📁 Project Structure

```
vision-ai/
├── backend/
│   ├── app/
│   │   ├── api/             # REST routes & WebSocket handler
│   │   ├── core/            # App configuration & settings
│   │   ├── database/        # SQLite models & database session
│   │   ├── detectors/       # MediaPipe pose, face & hand detectors
│   │   ├── movement/        # Temporal movement classifier & state machines
│   │   ├── schemas/         # Pydantic request/response schemas
│   │   ├── services/        # Session manager & background pipeline
│   │   └── main.py          # FastAPI application entry point
│   ├── tests/               # Pytest unit & integration tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Header/        # Top navigation & system status
│   │   │   ├── InputSwitcher/ # Camera selector & video drag-drop
│   │   │   ├── VideoPanel/    # Video player, HUD overlays, controls
│   │   │   ├── Terminal/      # Chronological event log & filters
│   │   │   ├── StatusPanel/   # Performance metrics & model status
│   │   │   └── Settings/      # Configuration modal
│   │   ├── hooks/             # Custom WebSocket hook (useAnalysisWS)
│   │   ├── services/          # API client
│   │   ├── types/             # TypeScript type definitions
│   │   ├── App.tsx            # Main application layout
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
└── scripts/
    ├── start.ps1
    └── start.bat
```

---

## 🧪 Testing

Run backend classifier and geometric tests:
```powershell
cd backend
pytest tests/ -v
```
All 10 unit tests cover:
- Arm raise detection
- Standing posture classification
- Duplicate cooldown prevention
- Confidence ranges
- State machine reset
- 3D joint angle calculation
- Wave oscillation detection
