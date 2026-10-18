"""
VISION AI — Core Configuration
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings:
    APP_NAME: str = "VISION AI"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Intelligent Human Movement Analysis System"

    # Database
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'vision_ai.db'}"

    # Upload settings
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_VIDEO_EXTENSIONS: set = {".mp4", ".webm", ".avi", ".mov", ".mkv"}

    # CORS
    CORS_ORIGINS: list = ["http://localhost:5173", "http://127.0.0.1:5173",
                          "http://localhost:3000", "http://127.0.0.1:3000"]

    # Processing
    DEFAULT_INFERENCE_WIDTH: int = 640
    DEFAULT_INFERENCE_HEIGHT: int = 480
    MAX_FRAME_QUEUE_SIZE: int = 5

    # Model paths (relative to project root app/assets/models)
    MODELS_DIR: Path = BASE_DIR.parent.parent / "app" / "assets" / "models"

    # Detection thresholds
    DEFAULT_DETECTION_CONFIDENCE: float = 0.5
    DEFAULT_TRACKING_CONFIDENCE: float = 0.5

    # Event retention
    MAX_EVENTS_PER_SESSION: int = 10000

    # CUDA
    @staticmethod
    def get_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


settings = Settings()

# Ensure upload dir exists
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
