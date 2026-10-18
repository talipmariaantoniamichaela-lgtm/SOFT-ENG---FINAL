"""
VISION AI — FastAPI Application Entry Point
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.database.models import init_db
from app.api.routes import router
from app.api.websocket import ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("=" * 55)
    print("  VISION AI — Intelligent Human Movement Analysis")
    print(f"  Version  : {settings.APP_VERSION}")
    print(f"  Device   : {settings.get_device()}")
    print(f"  Uploads  : {settings.UPLOAD_DIR}")
    print("=" * 55)
    init_db()
    yield
    # Shutdown
    print("[VISION AI] Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST API
app.include_router(router, prefix="/api")

# WebSocket
app.include_router(ws_router)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )
