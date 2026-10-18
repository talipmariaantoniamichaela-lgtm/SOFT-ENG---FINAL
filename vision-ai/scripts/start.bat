@echo off
echo =========================================
echo   Starting VISION AI System Servers
echo =========================================

set ROOT=%~dp0..

start "VISION AI Backend" cmd /k "cd /d %ROOT%\backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload"
start "VISION AI Frontend" cmd /k "cd /d %ROOT%\frontend && npm run dev -- --host 0.0.0.0 --port 5173"

echo.
echo Servers started:
echo   - Frontend: http://localhost:5173
echo   - Backend:  http://localhost:8080
echo   - API Docs: http://localhost:8080/docs
echo.
