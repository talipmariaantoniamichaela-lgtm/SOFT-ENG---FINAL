# VISION AI - Start both backend and frontend servers
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  Starting VISION AI System Servers      " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

$Root = Split-Path -Parent $PSScriptRoot

Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\backend'; python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\frontend'; npm run dev -- --host 0.0.0.0 --port 5173"

Write-Host "`nServers launched in dedicated windows:" -ForegroundColor Green
Write-Host "  - Frontend: http://localhost:5173" -ForegroundColor Yellow
Write-Host "  - Backend:  http://localhost:8080" -ForegroundColor Yellow
Write-Host "  - API Docs: http://localhost:8080/docs`n" -ForegroundColor Yellow
