Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Starting AI Vehicle Speed Tracker Server..." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

if (Test-Path "speed_tracker_env\python.exe") {
    & "speed_tracker_env\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
} elseif (Test-Path "speed_tracker_env\bin\python.exe") {
    & "speed_tracker_env\bin\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
} else {
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
}
