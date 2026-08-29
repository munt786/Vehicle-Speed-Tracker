@echo off
echo ========================================================
echo Starting AI Vehicle Speed Tracker Server...
echo ========================================================
if exist "speed_tracker_env\python.exe" (
    "speed_tracker_env\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
) else if exist "speed_tracker_env\bin\python.exe" (
    "speed_tracker_env\bin\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
) else (
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
)
pause
