#!/bin/bash
echo "========================================================"
echo "Starting AI Vehicle Speed Tracker Server..."
echo "========================================================"

if [ -f "speed_tracker_env/python.exe" ]; then
    speed_tracker_env/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
elif [ -f "speed_tracker_env/bin/python" ]; then
    speed_tracker_env/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000
elif [ -f "speed_tracker_env/Scripts/python" ]; then
    speed_tracker_env/Scripts/python -m uvicorn main:app --host 0.0.0.0 --port 8000
else
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
fi
