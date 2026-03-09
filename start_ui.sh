#!/bin/bash

# Kill any existing UI server instances
echo "Killing existing UI server instances..."
PIDs=$(pgrep -f "ui_app" 2>/dev/null)
if [ -n "$PIDs" ]; then
    echo "Found PIDs: $PIDs"
    echo "$PIDs" | xargs kill -9 2>/dev/null
    echo "Killed processes"
else
    echo "No existing processes found"
fi

# Also try to kill any Python processes on port 5004
echo "Checking for processes on port 5004..."
PORT_PID=$(lsof -ti:5004 2>/dev/null)
if [ -n "$PORT_PID" ]; then
    echo "Found process on port 5004: $PORT_PID"
    kill -9 $PORT_PID 2>/dev/null
    echo "Killed process on port 5004"
fi

# Wait a moment for processes to terminate
sleep 2

# Start the UI server
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Starting UI server from: $SCRIPT_DIR"
cd "$SCRIPT_DIR"

# Activate virtual environment and start the server
source "$SCRIPT_DIR/.venv/bin/activate"
python ui_app.py &

# Get the PID of the background process
UI_PID=$!
echo "UI server started with PID: $UI_PID"
echo "Access http://localhost:5004 in your browser"
