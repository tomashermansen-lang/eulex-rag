#!/bin/bash
# Start both backend and frontend for development

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$SCRIPT_DIR/.server_pids"

# Ensure homebrew and common paths are available
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Activate Python virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Kill any previously started servers from this script
if [ -f "$PID_FILE" ]; then
    echo -e "${YELLOW}Stopping previous servers...${NC}"
    while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            kill $pid 2>/dev/null && echo -e "  Killed process $pid"
        fi
    done < "$PID_FILE"
    rm -f "$PID_FILE"
    sleep 1
fi

# Default ports
BACKEND_PORT=8000
FRONTEND_PORT=5173

echo -e "${GREEN}Starting EuLex React UI...${NC}"

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed.${NC}"
    echo -e "${YELLOW}Install with: brew install node${NC}"
    exit 1
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm is not installed.${NC}"
    echo -e "${YELLOW}Install with: brew install node${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Node.js $(node --version)${NC}"
echo -e "${GREEN}✓ npm $(npm --version)${NC}"

# Check Python dependencies (FastAPI, uvicorn)
if ! python -c "import fastapi, uvicorn" &> /dev/null; then
    echo -e "${YELLOW}Installing FastAPI dependencies...${NC}"
    pip install fastapi uvicorn[standard] pydantic python-multipart
fi

# Check if node_modules exists
if [ ! -d "$SCRIPT_DIR/frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd "$SCRIPT_DIR/frontend" && npm install
fi

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    rm -f "$PID_FILE"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Start backend
echo -e "${GREEN}Starting backend on port $BACKEND_PORT...${NC}"
cd "$SCRIPT_DIR/backend" && uvicorn main:app --reload --port $BACKEND_PORT --reload-dir "$PROJECT_ROOT/src" --reload-dir "$SCRIPT_DIR/backend" &
BACKEND_PID=$!

# Wait for backend to be ready
sleep 2

# Start frontend
echo -e "${GREEN}Starting frontend on port $FRONTEND_PORT...${NC}"
cd "$SCRIPT_DIR/frontend" && VITE_API_PORT=$BACKEND_PORT npm run dev -- --port $FRONTEND_PORT &
FRONTEND_PID=$!

# Save PIDs for cleanup on next run
echo "$BACKEND_PID" > "$PID_FILE"
echo "$FRONTEND_PID" >> "$PID_FILE"

echo -e "\n${GREEN}Services started:${NC}"
echo -e "  Backend API: http://localhost:$BACKEND_PORT"
echo -e "  Frontend:    http://localhost:$FRONTEND_PORT"
echo -e "  API Docs:    http://localhost:$BACKEND_PORT/api/docs"
echo -e "\nPress Ctrl+C to stop all services\n"

# Wait for processes
wait
