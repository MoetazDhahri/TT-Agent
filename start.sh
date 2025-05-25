#!/usr/bin/env bash

# Start script for Tunisie Telecom Q&A application
# This script starts both the backend and frontend servers

# Define colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo -e "${GREEN}=== Tunisie Telecom Q&A App Starter ===${NC}"
echo -e "${BLUE}Starting services...${NC}"

# Check for required commands
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is required but not installed.${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}Error: npm is required but not installed.${NC}"
    exit 1
fi

# Navigate to the project root directory
cd "$(dirname "$0")"

# Start backend server in background
echo -e "${BLUE}Starting Python backend server...${NC}"
cd backend
python3 run_server.py &
BACKEND_PID=$!
cd ..

# Wait a moment to let the backend start
sleep 2

# Install frontend dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}Installing frontend dependencies...${NC}"
    npm install
fi

# Start frontend development server
echo -e "${BLUE}Starting Next.js frontend...${NC}"
npm run dev &
FRONTEND_PID=$!

echo -e "${GREEN}Services started:${NC}"
echo -e "  Backend running at ${BLUE}http://localhost:8000${NC} (PID: $BACKEND_PID)"
echo -e "  Frontend running at ${BLUE}http://localhost:3000${NC} (PID: $FRONTEND_PID)"
echo -e "${GREEN}Press Ctrl+C to stop all services${NC}"

# Handle shutdown
cleanup() {
    echo -e "\n${BLUE}Shutting down services...${NC}"
    kill $FRONTEND_PID 2>/dev/null
    kill $BACKEND_PID 2>/dev/null
    echo -e "${GREEN}Services stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Keep script running until interrupted
wait
