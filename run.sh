#!/bin/bash

# Start both backend and frontend for Tunisie Telecom Q&A Chatbot
echo "Starting Tunisie Telecom Q&A Chatbot"

# Kill any existing processes
echo "Stopping any existing processes..."
pkill -f "python run_server.py" >/dev/null 2>&1
pkill -f "next" >/dev/null 2>&1

# Start the backend server
echo "Starting backend server..."
cd "$(dirname "$0")/backend"
nohup python run_server.py > backend.log 2>&1 &
BACKEND_PID=$!
cd ..

echo "Backend server started with PID: $BACKEND_PID"
echo "Backend logs: tail -f backend/backend.log"

# Wait for backend to initialize and verify it's running
echo "Waiting for backend to initialize..."
max_retries=10
retry_count=0
backend_ready=false

while [ $retry_count -lt $max_retries ]; do
  echo "Checking if backend is ready (attempt $(($retry_count + 1))/$max_retries)..."
  
  if curl -s "http://localhost:8000/health" | grep -q "status.*ok"; then
    backend_ready=true
    echo "Backend is running and healthy!"
    break
  fi
  
  retry_count=$(($retry_count + 1))
  sleep 2
done

if [ "$backend_ready" = false ]; then
  echo "WARNING: Backend health check failed after $max_retries attempts."
  echo "The application may not function correctly."
  echo "Check backend logs with: tail -f backend/backend.log"
fi

# Start the frontend
echo "Starting frontend..."
export NEXT_PUBLIC_API_URL="http://localhost:8000"
npm run dev

# Cleanup function
cleanup() {
  echo "Stopping services..."
  kill $BACKEND_PID 2>/dev/null
  echo "Services stopped"
}

# Register cleanup function
trap cleanup EXIT

# Wait for the frontend to exit
wait
