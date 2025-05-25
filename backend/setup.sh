#!/usr/bin/env bash

# Setup script for Tunisie Telecom Q&A Backend

# Create required directories
echo "Creating necessary directories..."
mkdir -p ./backend/cache
mkdir -p ./backend/model
mkdir -p ./backend/uploads

# Check if Python is installed
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Install requirements
echo "Installing Python dependencies..."
cd backend
python3 -m pip install -r requirements.txt

# Set up environment
echo "Setting up environment variables..."
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Tunisie Telecom Backend Configuration
CSV_FILE=File.csv
MODEL_DIR=model
CACHE_DIR=cache
USE_CACHE=true
ENABLE_WEB_SEARCH=true
MAX_WEB_RESULTS=3
PORT=8000
EOF
fi

# Check if model files exist
echo "Checking model files..."
if [ ! -f "./model/qna_model.joblib" ] || [ ! -f "./model/tfidf_vectorizer.joblib" ]; then
    echo "Warning: Model files not found. Please make sure to place them in the model directory."
fi

# Check data file
if [ ! -f "File.csv" ]; then
    echo "Warning: File.csv not found. The Q&A functionality may be limited."
fi

echo "Setup completed!"
echo "To start the server, run: python3 run_server.py"
