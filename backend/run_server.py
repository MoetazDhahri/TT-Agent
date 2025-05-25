#!/usr/bin/env python3
import os
import sys
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TT_Backend")

# Make sure the required directories exist
os.makedirs("./model", exist_ok=True)
os.makedirs("./cache", exist_ok=True)
os.makedirs("./uploads", exist_ok=True)

# Import the FastAPI app, fallback to simple_backend if main backend fails
try:
    # First try the simple backend which is guaranteed to work
    from simple_backend import app
    logger.info("Using simple_backend.py for API endpoints")
except Exception as e:
    logger.error(f"Failed to import app from simple_backend.py: {str(e)}")
    try:
        # Try the full backend if simple one fails
        from backend import app
        logger.info("Successfully imported the FastAPI app from backend.py")
    except Exception as e:
        logger.error(f"Failed to import app from backend.py: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting Tunisie Telecom Q&A Backend Server...")
    print("API will be available at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)
