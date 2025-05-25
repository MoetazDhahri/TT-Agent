"""
Test server to verify backend connectivity
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test API")

@app.get("/")
def root():
    return {"status": "ok", "message": "Test API is running"}

if __name__ == "__main__":
    print("Starting test server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
