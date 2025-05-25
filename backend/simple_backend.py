"""
Simplified backend for Tunisie Telecom Q&A Chatbot
This serves as a compatibility layer between the frontend and full backend
"""

import os
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TT_API")

# Initialize FastAPI app
app = FastAPI(title="Tunisie Telecom Q&A API", 
              description="Ask questions about Tunisie Telecom services")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AskRequest(BaseModel):
    question: str
    language: Optional[str] = "fr"

class AskResponse(BaseModel):
    answer: str
    matched_question: Optional[str] = None
    confidence: Optional[float] = None
    informative: Optional[bool] = None
    source: Optional[str] = None

# Mock responses - you'd replace these with actual backend calls
MOCK_RESPONSES = {
    "default": {
        "answer": "Je suis l'assistant virtuel de Tunisie Telecom. Comment puis-je vous aider?",
        "matched_question": None,
        "confidence": 0.95,
        "informative": True,
        "source": "DEFAULT"
    },
    "recharge": {
        "answer": "Pour recharger votre ligne Tunisie Telecom, composez *123*code_recharge#",
        "matched_question": "Comment recharger ma ligne?",
        "confidence": 0.92,
        "informative": True,
        "source": "LOCAL_HIGH_CONF"
    },
    "forfait": {
        "answer": "Vous pouvez consulter tous nos forfaits sur www.tunisietelecom.tn/particulier/mobile/forfaits/",
        "matched_question": "Quels sont les forfaits disponibles?",
        "confidence": 0.89,
        "informative": True,
        "source": "LOCAL_HIGH_CONF"
    },
    "internet": {
        "answer": "Pour activer un forfait internet, envoyez un SMS au 1144 avec le code correspondant au forfait souhaité.",
        "matched_question": "Comment activer un forfait internet?",
        "confidence": 0.85,
        "informative": True,
        "source": "LOCAL_HIGH_CONF"
    },
    "contact": {
        "answer": "Vous pouvez contacter le service client de Tunisie Telecom au 1200 ou visiter une agence commerciale.",
        "matched_question": "Comment contacter le service client?",
        "confidence": 0.88,
        "informative": True,
        "source": "LOCAL_HIGH_CONF"
    }
}

# Simple in-memory file handling
class SimpleFileHandler:
    def __init__(self):
        os.makedirs("./uploads", exist_ok=True)

    async def process_uploads(self, files: List[UploadFile]):
        if not files:
            return []
            
        results = []
        for file in files:
            results.append({
                "filename": file.filename,
                "content": f"Processed content from {file.filename}",
                "type": self._get_file_type(file.filename),
                "size": 0  # Placeholder
            })
                
        return results
        
    def _get_file_type(self, filename):
        """Get file type from filename"""
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.txt', '.md', '.csv', '.json']:
            return "text"
        elif ext in ['.pdf']:
            return "pdf"
        elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
            return "image"
        else:
            return "unknown"

# Initialize file handler
file_handler = SimpleFileHandler()

# Routes
@app.get("/")
async def root():
    return {"message": "Tunisie Telecom Q&A API is running"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    logger.info(f"Processing question: {req.question}")
    
    # Simple keyword matching for demo
    if "recharge" in req.question.lower():
        response = MOCK_RESPONSES["recharge"]
    elif "forfait" in req.question.lower():
        response = MOCK_RESPONSES["forfait"]
    elif "internet" in req.question.lower():
        response = MOCK_RESPONSES["internet"]
    elif "contact" in req.question.lower() or "service client" in req.question.lower():
        response = MOCK_RESPONSES["contact"]
    else:
        response = MOCK_RESPONSES["default"]
    
    logger.info(f"Returning response with source: {response['source']}")
    return response

@app.post("/ask_with_files")
async def ask_with_files(
    question: str = Form(...),
    language: str = Form("fr"),
    files: List[UploadFile] = File(None)
):
    logger.info(f"Processing question with files: {question}")
    
    try:
        # Process any uploaded files
        processed_files = await file_handler.process_uploads(files)
        file_names = [f["filename"] for f in processed_files]
        
        # Use the same logic as regular ask but acknowledge files
        if "recharge" in question.lower():
            response = MOCK_RESPONSES["recharge"]
        elif "forfait" in question.lower():
            response = MOCK_RESPONSES["forfait"]
        elif "internet" in question.lower():
            response = MOCK_RESPONSES["internet"]
        elif "contact" in question.lower():
            response = MOCK_RESPONSES["contact"]
        else:
            response = MOCK_RESPONSES["default"]
        
        # Acknowledge the files in the response
        if files and len(files) > 0:
            response["answer"] = f"J'ai bien reçu vos fichiers ({', '.join(file_names)}). {response['answer']}"
        
        return {
            **response,
            "processed_files": [{"filename": f["filename"], "type": f["type"]} for f in processed_files]
        }
        
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
