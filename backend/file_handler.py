"""
File Upload Handler for TT Agent Backend
"""
import os
from fastapi import UploadFile, File, HTTPException
from typing import List
import logging

logger = logging.getLogger("TT_QA")

class FileUploadHandler:
    def __init__(self, upload_dir="./uploads"):
        """Initialize file upload handler"""
        self.upload_dir = upload_dir
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            
    async def process_uploads(self, files: List[UploadFile] = None):
        """Process uploaded files and extract their content"""
        if not files:
            return []
            
        results = []
        for file in files:
            try:
                # Save the file temporarily
                file_path = os.path.join(self.upload_dir, file.filename)
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                # Extract content based on file type
                file_type = self._get_file_type(file.filename)
                extracted_text = self._extract_text(file_path, file_type)
                
                results.append({
                    "filename": file.filename,
                    "content": extracted_text,
                    "type": file_type,
                    "size": os.path.getsize(file_path)
                })
                
                # Clean up temporary file
                os.remove(file_path)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                results.append({
                    "filename": file.filename,
                    "error": str(e)
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
        elif ext in ['.doc', '.docx']:
            return "word"
        elif ext in ['.xls', '.xlsx']:
            return "excel"
        elif ext in ['.ppt', '.pptx']:
            return "powerpoint"
        else:
            return "unknown"
            
    def _extract_text(self, file_path, file_type):
        """Extract text from file based on its type"""
        try:
            if file_type == "text":
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            elif file_type == "pdf":
                # In a production app, you'd use PyPDF2 or similar here
                return "PDF text extraction not implemented"
            elif file_type == "image":
                # In a production app, you'd use OCR here
                return "Image text extraction not implemented"
            elif file_type in ["word", "excel", "powerpoint"]:
                # In a production app, you'd use appropriate libraries here
                return f"{file_type.capitalize()} text extraction not implemented"
            else:
                return "File type not supported for text extraction"
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            return f"Error extracting text: {str(e)}"
