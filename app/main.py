import os
import uuid
import time
import json
import asyncio
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from pathlib import Path

# Change the import to be relative to the app package
from .reference_checker import (
    extract_relevant_text, extract_references_section, process_main_content,
    process_references_section, get_all_article_pdfs, process_articles_with_verification,
    save_articles_zip, WORD_LIMIT
)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the current directory
current_dir = Path(__file__).parent

# Mount the static directory
app.mount("/static", StaticFiles(directory=str(current_dir / "static")), name="static")

# Create necessary directories
os.makedirs(current_dir / "static", exist_ok=True)
os.makedirs(current_dir / "uploads", exist_ok=True)
os.makedirs(current_dir / "processing", exist_ok=True)

# Configuration
UPLOAD_DIR = str(current_dir / "uploads")
PROCESSING_DIR = str(current_dir / "processing")

# State management
class ProcessingStatus(BaseModel):
    status: str
    progress: Dict[str, str]
    output_dir: Optional[str] = None

# Global state
processing_queue = asyncio.Queue()
request_status: Dict[str, ProcessingStatus] = {}
executor = ThreadPoolExecutor(max_workers=1)

# Root endpoint to serve index.html
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(current_dir / "static" / "index.html")

# Rest of your endpoints remain the same...
@app.post("/submit")
async def submit_pdf(file: UploadFile = File(...)):
    try:
        request_id = str(uuid.uuid4())
        request_dir = os.path.join(PROCESSING_DIR, request_id)
        os.makedirs(request_dir, exist_ok=True)
        
        pdf_path = os.path.join(request_dir, "article.pdf")
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
        
        request_status[request_id] = ProcessingStatus(
            status="pending",
            progress={"stage": "Queued"},
            output_dir=request_dir
        )
        
        await processing_queue.put((request_id, pdf_path))
        return {"request_id": request_id}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status/{request_id}")
async def get_status(request_id: str):
    if request_id not in request_status:
        raise HTTPException(status_code=404, detail="Request not found")
    return request_status[request_id]

@app.get("/status")
async def get_all_status():
    return {"requests": {k: v.dict() for k, v in request_status.items()}}

@app.get("/download/{request_id}/{filename}")
async def download_file(request_id: str, filename: str):
    if request_id not in request_status:
        raise HTTPException(status_code=404, detail="Request not found")
    
    status = request_status[request_id]
    if status.status != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    file_path = os.path.join(status.output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path)

async def process_request(request_id: str, pdf_path: str):
    """Process a single PDF request"""
    try:
        status = request_status[request_id]
        output_dir = status.output_dir
        
        # Update status to processing
        status.status = "processing"
        status.progress["stage"] = "Extracting text"
        
        # Extract text
        main_text = extract_relevant_text(pdf_path, word_limit=WORD_LIMIT)
        references_text = extract_references_section(pdf_path)
        
        if not main_text or not references_text:
            raise Exception("Text extraction failed")
        
        # Process main content
        status.progress["stage"] = "Processing main content"
        main_data = process_main_content(main_text)
        sentences_data = main_data.get("sentences", [])
        
        # Process references
        status.progress["stage"] = "Processing references"
        ref_data = process_references_section(references_text)
        references = ref_data.get("references", {})
        
        # Download PDFs
        status.progress["stage"] = "Downloading referenced PDFs"
        get_all_article_pdfs(references, output_dir)
        
        # Verify sentences
        status.progress["stage"] = "Verifying sentences"
        articles_dir = os.path.join(output_dir, "articles")
        sentences_data = process_articles_with_verification(articles_dir, sentences_data)
        
        # Save output
        status.progress["stage"] = "Saving results"
        final_output = {"sentences": sentences_data, "references": references}
        
        json_path = os.path.join(output_dir, "verified.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        # Update status to completed
        status.status = "completed"
        status.progress["stage"] = "Completed"
        
    except Exception as e:
        status.status = "error"
        status.progress["stage"] = f"Error: {str(e)}"

async def process_queue():
    """Background task to process the queue"""
    while True:
        try:
            request_id, pdf_path = await processing_queue.get()
            await process_request(request_id, pdf_path)
        except Exception as e:
            print(f"Error processing request: {e}")
        finally:
            processing_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)