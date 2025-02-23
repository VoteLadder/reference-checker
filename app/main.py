import fitz
import os
import uuid
import time
import json
import asyncio
import logging
import aiohttp
from typing import Dict, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from pathlib import Path
from zipfile import ZipFile
from sqlalchemy.orm import Session
from datetime import datetime
from asyncio import QueueEmpty
from .database import get_db, ProcessingRequest, SessionLocal
from concurrent.futures import ThreadPoolExecutor
from .reference_checker import (
    extract_relevant_text, extract_references_section, process_main_content,
    process_references_section, get_all_article_pdfs, process_articles_with_verification,
    save_articles_zip, WORD_LIMIT, extract_title_and_summary
)

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    root_path="/website/reference_checker/reference-checker/api"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse(Path(current_dir / "static" / "index.html"))

# Get the current directory
current_dir = Path(__file__).parent

# Mount the static directory
app.mount("/static", StaticFiles(directory=str(current_dir / "static")), name="static")

# Create necessary directories
os.makedirs(current_dir / "static", exist_ok=True)
os.makedirs(current_dir / "uploads", exist_ok=True)
os.makedirs(current_dir / "processing", exist_ok=True)

# Processing queue
processing_queue = asyncio.Queue()
is_processing = False


@app.post("/submit")
async def submit_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        request_id = str(uuid.uuid4())
        request_dir = os.path.join("processing", request_id)
        os.makedirs(request_dir, exist_ok=True)

        # Save uploaded PDF
        pdf_path = os.path.join(request_dir, "article.pdf")
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)

        # Extract first 1000 words
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Get title and summary using the API
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, extract_title_and_summary, text)
        title, summary = result  # Unpack after awaiting

        if not title:
            # Fallback to basic extraction if API fails
            title = file.filename

        # Create database entry
        db_request = ProcessingRequest(
            request_id=request_id,
            status="pending",
            progress={"stage": "Queued"},
            output_dir=request_dir,
            original_filename=file.filename,
            article_title=title,
            summary=summary
        )

        db.add(db_request)
        db.commit()

        # Add to processing queue
        await processing_queue.put((request_id, pdf_path))

        return {"request_id": request_id}
    except Exception as e:
        logger.error(f"Error submitting PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/generate_summary")
async def generate_summary(request: Request, db: Session = Depends(get_db)):
    try:
        body = await request.json()
        request_id = body.get('request_id')
        if not request_id:
            raise HTTPException(status_code=400, detail="request_id is required")

        db_request = db.query(ProcessingRequest).filter(ProcessingRequest.request_id == request_id).first()
        if not db_request:
            raise HTTPException(status_code=404, detail="Request not found")

        pdf_path = os.path.join(db_request.output_dir, "article.pdf")
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="PDF not found")

        # Extract first page text for summary
        doc = fitz.open(pdf_path)
        text = doc[0].get_text()
        doc.close()

        # Call Gemini API for summary
        headers = {
            "Authorization": f"Bearer {os.getenv('MAIN_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": os.getenv('MAIN_MODEL'),
            "messages": [
                {
                    "role": "user",
                    "content": f"Please provide a concise 100-word summary of the following academic text:\n\n{text}"
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(os.getenv('MAIN_API_URL'), headers=headers, json=payload) as response:
                if response.status != 200:
                    raise HTTPException(status_code=500, detail="Error generating summary")
                data = await response.json()
                summary = data['choices'][0]['message']['content']
                return {"summary": summary}

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{request_id}")
async def get_status(request_id: str, db: Session = Depends(get_db)):
    request = db.query(ProcessingRequest).filter(ProcessingRequest.request_id == request_id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")
    return request

@app.get("/status")
async def get_all_status(db: Session = Depends(get_db)):
    requests = db.query(ProcessingRequest).order_by(ProcessingRequest.created_at.desc()).all()
    return {"requests": {req.request_id: {
        "status": req.status,
        "progress": req.progress,
        "created_at": req.created_at.isoformat(),
        "updated_at": req.updated_at.isoformat(),
        "original_filename": req.original_filename,
        "article_title": req.article_title,
        "summary": req.summary # Include summary in the response
    } for req in requests}}

@app.get("/download/{request_id}/{filename}")
async def download_file(request_id: str, filename: str, db: Session = Depends(get_db)):
    request = db.query(ProcessingRequest).filter(ProcessingRequest.request_id == request_id).first()
    if not request:
        raise HTTPException(status_code=404, detail="Request not found")

    if request.status != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")

    file_path = os.path.join(request.output_dir, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File {filename} not found")

    content_types = {
        '.json': 'application/json',
        '.txt': 'text/plain',
        '.zip': 'application/zip',
        '.pdf': 'application/pdf'
    }
    file_ext = os.path.splitext(filename)[1].lower()
    media_type = content_types.get(file_ext, 'application/octet-stream')

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type
    )

def save_json(path: str, data: dict):
    """Helper function to save JSON data"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_confirmations(path: str, final_output: dict):
    """Helper function to save all JSON data as a readable text file"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("Citation Confirmation Verifications\n")
        f.write("="*50 + "\n\n")

        for sent in final_output.get("sentences", []):
            f.write("Sentence:\n")
            f.write(sent.get("sentence", "") + "\n\n")

            if "verifications" in sent:
                f.write("Verifications:\n")
                for ver in sent["verifications"]:
                    f.write(f"Reference: {ver.get('reference', '')}\n")
                    f.write(f"Verdict: {ver.get('verdict', '')}\n")
                    f.write(f"Explanation: {ver.get('explanation', '')}\n")
                f.write("\n")
            f.write("-"*50 + "\n\n")  # Add line after every sentence


async def process_request(request_id: str, pdf_path: str, db: Session):
    """Process a single PDF request"""
    try:
        request = db.query(ProcessingRequest).filter(ProcessingRequest.request_id == request_id).first()
        output_dir = request.output_dir

        # Update status to processing
        request.status = "processing"
        request.progress = {"stage": "Extracting text"}
        db.commit()

        # Extract text and references
        loop = asyncio.get_event_loop()
        main_text = await loop.run_in_executor(None, lambda: extract_relevant_text(pdf_path, word_limit=WORD_LIMIT))
        references_text = await loop.run_in_executor(None, extract_references_section, pdf_path)

        if not main_text or not references_text:
            raise Exception("Text extraction failed")

        # Process main content
        request.progress = {"stage": "Processing main content"}
        db.commit()
        main_data = await loop.run_in_executor(None, lambda: process_main_content(main_text))
        sentences_data = main_data.get("sentences", [])

        # Process references
        request.progress = {"stage": "Processing references"}
        db.commit()
        ref_data = await loop.run_in_executor(None, lambda: process_references_section(references_text))
        references = ref_data.get("references", {})

        # Download PDFs
        request.progress = {"stage": "Downloading referenced PDFs"}
        db.commit()
        await loop.run_in_executor(None, lambda: get_all_article_pdfs(references, output_dir))

        # Verify sentences
        request.progress = {"stage": "Verifying sentences"}
        db.commit()
        articles_dir = os.path.join(output_dir, "articles")
        sentences_data = await loop.run_in_executor(
            None,
            lambda: process_articles_with_verification(articles_dir, sentences_data)
        )

        # Save outputs
        request.progress = {"stage": "Saving results"}
        db.commit()

        final_output = {"sentences": sentences_data, "references": references}

        # Save JSON output
        json_path = os.path.join(output_dir, "verified.json")
        await loop.run_in_executor(None, lambda: save_json(json_path, final_output))

        # Save confirmations text file
        conf_path = os.path.join(output_dir, "confirmations.txt")
        await loop.run_in_executor(None, lambda: save_confirmations(conf_path, final_output))

        # Create ZIP of articles
        await loop.run_in_executor(None, lambda: save_articles_zip(output_dir))

        # Update status to completed
        request.status = "completed"
        request.progress = {"stage": "Completed"}
        db.commit()

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        request.status = "error"
        request.progress = {"stage": f"Error: {str(e)}"}
        db.commit()

async def process_queue():
    """Background task to process the queue"""
    global is_processing
    db = SessionLocal()

    while True:
        try:
            if not is_processing:
                try:
                    # Get the next item from the queue
                    queue_item = await processing_queue.get()
                    is_processing = True

                    # Unpack the tuple after getting it from the queue
                    request_id, pdf_path = queue_item

                    try:
                        await process_request(request_id, pdf_path, db)
                    finally:
                        processing_queue.task_done()
                        is_processing = False

                except asyncio.QueueEmpty:
                    await asyncio.sleep(1)
            else:
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in queue processing: {str(e)}")
            is_processing = False
            await asyncio.sleep(1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_queue())

@app.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Return processing statistics."""
    total_processed = db.query(ProcessingRequest).filter(ProcessingRequest.status == "completed").count()

    # Calculate average processing time (very basic, improve as needed)
    completed_requests = db.query(ProcessingRequest).filter(ProcessingRequest.status == "completed").all()
    if completed_requests:
        total_processing_time = sum([(req.updated_at - req.created_at).total_seconds() for req in completed_requests if req.updated_at and req.created_at])
        average_time_seconds = total_processing_time / len(completed_requests) if completed_requests else 0
        average_time_minutes = f"{average_time_seconds / 60:.2f}"
    else:
        average_time_minutes = "N/A"


    return {
        "totalProcessed": total_processed,
        "averageTime": average_time_minutes # in minutes
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))  # Get port from env, default to 8001
    host = os.getenv("HOST", "0.0.0.0")    # Get host from env, default to 0.0.0.0
    uvicorn.run(app, host=host, port=port)

