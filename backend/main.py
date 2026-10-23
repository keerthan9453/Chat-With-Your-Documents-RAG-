import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
import cohere
import chromadb
import fitz

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.services.extractor import extract_document

MAX_FILE_SIZE = 20 * 1024 * 1024
logger = logging.getLogger(__name__)


def scan_docs_folder() -> dict:
    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    report = {"status": "ok", "documents_found": 0,
              "documents": [], "errors": []}

    if not docs_dir.is_dir():
        report["status"] = "missing_docs_folder"
        return report

    for document_path in sorted(docs_dir.iterdir()):
        if not document_path.is_file() or document_path.suffix.lower() not in {".pdf", ".docx"}:
            continue

        try:
            content = document_path.read_bytes()
            if len(content) > MAX_FILE_SIZE:
                raise ValueError("File exceeds the 20MB limit")

            sections = extract_document(document_path.name, content)
            report["documents"].append({
                "filename": document_path.name,
                "sections_extracted": len(sections),
                "sections": sections,
            })
        except Exception as exc:
            report["errors"].append(
                {"filename": document_path.name, "detail": str(exc)})

    report["documents_found"] = len(report["documents"])
    if report["errors"] and not report["documents"]:
        report["status"] = "processing_errors"
    elif not report["documents"]:
        report["status"] = "no_supported_documents"

    return report


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.docs_report = scan_docs_folder()
    logger.info(
        "Startup docs scan complete: %s supported document(s) found",
        app.state.docs_report["documents_found"],
    )
    yield

load_dotenv()
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
chroma_client = chromadb.Client()

app = FastAPI(title="DocuMint AI", lifespan=lifespan)


@app.get("/")
async def get_startup_docs_report():
    return app.state.docs_report


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or ""

    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Invalid file type. "
                            "Only PDF and DOCX files are allowed.")

    content = await file.read(MAX_FILE_SIZE + 1)

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, detail="File size exceeds the maximum limit of 20MB.")

    try:

        pages = extract_document(filename, content)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error occurred while extracting document content.")

    temp_file_path = f"temp_{filename}"

    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(content)
    return {"filename": filename, "sections": pages, "sections_extracted": len(pages)}
