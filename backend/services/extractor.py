from docx import Document
import io
import os
from dotenv import load_dotenv
import cohere
import chromadb
import fitz

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

load_dotenv()
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
chroma_client = chromadb.Client()


def classify_pdf_page(page) -> str:
    """
    Simple heuristic:
    Detect whether a page has meaningful extractable text.
    """
    text = page.get_text("text").strip()

    if len(text) >= 100:
        return "text_based"

    return "needs_ocr"


def extract_pdf(pdf_bytes: bytes) -> list[dict]:
    pages = []

    try:
        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        ) as document:

            for page_number, page in enumerate(document, 1):
                classification = classify_pdf_page(page)

                if classification == "text_based":
                    text = page.get_text("text")
                else:
                    # OCR is not implemented yet
                    text = page.get_text("text")

                pages.append({
                    "page": page_number,
                    "text": text.strip(),
                    "extraction_method": classification
                })

    except Exception as exc:
        raise ValueError("Unable to read PDF") from exc

    return pages


def extract_docx(docx_bytes: bytes) -> list[dict]:
    try:
        document = Document(io.BytesIO(docx_bytes))
    except Exception as exc:
        raise ValueError("Unable to read DOCX") from exc

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [{
        "page": None,
        "text": text,
        "extraction_method": "docx_text"
    }]


def extract_document(filename: str, content: bytes):
    extension = filename.lower().rsplit(".", 1)[-1]

    if extension == "pdf":
        return extract_pdf(content)

    if extension == "docx":
        return extract_docx(content)

    raise ValueError("Unsupported document format")
