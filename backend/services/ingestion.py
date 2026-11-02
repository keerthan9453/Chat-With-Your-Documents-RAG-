

from backend.services.vector_store import store_chunks
from backend.services.chunker import chunk_sections
from backend.services.embedder import embed_chunks
from backend.services.extractor import extract_document


def process_document(filename: str, content: bytes) -> dict:
    sections = extract_document(filename, content)
    chunks = chunk_sections(sections, filename)
    embedded_chunks = embed_chunks(chunks)
    stored_chunks = store_chunks(embedded_chunks)

    return {
        "filename": filename,
        "chunks": stored_chunks,
        "sections_extracted": len(sections),
        "chunks_created": len(chunks)
    }
