import os
from pathlib import Path

import cohere
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise ValueError("Cohere API key not found in environment variables.")

co = cohere.ClientV2(api_key=api_key)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]

    response = co.embed(
        model="embed-v4.0",
        texts=texts,
        input_type="search_document",
        embedding_types=["float"],
        output_dimension=1024
    )

    vectors = response.embeddings.float

    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    return chunks
