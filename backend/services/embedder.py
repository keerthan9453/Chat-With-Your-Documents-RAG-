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


def embed_query(question: str) -> list[float]:
    response = co.embed(
        model="embed-v4.0",
        texts=[question],
        input_type="search_query",
        embedding_types=["float"],
        output_dimension=1024
    )

    return response.embeddings.float[0]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    if not chunks:
        return []

    batch_size = 32

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [chunk["text"] for chunk in batch]

        response = co.embed(
            model="embed-v4.0",
            texts=texts,
            input_type="search_document",
            embedding_types=["float"],
            output_dimension=1024
        )

        vectors = response.embeddings.float

        if len(vectors) != len(batch):
            raise RuntimeError("Embedding count mismatch")

        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector

    return chunks
