
import chromadb
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "chroma_data"

client = chromadb.PersistentClient(path=str(DB_PATH))

collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)


def store_chunks(chunks: list[dict]) -> int:
    if not chunks:
        return 0

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],

        documents=[chunk["text"] for chunk in chunks],

        embeddings=[chunk["embedding"] for chunk in chunks],

        metadatas=[
            {
                "filename": chunk["filename"],
                "page": chunk["page"] or 0
            }
            for chunk in chunks
        ]
    )

    return len(chunks)


def search_vectors(
    query_vector: list[float],
    top_k: int = 5
):
    count = collection.count()

    if count == 0:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"]
    )

    return results
