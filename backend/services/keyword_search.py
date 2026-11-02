
import re
from rank_bm25 import BM25Okapi

from backend.services.vector_store import collection


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def keyword_search(query: str, top_k: int = 10) -> list[dict]:
    data = collection.get(
        include=["documents", "metadatas"]
    )

    documents = data["documents"]
    if not documents:
        return []

    tokenized_docs = [
        tokenize(document) for document in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    return [
        {
            "id": data["ids"][i],
            "text": documents[i],
            "metadata": data["metadatas"][i],
            "score": float(scores[i])
        }
        for i in ranked[:top_k]
        if scores[i] > 0
    ]
