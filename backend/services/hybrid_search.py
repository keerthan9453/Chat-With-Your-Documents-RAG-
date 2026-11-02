
from backend.services.embedder import embed_query
from backend.services.keyword_search import keyword_search
from backend.services.vector_store import search_vectors


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int = 60
) -> list[dict]:
    fused = {}

    for ranking in rankings:
        for rank, document in enumerate(ranking, start=1):
            doc_id = document["id"]

            if doc_id not in fused:
                fused[doc_id] = {
                    **document,
                    "rrf_score": 0.0
                }

            fused[doc_id]["rrf_score"] += 1 / (k + rank)

    return sorted(
        fused.values(),
        key=lambda doc: doc["rrf_score"],
        reverse=True
    )


def hybrid_search(question: str, top_k: int = 10):
    vector = embed_query(question)

    results = search_vectors(vector, top_k=top_k)

    dense = [
        {
            "id": chunk_id,
            "text": text,
            "metadata": metadata
        }
        for chunk_id, text, metadata in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0]
        )
    ]

    sparse = keyword_search(question, top_k=top_k)

    fused = reciprocal_rank_fusion([dense, sparse])

    return fused[:top_k]
