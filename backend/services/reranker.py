
from backend.services.embedder import co


def rerank_chunks(
    question: str,
    chunks: list[dict],
    top_n: int = 5
) -> list[dict]:

    if not chunks:
        return []

    response = co.rerank(
        model="rerank-v5.0-fast",
        query=question,
        documents=[chunk["text"] for chunk in chunks],
        top_n=min(top_n, len(chunks))
    )

    reranked = []

    for item in response.results:
        original_chunk = chunks[item.index]

        reranked.append({
            **original_chunk,
            "rerank_score": item.relevance_score
        })

    return reranked
