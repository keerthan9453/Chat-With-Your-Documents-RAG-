
def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int
) -> float:
    if not relevant_ids:
        raise ValueError("Relevant IDs must not be empty")

    retrieved = set(retrieved_ids[:k])

    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: set[str]
) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0
