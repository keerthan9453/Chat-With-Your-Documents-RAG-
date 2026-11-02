from backend.services.embedder import co


def rewrite_query(question: str) -> str:
    """Rewrite for retrieval, while preserving the user's intent.

    One extra Cohere chat call is made when enabled.
    """
    response = co.chat(
        model="command-a-plus-05-2026",
        messages=[
            {"role": "system", "content": (
                "Rewrite the user's question into a concise standalone search query. "
                "Preserve names, numbers, technical terms and intent. "
                "Do not answer the question, speculate or add facts. "
                "Output only the rewritten query."
            )},
            {"role": "user", "content": question},
        ],
        thinking={"type": "disabled"},
        temperature=0,
        max_tokens=120,
    )
    rewritten = "".join(
        block.text for block in (response.message.content or [])
        if block.type == "text" and block.text
    ).strip()
    if not rewritten or rewritten.startswith("<EOS_TOKEN>"):
        return question
    return rewritten
