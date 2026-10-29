
def chunk_sections(
    sections: list[dict],
    filename: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> list[dict]:

    chunks = []

    for section in sections:
        text = section["text"]
        page = section["page"]

        # TODO 1: Ignore empty text
        if not text.strip():
            continue

        if chunk_size <= overlap:
            raise ValueError("Chunk size must be greater than overlap.")
        if not 0 <= overlap < chunk_size:
            raise ValueError(
                "Overlap must be non-negative and less than chunk size.")

        # TODO 2: Calculate step size
        step_size = chunk_size - overlap

        # TODO 3: Loop through text
        # with overlapping windows
        chunks.append(chunk)

        for i in range(0, len(text), step_size):
            if i + chunk_size >= len(text):
                break
            chunk_text = text[i:i + chunk_size]
            if not chunk_text.strip():
                continue

            # TODO 4: Store each chunk with:
            # - unique chunk ID
            # - filename
            # - page
            # - text
            chunk = {
                "id": f"{filename}_{page}_{i}",
                "filename": filename,
                "page": page,
                "text": chunk_text
            }
            chunks.append(chunk)

    return chunks
