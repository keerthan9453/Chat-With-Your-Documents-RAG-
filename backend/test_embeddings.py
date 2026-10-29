
import numpy as np
from services.embedder import co

sentences = [
    "A process is a running program.",
    "A program currently being executed.",
    "The weather is sunny today."
]

response = co.embed(
    model="embed-v4.0",
    texts=sentences,
    input_type="search_query",
    embedding_types=["float"],
    output_dimension=1024
)

vectors = response.embeddings.float


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


print("Vector dimensions:", len(vectors[0]))

print(
    "Similar sentences:",
    cosine_similarity(vectors[0], vectors[1])
)

print(
    "Unrelated sentences:",
    cosine_similarity(vectors[0], vectors[2])
)
