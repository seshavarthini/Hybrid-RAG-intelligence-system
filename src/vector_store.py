
from pathlib import Path

import faiss
import numpy as np

from src.chunking import Chunk
from src.embeddings import EmbeddingModel


class VectorStore:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks provided.")

        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_model.encode(texts)

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Build the vector store first.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_vector = self.embedding_model.encode([query])
        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            chunk = self.chunks[int(index)]
            results.append({
                "text": chunk.text,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "score": float(score),
            })

        return results

    def save(self, directory: str = "artifacts") -> None:
        if self.index is None:
            raise RuntimeError("Build the vector store first.")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "faiss.index"))

        # Save matching chunk metadata and text.
        import pickle
        with (path / "chunks.pkl").open("wb") as file:
            pickle.dump(self.chunks, file)

