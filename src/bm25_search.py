
import re

from rank_bm25 import BM25Okapi

from src.chunking import Chunk


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Search:
    def __init__(self):
        self.chunks = []
        self.index = None

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks provided.")

        self.chunks = list(chunks)
        tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(tokenized_chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Build the BM25 index first.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)
        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            {
                "text": self.chunks[i].text,
                "filename": self.chunks[i].filename,
                "page_number": self.chunks[i].page_number,
                "score": float(scores[i]),
            }
            for i in ranked_indices
        ]
