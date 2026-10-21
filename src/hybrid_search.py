
from src.bm25_search import BM25Search
from src.vector_store import VectorStore


class HybridSearch:
    def __init__(
        self,
        bm25_search: BM25Search,
        vector_store: VectorStore,
        rrf_k: int = 60,
    ):
        self.bm25_search = bm25_search
        self.vector_store = vector_store
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Retrieve a wider candidate list from both search methods.
        bm25_results = self.bm25_search.search(query, top_k=top_k * 3)
        vector_results = self.vector_store.search(query, top_k=top_k * 3)

        fused_scores = {}
        chunk_lookup = {}

        # Fuse rankings using Reciprocal Rank Fusion (RRF).
        for results in (bm25_results, vector_results):
            for rank, result in enumerate(results, start=1):
                key = (
                    result["filename"],
                    result["page_number"],
                    result["text"],
                )

                fused_scores[key] = fused_scores.get(key, 0.0) + (
                    1.0 / (self.rrf_k + rank)
                )
                chunk_lookup[key] = result

        ranked_keys = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[:top_k]

        output = []
        for key in ranked_keys:
            result = dict(chunk_lookup[key])
            result["rrf_score"] = fused_scores[key]
            output.append(result)

        return output
