
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[dict], top_k: int = 3) -> list[dict]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not results:
            return []

        pairs = [(query, result["text"]) for result in results]
        scores = self.model.predict(pairs)

        reranked = []
        for result, score in zip(results, scores):
            item = dict(result)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]
