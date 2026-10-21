
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")
