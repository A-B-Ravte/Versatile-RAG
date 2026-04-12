from typing import List
from sentence_transformers import SentenceTransformer
from embedding.base_embedding import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]):
        return self.model.encode(texts)

    def embed_query(self, query: str):
        return self.model.encode([query])[0]