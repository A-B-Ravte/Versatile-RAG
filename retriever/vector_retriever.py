import numpy as np
from typing import List, Dict, Any
from dto.chunks_dto import Chunk
from retriever.base_retriever import BaseRetriever
from embedding.base_embedding import BaseEmbedding


class VectorRetriever(BaseRetriever):
    def __init__(self, embedding_model: BaseEmbedding):
        self.embedding_model = embedding_model

    @staticmethod
    def cosine_similarity(vec1, vec2) -> float:
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denominator == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / denominator)

    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.embed_query(query)

        scored_chunks = []

        for chunk in chunks:
            if chunk.embedding is None:
                continue

            score = self.cosine_similarity(query_embedding, chunk.embedding)

            scored_chunks.append({
                "score": score,
                "chunk": chunk
            })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]