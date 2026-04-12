from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from reranker.base_reranker import BaseReranker


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        if not retrieved_chunks:
            return []

        pairs = [
            (query, item["chunk"].chunk_text)
            for item in retrieved_chunks
        ]

        scores = self.model.predict(pairs)

        reranked_results = []

        for item, score in zip(retrieved_chunks, scores):
            reranked_results.append({
                "score": float(score),
                "chunk": item["chunk"],
                "original_score": item.get("score"),
                "bm25_score": item.get("bm25_score"),
                "vector_score": item.get("vector_score")
            })

        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results[:top_k]