from typing import List, Dict, Any

from dto.chunks_dto import Chunk
from retriever.base_retriever import BaseRetriever
from retriever.vector_retriever import VectorRetriever
from retriever.bm25_retriever import BM25Retriever
from embedding.base_embedding import BaseEmbedding


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        embedding_model: BaseEmbedding,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5
    ):
        self.bm25_retriever = BM25Retriever()
        self.vector_retriever = VectorRetriever(embedding_model)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def _normalize_scores(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        if not results:
            return {}

        scores = [item["score"] for item in results]
        min_score = min(scores)
        max_score = max(scores)

        normalized = {}

        for item in results:
            chunk_id = item["chunk"].chunk_id
            score = item["score"]

            if max_score == min_score:
                normalized[chunk_id] = 1.0
            else:
                normalized[chunk_id] = (score - min_score) / (max_score - min_score)

        return normalized

    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[Dict[str, Any]]:
        bm25_results = self.bm25_retriever.retrieve(query, chunks, top_k=len(chunks))
        vector_results = self.vector_retriever.retrieve(query, chunks, top_k=len(chunks))

        bm25_scores = self._normalize_scores(bm25_results)
        vector_scores = self._normalize_scores(vector_results)

        chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        all_chunk_ids = set(bm25_scores.keys()) | set(vector_scores.keys())

        hybrid_results = []

        for chunk_id in all_chunk_ids:
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            vector_score = vector_scores.get(chunk_id, 0.0)

            final_score = (
                self.bm25_weight * bm25_score +
                self.vector_weight * vector_score
            )

            hybrid_results.append({
                "score": final_score,
                "chunk": chunk_lookup[chunk_id],
                "bm25_score": bm25_score,
                "vector_score": vector_score
            })

        hybrid_results.sort(key=lambda x: x["score"], reverse=True)
        return hybrid_results[:top_k]