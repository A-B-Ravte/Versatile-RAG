from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, retrieved_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
        pass