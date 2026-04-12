from abc import ABC, abstractmethod
from typing import List
from dto.chunks_dto import Chunk

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 3):
        pass