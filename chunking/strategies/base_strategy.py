from abc import ABC, abstractmethod
from typing import List
from dto.chunks_dto import Chunk

class BaseChunker(ABC):
    @abstractmethod
    def split(self, chunks : List[Chunk]) -> List[Chunk]:
        pass

