from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BasePromptBuilder(ABC):
    @abstractmethod
    def build(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        pass