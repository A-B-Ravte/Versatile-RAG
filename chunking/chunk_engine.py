import uuid
from typing import List, Dict, Any
from dto.page_dto import Page
from dto.chunks_dto import Chunk, Metadata
from chunking.strategies.base_strategy import BaseChunker

class ChunkEngine:
    def __init__(self, strategies: List[BaseChunker]):
        self.strategies = strategies

    def process(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        current_chunks = []
        for page in pages:
            
            meta = Metadata(
                page_no=page.page_no,
                source=page.source,
                parent_id=None
            )
            
            current_chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_text=page.text, 
                metadata=meta
            ))
        for strategy in self.strategies:
            current_chunks = strategy.split(current_chunks)
            
        return current_chunks