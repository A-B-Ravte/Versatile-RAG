import uuid
from typing import List
from dto.chunks_dto import Chunk, Metadata
from .base_strategy import BaseChunker

class SimpleChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, chunks: List[Chunk]) -> List[Chunk]:
        new_chunks = []
        for parent in chunks:
            text = parent.chunk_text
            text_size = len(text)
            start = 0
            
            while start < text_size:
                end = start + self.chunk_size
                chunk_segment = text[start:end]
                
                if chunk_segment:
                    new_meta = Metadata(
                        page_no=parent.metadata.page_no,
                        source=parent.metadata.source,
                        parent_id=parent.chunk_id
                    )
                    new_chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        chunk_text=chunk_segment,
                        metadata=new_meta
                    ))
                
                if end >= text_size:
                    break
                
                start = end - self.overlap
        return new_chunks