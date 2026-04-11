import uuid
from typing import List
from dto.chunks_dto import Chunk, Metadata
from .base_strategy import BaseChunker

class ParagraphChunker(BaseChunker):
    def split(self, chunks: List[Chunk]) -> List[Chunk]:
        new_chunks = []
        
        for parent in chunks:
            
            paragraphs = [p.strip() for p in parent.chunk_text.split("\n\n") if p.strip()]
            
            for i, p_text in enumerate(paragraphs):
                # Re-use metadata from parent, but update parent_id
                new_meta = Metadata(
                    page_no=parent.metadata.page_no,
                    source=parent.metadata.source,
                    parent_id=parent.chunk_id
                )
                
                new_chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    chunk_text=p_text,
                    metadata=new_meta
                ))
        return new_chunks