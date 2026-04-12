from typing import List
from dto.chunks_dto import Chunk
from embedding.base_embedding import BaseEmbedding


class EmbeddingProcessor:
    def __init__(self, embedding_model: BaseEmbedding):
        self.embedding_model = embedding_model

    def process(self, chunks: List[Chunk]) -> List[Chunk]:
        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i].tolist()

        return chunks