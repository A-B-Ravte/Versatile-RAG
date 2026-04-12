import math
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any

from dto.chunks_dto import Chunk
from retriever.base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def _build_index(self, chunks: List[Chunk]) -> None:
        self.chunk_tokens: List[List[str]] = []
        self.term_frequencies: List[Counter] = []
        self.document_frequencies = defaultdict(int)
        self.doc_lengths: List[int] = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.chunk_text)
            self.chunk_tokens.append(tokens)

            tf = Counter(tokens)
            self.term_frequencies.append(tf)

            self.doc_lengths.append(len(tokens))

            for term in tf.keys():
                self.document_frequencies[term] += 1

        self.total_docs = len(chunks)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.total_docs if self.total_docs > 0 else 0
        )

    def _idf(self, term: str) -> float:
        df = self.document_frequencies.get(term, 0)
        if df == 0:
            return 0.0

        return math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))

    def _score(self, query_tokens: List[str], doc_index: int) -> float:
        score = 0.0
        tf = self.term_frequencies[doc_index]
        doc_length = self.doc_lengths[doc_index]

        for term in query_tokens:
            term_freq = tf.get(term, 0)
            if term_freq == 0:
                continue

            idf = self._idf(term)

            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, chunks: List[Chunk], top_k: int = 3) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        self._build_index(chunks)

        query_tokens = self._tokenize(query)
        scored_chunks = []

        for i, chunk in enumerate(chunks):
            score = self._score(query_tokens, i)

            scored_chunks.append({
                "score": score,
                "chunk": chunk
            })

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]