from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

from llamaindex_phase2.base_pipeline import BaseRAGPipeline


class Method2SentenceSplitterPipeline(BaseRAGPipeline):
    def build_query_engine(self):
        splitter = SentenceSplitter(
            chunk_size=256,
            chunk_overlap=50
        )

        index = VectorStoreIndex.from_documents(
            self.documents,
            transformations=[splitter]
        )

        return index.as_query_engine(similarity_top_k=3)


if __name__ == "__main__":
    pipeline = Method2SentenceSplitterPipeline(
        doc_path="D:/aakash/Agentic AI/Data/JD.pdf",
        llm_name="llama_cpp",
        query="What qualifications are required?"
    )
    pipeline.execute()