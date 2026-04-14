from llama_index.core import VectorStoreIndex

from llamaindex_phase2.base_pipeline import BaseRAGPipeline


class Method1SimpleIndexPipeline(BaseRAGPipeline):
    def build_query_engine(self):
        index = VectorStoreIndex.from_documents(self.documents)
        return index.as_query_engine(similarity_top_k=3)


if __name__ == "__main__":
    pipeline = Method1SimpleIndexPipeline(
        doc_path="D:/aakash/Agentic AI/Data/JD.pdf",
        llm_name="llama_cpp",
        query="What qualifications are required?"
    )
    pipeline.execute()