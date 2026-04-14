from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llamaindex_phase2.base_pipeline import BaseRAGPipeline


class Method4VectorRetrieverPipeline(BaseRAGPipeline):
    def build_query_engine(self):
        parser = SemanticSplitterNodeParser(
            embed_model=Settings.embed_model,
            breakpoint_percentile_threshold=85
        )

        nodes = parser.get_nodes_from_documents(self.documents)
        print(f"Created {len(nodes)} semantic chunks")

        index = VectorStoreIndex(nodes)

        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5
        )

        return RetrieverQueryEngine.from_args(retriever)


if __name__ == "__main__":
    pipeline = Method4VectorRetrieverPipeline(
        doc_path="D:/aakash/Agentic AI/Data/JD.pdf",
        llm_name="llama_cpp",
        query="What qualifications are required?"
    )
    pipeline.execute()