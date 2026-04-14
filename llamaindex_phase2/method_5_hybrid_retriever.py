from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llamaindex_phase2.base_pipeline import BaseRAGPipeline


class Method5HybridRetrieverPipeline(BaseRAGPipeline):
    def build_query_engine(self):
        parser = SemanticSplitterNodeParser(
            embed_model=Settings.embed_model,
            breakpoint_percentile_threshold=85
        )

        nodes = parser.get_nodes_from_documents(self.documents)
        print(f"Created {len(nodes)} semantic chunks")

        index = VectorStoreIndex(nodes)

        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5
        )

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=5
        )

        hybrid_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            mode=FUSION_MODES.RECIPROCAL_RANK,
            similarity_top_k=5
        )

        return RetrieverQueryEngine.from_args(hybrid_retriever)


if __name__ == "__main__":
    pipeline = Method5HybridRetrieverPipeline(
        doc_path="D:/aakash/Agentic AI/Data/JD.pdf",
        llm_name="llama_cpp",
        query="What qualifications are required?"
    )
    pipeline.execute()