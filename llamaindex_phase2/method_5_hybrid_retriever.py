from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine

from common import (
    parse_args,
    configure_embedding,
    configure_llm,
    load_documents,
    print_response,
)


def run(doc_path: str, llm_name: str, query: str):
    configure_embedding()
    configure_llm(llm_name)

    documents = load_documents(doc_path)

    parser = SemanticSplitterNodeParser(
        embed_model=Settings.embed_model,
        breakpoint_percentile_threshold=85
    )

    nodes = parser.get_nodes_from_documents(documents)
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

    query_engine = RetrieverQueryEngine.from_args(hybrid_retriever)
    response = query_engine.query(query)

    print_response(response)


if __name__ == "__main__":
    args = parse_args(default_query="What qualifications are required?")
    run(args.doc_path, args.llm, args.query)