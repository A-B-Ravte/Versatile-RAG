from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.retrievers import VectorIndexRetriever
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

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=5
    )

    query_engine = RetrieverQueryEngine.from_args(retriever)
    response = query_engine.query(query)

    print_response(response)


if __name__ == "__main__":
    args = parse_args(default_query="What qualifications are required?")
    run(args.doc_path, args.llm, args.query)