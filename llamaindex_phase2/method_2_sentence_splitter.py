from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

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

    splitter = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=50
    )

    index = VectorStoreIndex.from_documents(
        documents,
        transformations=[splitter]
    )

    query_engine = index.as_query_engine(similarity_top_k=3)

    response = query_engine.query(query)
    print_response(response)


if __name__ == "__main__":
    args = parse_args(default_query="What qualifications are required?")
    run(args.doc_path, args.llm, args.query)    