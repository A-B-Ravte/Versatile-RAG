from llama_index.core import VectorStoreIndex

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

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    response = query_engine.query(query)
    print_response(response)


if __name__ == "__main__":
    args = parse_args(default_query="What qualifications are required?")
    run(args.doc_path, args.llm, args.query)