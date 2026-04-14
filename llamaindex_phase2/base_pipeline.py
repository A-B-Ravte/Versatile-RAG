from abc import ABC, abstractmethod

from llamaindex_phase2.common import (
    configure_embedding,
    configure_llm,
    load_documents,
    print_response,
)


class BaseRAGPipeline(ABC):
    def __init__(self, doc_path: str, llm_name: str, query: str):
        self.doc_path = doc_path
        self.llm_name = llm_name
        self.query = query

        configure_embedding()
        configure_llm(llm_name)

        self.documents = load_documents(doc_path)

    @abstractmethod
    def build_query_engine(self):
        pass

    def execute(self):
        query_engine = self.build_query_engine()
        response = query_engine.query(self.query)
        print_response(response)
        return response