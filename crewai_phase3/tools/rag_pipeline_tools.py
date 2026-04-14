from typing import Type, ClassVar
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from llamaindex_phase2.method_1_simple_index import Method1SimpleIndexPipeline
from llamaindex_phase2.method_2_sentence_splitter import Method2SentenceSplitterPipeline
from llamaindex_phase2.method_3_semantic_splitter import Method3SemanticSplitterPipeline
from llamaindex_phase2.method_4_vector_retriever import Method4VectorRetrieverPipeline
from llamaindex_phase2.method_5_hybrid_retriever import Method5HybridRetrieverPipeline


class PipelineToolInput(BaseModel):
    doc_path: str = Field(..., description="Absolute path to the input PDF document")
    query: str = Field(..., description="User query to run against the document")
    llm_name: str = Field(default="llama_cpp", description="llm backend: llama_cpp or gemini")


class _BasePipelineTool(BaseTool):
    args_schema: Type[BaseModel] = PipelineToolInput
    pipeline_cls: ClassVar[type | None] = None

    def _run(self, doc_path: str, query: str, llm_name: str = "llama_cpp") -> str:
        pipeline = self.pipeline_cls(
            doc_path=doc_path,
            llm_name=llm_name,
            query=query
        )
        response = pipeline.execute()
        return str(response)


class Method1Tool(_BasePipelineTool):
    name: str = "method_1_simple_index_tool"
    description: str = "Use simple VectorStoreIndex plus QueryEngine. Best for fast, low-complexity RAG."
    pipeline_cls: ClassVar[type] = Method1SimpleIndexPipeline


class Method2Tool(_BasePipelineTool):
    name: str = "method_2_sentence_splitter_tool"
    description: str = "Use SentenceSplitter plus VectorStoreIndex plus QueryEngine. Best when overlap and controlled chunking help."
    pipeline_cls: ClassVar[type] = Method2SentenceSplitterPipeline


class Method3Tool(_BasePipelineTool):
    name: str = "method_3_semantic_splitter_tool"
    description: str = "Use SemanticSplitterNodeParser plus VectorStoreIndex plus QueryEngine. Best for semantically dense documents."
    pipeline_cls: ClassVar[type] = Method3SemanticSplitterPipeline


class Method4Tool(_BasePipelineTool):
    name: str = "method_4_vector_retriever_tool"
    description: str = "Use SemanticSplitterNodeParser plus explicit VectorIndexRetriever plus QueryEngine. Best when you want retriever control."
    pipeline_cls: ClassVar[type] = Method4VectorRetrieverPipeline


class Method5Tool(_BasePipelineTool):
    name: str = "method_5_hybrid_retriever_tool"
    description: str = "Use SemanticSplitterNodeParser plus VectorIndexRetriever plus BM25Retriever plus QueryFusionRetriever plus QueryEngine. Best for mixed semantic and keyword queries."
    pipeline_cls: ClassVar[type] = Method5HybridRetrieverPipeline