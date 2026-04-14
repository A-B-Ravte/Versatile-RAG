'''
from llamaindex_phase2.method_3_semantic_splitter import Method3SemanticSplitterPipeline


pipeline = Method3SemanticSplitterPipeline(
    doc_path="D:/aakash/Agentic AI/Data/JD.pdf",
    llm_name="llama_cpp",
    query="What qualifications are required?"
)
pipeline.execute()

'''

from crewai_phase3.flows.adaptive_rag_flow import run_adaptive_rag

print("🤖 Thinking... selecting best RAG pipeline...")
result = run_adaptive_rag(
    doc_path="D:/aakash/Agentic AI/Data/JD.pdf",
    query="What qualifications are required?",
    llm_name="llama_cpp"
)

print(result)

