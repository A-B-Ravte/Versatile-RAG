from llamaindex_phase2.method_3_semantic_splitter import Method3SemanticSplitterPipeline


pipeline = Method3SemanticSplitterPipeline(
    doc_path="D:/aakash/Agentic AI/Data/JD.pdf",
    llm_name="llama_cpp",
    query="What qualifications are required?"
)
pipeline.execute()