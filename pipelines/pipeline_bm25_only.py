from data_loader.document_reader import SingleDocument
from chunking.chunk_engine import ChunkEngine
from chunking.strategies.paragraph_split import ParagraphChunker
from chunking.strategies.simple_chunker import SimpleChunker
from retriever.bm25_retriever import BM25Retriever
from prompt.rag_prompt_builder import RAGPromptBuilder
from generation.response_generator import ResponseGenerator
from llm.llamma_cpp_wrapper import LlammaLocal


def run():
    pdf_path = r'D:/aakash/Agentic AI/Versatile_RAG/data/documets/sample.pdf'
    model_path = r"models/LlammaLocalModel/qwen2.5-coder-7b-instruct.Q4_K_M.gguf"
    query = "What does the book say about learning code?"

    loader = SingleDocument(pdf_path)
    pages = loader.read_document()

    chunk_engine = ChunkEngine(
        strategies=[
            ParagraphChunker(),
            SimpleChunker(chunk_size=200, overlap=50)
        ]
    )
    chunks = chunk_engine.process(pages)

    retriever = BM25Retriever()
    retrieved = retriever.retrieve(query, chunks, top_k=3)

    prompt_builder = RAGPromptBuilder()
    final_prompt = prompt_builder.build(query, retrieved)

    llm = LlammaLocal.get_instance(model_path)
    generator = ResponseGenerator(llm)
    answer = generator.generate(final_prompt)

    print("\n=== BM25 ONLY PIPELINE ===\n")
    print("Query:", query)
    print("\nFinal Answer:\n")
    print(answer)


if __name__ == "__main__":
    run()