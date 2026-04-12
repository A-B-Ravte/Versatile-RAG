from data_loader.document_reader import SingleDocument
from chunking.chunk_engine import ChunkEngine
from chunking.strategies.paragraph_split import ParagraphChunker
from chunking.strategies.simple_chunker import SimpleChunker
from embedding.sentence_transformer_engine import SentenceTransformerEmbedding
from embedding.embedding_processor import EmbeddingProcessor
from retriever.hybrid_retriever import HybridRetriever
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

    embedding_model = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
    embedding_processor = EmbeddingProcessor(embedding_model)
    chunks = embedding_processor.process(chunks)

    retriever = HybridRetriever(
        embedding_model=embedding_model,
        bm25_weight=0.5,
        vector_weight=0.5
    )
    retrieved = retriever.retrieve(query, chunks, top_k=3)

    prompt_builder = RAGPromptBuilder()
    final_prompt = prompt_builder.build(query, retrieved)

    llm = LlammaLocal.get_instance(model_path)
    generator = ResponseGenerator(llm)
    answer = generator.generate(final_prompt)

    print("\n=== HYBRID PIPELINE ===\n")
    print("Query:", query)
    print("\nFinal Answer:\n")
    print(answer)


if __name__ == "__main__":
    run()