from data_loader.document_reader import SingleDocument
from chunking.chunk_engine import ChunkEngine
from chunking.strategies.paragraph_split import ParagraphChunker
from chunking.strategies.simple_chunker import SimpleChunker
from embedding.sentence_transformer_engine import SentenceTransformerEmbedding
from embedding.embedding_processor import EmbeddingProcessor
from retriever.vector_retriever import VectorRetriever
from prompt.rag_prompt_builder import RAGPromptBuilder
from generation.response_generator import ResponseGenerator
from llm.llamma_cpp_wrapper import LlammaLocal
from llm.gemini_wrapper import GeminiLLM

pdf_path = r'D:/aakash/Agentic AI/Versatile_RAG/data/documets/sample.pdf'

loader = SingleDocument(pdf_path)
pages = loader.read_document()

engine = ChunkEngine(strategies=[
    SimpleChunker(chunk_size=200, overlap=50)
])

chunks = engine.process(pages)

embedding_model = SentenceTransformerEmbedding("all-MiniLM-L6-v2")
embedding_processor = EmbeddingProcessor(embedding_model)
chunks = embedding_processor.process(chunks)

retriever = VectorRetriever(embedding_model)

query = "What are the 12 rules of coding?"
retrieved = retriever.retrieve(query, chunks, top_k=5)

for data in retrieved:
    print(f"chunk is {data['chunk']}")
    print(f"score is {data['score']}")

prompt_builder = RAGPromptBuilder()
final_prompt = prompt_builder.build(query, retrieved)

#Gemini Call
llm = GeminiLLM(model_name="gemini-2.5-flash")
generator = ResponseGenerator(llm)

answer = generator.generate(final_prompt)

'''
#local llamma llm model call
model_path = r"models/LlammaLocalModel/qwen2.5-coder-7b-instruct.Q4_K_M.gguf"
llm = LlammaLocal.get_instance(model_path)
generator = ResponseGenerator(llm)

answer = generator.generate(final_prompt)
'''

print("\nFinal Answer:\n")
print(answer)