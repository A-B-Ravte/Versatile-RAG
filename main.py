from data_loader.document_reader import SingleDocument
from chunking.chunk_engine import ChunkEngine
from chunking.strategies.paragraph_split import ParagraphChunker
from chunking.strategies.simple_chunker import SimpleChunker
from embedding.sentence_transformer_engine import SentenceTransformerEmbedding
from embedding.embedding_processor import EmbeddingProcessor
from retriever.vector_retriever import VectorRetriever

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
results = retriever.retrieve(query, chunks, top_k=5)

for item in results:
    print("=" * 80)
    print("Score:", item["score"])
    print("Page:", item["chunk"].metadata.page_no)
    print(item["chunk"].chunk_text)