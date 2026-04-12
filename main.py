from llm.llamma_cpp_wrapper import LlammaLocal
from data_loader.document_reader import SingleDocument
from chunking.chunk_engine import ChunkEngine
from chunking.strategies.paragraph_split import ParagraphChunker
from chunking.strategies.simple_chunker import SimpleChunker
from sentence_transformers import SentenceTransformer

#============ model - infrence test ========================
'''
model_path = r"models/LlammaLocalModel/qwen2.5-coder-7b-instruct.Q4_K_M.gguf"
llm = LlammaLocal.get_instance(model_path)

final_prompt = "write a python code to get substring from string."

response = llm(
        final_prompt,
        max_tokens=200,
        temperature=0.01,
        top_p=0.9,
        repeat_penalty=1.1
    )

print(response['choices'][0]['text'])


#============  data_loader test ========================

pdf_path = r'D:/aakash/Agentic AI/Versatile_RAG/data/documets/sample.pdf'

data_load = SingleDocument(pdf_path)

pages = data_load.read_document()


#==================chunking test====================    

strategy_1 = ParagraphChunker()

strategy_2 = SimpleChunker(chunk_size=200, overlap=50)

engine = ChunkEngine(strategies=[strategy_1, strategy_2])

final_chunks = engine.process(pages)

print(f"Total Chunks Created: {len(final_chunks)}")
print(final_chunks[0]) # See the structure
'''

query = 'I am the new agentic bot in the future'

model = SentenceTransformer("all-MiniLM-L6-v2")

chunk_texts = [query]

embedding = model.encode(chunk_texts)[0]
print(embedding)