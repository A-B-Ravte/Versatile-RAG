# 🚀 Versatile Autonomous Adaptive Agentic RAG

A production-ready, **domain-agnostic Retrieval-Augmented Generation (RAG) platform** that can work with **any document and any query**. Instead of forcing one fixed pipeline for every use case, the system is designed to **adaptively choose the right chunking and retrieval strategy** based on document structure and query intent.

The project evolves across **three versions**:

- **Version 1:** Pure Python RAG built from scratch
- **Version 2:** Framework-based RAG using **LlamaIndex**
- **Version 3:** Adaptive agentic orchestration using **CrewAI**

---

## Why this project?

Most RAG systems are rigid:

- one chunking strategy for every document
- one retriever for every query
- one fixed pipeline regardless of cost, accuracy, or query type

Real-world enterprise documents are not like that.

- A short job description does not need the same pipeline as a long policy document.
- Exact-match queries benefit from **BM25**.
- Conceptual questions benefit from **vector retrieval**.
- Mixed queries often need **hybrid retrieval**.
- Dense, unstructured documents benefit from **semantic chunking**.

This project solves that by building a **plug-and-play RAG architecture** where pipelines are modular, swappable, and later selectable by an autonomous agent.

---

## Core idea

> **Any document, any query — the system decides the right way to retrieve the answer.**

The platform is built so that:

- chunking strategies are replaceable
- retrievers are replaceable
- LLM backends are replaceable
- framework implementations can coexist with raw Python implementations
- agentic systems can use RAG pipelines as tools

---

## Version evolution

### Version 1 — Pure Python RAG Engine

Built from scratch to understand and control every layer of the RAG pipeline.

#### Implemented
- PDF reading with **PyMuPDF**
- DTO-driven design for pages and chunks
- Custom chunking strategies
  - paragraph-based chunking
  - fixed-size chunking with overlap
- Embedding generation using **Sentence Transformers**
- Pure Python / NumPy vector similarity retrieval
- BM25 retrieval
- Hybrid retrieval
- Reranking-ready architecture
- Local LLM inference using **llama.cpp**
- Modular pipeline structure

#### Purpose
To understand RAG from first principles before using frameworks.

---

### Version 2 — Framework-based RAG with LlamaIndex

Migrated the architecture into **LlamaIndex** while preserving the same modular thinking.

#### Implemented
- PDF loading with **PyMuPDFReader**
- `SentenceSplitter`
- `SemanticSplitterNodeParser`
- `VectorStoreIndex`
- `VectorIndexRetriever`
- `BM25Retriever`
- `QueryFusionRetriever`
- Local `LlamaCPP` integration
- Multiple framework-based RAG pipelines

#### Purpose
To move from custom implementation to production-style framework pipelines without losing architecture control.

---

### Version 3 — Adaptive Agentic RAG with CrewAI

Built an agentic layer on top of the RAG platform.

#### Implemented
- A selector agent that decides which pipeline to use
- Tool-based pipeline execution
- CrewAI integration
- Dynamic method selection based on query and document characteristics

#### Current flow
- Agent analyzes the query
- Agent selects the best available RAG method
- Selected pipeline runs
- Final answer is returned

#### Purpose
To evolve from static RAG to **adaptive agentic RAG**.

---

## Architecture overview

### High-level flow

```text
Document + Query
        │
        ▼
┌──────────────────────────────┐
│     Selector / Decision      │
│   (CrewAI agent in V3)       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Chunking Strategy       │
│  Paragraph / Fixed / Semantic│
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│      Retrieval Strategy      │
│  Vector / BM25 / Hybrid      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│       LLM Answer Layer       │
│   llama.cpp / Gemini / etc.  │
└──────────────────────────────┘
```

---

## Project structure

```text
Versatile_RAG/
│
├── main.py
├── main2.py
│
├── chunking/
│   ├── chunk_engine.py
│   └── strategies/
│       ├── base_strategy.py
│       ├── paragraph_split.py
│       └── simple_chunker.py
│
├── data/
│   └── documents/
│
├── data_loader/
│   └── document_reader.py
│
├── dto/
│   ├── chunks_dto.py
│   └── page_dto.py
│
├── embedding/
│   ├── base_embedding.py
│   ├── embedding_processor.py
│   └── sentence_transformer_engine.py
│
├── generation/
│   └── response_generator.py
│
├── llm/
│   ├── base_llm.py
│   ├── gemini_wrapper.py
│   └── llamma_cpp_wrapper.py
│
├── models/
│   └── LlammaLocalModel/
│       └── qwen2.5-coder-7b-instruct.Q4_K_M.gguf
│
├── pipelines/
│   ├── pipeline_vector_only.py
│   ├── pipeline_bm25_only.py
│   ├── pipeline_hybrid.py
│   └── pipeline_hybrid_rerank.py
│
├── prompt/
│   ├── base_prompt.py
│   └── rag_prompt_builder.py
│
├── reranker/
│   ├── base_reranker.py
│   └── cross_encoder_reranker.py
│
├── retriever/
│   ├── base_retriever.py
│   ├── vector_retriever.py
│   ├── bm25_retriever.py
│   └── hybrid_retriever.py
│
├── llamaindex_phase2/
│   ├── common.py
│   ├── base_pipeline.py
│   ├── method_1_simple_index.py
│   ├── method_2_sentence_splitter.py
│   ├── method_3_semantic_splitter.py
│   ├── method_4_vector_retriever.py
│   └── method_5_hybrid_retriever.py
│
├── crewai_phase3/
│   ├── agents/
│   │   └── selector_agent.py
│   ├── flows/
│   │   └── adaptive_rag_flow.py
│   └── tools/
│       └── rag_pipeline_tools.py
│
└── README.md
```

---

## Tech stack

### Core language and backend
- Python
- FastAPI (used in related projects / future serving direction)

### Retrieval and embeddings
- Sentence Transformers
- NumPy
- BM25
- Hybrid Retrieval
- LlamaIndex

### LLMs
- llama.cpp
- Gemini API
- LlamaCPP integration in LlamaIndex

### Agent framework
- CrewAI

### Parsing / document loading
- PyMuPDF
- PyMuPDFReader

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/A-B-Ravte/Versatile-RAG.git
cd Versatile-RAG
```

### 2. Create virtual environment

#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If you have a `requirements.txt`, use:

```bash
pip install -r requirements.txt
```

If not, install the major dependencies manually:

```bash
pip install python-dotenv pymupdf numpy sentence-transformers transformers
pip install llama-cpp-python
pip install llama-index
pip install llama-index-readers-file
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-llama-cpp
pip install llama-index-llms-google-genai
pip install llama-index-retrievers-bm25
pip install crewai
```

---

## Environment configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> If you are running only local `llama.cpp` pipelines, Gemini is optional.

---

## Running the project

## Version 1 — Pure Python pipelines

These pipelines use your raw architecture.

### 1. Vector-only pipeline
```bash
python pipelines/pipeline_vector_only.py
```

### 2. BM25-only pipeline
```bash
python pipelines/pipeline_bm25_only.py
```

### 3. Hybrid pipeline
```bash
python pipelines/pipeline_hybrid.py
```

### 4. Hybrid + rerank pipeline
```bash
python pipelines/pipeline_hybrid_rerank.py
```

---

## Version 2 — LlamaIndex pipelines

These methods expose progressively advanced framework-based RAG pipelines.

### Method 1 — Simple index
```bash
python llamaindex_phase2/method_1_simple_index.py
```

### Method 2 — Sentence splitter
```bash
python llamaindex_phase2/method_2_sentence_splitter.py
```

### Method 3 — Semantic splitter
```bash
python llamaindex_phase2/method_3_semantic_splitter.py
```

### Method 4 — Vector retriever
```bash
python llamaindex_phase2/method_4_vector_retriever.py
```

### Method 5 — Hybrid retriever
```bash
python llamaindex_phase2/method_5_hybrid_retriever.py
```

> These files are class-based so they can later be used as tools by the agentic layer.

---

## Version 3 — CrewAI adaptive agent

Run the adaptive selector flow:

```bash
python main2.py
```

### What happens
- The CrewAI selector agent analyzes the query
- It chooses the best pipeline method
- It invokes that pipeline as a tool
- The selected method returns the final answer

---

## Available RAG methods in Phase 2 / Phase 3

### Method 1
**Simple `VectorStoreIndex + QueryEngine`**

Use when:
- document is simple
- query is direct
- low cost / simple retrieval is enough

### Method 2
**`SentenceSplitter + VectorStoreIndex + QueryEngine`**

Use when:
- fixed chunk size with overlap is enough
- document is moderately structured

### Method 3
**`SemanticSplitterNodeParser + VectorStoreIndex + QueryEngine`**

Use when:
- document is semantically dense
- better chunk quality is needed

### Method 4
**`SemanticSplitterNodeParser + VectorIndexRetriever + QueryEngine`**

Use when:
- explicit vector retrieval control is needed

### Method 5
**`SemanticSplitterNodeParser + VectorIndexRetriever + BM25Retriever + QueryFusionRetriever + QueryEngine`**

Use when:
- query mixes semantic meaning and keyword matching
- hybrid retrieval gives better recall

---

## Design principles

### 1. Architecture first
Frameworks should fit the architecture — not replace it.

### 2. Plug-and-play design
Each layer is replaceable:
- chunker
- retriever
- LLM
- framework backend

### 3. Progressively agentic
The system evolves from:
- static RAG
- to advanced RAG
- to adaptive agentic RAG

### 4. Domain-agnostic
The same architecture can be applied to:
- resumes
- job descriptions
- contracts
- policies
- BFSI documents
- enterprise knowledge bases

---

## Demo use case

A simple public-safe demo:

- Pass a resume PDF
- Ask: **“What technical skills are mentioned in this resume relevant to AI and backend engineering?”**
- Agent selects the best pipeline automatically
- Returns the final answer

This demonstrates:
- dynamic pipeline selection
- real document understanding
- adaptive retrieval

---

## Roadmap

### Next planned improvements
- Validator agent
- RAGAS-based evaluation
- response faithfulness checks
- retry / fallback logic
- self-correcting agentic workflow
- multi-agent orchestration
- vector database integration (Qdrant / pgvector)

---

## Why this project matters

This project is not just “chat with PDF”.

It is an attempt to solve a real problem:

> **Different documents and different queries require different RAG pipelines.**

Instead of hardcoding one approach, this platform is built to **adapt**.

---

## Author

**Aakash Ravte**  
Senior Software Engineer | Agentic AI | GenAI | Intelligent Automation Systems

- LinkedIn: https://www.linkedin.com/in/aakash-ravte-543453189
- GitHub: https://github.com/A-B-Ravte
- Email: letsmailravte@gmail.com
