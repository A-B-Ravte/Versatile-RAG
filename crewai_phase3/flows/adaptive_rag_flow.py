from crewai import Task, Crew, Process

from crewai_phase3.agents.selector_agent import build_selector_agent


def build_selection_task(doc_path: str, query: str, llm_name: str = "llama_cpp") -> Task:
    return Task(
        description=f"""
You are given:
- Document path: {doc_path}
- Query: {query}
- Target LLM backend for pipeline execution: {llm_name}

Your job:
1. Reason about the document and query.
2. Choose the best pipeline tool.
3. Call exactly one pipeline tool.
4. Return the final answer from that tool.

Selection rules:
- Prefer Method 1 for simple, short, low-cost use cases.
- Prefer Method 2 when controlled chunking with overlap is enough.
- Prefer Method 3 for semantically dense documents where smart chunking matters.
- Prefer Method 4 when semantic chunking is needed and explicit vector retrieval control helps.
- Prefer Method 5 when the query likely needs both semantic retrieval and keyword matching.

Do not choose a more complex method unless needed.
""",
        expected_output="The final answer returned by the selected RAG pipeline.",
        agent=build_selector_agent(),
    )


def run_adaptive_rag(doc_path: str, query: str, llm_name: str = "llama_cpp"):
    task = build_selection_task(doc_path, query, llm_name)

    crew = Crew(
        agents=[build_selector_agent()],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff()

