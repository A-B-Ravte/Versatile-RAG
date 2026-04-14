from crewai import Task, Crew, Process

from crewai_phase3.agents.selector_agent import build_selector_agent


def build_selection_task(doc_path: str, query: str, llm_name: str = "llama_cpp") -> Task:
    return Task(
        description=f"""
You must follow this process step by step:

1. Analyze the document and query.
2. Decide which method is best.
3. Explain WHY you selected that method.
4. Call the correct tool.
5. Return the final answer.

IMPORTANT:
- First clearly state: "Selected Method: X"
- Then explain reasoning
- Then call the tool

Query: {query}
Document Path: {doc_path}
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
        tracing=False,
    )

    return crew.kickoff()

