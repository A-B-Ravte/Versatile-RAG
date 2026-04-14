import os
from dotenv import load_dotenv

from crewai import Agent, LLM

from crewai_phase3.tools.rag_pipeline_tools import (
    Method1Tool,
    Method2Tool,
    Method3Tool,
    Method4Tool,
    Method5Tool,
)

load_dotenv()


def build_selector_agent() -> Agent:
    llm = LLM(
        model="gemini/gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.1,
    )

    return Agent(
        role="Adaptive RAG Pipeline Selector",
        goal=(
            "Choose the most cost-effective and accurate RAG pipeline method "
            "based on document characteristics and query type, then execute it."
        ),
        backstory=(
            "You are an expert retrieval architect. "
            "You understand chunking, vector search, BM25, hybrid fusion, and retrieval cost trade-offs. "
            "You always choose the simplest method that can answer well."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        tools=[
            Method1Tool(),
            Method2Tool(),
            Method3Tool(),
            Method4Tool(),
            Method5Tool(),
        ],
    )