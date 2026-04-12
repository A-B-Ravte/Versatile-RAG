from typing import List, Dict, Any
from prompt.base_prompt import BasePromptBuilder


class RAGPromptBuilder(BasePromptBuilder):
    def build(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_parts = []

        for item in retrieved_chunks:
            chunk = item["chunk"]
            page_no = chunk.metadata.page_no
            text = chunk.chunk_text

            context_parts.append(f"[Page {page_no}]\n{text}")

        context = "\n\n".join(context_parts)

        final_prompt = f"""
You are a helpful assistant.

Answer the user's question only using the context below.
If the answer is not present in the context, say:
"I could not find the answer in the document."

Always be factual and concise.

Context:
{context}

Question:
{query}

Answer:
"""
        return final_prompt.strip()