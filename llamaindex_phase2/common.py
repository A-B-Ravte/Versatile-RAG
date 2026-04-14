import argparse
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

from llama_index.core import Settings
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.llama_cpp import LlamaCPP


LOCAL_MODEL_PATH = (
    "D:/aakash/Agentic AI/Versatile_RAG/models/"
    "LlammaLocalModel/qwen2.5-coder-7b-instruct.Q4_K_M.gguf"
)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args(default_query: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_path", required=True, help="Path to PDF document")
    parser.add_argument(
        "--llm",
        required=True,
        choices=["llama_cpp", "gemini"],
        help="LLM backend to use"
    )
    parser.add_argument(
        "--query",
        default=default_query,
        help="User query for the RAG pipeline"
    )
    return parser.parse_args()


def configure_embedding():
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def configure_llm(llm_name: str):
    if llm_name == "gemini":
        load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")

        Settings.llm = GoogleGenAI(
            model="models/gemini-2.5-flash",
            temperature=0.1
        )

    elif llm_name == "llama_cpp":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

        def messages_to_prompt(messages):
            messages = [{"role": m.role.value, "content": m.content} for m in messages]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        def completion_to_prompt(completion):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": completion}],
                tokenize=False,
                add_generation_prompt=True,
            )

        Settings.llm = LlamaCPP(
            model_path=LOCAL_MODEL_PATH,
            temperature=0.1,
            max_new_tokens=256,
            context_window=4096,
            generate_kwargs={
                "top_p": 0.9,
                "repeat_penalty": 1.2,
            },
            model_kwargs={
                "n_threads": 4,
                "n_ctx": 4096,
                "verbose": False,
            },
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )

    else:
        raise ValueError(f"Unsupported llm backend: {llm_name}")


def load_documents(doc_path: str):
    reader = PyMuPDFReader()
    return reader.load(file_path=doc_path)


def print_response(response):
    print("\n=== ANSWER ===\n")
    print(response)
    
    if hasattr(response, "source_nodes"):
        print("\n=== RETRIEVED NODES ===\n")
        for i, node in enumerate(response.source_nodes, start=1):
            page = node.metadata.get("page_label", "N/A")
            score = getattr(node, "score", None)
            score_str = f"{score:.3f}" if score is not None else "N/A"

            print(f"[{i}] Page: {page} | Score: {score_str}")
            print(node.text[:200])
            print("-" * 80)
            